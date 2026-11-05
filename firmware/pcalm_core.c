/*
 * Thought experiment: one SpiNNaker2 processing element running PC-ALM.
 *
 * This is not a chip image. It does not include s2app.h, it is not built by
 * the py-spinnaker2 CMake flow, and PcalmBrian2Backend does not load it.
 * The stock application it is modeled on is
 * lif_neuron/main.c in the py-spinnaker2 runtime:
 *
 *   packet_received_callback   one multicast key, one synaptic row
 *   timer_callback             drain the NoC, then neuron_do_timestep_update
 *
 * A spike still only contributes one term of W h_prev. The residual, the
 * combined signal c = λ + ρ r, the activity step, and λ ← λ + α r run in the
 * timer tick, after that dot product is finished. That is the same order as
 * forward_sweep in lpc_s2/algorithm.py. The weight step stays off this core:
 * the Python schedule applies it only after inference returns.
 *
 * The host check at the bottom compiles on the machine that runs the demo:
 *
 *   cc -DPCALM_HOST_CHECK -O2 -o /tmp/pcalm_core firmware/pcalm_core.c -lm
 *   /tmp/pcalm_core
 *
 * On a Cortex-M4F the same loops would use float and the toolchain's expf.
 * The check uses double so it can sit next to the Python identity chain.
 */

#include <math.h>
#include <stdint.h>
#include <string.h>

#define PCALM_MAX_UNITS 8
#define PCALM_MAX_HIDDEN 2

enum {
    PCALM_IDENTITY = 0,
    PCALM_TANH = 1,
    PCALM_RELU = 2
};

typedef struct {
    uint32_t n_hidden;
    uint32_t width[PCALM_MAX_HIDDEN + 1]; /* width[0] is the clamped input */
    uint32_t readout_n;
    uint32_t T;
    int activation;
    int use_spike_dot; /* layer 0 dot product arrives as packets, not DTCM */
    double eta_h;
    double alpha;
    double rho;
    double x[PCALM_MAX_UNITS];
    double y[PCALM_MAX_UNITS];
    double h[PCALM_MAX_HIDDEN][PCALM_MAX_UNITS];
    double lambda[PCALM_MAX_HIDDEN][PCALM_MAX_UNITS];
    double grad[PCALM_MAX_HIDDEN][PCALM_MAX_UNITS];
    double hidden_w[PCALM_MAX_HIDDEN][PCALM_MAX_UNITS][PCALM_MAX_UNITS];
    double readout_w[PCALM_MAX_UNITS][PCALM_MAX_UNITS];
    uint32_t sweep; /* 0 = feasible init, then T-1 primal-dual, then one primal */
} pcalm_core_t;

static pcalm_core_t core;

/* Filled by the previous core before it emits a key. A stock spike packet
   carries the source index, not the real-valued activity. */
static double h_prev_mailbox[PCALM_MAX_UNITS];
static double spike_pre[PCALM_MAX_UNITS];
static uint32_t pre_terms;
static uint32_t spike_queue[PCALM_MAX_UNITS];
static uint32_t spike_queue_count;
static uint32_t systicks;

static double pcalm_sigma(double pre)
{
    if (core.activation == PCALM_TANH) {
        return tanh(pre);
    }
    if (core.activation == PCALM_RELU) {
        return pre > 0.0 ? pre : 0.0;
    }
    return pre;
}

static double pcalm_sigma_prime(double pre)
{
    if (core.activation == PCALM_TANH) {
        double value = tanh(pre);
        return 1.0 - value * value;
    }
    if (core.activation == PCALM_RELU) {
        return pre > 0.0 ? 1.0 : 0.0;
    }
    return 1.0;
}

static double pcalm_dot(const double *row, const double *vector, uint32_t length)
{
    double sum = 0.0;
    uint32_t index;

    for (index = 0; index < length; index++) {
        sum += row[index] * vector[index];
    }
    return sum;
}

static const double *pcalm_presynaptic(uint32_t layer)
{
    if (layer == 0) {
        return core.x;
    }
    return core.h[layer - 1];
}

/* Gradients from the state at the start of the sweep. Layer 0's spike
   accumulator is not read here; a partial packet sum must not enter c. */
static void pcalm_activity_gradients(void)
{
    double residual[PCALM_MAX_HIDDEN][PCALM_MAX_UNITS];
    double pre[PCALM_MAX_HIDDEN][PCALM_MAX_UNITS];
    double prediction[PCALM_MAX_UNITS];
    uint32_t layer;
    uint32_t unit;
    uint32_t top = core.n_hidden - 1;

    for (layer = 0; layer < core.n_hidden; layer++) {
        const double *previous = pcalm_presynaptic(layer);
        for (unit = 0; unit < core.width[layer + 1]; unit++) {
            pre[layer][unit] = pcalm_dot(
                core.hidden_w[layer][unit], previous, core.width[layer]);
            residual[layer][unit] =
                core.h[layer][unit] - pcalm_sigma(pre[layer][unit]);
        }
    }
    for (unit = 0; unit < core.readout_n; unit++) {
        prediction[unit] = pcalm_dot(
            core.readout_w[unit], core.h[top], core.width[top + 1]);
    }
    for (layer = 0; layer < core.n_hidden; layer++) {
        for (unit = 0; unit < core.width[layer + 1]; unit++) {
            double combined =
                core.lambda[layer][unit] + core.rho * residual[layer][unit];
            double pulled = 0.0;
            uint32_t next;

            if (layer + 1 < core.n_hidden) {
                for (next = 0; next < core.width[layer + 2]; next++) {
                    double slope = pcalm_sigma_prime(pre[layer + 1][next]);
                    double next_combined = core.lambda[layer + 1][next]
                        + core.rho * residual[layer + 1][next];
                    pulled += core.hidden_w[layer + 1][next][unit] * slope * next_combined;
                }
            } else {
                for (next = 0; next < core.readout_n; next++) {
                    double supervised = core.y[next] - prediction[next];
                    pulled += core.readout_w[next][unit] * supervised;
                }
            }
            core.grad[layer][unit] = combined - pulled;
        }
    }
}

/* One input-to-output sweep. λ for layer i is updated before layer i+1. */
static void pcalm_forward_sweep(int update_activities, int update_multipliers)
{
    double committed[PCALM_MAX_UNITS];
    uint32_t committed_n = core.width[0];
    uint32_t layer;
    uint32_t unit;

    if (update_activities) {
        pcalm_activity_gradients();
    }
    memcpy(committed, core.x, sizeof(double) * committed_n);

    for (layer = 0; layer < core.n_hidden; layer++) {
        uint32_t width = core.width[layer + 1];
        double next_committed[PCALM_MAX_UNITS];

        for (unit = 0; unit < width; unit++) {
            double activated;
            double activity;

            if (layer == 0 && core.use_spike_dot) {
                activated = pcalm_sigma(spike_pre[unit]);
            } else {
                activated = pcalm_sigma(pcalm_dot(
                    core.hidden_w[layer][unit], committed, committed_n));
            }
            if (update_activities) {
                activity = core.h[layer][unit] - core.eta_h * core.grad[layer][unit];
            } else {
                activity = activated;
            }
            /* Same residual and dual step as algorithm.py: r = h − σ(W h_prev). */
            if (update_multipliers) {
                double residual = activity - activated;
                core.lambda[layer][unit] += core.alpha * residual;
            }
            core.h[layer][unit] = activity;
            next_committed[unit] = activity;
        }
        memcpy(committed, next_committed, sizeof(double) * width);
        committed_n = width;
    }
}

/*
 * Stock signature: the NoC delivers a routing key. The callback adds one
 * column of W_1 h_prev into the layer-0 accumulator and returns. It does
 * not form r, c, or λ. A partial sum is not a residual.
 */
static void packet_received_callback(uint32_t key)
{
    uint32_t source = key;
    uint32_t unit;

    if (source >= core.width[0] || core.n_hidden == 0) {
        return;
    }
    for (unit = 0; unit < core.width[1]; unit++) {
        spike_pre[unit] += core.hidden_w[0][unit][source] * h_prev_mailbox[source];
    }
    pre_terms++;
}

static void receive_spikes(void (*callback)(uint32_t key))
{
    uint32_t index;

    for (index = 0; index < spike_queue_count; index++) {
        callback(spike_queue[index]);
    }
    spike_queue_count = 0;
}

static void pcalm_clear_spike_dot(void)
{
    memset(spike_pre, 0, sizeof(spike_pre));
    pre_terms = 0;
}

/*
 * Timer tick. Same skeleton as lif_neuron: drain packets, then the state
 * update that used to be neuron_do_timestep_update. One tick is one forward
 * sweep. The Lagrangian step is skipped while layer 0's dot product is short.
 */
void timer_callback(void)
{
    int update_activities;
    int update_multipliers;

    systicks++;
    receive_spikes(packet_received_callback);
    if (core.use_spike_dot && pre_terms != core.width[0]) {
        return;
    }
    if (core.sweep > core.T) {
        return;
    }

    update_activities = core.sweep > 0;
    update_multipliers = core.sweep < core.T;
    pcalm_forward_sweep(update_activities, update_multipliers);
    core.sweep++;
    pcalm_clear_spike_dot();
}

void pcalm_reset(const pcalm_core_t *initial)
{
    core = *initial;
    core.sweep = 0;
    systicks = 0;
    spike_queue_count = 0;
    memset(h_prev_mailbox, 0, sizeof(h_prev_mailbox));
    pcalm_clear_spike_dot();
}

void pcalm_set_mailbox(uint32_t source, double value)
{
    if (source < PCALM_MAX_UNITS) {
        h_prev_mailbox[source] = value;
    }
}

void pcalm_push_spike(uint32_t key)
{
    if (spike_queue_count < PCALM_MAX_UNITS) {
        spike_queue[spike_queue_count++] = key;
    }
}

double pcalm_hidden(uint32_t layer, uint32_t unit)
{
    return core.h[layer][unit];
}

double pcalm_lambda(uint32_t layer, uint32_t unit)
{
    return core.lambda[layer][unit];
}

uint32_t pcalm_sweep(void)
{
    return core.sweep;
}

#ifdef PCALM_HOST_CHECK

#include <stdio.h>

static int close_to(double value, double expected)
{
    double gap = value - expected;
    if (gap < 0.0) {
        gap = -gap;
    }
    return gap < 1e-9;
}

/* Figure 1 scalar chain, T = 2: one primal-dual sweep writes λ = α r. */
static int check_local_sweep(void)
{
    pcalm_core_t initial;
    int tick;

    memset(&initial, 0, sizeof(initial));
    initial.n_hidden = 1;
    initial.width[0] = 1;
    initial.width[1] = 1;
    initial.readout_n = 1;
    initial.T = 2;
    initial.activation = PCALM_IDENTITY;
    initial.eta_h = 0.05;
    initial.alpha = 0.4;
    initial.rho = 1.0;
    initial.x[0] = 1.0;
    initial.y[0] = 1.0;
    initial.hidden_w[0][0][0] = 0.5;
    initial.readout_w[0][0] = 1.5;
    pcalm_reset(&initial);

    for (tick = 0; tick < 3; tick++) {
        timer_callback();
    }
    /* h after the primal-dual sweep is 0.51875 and λ = 0.4 * 0.01875.
       The third tick is the final primal sweep and must leave λ alone. */
    if (!close_to(pcalm_lambda(0, 0), 0.0075)) {
        fprintf(stderr, "local sweep λ = %.12f, expected 0.0075\n", pcalm_lambda(0, 0));
        return 1;
    }
    if (pcalm_sweep() != 3) {
        fprintf(stderr, "local sweep finished on sweep %u\n", pcalm_sweep());
        return 1;
    }
    return 0;
}

/* No complete dot product: the timer tick must not move λ. */
static int check_partial_spike_holds_lambda(void)
{
    pcalm_core_t initial;

    memset(&initial, 0, sizeof(initial));
    initial.n_hidden = 1;
    initial.width[0] = 1;
    initial.width[1] = 1;
    initial.readout_n = 1;
    initial.T = 2;
    initial.activation = PCALM_IDENTITY;
    initial.use_spike_dot = 1;
    initial.eta_h = 0.05;
    initial.alpha = 0.4;
    initial.rho = 1.0;
    initial.x[0] = 1.0;
    initial.y[0] = 1.0;
    initial.hidden_w[0][0][0] = 0.5;
    initial.readout_w[0][0] = 1.5;
    pcalm_reset(&initial);

    timer_callback();
    if (pcalm_sweep() != 0 || !close_to(pcalm_lambda(0, 0), 0.0)) {
        fprintf(stderr, "partial spike updated the sweep\n");
        return 1;
    }

    pcalm_set_mailbox(0, 1.0);
    pcalm_push_spike(0);
    timer_callback();
    if (pcalm_sweep() != 1 || !close_to(pcalm_hidden(0, 0), 0.5)) {
        fprintf(stderr, "spike dot init h = %.12f\n", pcalm_hidden(0, 0));
        return 1;
    }
    if (!close_to(pcalm_lambda(0, 0), 0.0)) {
        fprintf(stderr, "init residual should leave λ at 0\n");
        return 1;
    }
    return 0;
}

int main(void)
{
    if (check_local_sweep() != 0) {
        return 1;
    }
    if (check_partial_spike_holds_lambda() != 0) {
        return 1;
    }
    printf("pcalm core sketch: lambda stays in the timer tick\n");
    return 0;
}

#endif
