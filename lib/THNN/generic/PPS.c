#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/PPS.c"
#else

void THNN_(PPS_updateOutput)(
    THNNState *state,
    THTensor *input,
    THTensor *output,
    int order)
{
    THTensor_(resizeAs)(output, input);

    TH_TENSOR_APPLY2(real, output, real, input,
        real s = 1./(1.+ exp(- *input_data));
        *output_data = s;
        if (order > 0) {
            s *= s;
        }

        if (order == 1) {
            *output_data += -s;
        } else if (order == 2) {
            *output_data += -3*s;
            s *= s;
            *output_data += 2*s;
        } else if (order == 3) {
            *output_data += -7*s;
            s *= s;
            *output_data += 12*s;
            s *= s;
            *output_data += -6*s;
        } else if (order == 4) {
            *output_data += -15*s;
            s *= s;
            *output_data += 50*s;
            s *= s;
            *output_data += -60*s;
            s *= s;
            *output_data += 24*s;
        } else if (order == 5) {
            *output_data += -31*s;
            s *= s;
            *output_data += 180*s;
            s *= s;
            *output_data += -390*s;
            s *= s;
            *output_data += 360*s;
            s *= s;
            *output_data += -120*s;
        }
    );

}

void THNN_(PPS_updateGradInput)(
    THNNState *state,
    THTensor *input,
    THTensor *gradOutput,
    THTensor *gradInput,
    THTensor *output,
    int order)
{
    THTensor_(resizeAs)(gradInput, output);
    TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,
        real s = 1./(1.+ exp(- *input_data));
        real derivative = s;
        s *= s;
        if (order == 1) {
            derivative += -s;
        } else if (order == 2) {
            derivative += -3*s;
            s *= s;
            derivative += 2*s;
        } else if (order == 3) {
            derivative += -7*s;
            s *= s;
            derivative += 12*s;
            s *= s;
            derivative += -6*s;
        } else if (order == 4) {
            derivative += -15*s;
            s *= s;
            derivative += 50*s;
            s *= s;
            derivative += -60*s;
            s *= s;
            derivative += 24*s;
        } else if (order == 5) {
            derivative += -31*s;
            s *= s;
            derivative += 180*s;
            s *= s;
            derivative += -390*s;
            s *= s;
            derivative += 360*s;
            s *= s;
            derivative += -120*s;
        }
        *gradInput_data = *gradOutput_data * derivative;
    );
}


#endif
