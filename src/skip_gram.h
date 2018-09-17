#pragma once

#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <float.h>

inline float sigmoid(float x) 
{
	return 1 / (1 + exp(-x));
}

inline float logsigmoid(float x) 
{
	return -log(1 + exp(-x));
}

#define in_offset(In, x, k, M, T) (In) + (x)*(M)*(T) + (k)*(M)

//assuming everything is indexed from 1 like in julia
float inplace_update(float* In, float* Out,
	int64_t M, int64_t T, double* z,
	int64_t x,
	int32_t* path, int8_t* code, int64_t length,
	float* in_grad, float* out_grad,
	float lr, float sense_treshold) {

	--x;

	float pr = 0;

	for (int k = 0; k < T; ++k)
		for (int i = 0; i < M; ++i)
			in_grad[k*M + i] = 0;

	for (int n = 0; n < length && code[n] != -1; ++n) {
		float* out = Out + (path[n] - 1)*M;

		for (int i = 0; i < M; ++i)
			out_grad[i] = 0;

		for (int k = 0; k < T; ++k) {
			if (z[k] < sense_treshold) continue;

			float* in = in_offset(In, x, k, M, T);

			float f = 0;
			for (int i = 0; i < M; ++i)
				f += in[i] * out[i];

			pr += z[k] * logsigmoid(f * (1 - 2 * code[n]));

			float d = 1 - code[n] - sigmoid(f);
			float g = z[k] * lr * d;

			for (int i = 0; i < M; ++i) {
				in_grad[k*M + i] += g * out[i];
				out_grad[i] += g * in[i];
			}
		}

		for (int i = 0; i < M; ++i)
			out[i] += out_grad[i];
	}

	for (int k = 0; k < T; ++k) {
		if (z[k] < sense_treshold) continue;
		float* in = in_offset(In, x, k, M, T);
		for (int i = 0; i < M; ++i)
			in[i] += in_grad[k*M + i];
	}

	return pr;
}

float skip_gram(float* In, float* Out,
	int64_t M,
	int32_t* path, int8_t* code, int length) {

	float pr = 0;

	for (int n = 0; n < length && code[n] != -1; ++n) {
		float* out = Out + (path[n] - 1)*M;

		float f = 0;
		for (int i = 0; i < M; ++i)
			f += In[i] * out[i];

		pr += logsigmoid(f * (1 - 2 * code[n]));
	}

	return pr;
}

void update_z(float* In, float* Out,
	int64_t M, int64_t T, double* z,
	int64_t x,
	int32_t* path, int8_t* code, int64_t length) {

	--x;

	for (int n = 0; n < length && code[n] != -1; ++n) {
		float* out = Out + (path[n] - 1)*M;

		for (int k = 0; k < T; ++k) {
			float* in = in_offset(In, x, k, M, T);

			float f = 0;
			for (int i = 0; i < M; ++i)
				f += in[i] * out[i];

			z[k] += logsigmoid(f * (1 - 2 * code[n]));
		}
	}
}


template<class _T1, class _T2>
float log_skip_gram(const VectorModel& vm, _T1 w, _T2 s, _T1 v)
{
	return skip_gram(&vm.in[{0, s, w}], &wm.out[0], wm.M(), &wm.path[{0, v}], &vm.code[{0, v}], vm.code.size(0));
}
