#pragma once

#include "mathUtils.h"
#include "VectorModel.h"

float skip_gram(const float* In, const float* Out,
	int64_t M,
	const int32_t* path, const int8_t* code, int length) {

	float pr = 0;

	for (int n = 0; n < length && code[n] != -1; ++n) {
		const float* out = Out + path[n] * M;

		float f = 0;
		for (int i = 0; i < M; ++i)
			f += In[i] * out[i];

		pr += logsigmoid(f * (1 - 2 * code[n]));
	}

	return pr;
}

template<class _T1, class _T2>
float log_skip_gram(const VectorModel& vm, _T1 w, _T2 s, _T1 v)
{
	return skip_gram(&vm.in[{0, s, w}], &wm.out[0], wm.M(), &wm.path[{0, v}], &vm.code[{0, v}], vm.code.size(0));
}
