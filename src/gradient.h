#pragma once

#include "VectorModel.h"
#include "mathUtils.h"
#include "stick_breaking.h"


#define in_offset(In, x, k, M, T) (In) + (x)*(M)*(T) + (k)*(M)

//assuming everything is indexed from 1 like in julia
static float inplace_update(float* In, float* Out,
	int64_t M, int64_t T, double* z,
	int64_t x,
	int32_t* path, int8_t* code, int64_t length,
	float* in_grad, float* out_grad,
	float lr, float sense_treshold) {

	//--x;

	float pr = 0;

	for (int k = 0; k < T; ++k)
		for (int i = 0; i < M; ++i)
			in_grad[k*M + i] = 0;

	for (int n = 0; n < length && code[n] != -1; ++n) {
		float* out = Out + path[n]*M;

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

auto in_place_update(VectorModel& vm, size_t x, size_t y, std::vector<double>& z,
	float lr, mdvector<float, 2>& in_grad, std::vector<float>& out_grad, 
	float sense_threshold)
{
	return inplace_update(vm.in.data(), vm.out.data(), vm.M(), vm.T(), z.data(), x, 
		&vm.path[{0, y}], &vm.code[{0, y}], vm.code.size()[0], 
		in_grad.data(), out_grad.data(), lr, sense_threshold);
}

auto var_init_z(const VectorModel& vm, int x, std::vector<double>& z)
{
	return expected_logpi(z, vm, x);
}


void update_z(const float* In, const float* Out,
	int64_t M, int64_t T, double* z,
	int64_t x,
	const int32_t* path, const int8_t* code, int64_t length) {

	//--x;

	for (int n = 0; n < length && code[n] != -1; ++n) {
		const float* out = Out + path[n] * M;

		for (int k = 0; k < T; ++k) {
			const float* in = in_offset(In, x, k, M, T);

			float f = 0;
			for (int i = 0; i < M; ++i)
				f += in[i] * out[i];

			z[k] += logsigmoid(f * (1 - 2 * code[n]));
		}
	}
}


auto var_update_z(const VectorModel& vm, size_t x, size_t y, 
	std::vector<double>& z, int num_meanings)
{
	return update_z(vm.in.data(), vm.out.data(), vm.M(), num_meanings, z.data(), x,
		&vm.path[{0, y}], &vm.code[{0, y}], vm.path.size()[0]);
}

auto var_update_counts(VectorModel& vm, size_t x, std::vector<double>& local_counts, double lr)
{
	for (size_t k = 0; k < vm.T(); ++k)
	{
		vm.counts[{k, x}] += lr * (local_counts[k] * vm.frequencies[x] - vm.counts[{k, x}]);
	}
}

void inplace_train_vectors(VectorModel& vm, const std::vector<size_t>& doc,
	int window_length, float start_lr, float total_words,
	std::vector<int64_t>& words_read, std::vector<float>& total_ll,
	int batch = 10000, bool context_cut = true, float sense_threshold = 1e-32)
{
	const size_t N = doc.size();
	mdvector<float, 2> in_grad{ {vm.M(), vm.T()} };
	std::vector<float> out_grad(vm.M());
	std::vector<double> z(vm.T());
	float senses = 0.f, max_senses = 0.f;

	std::uniform_int_distribution<> uid{ 0, window_length - 1 };
	std::mt19937_64 rg;

	for (size_t i = 0; i < N; ++i)
	{
		const auto& x = doc[i];
		float lr1 = std::max(start_lr * (1 - words_read[0] / (total_words + 1)), start_lr * 1e-4f);
		float lr2 = lr1;

		int random_reduce = context_cut ? uid(rg) : 0;
		int window = window_length - random_reduce;
		std::fill(z.begin(), z.end(), 0.f);

		float n_senses = var_init_z(vm, x, z);
		senses += n_senses;
		max_senses = std::max(max_senses, n_senses);

		size_t jBegin = 0, jEnd = N;
		if (i > window) jBegin = i - window;
		if (i + window < N) jEnd = i + window;

		for (auto j = jBegin; j < jEnd; ++j)
		{
			if (i == j) continue;
			var_update_z(vm, x, doc[j], z, vm.T());
			assert(isnormal(z[0]));
		}

		exp_normalize(z);

		for (auto j = jBegin; j < jEnd; ++j)
		{
			if (i == j) continue;
			const auto& y = doc[j];

			float ll = in_place_update(vm, x, y, z, lr1, in_grad, out_grad, 
				sense_threshold);
			assert(isnormal(ll));
			total_ll[1] += 1;
			total_ll[0] += (ll - total_ll[0]) / total_ll[1];
		}

		words_read[0] += 1;

		// variational update for q(pi_v)
		var_update_counts(vm, x, z, lr2);

		if (i % 1000 == 0)
		{
			float time_per_kword = batch / 1000.f;
			printf("%.2f%% %.4f %.4f %.4f %.2f/%.2f %.2f kwords/sec\n", 
				words_read[0] / (total_words / 100.f), total_ll[0], lr1, lr2, 
				(float)senses / (i + 1), max_senses, time_per_kword);
		}

		if (words_read[0] > total_words) break;
	}
}

