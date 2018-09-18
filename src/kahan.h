#pragma once

template<class _T>
struct Kahan
{
	T sum = 0, c = 0;

	void add(_T x)
	{
		auto y = x - c;
		auto t = sum + y;
		c = (t - sum) - y;
		sum = t;
	}
};

template<class _T>
struct MeanCounter
{
	int64_t n = 0;
	Kahan<_T> mean;

	const Kahan<_T>& add(_T x)
	{
		n++;
		mean.add((x - mean.sum) / n);
		return mean;
	}
};