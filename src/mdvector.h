#pragma once

#include <array>
#include <vector>
template<class _Type, size_t _Dimension = 1>
class mdvector : public std::vector<_Type>
{
	std::array<size_t, _Dimension> dimension = { {0,} }, indexer;
public:
	mdvector(const std::array<size_t, _Dimension>& _dim = { {0,} })
	{
		resize(_dim);
	}

	void resize(const std::array<size_t, _Dimension>& _dim)
	{
		dimension = _dim;
		size_t l = 1;
		for (size_t i = 0; i < _Dimension; ++i)
		{
			indexer[i] = l;
			l *= dimension[i];
		}
		std::vector<_Type>::resize(l);
	}

	_Type& operator[](const std::array<size_t, _Dimension>& idx)
	{
		size_t ridx = 0;
		for (size_t i = 0; i < _Dimension; ++i) ridx += indexer[i] * idx[i];
		return std::vector<_Type>::operator[](ridx);
	}

	const _Type& operator[](const std::array<size_t, _Dimension>& idx) const
	{
		size_t ridx = 0;
		for (size_t i = 0; i < _Dimension; ++i) ridx += indexer[i] * idx[i];
		return std::vector<_Type>::operator[](ridx);
	}

	const std::array<size_t, _Dimension>& size() const
	{
		return dimension;
	}
};