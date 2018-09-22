#pragma once

#include <iostream>
#include <vector>

template<class _Ty>
inline void writeToBinStream(std::ostream& os, const _Ty& v)
{
	static_assert(std::is_fundamental_v<_Ty>, "Only fundamental type can be written!");
	if (!os.write((const char*)&v, sizeof(_Ty))) throw std::ios_base::failure(std::string{ "writing type '" } +typeid(_Ty).name() + "' failed");
}

template<class _Ty>
inline void readFromBinStream(std::istream& is, _Ty& v)
{
	static_assert(std::is_fundamental_v<_Ty>, "Only fundamental type can be read!");
	if (!is.read((char*)&v, sizeof(_Ty))) throw std::ios_base::failure(std::string{ "reading type '" } +typeid(_Ty).name() + "' failed");
}

template<class _Ty>
inline _Ty readFromBinStream(std::istream& is)
{
	_Ty v;
	readFromBinStream(is, v);
	return v;
}


template<class _Ty1>
inline void writeToBinStream(std::ostream& os, const std::vector<_Ty1>& v)
{
	writeToBinStream<uint32_t>(os, v.size());
	for (auto& p : v)
	{
		writeToBinStream(os, p);
	}
}

template<class _Ty1>
inline void readFromBinStream(std::istream& is, std::vector<_Ty1>& v)
{
	size_t len = readFromBinStream<uint32_t>(is);
	v.clear();
	for (size_t i = 0; i < len; ++i)
	{
		v.emplace_back(readFromBinStream<_Ty1>(is));
	}
}
