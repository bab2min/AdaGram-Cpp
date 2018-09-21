#pragma once

#include <random>
#include <functional>
#include <Eigen/Dense>
#include "dictionary.h"
#include "Timer.h"

class AdaGramModel
{
	struct HierarchicalOuput
	{
		std::vector<int8_t> code;
		std::vector<uint32_t> path;
	};

	std::vector<size_t> frequencies;
	Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> code;
	Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic> path;
	Eigen::MatrixXf in, out, counts;
	size_t M; // dimension of word vector
	size_t T; // number of max prototype of word
	float alpha; // parameter for dirichlet process
	float d; // paramter for pitman-yor process
	float min_prob = 1e-3; // min probability of stick piece

	float sense_threshold = 1e-10;
	bool context_cut = true;

	size_t totalWords = 0;
	size_t procWords = 0;

	size_t totalLLCnt = 0;
	float totalLL = 0;

	std::mt19937_64 rg;
	WordDictionary<> vocabs;

	Timer timer;

	std::vector<HierarchicalOuput> buildHuffmanTree() const;
	void updateZ(Eigen::VectorXf& z, size_t x, size_t y) const;
	float inplaceUpdate(size_t x, size_t y, const Eigen::VectorXf& z, float lr);
	void updateCounts(size_t x, const Eigen::VectorXf& localCounts, float lr);
	std::pair<Eigen::VectorXf, size_t> getExpectedLogPi(size_t v) const;
	std::pair<Eigen::VectorXf, size_t> getExpectedPi(size_t v) const;
	void buildModel();
	void trainVectors(const uint32_t* ws, size_t N, size_t window_length, float start_lr);
public:

	AdaGramModel(size_t _M = 100, size_t _T = 5, float _alpha = 1e-1, float _d = 0, size_t seed = std::random_device()())
		: M(_M), T(_T), alpha(_alpha), d(_d), rg(seed)
	{}

	template<class _Iter, class _Pred>
	void buildVocab(_Iter begin, _Iter end, size_t minCnt = 20, _Pred test = [](const std::string&) {return true; })
	{
		WordDictionary<> rdict;
		std::vector<size_t> rfreqs;
		std::string word;
		for (; begin != end; ++begin)
		{
			if (!test(*begin)) continue;
			size_t id = rdict.getOrAdd(*begin);
			if (id >= rfreqs.size()) rfreqs.resize(id + 1);
			rfreqs[id]++;
		}

		for (size_t i = 0; i < rdict.size(); ++i)
		{
			if (rfreqs[i] < minCnt) continue;
			frequencies.emplace_back(rfreqs[i]);
			vocabs.add(rdict.getStr(i));
		}
		buildModel();
	}

	void train(const std::function<std::vector<std::string>(size_t, double&)>& reader, size_t window_length = 4, float start_lr = 0.025, size_t batch = 1000, size_t epoch = 1);

	void train(std::istream& is, size_t minCnt = 20);
	std::pair<Eigen::VectorXf, size_t> getExpectedPi(const std::string& word) const;
	std::vector<std::tuple<std::string, size_t, float>> nearestNeighbors(const std::string& word, size_t s, size_t K = 10, float min_count = 1.f) const;
};

