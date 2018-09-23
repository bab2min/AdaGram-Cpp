#pragma once

#include <random>
#include <functional>
#include <mutex>
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

	std::vector<size_t> frequencies; // (V)
	Eigen::MatrixXf in; // (M, T * V)
	Eigen::MatrixXf inNormalized; // (M, T * V)
	Eigen::MatrixXf out; // (M, V)
	Eigen::MatrixXf counts; // (T, V)
	Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> code; // (MAX_CODELENGTH, V)
	Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic> path; // (MAX_CODELENGTH, V)
	size_t M; // dimension of word vector
	size_t T; // number of max prototype of word
	float alpha; // parameter for dirichlet process
	float d; // paramter for pitman-yor process
	float min_prob = 1e-3; // min probability of stick piece

	float sense_threshold = 1e-10;
	bool context_cut = true;

	size_t totalWords = 0;
	size_t procWords = 0, lastProcWords = 0;
	size_t totalLLCnt = 0;
	double totalLL = 0;

	//float alphaFreqWeight = -.67f;

	std::mt19937_64 rg;
	WordDictionary<> vocabs;

	Timer timer;

	std::mutex mtx;

	std::vector<HierarchicalOuput> buildHuffmanTree() const;
	void updateZ(Eigen::VectorXf& z, size_t x, size_t y) const;
	float inplaceUpdate(size_t x, size_t y, const Eigen::VectorXf& z, float lr);
	void updateCounts(size_t x, const Eigen::VectorXf& localCounts, float lr);
	std::pair<Eigen::VectorXf, size_t> getExpectedLogPi(size_t v) const;
	std::pair<Eigen::VectorXf, size_t> getExpectedPi(size_t v) const;
	void buildModel();
	void trainVectors(const uint32_t* ws, size_t N, size_t window_length, float start_lr, size_t threadId = 0);
	void updateNormalizedVector();
public:

	AdaGramModel(size_t _M = 100, size_t _T = 5, float _alpha = 1e-1, float _d = 0, size_t seed = std::random_device()())
		: M(_M), T(_T), alpha(_alpha), d(_d), rg(seed)
	{}

	AdaGramModel(AdaGramModel&& o)
		: M(o.M), T(o.T), alpha(o.alpha), d(o.d), rg(o.rg),
		min_prob(o.min_prob), sense_threshold(o.sense_threshold), context_cut(o.context_cut),
		vocabs(std::move(o.vocabs)), frequencies(std::move(o.frequencies)),
		in(std::move(o.in)), out(std::move(o.out)), counts(std::move(o.counts)),
		code(std::move(o.code)), path(std::move(o.path)), inNormalized(std::move(o.inNormalized))
	{
	}

	AdaGramModel& operator=(AdaGramModel&& o)
	{
		M = o.M;
		T = o.T;
		alpha = o.alpha;
		d = o.d;
		rg = o.rg;
		min_prob = o.min_prob;
		sense_threshold = o.sense_threshold;
		context_cut = o.context_cut;
		vocabs = std::move(o.vocabs);
		frequencies = std::move(o.frequencies);
		in = std::move(o.in);
		out = std::move(o.out);
		counts = std::move(o.counts);
		code = std::move(o.code);
		path = std::move(o.path);
		inNormalized = std::move(o.inNormalized);
		return *this;
	}

	template<class _Iter, class _Pred, class _Transform>
	void buildVocab(_Iter begin, _Iter end, size_t minCnt = 20, 
		_Pred test = [](const std::string&) { return true; }, _Transform trans = [](const std::string& t) { return t; })
	{
		WordDictionary<> rdict;
		std::vector<size_t> rfreqs;
		std::string word;
		for (; begin != end; ++begin)
		{
			if (!test(*begin)) continue;
			size_t id = rdict.getOrAdd(trans(*begin));
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

	void train(const std::function<std::vector<std::string>(size_t)>& reader, size_t numWorkers = 0,
		size_t window_length = 4, float start_lr = 0.025, size_t batchSents = 1000, size_t epochs = 1);

	void buildTrain(std::istream& is, size_t minCnt = 20, 
		const std::function<bool(const std::string&)>& test = [](const std::string&) { return true; }, 
		const std::function<std::string(const std::string&)>& trans = [](const std::string& t) { return t; },
		size_t numWorkers = 0, size_t window_length = 4, float start_lr = 0.025, size_t batchSents = 1000, size_t epochs = 1);

	std::pair<Eigen::VectorXf, size_t> getExpectedPi(const std::string& word) const;
	std::vector<std::tuple<std::string, size_t, float>> nearestNeighbors(const std::string& word, size_t s,
		size_t K = 10, float min_count = 1.f) const;
	std::vector<float> disambiguate(const std::string& word, const std::vector<std::string>& context,
		bool use_prior = true, float min_prob = 1e-3) const;

	const std::vector<std::string>& getVocabs() const
	{
		return vocabs.getKeys();
	}

	void saveModel(std::ostream& os) const;
	static AdaGramModel loadModel(std::istream& is);
};

