#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <random>
#include <functional>
#include <mutex>
#include <memory>
#include <fstream>
#include <Eigen/Dense>
#include "dictionary.h"
#include "Timer.h"

namespace ag
{
	enum class Mode
	{
		hierarchical_softmax,
		negative_sampling
	};

	struct Record
	{
		std::vector<std::string> words;
	};

	using DataReader = std::function<Record()>;

	namespace util
	{
		struct AllPass
		{
			bool operator()(const std::string& o)
			{
				return true;
			}
		};

		struct NoTransform
		{
			std::string operator()(const std::string& o)
			{
				return o;
			}
		};

		template<typename _Filter, typename _Transformer = NoTransform>
		class FileLineReader
		{
			std::vector<std::string> files;
			size_t current_id = 0;
			std::unique_ptr<std::ifstream> ifs;
		public:
			FileLineReader(const std::vector<std::string>& _files)
				: files(_files), ifs(new std::ifstream{ files[0] })
			{
			}

			Record operator()()
			{
				Record rec;
				std::string line;
				_Filter filter;
				_Transformer transformer;
				while (current_id < files.size())
				{
					while (std::getline(*ifs, line))
					{
						std::istringstream iss{ line };
						std::istream_iterator<std::string> iBegin{ iss }, iEnd{};
						for (; iBegin != iEnd; ++iBegin)
						{
							if (filter(*iBegin)) rec.words.emplace_back(transformer(*iBegin));
						}
						if (rec.words.empty()) continue;
						return rec;
					}
					if (++current_id >= files.size()) break;
					ifs = std::unique_ptr<std::ifstream>(new std::ifstream{ files[current_id] });
				}
				return rec;
			}

			static std::function<DataReader()> generator(const std::string& _files)
			{
				return [=]()
				{
					auto r = std::make_shared<FileLineReader>(std::vector<std::string>{ _files });
					return [=]() { return r->operator()(); };
				};
			}

			static std::function<DataReader()> generator(const std::vector<std::string>& _files)
			{
				return [=]()
				{
					auto r = std::make_shared<FileLineReader>(_files);
					return [=]() { return r->operator()(); };
				};
			}

		};

		using BasicLineReader = FileLineReader<AllPass, NoTransform>;
	}

	struct ThreadLocalBase
	{
		std::mt19937_64 rg;
		Eigen::MatrixXf update_in, update_out;
		std::unordered_map<uint32_t, uint32_t> update_out_idx;
		std::unordered_set<uint32_t> update_out_idx_hash;
	};

	template<typename _Derived, typename _ThreadLocalData>
	class AdaGramBase
	{
	protected:
		struct Report
		{
			float ll = 0, avg_senses = 0;
			size_t proc_words = 0;
			size_t max_senses = 0;
		};

		std::vector<size_t> frequencies; // (V)
		Eigen::MatrixXf in; // (M, T * V)
		Eigen::MatrixXf inNormalized; // (M, T * V)
		Eigen::MatrixXf out; // (M, V)
		Eigen::MatrixXf counts; // (T, V)

		size_t M; // dimension of word vector
		size_t T; // number of max prototype of word
		float alpha; // parameter for dirichlet process
		float d; // paramter for pitman-yor process
		float subsampling;
		float min_prob = 1e-3; // min probability of stick piece

		float sense_threshold = 1e-10;
		bool context_cut = true;

		_ThreadLocalData globalData;
		WordDictionary<> vocabs;

		void updateCounts(size_t x, const Eigen::VectorXf& localCounts, float lr);
		std::pair<Eigen::VectorXf, size_t> getExpectedLogPi(size_t v) const;
		std::pair<Eigen::VectorXf, size_t> getExpectedPi(size_t v) const;
		void buildModel();
		template<bool _multi>
		Report trainVectors(const uint32_t* ws, size_t len, size_t window_length, float start_lr, float end_lr, 
			_ThreadLocalData& ld, size_t num_workers, std::mutex* mtx_in, std::mutex* mtx_out);
		void updateNormalizedVector();
	public:

		AdaGramBase(size_t _M = 100, size_t _T = 5, float _alpha = 1e-1, float _d = 0,
			float _subsampling = 1e-4, size_t seed = std::random_device()())
			: M(_M), T(_T), alpha(_alpha), d(_d), subsampling(_subsampling)
		{
			globalData.rg = std::mt19937_64{ seed };
		}

		void buildVocab(const std::function<DataReader()>& reader, size_t min_cnt = 10);

		void train(const std::function<DataReader()>& reader, size_t num_workers = 0,
			size_t window_length = 4, float start_lr = 0.025, float end_lr = 0.00025, 
			size_t batch_sents = 1000, size_t epochs = 1, size_t report = 100000);

		void buildTrain(const std::function<DataReader()>& reader, size_t min_cnt = 10,
			size_t num_workers = 0, size_t window_length = 4, float start_lr = 0.025, float end_lr = 0.00025, 
			size_t batch_sents = 1000, size_t epochs = 1);

		std::pair<Eigen::VectorXf, size_t> getExpectedPi(const std::string& word) const;

		std::vector<std::tuple<std::string, size_t, float>> nearestNeighbors(const std::string& word, size_t s,
			size_t K = 10, float min_count = 1.f) const;

		std::vector<float> disambiguate(const std::string& word, const std::vector<std::string>& context,
			bool use_prior = true, float min_prob = 1e-3) const;

		std::vector<std::tuple<std::string, size_t, float>> mostSimilar(
			const std::vector<std::pair<std::string, size_t>>& positive_words,
			const std::vector<std::pair<std::string, size_t>>& negative_words,
			size_t K = 10, float min_count = 1.f) const;

		const std::vector<std::string>& getVocabs() const
		{
			return vocabs.getKeys();
		}

		void saveModel(std::ostream& os) const;
		void loadModel(std::istream& is);
	};

	template<Mode _mode>
	class AdaGramModel
	{
	};

	template<>
	class AdaGramModel<Mode::hierarchical_softmax>
		: public AdaGramBase<AdaGramModel<Mode::hierarchical_softmax>, ThreadLocalBase>
	{
		using BaseClass = AdaGramBase<AdaGramModel<Mode::hierarchical_softmax>, ThreadLocalBase>;
		using ThreadLocalData = ThreadLocalBase;
		friend BaseClass;

		struct HuffmanResult
		{
			std::vector<int8_t> code;
			std::vector<uint32_t> path;
		};

		std::vector<HuffmanResult> buildHuffmanTree() const;

		Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> code; // (MAX_CODELENGTH, V)
		Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic> path; // (MAX_CODELENGTH, V)

		void _buildModel();
		void updateZ(size_t x, size_t y, Eigen::VectorXf& z) const;
		float inplaceUpdate(size_t x, size_t y, const Eigen::VectorXf& z, float lr);

		void initSharedForMulti(ThreadLocalData& ld, size_t window_length) const;
		void allocateCache(ThreadLocalData& ld, size_t y, size_t num_workers) const;
		float update(size_t x, size_t y, const Eigen::VectorXf& z, float lr, ThreadLocalData& data) const;
	public:
		using AdaGramBase<AdaGramModel<Mode::hierarchical_softmax>, ThreadLocalBase>::AdaGramBase;
	};

	template<>
	class AdaGramModel<Mode::negative_sampling>
		: public AdaGramBase<AdaGramModel<Mode::negative_sampling>, ThreadLocalBase>
	{
		using BaseClass = AdaGramBase<AdaGramModel<Mode::negative_sampling>, ThreadLocalBase>;
		using ThreadLocalData = ThreadLocalBase;
		friend BaseClass;

		std::discrete_distribution<uint32_t> unigramTable;
		size_t negativeSampleSize = 0;

		void _buildModel();
		void updateZ(size_t x, size_t y, Eigen::VectorXf& z, bool negative = false) const;
		float inplaceUpdate(size_t x, size_t y, const Eigen::VectorXf& z, float lr, bool negative = false);

		void initSharedForMulti(ThreadLocalData& ld, size_t window_length) const;
		void allocateCache(ThreadLocalData& ld, size_t y, size_t num_workers) const;
		float update(size_t x, size_t y, const Eigen::VectorXf& z, float lr, ThreadLocalData& data, bool negative = false) const;
	public:
		using AdaGramBase<AdaGramModel<Mode::negative_sampling>, ThreadLocalBase>::AdaGramBase;
	};
}
