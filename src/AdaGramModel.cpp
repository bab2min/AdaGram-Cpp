#include <numeric>
#include <iostream>
#include <iterator>
#include <unordered_set>
#include <cassert>

#include "AdaGramModel.h"
#include "mathUtils.h"
#include "IOUtils.h"
#include "ThreadPool.h"

using namespace std;
using namespace Eigen;
using namespace ag;


template<typename _Derived, typename _ThreadLocalData>
void AdaGramBase<_Derived, _ThreadLocalData>::buildModel()
{
	const size_t V = vocabs.size();
	// allocate & initialize model
	in = MatrixXf::Random(M, T * V) * (.5f / M);
	out = MatrixXf::Random(M, V) * (.5f / M);
	counts = MatrixXf::Zero(T, V);
	for (size_t v = 0; v < V; ++v) counts(0, v) = frequencies[v];
	
	static_cast<_Derived*>(this)->_buildModel();
}


template<typename _Derived, typename _ThreadLocalData>
pair<VectorXf, size_t> AdaGramBase<_Derived, _ThreadLocalData>::getExpectedLogPi(size_t v) const
{
	float vAlpha = alpha; // *pow(log(frequencies[v]), alphaFreqWeight);
	VectorXf pi = counts.col(v);
	size_t senses = 0;
	float r = 0.f, x = 1.f;
	double ts = pi.sum();
	for (size_t k = 0; k < T - 1; ++k)
	{
		ts = max(ts - pi[k], 0.);
		float a = 1 + pi[k] - d, b = vAlpha + (k + 1) * d + ts;
		pi[k] = meanlog_beta(a, b) + r;
		r += meanlog_mirror(a, b);

		float pi_k = mean_beta(a, b) * x;
		x = max(x - pi_k, 0.f);
		if (pi_k >= min_prob) senses++;
	}
	pi[T - 1] = r;
	if (x >= min_prob) senses++;
	return make_pair(move(pi), senses);

}

template<typename _Derived, typename _ThreadLocalData>
pair<VectorXf, size_t> AdaGramBase<_Derived, _ThreadLocalData>::getExpectedPi(size_t v) const
{
	float vAlpha = alpha; // *pow(log(frequencies[v]), alphaFreqWeight);
	VectorXf pi(T);
	size_t senses = 0;
	float r = 1.f;
	float ts = counts.col(v).sum();
	for (size_t k = 0; k < T - 1; ++k)
	{
		ts = max(ts - counts(k, v), 0.f);
		float a = 1 + counts(k, v) - d, b = vAlpha + (k + 1) * d + ts;
		pi[k] = mean_beta(a, b) * r;
		if (pi[k] >= min_prob) senses++;
		r = max(r - pi[k], 0.f);
	}
	pi[T - 1] = r;
	if (r >= min_prob) senses++;
	return make_pair(move(pi), senses);
}


template<typename _Derived, typename _ThreadLocalData>
void AdaGramBase<_Derived, _ThreadLocalData>::updateCounts(size_t x, const VectorXf & localCounts, float lr)
{
	counts.col(x) += lr * (localCounts * frequencies[x] - counts.col(x));
}

inline void softmax_inplace(VectorXf& z)
{
	z = (z.array() - z.maxCoeff()).exp();
	z /= z.sum();
}

template<typename _Derived, typename _ThreadLocalData>
template<bool _multi>
auto AdaGramBase<_Derived, _ThreadLocalData>::trainVectors(const word_id_t * ws, 
	size_t len, size_t window_length, 
	float start_lr, float end_lr,
	_ThreadLocalData& ld, size_t num_workers, 
	mt19937_64& rg, mutex* mtx_in, mutex* mtx_out) -> Report
{
	Report ret;
	size_t ll_cnt = 0;
	uniform_int_distribution<size_t> uid{ 0, window_length > 2 ? window_length - 2 : 0 };

	for (size_t i = 0; i < len; ++i)
	{
		const auto& x = ws[i];
		float lr = max(start_lr + (end_lr - start_lr) * i / len, start_lr * 1e-4f);

		int random_reduce = context_cut ? uid(rg) : 0;
		int window = window_length - random_reduce;
		size_t jBegin = 0, jEnd = len - 1;
		if (i > window) jBegin = i - window;
		if (i + window < len) jEnd = i + window;

		// updating z, which represents probabilities of each sense
		auto t = getExpectedLogPi(x);
		VectorXf& z = t.first;			
		ret.avg_senses += t.second;
		ret.max_senses = max(ret.max_senses, t.second);

		MatrixXf fs(T * static_cast<_Derived*>(this)->getFWidth(), jEnd - jBegin + 1);
		for (auto j = jBegin; j <= jEnd; ++j)
		{
			if (i == j) continue;
			static_cast<_Derived*>(this)->template updateZ<_multi>(ld, rg, num_workers, x, ws[j], z, fs.col(j - jBegin).data());
			assert(isfinite(z[0]));
		}

		softmax_inplace(z);
		
		// update in, out vector
		for (auto j = jBegin; j <= jEnd; ++j)
		{
			if (i == j) continue;
			float ll;
			if(_multi) ll = static_cast<_Derived*>(this)->update(ld, x, ws[j], z, fs.col(j - jBegin).data(), lr);
			else ll = static_cast<_Derived*>(this)->inplaceUpdate(ld, x, ws[j], z, fs.col(j - jBegin).data(), lr);
			assert(isfinite(ll));
			ll_cnt++;
			ret.ll += (ll - ret.ll) / ll_cnt;
		}

		if(_multi)
		{
			{
				lock_guard<mutex> lock{mtx_in[x % num_workers]};
				// variational update for q(pi_v)
				updateCounts(x, z, lr);	
				in.block(0, x * T, M, T) += ld.update_in;
			}

			for(size_t hash : ld.update_out_idx_hash)
			{
				lock_guard<mutex> lock{mtx_out[hash]};
				for(auto& p : ld.update_out_idx)
				{
					if(p.first % num_workers != hash) continue;
					out.col(p.first) += ld.update_out.col(p.second);
				}
			}

			ld.update_in.setZero();
			ld.update_out.setZero();
			ld.update_out_idx.clear();
			ld.update_out_idx_hash.clear();
		}
		else
		{
			// variational update for q(pi_v)
			updateCounts(x, z, lr);
		}
	}
	ret.proc_words = len;
	ret.avg_senses /= len;
	return ret;
}


template<typename _Derived, typename _ThreadLocalData>
void AdaGramBase<_Derived, _ThreadLocalData>::updateNormalizedVector()
{
	inNormalized = MatrixXf::Zero(in.rows(), in.cols());
	for (size_t i = 0; i < T * vocabs.size(); ++i)
	{
		inNormalized.col(i) = in.col(i).normalized();
	}
}

template<typename _Derived, typename _ThreadLocalData>
void AdaGramBase<_Derived, _ThreadLocalData>::buildVocab(const std::function<DataReader()>& reader, size_t min_cnt)
{
	WordDictionary<> rdict;
	std::vector<size_t> rfreqs;
	std::string word;
	auto rr = reader();
	while(1)
	{
		auto rec = rr();
		if (rec.words.empty()) break;
		for (auto& w : rec.words)
		{
			size_t id = rdict.getOrAdd(w);
			if (id >= rfreqs.size()) rfreqs.resize(id + 1);
			rfreqs[id]++;
		}
	}

	vector<pair<size_t, string>> pair_to_sort;

	for (size_t i = 0; i < rdict.size(); ++i)
	{
		if (rfreqs[i] < min_cnt) continue;
		pair_to_sort.emplace_back(rfreqs[i], rdict.getStr(i));
	}

	sort(pair_to_sort.rbegin(), pair_to_sort.rend());

	for (auto& p : pair_to_sort)
	{
		frequencies.emplace_back(p.first);
		vocabs.add(p.second);
	}
	buildModel();
}

template<typename _Derived, typename _ThreadLocalData>
void AdaGramBase<_Derived, _ThreadLocalData>::train(const function<DataReader()>& reader, 
	size_t num_workers, size_t window_length, float start_lr, float end_lr, 
	size_t batch, size_t epoch, size_t report)
{
	if (!num_workers) num_workers = thread::hardware_concurrency();
	ThreadPool workers{ num_workers };
	vector<_ThreadLocalData> ld;
	vector<mt19937_64> rgs;
	unique_ptr<mutex[]> mtx_in, mtx_out;
	if (num_workers > 1)
	{
		ld.resize(num_workers, static_cast<_Derived*>(this)->createLocalData(window_length));
		for (size_t i = 0; i < num_workers; ++i) rgs.emplace_back(rg());
		mtx_in = unique_ptr<mutex[]>(new mutex[num_workers]);
		mtx_out = unique_ptr<mutex[]>(new mutex[num_workers]);
	}
	else
	{
		globalData = static_cast<_Derived*>(this)->createLocalData(window_length);
	}

	double avg_ll = 0, avg_senses = 0;
	size_t totW = accumulate(frequencies.begin(), frequencies.end(), 0);
	size_t total_word_cnt = epoch * totW;
	size_t proc_word_cnt = 0, report_word_cnt = 0, max_senses = 0, skip_cnt = 0;
	vector<vector<word_id_t>> collections;

	const auto& getLR = [&]()
	{
		return start_lr + (end_lr - start_lr) * proc_word_cnt / total_word_cnt;
	};

	const auto& updateStat = [&](const Report& result, size_t collection_cnt)
	{
		size_t bf = proc_word_cnt;
		proc_word_cnt += result.proc_words
			+ skip_cnt * collection_cnt / collections.size() - skip_cnt * (collection_cnt - 1) / collections.size();
		report_word_cnt += result.proc_words;
		avg_ll += (result.ll - avg_ll) * result.proc_words / report_word_cnt;
		avg_senses += (result.avg_senses - avg_senses) * result.proc_words / report_word_cnt;
		max_senses = max(max_senses, result.max_senses);

		if (report && bf / report < proc_word_cnt / report)
		{
			fprintf(stderr, "%.2f%% ll:%4.4f, avg_senses:%3.3f, max_senses:%zd\n",
				100.f * proc_word_cnt / total_word_cnt, avg_ll, avg_senses, max_senses);
			report_word_cnt = 0;
			avg_ll = 0;
			avg_senses = 0;
			max_senses = 0;
		}
	};

	const auto& procCollection = [&]()
	{
		if (collections.empty()) return;
		shuffle(collections.begin(), collections.end(), rg);
		if (num_workers > 1)
		{
			vector<future<Report>> futures;
			futures.reserve(collections.size());
			for (auto& d : collections)
			{
				futures.emplace_back(workers.enqueue([&](size_t thread_id)
				{
					return trainVectors<true>(d.data(), d.size(), window_length, getLR(), getLR(), ld[thread_id],
						num_workers, rgs[thread_id], mtx_in.get(), mtx_out.get());
				}));
			}
			size_t collection_cnt = 0;
			for (auto& f : futures)
			{
				Report result = f.get();
				collection_cnt++;
				updateStat(result, ++collection_cnt);
			}
		}
		else
		{
			size_t collection_cnt = 0;
			for (auto& d : collections)
			{
				Report result = trainVectors<false>(d.data(), d.size(), window_length, getLR(), getLR(), globalData,
					1, rg, nullptr, nullptr);
				updateStat(result, ++collection_cnt);
			}
		}
		collections.clear();
		skip_cnt = 0;
	};

	for (size_t e = 0; e < epoch; ++e)
	{
		auto rr = reader();
		for (size_t id = 0; ; ++id)
		{
			auto rec = rr();
			if (rec.words.empty()) break;

			vector<uint32_t> doc;
			doc.reserve(rec.words.size());
			for (auto& w : rec.words)
			{
				auto id = vocabs.get(w);
				if (id < 0) continue;
				float ww = subsampling / (frequencies[id] / (float)totW);
				if (subsampling > 0 &&
					generate_canonical<float, 24>(rg) > sqrt(ww) + ww)
				{
					skip_cnt += 1;
					continue;
				}
				doc.emplace_back(id);
			}

			if (doc.size() < 2)
			{
				skip_cnt += doc.size();
				continue;
			}

			collections.emplace_back(move(doc));
			if (collections.size() >= batch)
			{
				procCollection();
			}
		}
	}
	procCollection();

	updateNormalizedVector();
}

template<typename _Derived, typename _ThreadLocalData>
void AdaGramBase<_Derived, _ThreadLocalData>::buildTrain(const function<DataReader()>& reader, size_t min_cnt,
	size_t num_workers, size_t window_length, float start_lr, float end_lr, size_t batch_sents, size_t epochs, size_t report)
{
	buildVocab(reader, min_cnt);
	train(reader, num_workers, window_length, start_lr, end_lr, batch_sents, epochs, report);
}

template<typename _Derived, typename _ThreadLocalData>
pair<VectorXf, size_t> AdaGramBase<_Derived, _ThreadLocalData>::getExpectedPi(const string & word) const
{
	size_t wv = vocabs.get(word);
	if (wv == (size_t)-1) return {};
	return getExpectedPi(wv);
}

template<typename _Derived, typename _ThreadLocalData>
vector<tuple<string, size_t, float>> AdaGramBase<_Derived, _ThreadLocalData>::nearestNeighbors(const string & word, size_t ws, size_t K, float min_count) const
{
	const size_t V = vocabs.size();
	size_t wv = vocabs.get(word);
	if (wv == (size_t)-1) return {};
	auto vec = inNormalized.col(wv * T + ws);

	if (counts(ws, wv) < min_count) return {};

	vector<tuple<string, size_t, float>> top;
	MatrixXf sim(T, V);
	for (size_t v = 0; v < V; ++v)
	{
		for (size_t s = 0; s < T; ++s)
		{
			if ((v == wv && s == ws) || counts(s, v) < min_count)
			{
				sim(s, v) = -INFINITY;
				continue;
			}
			sim(s, v) = inNormalized.col(v * T + s).dot(vec);
		}
	}

	for (size_t k = 0; k < K; ++k)
	{
		size_t idx = max_element(sim.data(), sim.data() + sim.size()) - sim.data();
		top.emplace_back(vocabs.getStr(idx / T), idx % T, sim.data()[idx]);
		sim.data()[idx] = -INFINITY;
	}
	return top;
}

template<typename _Derived, typename _ThreadLocalData>
vector<float> AdaGramBase<_Derived, _ThreadLocalData>::disambiguate(const string & word, const vector<string>& context, bool use_prior, float min_prob) const
{
	const size_t V = vocabs.size();
	size_t wv = vocabs.get(word);
	if (wv == (size_t)-1) return {};
	_ThreadLocalData ld;
	VectorXf z = VectorXf::Zero(T);
	if (use_prior)
	{
		z = getExpectedPi(wv).first;
		for (size_t k = 0; k < T; ++k)
		{
			if (z[k] < min_prob) z[k] = 0;
			z[k] = log(z[k]);
		}
	}

	mt19937_64 rg{ 1 };
	for (auto& y : context)
	{
		size_t yId = vocabs.get(y);
		if (yId == (size_t)-1) continue;
		static_cast<const _Derived*>(this)->template updateZ<false>(ld, rg, 1, wv, yId, z, nullptr);
	}

	softmax_inplace(z);
	return { z.data(), z.data() + T };
}

template<typename _Derived, typename _ThreadLocalData>
vector<tuple<string, size_t, float>> AdaGramBase<_Derived, _ThreadLocalData>::mostSimilar(const vector<pair<string, size_t>>& positiveWords, 
	const vector<pair<string, size_t>>& negativeWords, size_t K, float min_count) const
{
	VectorXf vec = VectorXf::Zero(M);
	const size_t V = vocabs.size();
	unordered_set<size_t> uniqs;
	for (auto& p : positiveWords)
	{
		size_t wv = vocabs.get(p.first);
		if (wv == (size_t)-1) return {};
		if (counts(p.second, wv) < min_count) return {};
		vec += inNormalized.col(wv * T + p.second);
		uniqs.emplace(wv * T + p.second);
	}

	for (auto& p : negativeWords)
	{
		size_t wv = vocabs.get(p.first);
		if (wv == (size_t)-1) return {};
		if (counts(p.second, wv) < min_count) return {};
		vec -= inNormalized.col(wv * T + p.second);
		uniqs.emplace(wv * T + p.second);
	}

	vec.normalize();

	vector<tuple<string, size_t, float>> top;
	MatrixXf sim(T, V);
	for (size_t v = 0; v < V; ++v)
	{
		for (size_t s = 0; s < T; ++s)
		{
			if (uniqs.count(v * T + s) || counts(s, v) < min_count)
			{
				sim(s, v) = -INFINITY;
				continue;
			}
			sim(s, v) = inNormalized.col(v * T + s).dot(vec);
		}
	}

	for (size_t k = 0; k < K; ++k)
	{
		size_t idx = max_element(sim.data(), sim.data() + sim.size()) - sim.data();
		top.emplace_back(vocabs.getStr(idx / T), idx % T, sim.data()[idx]);
		sim.data()[idx] = -INFINITY;
	}
	return top;
}

template<typename _Derived, typename _ThreadLocalData>
void AdaGramBase<_Derived, _ThreadLocalData>::saveModel(ostream & os) const
{
	writeToBinStream(os, (uint32_t)M);
	writeToBinStream(os, (uint32_t)T);
	writeToBinStream(os, alpha);
	writeToBinStream(os, d);
	writeToBinStream(os, sense_threshold);
	writeToBinStream(os, (uint32_t)context_cut);
	//writeToBinStream(os, (uint32_t)code.rows());
	vocabs.writeToFile(os);
	writeToBinStream(os, frequencies);
	writeToBinStream(os, in);
	writeToBinStream(os, out);
	writeToBinStream(os, counts);
	//writeToBinStream(os, code);
	//writeToBinStream(os, path);
}

template<typename _Derived, typename _ThreadLocalData>
void AdaGramBase<_Derived, _ThreadLocalData>::loadModel(istream & is)
{
	size_t M = readFromBinStream<uint32_t>(is);
	size_t T = readFromBinStream<uint32_t>(is);
	float alpha = readFromBinStream<float>(is);
	float d = readFromBinStream<float>(is);
	float sense_threshold = readFromBinStream<float>(is);
	bool context_cut = readFromBinStream<uint32_t>(is);
	size_t max_length = readFromBinStream<uint32_t>(is);

	sense_threshold = sense_threshold;
	context_cut = context_cut;
	vocabs.readFromFile(is);
	size_t V = vocabs.size();
	in.resize(M, T*V);
	out.resize(M, V);
	counts.resize(T, V);
	//code.resize(max_length, V);
	//path.resize(max_length, V);

	readFromBinStream(is, frequencies);
	readFromBinStream(is, in);
	readFromBinStream(is, out);
	readFromBinStream(is, counts);
	//readFromBinStream(is, code);
	//readFromBinStream(is, path);

	updateNormalizedVector();
}


struct HSoftmaxNode
{
	uint32_t parent = -1;
	bool branch = false;

	static vector<pair<int32_t, int8_t>> softmax_path(
		const vector<HSoftmaxNode>& nodes, size_t V, size_t id)
	{
		vector<pair<int32_t, int8_t>> ret;
		while (nodes[id].parent != (uint32_t)-1)
		{
			const auto& node = nodes[id];
			assert(node.parent >= V);
			ret.emplace_back(2 * V - 2 - node.parent, node.branch ? 1 : -1);
			id = node.parent;
		}
		return ret;
	}
};


auto AdaGramModel<Mode::hierarchical_softmax>::buildHuffmanTree() const 
	-> vector<HuffmanResult>
{
	auto V = vocabs.size();
	auto nodes = vector<HSoftmaxNode>(V);
	vector<pair<size_t, size_t>> heap;

	for (size_t i = 0; i < V; ++i)
	{
		heap.emplace_back(i, frequencies[i]);
	}
	size_t heapSize = V;

	const auto& freq_cmp = [](const pair<size_t, size_t>& a, const pair<size_t, size_t>& b)
	{
		return a.second > b.second;
	};
	make_heap(heap.begin(), heap.end(), freq_cmp);

	size_t L = V;
	while (heapSize > 1)
	{
		nodes.emplace_back();
		size_t freq = 0;

		pop_heap(heap.begin(), heap.begin() + heapSize, freq_cmp);
		heapSize--;
		nodes[heap[heapSize].first].parent = L;
		nodes[heap[heapSize].first].branch = true;
		freq += heap[heapSize].second;

		pop_heap(heap.begin(), heap.begin() + heapSize, freq_cmp);
		heapSize--;
		nodes[heap[heapSize].first].parent = L;
		nodes[heap[heapSize].first].branch = false;
		freq += heap[heapSize].second;

		heap[heapSize] = { nodes.size() - 1, freq };
		heapSize++;
		push_heap(heap.begin(), heap.begin() + heapSize, freq_cmp);
		L++;
	}


	auto outputs = vector<HuffmanResult>(V);
	for (auto v = 0; v < V; ++v)
	{
		vector<int8_t> code;
		vector<uint32_t> path;
		for (auto& p : HSoftmaxNode::softmax_path(nodes, V, v))
		{
			code.emplace_back(p.second);
			path.emplace_back(p.first);
		}
		outputs[v] = HuffmanResult{ code, path };
	}
	return outputs;
}

void AdaGramModel<Mode::hierarchical_softmax>::_buildModel()
{
	const size_t V = vocabs.size();

	auto outputs = buildHuffmanTree();
	size_t max_length = max_element(outputs.begin(), outputs.end(), [](const HuffmanResult& a, const HuffmanResult& b)
	{
		return a.code.size() < b.code.size();
	})->code.size();
	path = Matrix<uint32_t, Dynamic, Dynamic>::Zero(max_length, V);
	code = Matrix<int8_t, Dynamic, Dynamic>::Zero(max_length, V);

	for (size_t v = 0; v < V; ++v)
	{
		copy(outputs[v].code.begin(), outputs[v].code.end(), code.col(v).data());
		copy(outputs[v].path.begin(), outputs[v].path.end(), path.col(v).data());
	}
}

template<bool _multi>
void AdaGramModel<Mode::hierarchical_softmax>::updateZ(ThreadLocalData & ld, mt19937_64 & rg, size_t num_workers, word_id_t x, word_id_t y, VectorXf & z, float * f) const
{
	for (int n = 0; n < code.rows() && code(n, y); ++n)
	{
		auto node_id = path(n, y);
		if (_multi && ld.update_out_idx.find(node_id) == ld.update_out_idx.end())
		{
			ld.update_out_idx.emplace(node_id, ld.update_out_idx.size());
			ld.update_out_idx_hash.emplace(node_id % num_workers);
		}

		auto ocol = out.col(node_id);
		for (int k = 0; k < T; ++k)
		{
			float dot = in.col(x * T + k).dot(ocol);
			if (f) f[n * T + k] = dot;
			z[k] += logsigmoid(dot * code(n, y));
		}
	}
}

float AdaGramModel<Mode::hierarchical_softmax>::inplaceUpdate(ThreadLocalData& ld, word_id_t x, word_id_t y, const VectorXf& z, const float* f, float lr)
{
	MatrixXf in_grad = MatrixXf::Zero(M, T);
	VectorXf out_grad = VectorXf::Zero(M);

	float pr = 0;
	for (int n = 0; n < code.rows() && code(n, y); ++n)
	{
		auto outcol = out.col(path(n, y));
		out_grad.setZero();

		for (int k = 0; k < T; ++k)
		{
			if (z[k] < sense_threshold) continue;
			auto incol = in.col(x * T + k);
			pr += z[k] * logsigmoid(f[n * T + k] * code(n, y));

			float d = (1 + code(n, y)) / 2 - sigmoid(f[n * T + k]);
			float g = z[k] * lr * d;

			in_grad.col(k) += g * outcol;
			out_grad += g * incol;
		}
		outcol += out_grad;
	}

	for (int k = 0; k < T; ++k)
	{
		if (z[k] < sense_threshold) continue;
		auto incol = in.col(x * T + k);
		incol += in_grad.col(k);
	}
	return pr;
}


float AdaGramModel<Mode::hierarchical_softmax>::update(ThreadLocalData& ld, word_id_t x, word_id_t y, const VectorXf& z, const float* f, float lr) const
{
	float pr = 0;
	for (int n = 0; n < code.rows() && code(n, y); ++n)
	{
		auto outcol = out.col(path(n, y));
		size_t cache_id = ld.update_out_idx.find(path(n, y))->second;

		for (int k = 0; k < T; ++k)
		{
			if (z[k] < sense_threshold) continue;
			auto incol = in.col(x * T + k);
			pr += z[k] * logsigmoid(f[n * T + k] * code(n, y));

			float d = (1 + code(n, y)) / 2 - sigmoid(f[n * T + k]);
			float g = z[k] * lr * d;

			ld.update_in.col(k) += g * outcol;
			ld.update_out.col(cache_id) += g * incol;
		}
	}

	return pr;
}

size_t AdaGramModel<Mode::hierarchical_softmax>::getFWidth() const
{
	return code.rows();
}

auto AdaGramModel<Mode::hierarchical_softmax>::createLocalData(size_t window_length) const -> ThreadLocalData
{
	ThreadLocalData ld;
	ld.update_in = MatrixXf::Zero(M, T);
	ld.update_out = MatrixXf::Zero(M, window_length * 2 * getFWidth());
	return ld;
}



void AdaGramModel<Mode::negative_sampling>::_buildModel()
{
	vector<double> weights;
	transform(frequencies.begin(), frequencies.end(), back_inserter(weights), [](auto w) { return pow(w, 0.75); });
	unigram_table = discrete_distribution<word_id_t>(weights.begin(), weights.end());
}

template<typename _Eigen>
float logsum(_Eigen&& arr)
{
	float m = arr.maxCoeff();
	return m + log((arr - m).exp().sum());
}

template<bool _multi>
void AdaGramModel<Mode::negative_sampling>::updateZ(ThreadLocalData& ld, mt19937_64& rg, size_t num_workers, word_id_t x, word_id_t y, VectorXf & z, float* f) const
{
	ld.consumed = 0;
	Eigen::ArrayXXf negs{ negative_sample_size, T };
	for(size_t n = 0; n <= negative_sample_size; ++n)
	{
		word_id_t w;
		if(n) 
		{
			w = unigram_table(rg);
			ld.negative_samples[ld.produced++] = w;
		}
		else w = y;

		if (_multi && ld.update_out_idx.find(w) == ld.update_out_idx.end())
		{
			ld.update_out_idx.emplace(w, ld.update_out_idx.size());
			ld.update_out_idx_hash.emplace(w % num_workers);
		}

		for (int k = 0; k < T; ++k)
		{
			float dot = in.col(x * T + k).dot(out.col(w));
			if(f) f[n * T + k] = dot;
			if (n == 0)
			{
				z[k] += logsigmoid(dot);
			}
			else
			{
				negs(n - 1, k) = logsigmoid(-dot);
			}
		}
	}

	for (int k = 0; k < T; ++k)
	{
		z[k] += logsum(negs.col(k));
	}
}


float AdaGramModel<Mode::negative_sampling>::inplaceUpdateSingle(ThreadLocalData& ld, word_id_t x, word_id_t y, const VectorXf & z, const float* f, float lr, bool negative)
{
	MatrixXf in_grad = MatrixXf::Zero(M, T);
	VectorXf out_grad = VectorXf::Zero(M);

	float pr = 0;
	auto outcol = out.col(y);
	for (int k = 0; k < T; ++k)
	{
		if (z[k] < sense_threshold) continue;
		auto incol = in.col(x * T + k);
		pr += z[k] * logsigmoid(f[k] * (negative ? -1 : 1));

		float d = (negative ? 0 : 1) - sigmoid(f[k]);
		float g = z[k] * lr * d;

		in_grad.col(k) += g * outcol;
		out_grad += g * incol;
	}
	outcol += out_grad;

	for (int k = 0; k < T; ++k)
	{
		if (z[k] < sense_threshold) continue;
		in.col(x * T + k) += in_grad.col(k);
	}
	return pr;
}

float AdaGramModel<Mode::negative_sampling>::inplaceUpdate(ThreadLocalData& ld, word_id_t x, word_id_t y, const VectorXf & z, const float* f, float lr)
{
	float pr = 0;
	ld.produced = 0;
	for(size_t n = 0; n <= negative_sample_size; ++n)
	{
		word_id_t w;
		if(n)
		{
			w = ld.negative_samples[ld.consumed++];
		}
		else w = y;
		pr += inplaceUpdateSingle(ld, x, w, z, f + n * T, lr, !!n);
	}
	return pr;
}

float AdaGramModel<Mode::negative_sampling>::updateSingle(ThreadLocalData& ld, word_id_t x, word_id_t y, const VectorXf& z, const float* f, float lr, bool negative) const
{
	float pr = 0;
	size_t cache_id = ld.update_out_idx.find(y)->second;
	auto outcol = out.col(y);
	for (int k = 0; k < T; ++k)
	{
		if (z[k] < sense_threshold) continue;
		auto incol = in.col(x * T + k);
		pr += z[k] * logsigmoid(f[k] * (negative ? -1 : 1));

		float d = (negative ? 0 : 1) - sigmoid(f[k]);
		float g = z[k] * lr * d;

		ld.update_in.col(k) += g * outcol;
		ld.update_out.col(cache_id) += g * incol;
	}
	return pr;
}

float AdaGramModel<Mode::negative_sampling>::update(ThreadLocalData& ld, word_id_t x, word_id_t y, const VectorXf& z, const float* f, float lr) const
{
	float pr = 0;
	ld.produced = 0;
	for(size_t n = 0; n <= negative_sample_size; ++n)
	{
		word_id_t w;
		if(n)
		{
			w = ld.negative_samples[ld.consumed++];
		}
		else w = y;
		pr += updateSingle(ld, x, w, z, f + n * T, lr, !!n);
	}
	return pr;
}

size_t AdaGramModel<Mode::negative_sampling>::getFWidth() const
{
	return negative_sample_size + 1;
}


auto AdaGramModel<Mode::negative_sampling>::createLocalData(size_t window_length) const -> ThreadLocalData
{
	ThreadLocalData ld;
	ld.update_in = MatrixXf::Zero(M, T);
	ld.update_out = MatrixXf::Zero(M, window_length * 2 * getFWidth());
	ld.negative_samples.resize(negative_sample_size * window_length * 2);
	return ld;
}

void AdaGramModel<Mode::negative_sampling>::allocateCache(ThreadLocalData& ld, word_id_t y, size_t num_workers) const
{
	if (ld.update_out_idx.find(y) == ld.update_out_idx.end())
	{
		ld.update_out_idx.emplace(y, ld.update_out_idx.size());
		ld.update_out_idx_hash.emplace(y % num_workers);
	}
}

template class ag::AdaGramBase<AdaGramModel<Mode::hierarchical_softmax>, ThreadLocalBase>;
template class ag::AdaGramBase<AdaGramModel<Mode::negative_sampling>, ThreadLocalNS>;
