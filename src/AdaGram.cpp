// AdaGram-Cpp.cpp : Defines the entry point for the console application.
//

#include <vector>
#include <array>
#include <map>
#include <algorithm>
#include <unordered_map>
#include <numeric>
#include <string>
#include <iostream>
#include <fstream>

#include "VectorModel.h"
#include "dictionary.h"
#include "util.h"

using namespace std;

int main()
{
	WordDictionary<> dict;
	vector<size_t> doc, freqs;
	{
		WordDictionary<> rdict;
		string word;
		vector<size_t> rdoc;
		ifstream ifs{ "data/enwiki10000.txt" };
		while (ifs >> word)
		{
			if (word == ".") continue;
			rdoc.emplace_back(rdict.getOrAdd(word));
		}

		vector<size_t> rfreqs(rdict.size()), rmap(rdict.size(), (size_t)-1);
		for (auto x : rdoc)
		{
			rfreqs[x]++;
		}

		for (size_t i = 0; i < rdict.size(); ++i)
		{
			if (rfreqs[i] >= 20)
			{
				rmap[i] = dict.add(rdict.getStr(i));
				freqs.emplace_back(rfreqs[i]);
			}
		}

		for (auto x : rdoc)
		{
			if (rmap[x] != (size_t)-1) doc.emplace_back(rmap[x]);
		}
	}

	VectorModel vm{ freqs, 100, 5, .1f, 0 };

	for (size_t v = 0; v < vm.V(); ++v)
	{
		vm.counts[{0, v}] = vm.frequencies[v];
	}

	vector<int64_t> words_read(1);
	vector<float> total_ll(2);
	while (words_read[0] < doc.size())
	{
		inplace_train_vectors(vm, &doc[words_read[0]], min(64000llu, doc.size() - words_read[0]), 4, 0.025, doc.size(), words_read, total_ll);
	}
	cout << "Finish!" << endl;
	string word;
	while (cin >> word)
	{
		size_t s = 0;
		if (word.find("__") != string::npos)
		{
			s = stoi(word.substr(word.find("__") + 2));
			word = word.substr(0, word.find("__"));
		}

		if (dict.get(word) >= 0)
		{
			for (auto p : expected_pi(vm, dict.get(word)))
			{
				cout << p << ", ";
			}
			cout << endl;

			for (auto& p : nearest_neighbors(vm, dict, word, s, 20))
			{
				cout << get<0>(p) << "__" << get<1>(p) << '\t' << get<2>(p) << endl;
			}
		}
		else
		{
			cout << "Unknown words" << endl;
		}
		cout << endl;
	}
    return 0;
}

