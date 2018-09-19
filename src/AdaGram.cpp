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
		ifstream ifs{ "D:/enwiki3000.txt" };
		while (ifs >> word)
		{
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

	VectorModel vm{ freqs, 50, 2, 0.3, 0 };

	vector<int64_t> words_read(1);
	vector<float> total_ll(2);
	inplace_train_vectors(vm, doc, 4, 0.025, doc.size(), words_read, total_ll);

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

