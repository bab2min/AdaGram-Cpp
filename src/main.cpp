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
#include <sstream>
#include <iterator>
#include <fstream>

#include "AdaGramModel.h"

using namespace std;

struct KoreanStopwordFilter
{
	bool operator()(const string& o)
	{
		if (o.find("/NN") != string::npos) return true;
		if (o.find("/VA") != string::npos) return true;
		if (o.find("/VV") != string::npos) return true;
		if (o.find("/MM") != string::npos) return true;
		if (o.find("/MAG") != string::npos) return true;
		if (o.find("/XR") != string::npos) return true;
		if (o.find("/SL") != string::npos) return true;
		return false;
	}
};

struct PuncFilter
{
	bool operator()(const string& o)
	{
		return o != ".";
	}
};

struct KoreanTagTransformer
{
	string operator()(const string& o)
	{
		if (o.find("/NN") != string::npos) return o.substr(0, o.size() - 1);
		return o;
	}
};


int main()
{
	ag::AdaGramModel<ag::Mode::hierarchical_softmax> agm{ 300, 5, .05f, 0, 1e-4 };
	if (1)
	{
		Timer timer;
		//auto gen = ag::util::FileLineReader<KoreanStopwordFilter, KoreanTagTransformer>::generator("G:/namu_tagged.txt");
		auto gen = ag::util::FileLineReader<PuncFilter>::generator("data/enwiki10000.txt");
		agm.buildTrain(gen, 10, 8, 5, 0.025, 0.00025, 100000, 2);
		cout << "Finished in " << timer.getElapsed() << " sec" << endl;
		ofstream ofs{ "namu_subsampling.mdl", ios_base::binary };
		agm.saveModel(ofs);
	}
	else
	{
		ifstream ifs{ "namu_subsampling.mdl", ios_base::binary };
		agm.loadModel(ifs);

		/*
		ofstream ofs{ "namuTest_subsampling.txt" };
		for (auto& w : agm.getVocabs())
		{
			auto pi = agm.getExpectedPi(w).first;
			for (size_t s = 0; s < 5; ++s)
			{
				auto nn = agm.nearestNeighbors(w, s, 15);
				if (nn.empty()) continue;
				if (pi[s] < 0.1f) continue;
				ofs << "== " << w << '\t' << s << '\t' << pi[s] << endl;
				for (auto& n : nn)
				{
					ofs << get<0>(n) << '\t' << get<1>(n) << '\t' << get<2>(n) << endl;
				}
				ofs << endl;
			}
		}
		*/
	}

	string line;
	while (cout << ">> ", getline(cin, line))
	{
		vector<pair<string, size_t>> positives, negatives;
		istringstream iss{ line };
		istream_iterator<string> wBegin{ iss }, wEnd{};
		bool sign = false;
		for (; wBegin != wEnd; ++wBegin)
		{
			auto word = *wBegin;
			if (word == "-")
			{
				sign = true;
			}
			else if (word == "+")
			{
				sign = false;
			}
			else
			{
				size_t s = 0;
				if (word.find("__") != string::npos)
				{
					s = stoi(word.substr(word.find("__") + 2));
					word = word.substr(0, word.find("__"));
				}
				(sign ? negatives : positives).emplace_back(word, s);
				sign = false;
			}
		}
		
		for (auto& p : positives)
		{
			auto pp = agm.getExpectedPi(p.first);
			cout << p.first << " : ";
			if (pp.second)
			{
				for (size_t i = 0; i < pp.first.size(); ++i)
				{
					cout << pp.first[i] << ", ";
				}
			}
			else
			{
				cout << "Unknown words ";
			}
			cout << endl;
		}
		for (auto& p : negatives)
		{
			auto pp = agm.getExpectedPi(p.first);
			cout << p.first << " : ";
			if (pp.second)
			{
				for (size_t i = 0; i < pp.first.size(); ++i)
				{
					cout << pp.first[i] << ", ";
				}
			}
			else
			{
				cout << "Unknown words ";
			}
			cout << endl;
		}
		cout << "================" << endl;
		for (auto& p : agm.mostSimilar(positives, negatives, 20))
		{
			cout << get<0>(p) << "__" << get<1>(p) << '\t' << get<2>(p) << endl;
		}
		cout << endl;
	}
    return 0;
}

