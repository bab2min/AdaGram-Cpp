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

#include "AdaGramModel.h"

using namespace std;

bool posSelect(const string& o)
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

string unifyNoun(const string& o)
{
	if (o.find("/NN") != string::npos) return o.substr(0, o.size() - 1);
	return o;
}

int main()
{
	AdaGramModel agm{ 300, 5, .1f, 0, 1e-4 };
	if (1)
	{
		Timer timer;
		ifstream ifs{ "D:/namu_tagged.txt" };
		agm.buildTrain(ifs, 10, posSelect, unifyNoun, 1, 5, 0.025, 100000, 2);
		cout << "Finished in " << timer.getElapsed() << " sec" << endl;
		ofstream ofs{ "namu_subsampling.mdl", ios_base::binary };
		agm.saveModel(ofs);
	}
	else
	{
		ifstream ifs{ "namu_subsampling.mdl", ios_base::binary };
		agm = AdaGramModel::loadModel(ifs);

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
	}

	string word;
	while (cout << ">> ", cin >> word)
	{
		size_t s = 0;
		if (word.find("__") != string::npos)
		{
			s = stoi(word.substr(word.find("__") + 2));
			word = word.substr(0, word.find("__"));
		}

		auto pp = agm.getExpectedPi(word);
		if (pp.second)
		{
			for (size_t i = 0; i < pp.first.size(); ++i)
			{
				cout << pp.first[i] << ", ";
			}
			cout << endl;

			for (auto& p : agm.nearestNeighbors(word, s, 20))
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

