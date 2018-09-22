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

int main()
{
	AdaGramModel agm{ 100, 5, .1f };
	if (1)
	{
		ifstream ifs{ "data/enwiki3000.txt" };
		istream_iterator<string> iBegin{ ifs }, iEnd{};
		ifs.seekg(0, ifs.end);
		size_t totalSize = ifs.tellg();
		ifs.seekg(0);

		agm.buildVocab(iBegin, iEnd, 20, [](const auto& o) { return o != "."; });
		agm.train([&ifs, totalSize](size_t id, double& progress)->vector<string>
		{
			if (id == 0)
			{
				ifs.clear();
				ifs.seekg(0);
			}
			string line;
			if (!getline(ifs, line)) return {};
			progress = ifs.tellg() / (double)totalSize;
			istringstream iss{ line };
			istream_iterator<string> iBegin{ iss }, iEnd{};
			return { iBegin, iEnd };
		}, 0, 4, 0.025, 2000, 2);
		cout << "Finish!" << endl;

		ofstream ofs{ "enwiki3000.mdl", ios_base::binary };
		agm.saveModel(ofs);
	}
	else
	{
		ifstream ifs{ "enwiki3000.mdl", ios_base::binary };
		agm = AdaGramModel::loadModel(ifs);
	}

	string word;
	while (cin >> word)
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

