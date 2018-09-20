#pragma once

#include <cassert>

struct HierarchicalSoftmaxNode
{
	int32_t parent = -1;
	bool branch = false;

	static std::vector<std::pair<int32_t, float>> softmax_path(
		const std::vector<HierarchicalSoftmaxNode>& nodes, int V, int id)
	{
		std::vector<std::pair<int32_t, float>> ret;
		while (1)
		{
			auto& node = nodes[id];
			if (node.parent == -1) break;
			assert(node.parent >= V);
			ret.emplace_back(node.parent - V, node.branch);
			id = node.parent;
		}
		return ret;
	}
};

struct HierarchicalOuput
{
	std::vector<int8_t> code;
	std::vector<int> path;

	size_t length() const
	{
		return path.size();
	}
};


auto build_huffman_tree(const std::vector<size_t>& freqs)
{
	auto V = freqs.size();
	auto nodes = std::vector<HierarchicalSoftmaxNode>(V);
	std::vector<std::pair<size_t, size_t>> heap;

	for (size_t i = 0; i < V; ++i)
	{
		heap.emplace_back(i, freqs[i]);
	}
	size_t heapSize = V;

	const auto& freq_cmp = [](const std::pair<size_t, size_t>& a, const std::pair<size_t, size_t>& b)
	{
		return a.second > b.second;
	};
	std::make_heap(heap.begin(), heap.end(), freq_cmp);

	size_t L = V;
	while (heapSize > 1)
	{
		nodes.emplace_back();
		size_t freq = 0;

		std::pop_heap(heap.begin(), heap.begin() + heapSize, freq_cmp);
		heapSize--;
		nodes[heap[heapSize].first].parent = L;
		nodes[heap[heapSize].first].branch = true;
		freq += heap[heapSize].second;

		std::pop_heap(heap.begin(), heap.begin() + heapSize, freq_cmp);
		heapSize--;
		nodes[heap[heapSize].first].parent = L;
		nodes[heap[heapSize].first].branch = false;
		freq += heap[heapSize].second;

		heap[heapSize] = { nodes.size() - 1, freq };
		heapSize++;
		std::push_heap(heap.begin(), heap.begin() + heapSize, freq_cmp);
		L++;
	}
	return nodes;
}

auto convert_huffman_tree(const std::vector<HierarchicalSoftmaxNode>& nodes, int V)
{
	auto outputs = std::vector<HierarchicalOuput>(V);
	for (auto v = 0; v < V; ++v)
	{
		std::vector<int8_t> code;
		std::vector<int> path;
		for (auto& p : HierarchicalSoftmaxNode::softmax_path(nodes, V, v))
		{
			code.emplace_back(int8_t(p.second + 0.5f));
			path.emplace_back(p.first);
		}
		outputs[v] = HierarchicalOuput{ code, path };
	}
	return outputs;
}
