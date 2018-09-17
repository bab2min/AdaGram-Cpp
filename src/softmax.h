#pragma once


struct HierarchicalSoftmaxNode
{
	int32_t parent = 0;
	bool branch = false;

	static vector<pair<int32_t, float>> softmax_path(const vector<HierarchicalSoftmaxNode>& nodes, int V, int id)
	{
		vector<pair<int32_t, float>> ret;
		while (1)
		{
			auto& node = nodes[id];
			if (node.parent == 0) break;
			assert(node.parent > V);
			ret.emplace_back(node.parent - V, node.branch);
			id = node.parent;
		}
		return ret;
	}
};

struct HierarchicalOuput
{
	vector<int8_t> code;
	vector<int> path;

	size_t length() const
	{
		return path.size();
	}
};


template<class Tf>
auto build_huffman_tree(const vector<Tf>& freqs)
{
	auto V = freqs.size();
	auto nodes = vector<HierarchicalSoftmaxNode>(V);
	/*
	freq_ord = By(wf -> wf[2])
	heap = heapify!([(nodes[v], freqs[v]) for v in 1:V], freq_ord)

	function pop_initialize!(parent::Int, branch::Bool)
		node = heappop!(heap, freq_ord)
		node[1].parent = Int32(parent)
		node[1].branch = branch
		return node[2]
	end

	L = V
	while length(heap) > 1
		L += 1
		node = HierarchicalSoftmaxNode()
		push!(nodes, node)

		freq = 1
		freq = pop_initialize!(L, true) + pop_initialize!(L, false)
		heappush!(heap, (node, freq), freq_ord)
	end

	@assert length(heap) == 1

	*/
	return nodes;
}

auto convert_huffman_tree(const vector<HierarchicalSoftmaxNode>& nodes, int V)
{
	auto outputs = vector<HierarchicalOuput>(V);
	for (auto v = 0; v < V; ++v)
	{
		vector<int8_t> code;
		vector<int> path;
		for (auto& p : HierarchicalSoftmaxNode::softmax_path(nodes, V, v))
		{
			code.emplace_back(int8_t(p.second + 0.5f));
			path.emplace_back(p.first);
		}
		outputs[v] = HierarchicalOuput{ code, path };
	}
	return outputs;
}
