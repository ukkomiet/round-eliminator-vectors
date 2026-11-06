Vibe-coded round elimination tools. 
round_eliminator.py is the normal tool (follows https://arxiv.org/abs/2410.20224)
round_eliminator_vectors.py is the modified one, that uses ordered multisets for configurations (now called vectors)

example usage described in the tool header comments

problems are described in the JSON-files as follows (example is sinkless orientation, degree 3, normal unordered multiset configurations):
```
{
    "alphabet": ["h", "t"],
    "node_constraint": [["t", "t", "t"], ["t", "t", "h"], ["t", "h", "h"]],
    "edge_constraint": [["h", "t"]]

}
```
