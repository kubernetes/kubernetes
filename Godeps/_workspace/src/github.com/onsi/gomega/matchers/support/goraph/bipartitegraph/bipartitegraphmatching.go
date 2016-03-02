package bipartitegraph

import . "github.com/onsi/gomega/matchers/support/goraph/node"
import . "github.com/onsi/gomega/matchers/support/goraph/edge"
import "github.com/onsi/gomega/matchers/support/goraph/util"

func (bg *BipartiteGraph) LargestMatching() (matching EdgeSet) {
	paths := bg.maximalDisjointSLAPCollection(matching)

	for len(paths) > 0 {
		for _, path := range paths {
			matching = matching.SymmetricDifference(path)
		}
		paths = bg.maximalDisjointSLAPCollection(matching)
	}

	return
}

func (bg *BipartiteGraph) maximalDisjointSLAPCollection(matching EdgeSet) (result []EdgeSet) {
	guideLayers := bg.createSLAPGuideLayers(matching)
	if len(guideLayers) == 0 {
		return
	}

	used := make(map[Node]bool)

	for _, u := range guideLayers[len(guideLayers)-1] {
		slap, found := bg.findDisjointSLAP(u, matching, guideLayers, used)
		if found {
			for _, edge := range slap {
				used[edge.Node1] = true
				used[edge.Node2] = true
			}
			result = append(result, slap)
		}
	}

	return
}

func (bg *BipartiteGraph) findDisjointSLAP(
	start Node,
	matching EdgeSet,
	guideLayers []NodeOrderedSet,
	used map[Node]bool,
) ([]Edge, bool) {
	return bg.findDisjointSLAPHelper(start, EdgeSet{}, len(guideLayers)-1, matching, guideLayers, used)
}

func (bg *BipartiteGraph) findDisjointSLAPHelper(
	currentNode Node,
	currentSLAP EdgeSet,
	currentLevel int,
	matching EdgeSet,
	guideLayers []NodeOrderedSet,
	used map[Node]bool,
) (EdgeSet, bool) {
	used[currentNode] = true

	if currentLevel == 0 {
		return currentSLAP, true
	}

	for _, nextNode := range guideLayers[currentLevel-1] {
		if used[nextNode] {
			continue
		}

		edge, found := bg.Edges.FindByNodes(currentNode, nextNode)
		if !found {
			continue
		}

		if matching.Contains(edge) == util.Odd(currentLevel) {
			continue
		}

		currentSLAP = append(currentSLAP, edge)
		slap, found := bg.findDisjointSLAPHelper(nextNode, currentSLAP, currentLevel-1, matching, guideLayers, used)
		if found {
			return slap, true
		}
		currentSLAP = currentSLAP[:len(currentSLAP)-1]
	}

	used[currentNode] = false
	return nil, false
}

func (bg *BipartiteGraph) createSLAPGuideLayers(matching EdgeSet) (guideLayers []NodeOrderedSet) {
	used := make(map[Node]bool)
	currentLayer := NodeOrderedSet{}

	for _, node := range bg.Left {
		if matching.Free(node) {
			used[node] = true
			currentLayer = append(currentLayer, node)
		}
	}

	if len(currentLayer) == 0 {
		return []NodeOrderedSet{}
	} else {
		guideLayers = append(guideLayers, currentLayer)
	}

	done := false

	for !done {
		lastLayer := currentLayer
		currentLayer = NodeOrderedSet{}

		if util.Odd(len(guideLayers)) {
			for _, leftNode := range lastLayer {
				for _, rightNode := range bg.Right {
					if used[rightNode] {
						continue
					}

					edge, found := bg.Edges.FindByNodes(leftNode, rightNode)
					if !found || matching.Contains(edge) {
						continue
					}

					currentLayer = append(currentLayer, rightNode)
					used[rightNode] = true

					if matching.Free(rightNode) {
						done = true
					}
				}
			}
		} else {
			for _, rightNode := range lastLayer {
				for _, leftNode := range bg.Left {
					if used[leftNode] {
						continue
					}

					edge, found := bg.Edges.FindByNodes(leftNode, rightNode)
					if !found || !matching.Contains(edge) {
						continue
					}

					currentLayer = append(currentLayer, leftNode)
					used[leftNode] = true
				}
			}

		}

		if len(currentLayer) == 0 {
			return []NodeOrderedSet{}
		} else {
			guideLayers = append(guideLayers, currentLayer)
		}
	}

	return
}
