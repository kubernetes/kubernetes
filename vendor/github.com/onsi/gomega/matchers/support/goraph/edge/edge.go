package edge

import . "github.com/onsi/gomega/matchers/support/goraph/node"

type Edge struct {
	Node1 int
	Node2 int
}

type EdgeSet []Edge

func (ec EdgeSet) Free(node Node) bool {
	for _, e := range ec {
		if e.Node1 == node.ID || e.Node2 == node.ID {
			return false
		}
	}

	return true
}

func (ec EdgeSet) Contains(edge Edge) bool {
	for _, e := range ec {
		if e == edge {
			return true
		}
	}

	return false
}

func (ec EdgeSet) FindByNodes(node1, node2 Node) (Edge, bool) {
	for _, e := range ec {
		if (e.Node1 == node1.ID && e.Node2 == node2.ID) || (e.Node1 == node2.ID && e.Node2 == node1.ID) {
			return e, true
		}
	}

	return Edge{}, false
}

func (ec EdgeSet) SymmetricDifference(ec2 EdgeSet) EdgeSet {
	edgesToInclude := make(map[Edge]bool)

	for _, e := range ec {
		edgesToInclude[e] = true
	}

	for _, e := range ec2 {
		edgesToInclude[e] = !edgesToInclude[e]
	}

	result := EdgeSet{}
	for e, include := range edgesToInclude {
		if include {
			result = append(result, e)
		}
	}

	return result
}
