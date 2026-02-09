package internal

import "github.com/onsi/ginkgo/v2/types"

type TreeNode struct {
	Node     Node
	Parent   *TreeNode
	Children TreeNodes
}

func (tn *TreeNode) AppendChild(child *TreeNode) {
	tn.Children = append(tn.Children, child)
	child.Parent = tn
}

func (tn *TreeNode) AncestorNodeChain() Nodes {
	if tn.Parent == nil || tn.Parent.Node.IsZero() {
		return Nodes{tn.Node}
	}
	return append(tn.Parent.AncestorNodeChain(), tn.Node)
}

type TreeNodes []*TreeNode

func (tn TreeNodes) Nodes() Nodes {
	out := make(Nodes, len(tn))
	for i := range tn {
		out[i] = tn[i].Node
	}
	return out
}

func (tn TreeNodes) WithID(id uint) *TreeNode {
	for i := range tn {
		if tn[i].Node.ID == id {
			return tn[i]
		}
	}

	return nil
}

func GenerateSpecsFromTreeRoot(tree *TreeNode) Specs {
	var walkTree func(nestingLevel int, lNodes Nodes, rNodes Nodes, trees TreeNodes) Specs
	walkTree = func(nestingLevel int, lNodes Nodes, rNodes Nodes, trees TreeNodes) Specs {
		tests := Specs{}

		nodes := make(Nodes, len(trees))
		for i := range trees {
			nodes[i] = trees[i].Node
			nodes[i].NestingLevel = nestingLevel
		}

		for i := range nodes {
			if !nodes[i].NodeType.Is(types.NodeTypesForContainerAndIt) {
				continue
			}
			leftNodes, rightNodes := nodes.SplitAround(nodes[i])
			leftNodes = leftNodes.WithoutType(types.NodeTypesForContainerAndIt)
			rightNodes = rightNodes.WithoutType(types.NodeTypesForContainerAndIt)

			leftNodes = lNodes.CopyAppend(leftNodes...)
			rightNodes = rightNodes.CopyAppend(rNodes...)

			if nodes[i].NodeType.Is(types.NodeTypeIt) {
				tests = append(tests, Spec{Nodes: leftNodes.CopyAppend(nodes[i]).CopyAppend(rightNodes...)})
			} else {
				treeNode := trees.WithID(nodes[i].ID)
				tests = append(tests, walkTree(nestingLevel+1, leftNodes.CopyAppend(nodes[i]), rightNodes, treeNode.Children)...)
			}
		}

		return tests
	}

	return walkTree(0, Nodes{}, Nodes{}, tree.Children)
}
