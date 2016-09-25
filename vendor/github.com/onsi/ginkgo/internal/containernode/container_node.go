package containernode

import (
	"math/rand"
	"sort"

	"github.com/onsi/ginkgo/internal/leafnodes"
	"github.com/onsi/ginkgo/types"
)

type subjectOrContainerNode struct {
	containerNode *ContainerNode
	subjectNode   leafnodes.SubjectNode
}

func (n subjectOrContainerNode) text() string {
	if n.containerNode != nil {
		return n.containerNode.Text()
	} else {
		return n.subjectNode.Text()
	}
}

type CollatedNodes struct {
	Containers []*ContainerNode
	Subject    leafnodes.SubjectNode
}

type ContainerNode struct {
	text         string
	flag         types.FlagType
	codeLocation types.CodeLocation

	setupNodes               []leafnodes.BasicNode
	subjectAndContainerNodes []subjectOrContainerNode
}

func New(text string, flag types.FlagType, codeLocation types.CodeLocation) *ContainerNode {
	return &ContainerNode{
		text:         text,
		flag:         flag,
		codeLocation: codeLocation,
	}
}

func (container *ContainerNode) Shuffle(r *rand.Rand) {
	sort.Sort(container)
	permutation := r.Perm(len(container.subjectAndContainerNodes))
	shuffledNodes := make([]subjectOrContainerNode, len(container.subjectAndContainerNodes))
	for i, j := range permutation {
		shuffledNodes[i] = container.subjectAndContainerNodes[j]
	}
	container.subjectAndContainerNodes = shuffledNodes
}

func (node *ContainerNode) BackPropagateProgrammaticFocus() bool {
	if node.flag == types.FlagTypePending {
		return false
	}

	shouldUnfocus := false
	for _, subjectOrContainerNode := range node.subjectAndContainerNodes {
		if subjectOrContainerNode.containerNode != nil {
			shouldUnfocus = subjectOrContainerNode.containerNode.BackPropagateProgrammaticFocus() || shouldUnfocus
		} else {
			shouldUnfocus = (subjectOrContainerNode.subjectNode.Flag() == types.FlagTypeFocused) || shouldUnfocus
		}
	}

	if shouldUnfocus {
		if node.flag == types.FlagTypeFocused {
			node.flag = types.FlagTypeNone
		}
		return true
	}

	return node.flag == types.FlagTypeFocused
}

func (node *ContainerNode) Collate() []CollatedNodes {
	return node.collate([]*ContainerNode{})
}

func (node *ContainerNode) collate(enclosingContainers []*ContainerNode) []CollatedNodes {
	collated := make([]CollatedNodes, 0)

	containers := make([]*ContainerNode, len(enclosingContainers))
	copy(containers, enclosingContainers)
	containers = append(containers, node)

	for _, subjectOrContainer := range node.subjectAndContainerNodes {
		if subjectOrContainer.containerNode != nil {
			collated = append(collated, subjectOrContainer.containerNode.collate(containers)...)
		} else {
			collated = append(collated, CollatedNodes{
				Containers: containers,
				Subject:    subjectOrContainer.subjectNode,
			})
		}
	}

	return collated
}

func (node *ContainerNode) PushContainerNode(container *ContainerNode) {
	node.subjectAndContainerNodes = append(node.subjectAndContainerNodes, subjectOrContainerNode{containerNode: container})
}

func (node *ContainerNode) PushSubjectNode(subject leafnodes.SubjectNode) {
	node.subjectAndContainerNodes = append(node.subjectAndContainerNodes, subjectOrContainerNode{subjectNode: subject})
}

func (node *ContainerNode) PushSetupNode(setupNode leafnodes.BasicNode) {
	node.setupNodes = append(node.setupNodes, setupNode)
}

func (node *ContainerNode) SetupNodesOfType(nodeType types.SpecComponentType) []leafnodes.BasicNode {
	nodes := []leafnodes.BasicNode{}
	for _, setupNode := range node.setupNodes {
		if setupNode.Type() == nodeType {
			nodes = append(nodes, setupNode)
		}
	}
	return nodes
}

func (node *ContainerNode) Text() string {
	return node.text
}

func (node *ContainerNode) CodeLocation() types.CodeLocation {
	return node.codeLocation
}

func (node *ContainerNode) Flag() types.FlagType {
	return node.flag
}

//sort.Interface

func (node *ContainerNode) Len() int {
	return len(node.subjectAndContainerNodes)
}

func (node *ContainerNode) Less(i, j int) bool {
	return node.subjectAndContainerNodes[i].text() < node.subjectAndContainerNodes[j].text()
}

func (node *ContainerNode) Swap(i, j int) {
	node.subjectAndContainerNodes[i], node.subjectAndContainerNodes[j] = node.subjectAndContainerNodes[j], node.subjectAndContainerNodes[i]
}
