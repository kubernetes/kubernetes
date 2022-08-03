package internal

import (
	"strings"

	"github.com/onsi/ginkgo/v2/types"
)

type Spec struct {
	Nodes Nodes
	Skip  bool
}

func (s Spec) SubjectID() uint {
	return s.Nodes.FirstNodeWithType(types.NodeTypeIt).ID
}

func (s Spec) Text() string {
	texts := []string{}
	for i := range s.Nodes {
		if s.Nodes[i].Text != "" {
			texts = append(texts, s.Nodes[i].Text)
		}
	}
	return strings.Join(texts, " ")
}

func (s Spec) FirstNodeWithType(nodeTypes types.NodeType) Node {
	return s.Nodes.FirstNodeWithType(nodeTypes)
}

func (s Spec) FlakeAttempts() int {
	flakeAttempts := 0
	for i := range s.Nodes {
		if s.Nodes[i].FlakeAttempts > 0 {
			flakeAttempts = s.Nodes[i].FlakeAttempts
		}
	}

	return flakeAttempts
}

type Specs []Spec

func (s Specs) HasAnySpecsMarkedPending() bool {
	for i := range s {
		if s[i].Nodes.HasNodeMarkedPending() {
			return true
		}
	}

	return false
}

func (s Specs) CountWithoutSkip() int {
	n := 0
	for i := range s {
		if !s[i].Skip {
			n += 1
		}
	}
	return n
}

func (s Specs) AtIndices(indices SpecIndices) Specs {
	out := make(Specs, len(indices))
	for i, idx := range indices {
		out[i] = s[idx]
	}
	return out
}
