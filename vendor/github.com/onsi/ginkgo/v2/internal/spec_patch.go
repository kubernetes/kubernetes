package internal

import (
	"github.com/onsi/ginkgo/v2/types"
)

func (s Spec) CodeLocations() []types.CodeLocation {
	return s.Nodes.CodeLocations()
}

func (s Spec) AppendText(text string) {
	s.Nodes[len(s.Nodes)-1].Text += text
}

func (s Spec) Labels() []string {
	var labels []string
	for _, n := range s.Nodes {
		labels = append(labels, n.Labels...)
	}

	return labels
}
