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
