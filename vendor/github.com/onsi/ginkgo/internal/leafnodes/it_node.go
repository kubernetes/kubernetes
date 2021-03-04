package leafnodes

import (
	"time"

	"github.com/onsi/ginkgo/internal/failer"
	"github.com/onsi/ginkgo/types"
)

type ItNode struct {
	runner *runner

	flag types.FlagType
	text string
}

func NewItNode(text string, body interface{}, flag types.FlagType, codeLocation types.CodeLocation, timeout time.Duration, failer *failer.Failer, componentIndex int) *ItNode {
	return &ItNode{
		runner: newRunner(body, codeLocation, timeout, failer, types.SpecComponentTypeIt, componentIndex),
		flag:   flag,
		text:   text,
	}
}

func (node *ItNode) Run() (outcome types.SpecState, failure types.SpecFailure) {
	return node.runner.run()
}

func (node *ItNode) Type() types.SpecComponentType {
	return types.SpecComponentTypeIt
}

func (node *ItNode) Text() string {
	return node.text
}

func (node *ItNode) Flag() types.FlagType {
	return node.flag
}

func (node *ItNode) CodeLocation() types.CodeLocation {
	return node.runner.codeLocation
}

func (node *ItNode) Samples() int {
	return 1
}
