package leafnodes

import (
	"github.com/onsi/ginkgo/types"
)

func (node *ItNode) SetText(text string) {
	node.text = text
}

func (node *ItNode) SetFlag(flag types.FlagType) {
	node.flag = flag
}
