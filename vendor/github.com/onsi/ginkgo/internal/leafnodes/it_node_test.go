package leafnodes_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/ginkgo/internal/leafnodes"
	. "github.com/onsi/gomega"

	"github.com/onsi/ginkgo/internal/codelocation"
	"github.com/onsi/ginkgo/types"
)

var _ = Describe("It Nodes", func() {
	It("should report the correct type, text, flag, and code location", func() {
		codeLocation := codelocation.New(0)
		it := NewItNode("my it node", func() {}, types.FlagTypeFocused, codeLocation, 0, nil, 3)
		Ω(it.Type()).Should(Equal(types.SpecComponentTypeIt))
		Ω(it.Flag()).Should(Equal(types.FlagTypeFocused))
		Ω(it.Text()).Should(Equal("my it node"))
		Ω(it.CodeLocation()).Should(Equal(codeLocation))
		Ω(it.Samples()).Should(Equal(1))
	})
})
