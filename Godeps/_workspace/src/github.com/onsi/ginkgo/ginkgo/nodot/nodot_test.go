package nodot_test

import (
	. "github.com/onsi/ginkgo/ginkgo/nodot"
	"strings"
)

var _ = Describe("ApplyNoDot", func() {
	var result string

	apply := func(input string) string {
		output, err := ApplyNoDot([]byte(input))
		Ω(err).ShouldNot(HaveOccurred())
		return string(output)
	}

	Context("when no declarations have been imported yet", func() {
		BeforeEach(func() {
			result = apply("")
		})

		It("should add headings for the various declarations", func() {
			Ω(result).Should(ContainSubstring("// Declarations for Ginkgo DSL"))
			Ω(result).Should(ContainSubstring("// Declarations for Gomega DSL"))
			Ω(result).Should(ContainSubstring("// Declarations for Gomega Matchers"))
		})

		It("should import Ginkgo's declarations", func() {
			Ω(result).Should(ContainSubstring("var It = ginkgo.It"))
			Ω(result).Should(ContainSubstring("var XDescribe = ginkgo.XDescribe"))
		})

		It("should import Ginkgo's types", func() {
			Ω(result).Should(ContainSubstring("type Done ginkgo.Done"))
			Ω(result).Should(ContainSubstring("type Benchmarker ginkgo.Benchmarker"))
			Ω(strings.Count(result, "type ")).Should(Equal(2))
		})

		It("should import Gomega's DSL and matchers", func() {
			Ω(result).Should(ContainSubstring("var Ω = gomega.Ω"))
			Ω(result).Should(ContainSubstring("var ContainSubstring = gomega.ContainSubstring"))
			Ω(result).Should(ContainSubstring("var Equal = gomega.Equal"))
		})

		It("should not import blacklisted things", func() {
			Ω(result).ShouldNot(ContainSubstring("GINKGO_VERSION"))
			Ω(result).ShouldNot(ContainSubstring("GINKGO_PANIC"))
			Ω(result).ShouldNot(ContainSubstring("GOMEGA_VERSION"))
		})
	})

	It("should be idempotent (module empty lines - go fmt can fix those for us)", func() {
		first := apply("")
		second := apply(first)
		first = strings.Trim(first, "\n")
		second = strings.Trim(second, "\n")
		Ω(first).Should(Equal(second))
	})

	It("should not mess with other things in the input", func() {
		result = apply("var MyThing = SomethingThatsMine")
		Ω(result).Should(ContainSubstring("var MyThing = SomethingThatsMine"))
	})

	Context("when the user has redefined a name", func() {
		It("should honor the redefinition", func() {
			result = apply(`
var _ = gomega.Ω
var When = ginkgo.It
            `)

			Ω(result).Should(ContainSubstring("var _ = gomega.Ω"))
			Ω(result).ShouldNot(ContainSubstring("var Ω = gomega.Ω"))

			Ω(result).Should(ContainSubstring("var When = ginkgo.It"))
			Ω(result).ShouldNot(ContainSubstring("var It = ginkgo.It"))

			Ω(result).Should(ContainSubstring("var Context = ginkgo.Context"))
		})
	})
})
