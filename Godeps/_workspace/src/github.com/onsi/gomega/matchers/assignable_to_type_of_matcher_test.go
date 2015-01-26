package matchers_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	. "github.com/onsi/gomega/matchers"
)

var _ = Describe("AssignableToTypeOf", func() {
	Context("When asserting assignability between types", func() {
		It("should do the right thing", func() {
			Ω(0).Should(BeAssignableToTypeOf(0))
			Ω(5).Should(BeAssignableToTypeOf(-1))
			Ω("foo").Should(BeAssignableToTypeOf("bar"))
			Ω(struct{ Foo string }{}).Should(BeAssignableToTypeOf(struct{ Foo string }{}))

			Ω(0).ShouldNot(BeAssignableToTypeOf("bar"))
			Ω(5).ShouldNot(BeAssignableToTypeOf(struct{ Foo string }{}))
			Ω("foo").ShouldNot(BeAssignableToTypeOf(42))
		})
	})

	Context("When asserting nil values", func() {
		It("should error", func() {
			success, err := (&AssignableToTypeOfMatcher{Expected: nil}).Match(nil)
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())
		})
	})
})
