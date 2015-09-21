package matchers_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	. "github.com/onsi/gomega/matchers"
)

var _ = Describe("BeEquivalentTo", func() {
	Context("when asserting that nil is equivalent to nil", func() {
		It("should error", func() {
			success, err := (&BeEquivalentToMatcher{Expected: nil}).Match(nil)

			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())
		})
	})

	Context("When asserting on nil", func() {
		It("should do the right thing", func() {
			Ω("foo").ShouldNot(BeEquivalentTo(nil))
			Ω(nil).ShouldNot(BeEquivalentTo(3))
			Ω([]int{1, 2}).ShouldNot(BeEquivalentTo(nil))
		})
	})

	Context("When asserting on type aliases", func() {
		It("should the right thing", func() {
			Ω(StringAlias("foo")).Should(BeEquivalentTo("foo"))
			Ω("foo").Should(BeEquivalentTo(StringAlias("foo")))
			Ω(StringAlias("foo")).ShouldNot(BeEquivalentTo("bar"))
			Ω("foo").ShouldNot(BeEquivalentTo(StringAlias("bar")))
		})
	})

	Context("When asserting on numbers", func() {
		It("should convert actual to expected and do the right thing", func() {
			Ω(5).Should(BeEquivalentTo(5))
			Ω(5.0).Should(BeEquivalentTo(5.0))
			Ω(5).Should(BeEquivalentTo(5.0))

			Ω(5).ShouldNot(BeEquivalentTo("5"))
			Ω(5).ShouldNot(BeEquivalentTo(3))

			//Here be dragons!
			Ω(5.1).Should(BeEquivalentTo(5))
			Ω(5).ShouldNot(BeEquivalentTo(5.1))
		})
	})
})
