package matchers_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	. "github.com/onsi/gomega/matchers"
)

var _ = Describe("BeEmpty", func() {
	Context("when passed a supported type", func() {
		It("should do the right thing", func() {
			Ω("").Should(BeEmpty())
			Ω(" ").ShouldNot(BeEmpty())

			Ω([0]int{}).Should(BeEmpty())
			Ω([1]int{1}).ShouldNot(BeEmpty())

			Ω([]int{}).Should(BeEmpty())
			Ω([]int{1}).ShouldNot(BeEmpty())

			Ω(map[string]int{}).Should(BeEmpty())
			Ω(map[string]int{"a": 1}).ShouldNot(BeEmpty())

			c := make(chan bool, 1)
			Ω(c).Should(BeEmpty())
			c <- true
			Ω(c).ShouldNot(BeEmpty())
		})
	})

	Context("when passed a correctly typed nil", func() {
		It("should be true", func() {
			var nilSlice []int
			Ω(nilSlice).Should(BeEmpty())

			var nilMap map[int]string
			Ω(nilMap).Should(BeEmpty())
		})
	})

	Context("when passed an unsupported type", func() {
		It("should error", func() {
			success, err := (&BeEmptyMatcher{}).Match(0)
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())

			success, err = (&BeEmptyMatcher{}).Match(nil)
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())
		})
	})
})
