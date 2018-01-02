package matchers_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	. "github.com/onsi/gomega/matchers"
)

var _ = Describe("HaveCap", func() {
	Context("when passed a supported type", func() {
		It("should do the right thing", func() {
			Ω([0]int{}).Should(HaveCap(0))
			Ω([2]int{1}).Should(HaveCap(2))

			Ω([]int{}).Should(HaveCap(0))
			Ω([]int{1, 2, 3, 4, 5}[:2]).Should(HaveCap(5))
			Ω(make([]int, 0, 5)).Should(HaveCap(5))

			c := make(chan bool, 3)
			Ω(c).Should(HaveCap(3))
			c <- true
			c <- true
			Ω(c).Should(HaveCap(3))

			Ω(make(chan bool)).Should(HaveCap(0))
		})
	})

	Context("when passed a correctly typed nil", func() {
		It("should operate succesfully on the passed in value", func() {
			var nilSlice []int
			Ω(nilSlice).Should(HaveCap(0))

			var nilChan chan int
			Ω(nilChan).Should(HaveCap(0))
		})
	})

	Context("when passed an unsupported type", func() {
		It("should error", func() {
			success, err := (&HaveCapMatcher{Count: 0}).Match(0)
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())

			success, err = (&HaveCapMatcher{Count: 0}).Match(nil)
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())
		})
	})
})
