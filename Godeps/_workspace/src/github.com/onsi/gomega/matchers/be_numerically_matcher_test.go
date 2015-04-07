package matchers_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	. "github.com/onsi/gomega/matchers"
)

var _ = Describe("BeNumerically", func() {
	Context("when passed a number", func() {
		It("should support ==", func() {
			Ω(uint32(5)).Should(BeNumerically("==", 5))
			Ω(float64(5.0)).Should(BeNumerically("==", 5))
			Ω(int8(5)).Should(BeNumerically("==", 5))
		})

		It("should not have false positives", func() {
			Ω(5.1).ShouldNot(BeNumerically("==", 5))
			Ω(5).ShouldNot(BeNumerically("==", 5.1))
		})

		It("should support >", func() {
			Ω(uint32(5)).Should(BeNumerically(">", 4))
			Ω(float64(5.0)).Should(BeNumerically(">", 4.9))
			Ω(int8(5)).Should(BeNumerically(">", 4))

			Ω(uint32(5)).ShouldNot(BeNumerically(">", 5))
			Ω(float64(5.0)).ShouldNot(BeNumerically(">", 5.0))
			Ω(int8(5)).ShouldNot(BeNumerically(">", 5))
		})

		It("should support <", func() {
			Ω(uint32(5)).Should(BeNumerically("<", 6))
			Ω(float64(5.0)).Should(BeNumerically("<", 5.1))
			Ω(int8(5)).Should(BeNumerically("<", 6))

			Ω(uint32(5)).ShouldNot(BeNumerically("<", 5))
			Ω(float64(5.0)).ShouldNot(BeNumerically("<", 5.0))
			Ω(int8(5)).ShouldNot(BeNumerically("<", 5))
		})

		It("should support >=", func() {
			Ω(uint32(5)).Should(BeNumerically(">=", 4))
			Ω(float64(5.0)).Should(BeNumerically(">=", 4.9))
			Ω(int8(5)).Should(BeNumerically(">=", 4))

			Ω(uint32(5)).Should(BeNumerically(">=", 5))
			Ω(float64(5.0)).Should(BeNumerically(">=", 5.0))
			Ω(int8(5)).Should(BeNumerically(">=", 5))

			Ω(uint32(5)).ShouldNot(BeNumerically(">=", 6))
			Ω(float64(5.0)).ShouldNot(BeNumerically(">=", 5.1))
			Ω(int8(5)).ShouldNot(BeNumerically(">=", 6))
		})

		It("should support <=", func() {
			Ω(uint32(5)).Should(BeNumerically("<=", 6))
			Ω(float64(5.0)).Should(BeNumerically("<=", 5.1))
			Ω(int8(5)).Should(BeNumerically("<=", 6))

			Ω(uint32(5)).Should(BeNumerically("<=", 5))
			Ω(float64(5.0)).Should(BeNumerically("<=", 5.0))
			Ω(int8(5)).Should(BeNumerically("<=", 5))

			Ω(uint32(5)).ShouldNot(BeNumerically("<=", 4))
			Ω(float64(5.0)).ShouldNot(BeNumerically("<=", 4.9))
			Ω(int8(5)).Should(BeNumerically("<=", 5))
		})

		Context("when passed ~", func() {
			Context("when passed a float", func() {
				Context("and there is no precision parameter", func() {
					It("should default to 1e-8", func() {
						Ω(5.00000001).Should(BeNumerically("~", 5.00000002))
						Ω(5.00000001).ShouldNot(BeNumerically("~", 5.0000001))
					})
				})

				Context("and there is a precision parameter", func() {
					It("should use the precision parameter", func() {
						Ω(5.1).Should(BeNumerically("~", 5.19, 0.1))
						Ω(5.1).Should(BeNumerically("~", 5.01, 0.1))
						Ω(5.1).ShouldNot(BeNumerically("~", 5.22, 0.1))
						Ω(5.1).ShouldNot(BeNumerically("~", 4.98, 0.1))
					})
				})
			})

			Context("when passed an int/uint", func() {
				Context("and there is no precision parameter", func() {
					It("should just do strict equality", func() {
						Ω(5).Should(BeNumerically("~", 5))
						Ω(5).ShouldNot(BeNumerically("~", 6))
						Ω(uint(5)).ShouldNot(BeNumerically("~", 6))
					})
				})

				Context("and there is a precision parameter", func() {
					It("should use precision paramter", func() {
						Ω(5).Should(BeNumerically("~", 6, 2))
						Ω(5).ShouldNot(BeNumerically("~", 8, 2))
						Ω(uint(5)).Should(BeNumerically("~", 6, 1))
					})
				})
			})
		})
	})

	Context("when passed a non-number", func() {
		It("should error", func() {
			success, err := (&BeNumericallyMatcher{Comparator: "==", CompareTo: []interface{}{5}}).Match("foo")
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())

			success, err = (&BeNumericallyMatcher{Comparator: "=="}).Match(5)
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())

			success, err = (&BeNumericallyMatcher{Comparator: "~", CompareTo: []interface{}{3.0, "foo"}}).Match(5.0)
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())

			success, err = (&BeNumericallyMatcher{Comparator: "==", CompareTo: []interface{}{"bar"}}).Match(5)
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())

			success, err = (&BeNumericallyMatcher{Comparator: "==", CompareTo: []interface{}{"bar"}}).Match("foo")
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())

			success, err = (&BeNumericallyMatcher{Comparator: "==", CompareTo: []interface{}{nil}}).Match(0)
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())

			success, err = (&BeNumericallyMatcher{Comparator: "==", CompareTo: []interface{}{0}}).Match(nil)
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())
		})
	})

	Context("when passed an unsupported comparator", func() {
		It("should error", func() {
			success, err := (&BeNumericallyMatcher{Comparator: "!=", CompareTo: []interface{}{5}}).Match(4)
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())
		})
	})
})
