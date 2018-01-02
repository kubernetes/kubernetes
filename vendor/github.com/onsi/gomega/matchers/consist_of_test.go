package matchers_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("ConsistOf", func() {
	Context("with a slice", func() {
		It("should do the right thing", func() {
			Ω([]string{"foo", "bar", "baz"}).Should(ConsistOf("foo", "bar", "baz"))
			Ω([]string{"foo", "bar", "baz"}).Should(ConsistOf("foo", "bar", "baz"))
			Ω([]string{"foo", "bar", "baz"}).Should(ConsistOf("baz", "bar", "foo"))
			Ω([]string{"foo", "bar", "baz"}).ShouldNot(ConsistOf("baz", "bar", "foo", "foo"))
			Ω([]string{"foo", "bar", "baz"}).ShouldNot(ConsistOf("baz", "foo"))
		})
	})

	Context("with an array", func() {
		It("should do the right thing", func() {
			Ω([3]string{"foo", "bar", "baz"}).Should(ConsistOf("foo", "bar", "baz"))
			Ω([3]string{"foo", "bar", "baz"}).Should(ConsistOf("baz", "bar", "foo"))
			Ω([3]string{"foo", "bar", "baz"}).ShouldNot(ConsistOf("baz", "bar", "foo", "foo"))
			Ω([3]string{"foo", "bar", "baz"}).ShouldNot(ConsistOf("baz", "foo"))
		})
	})

	Context("with a map", func() {
		It("should apply to the values", func() {
			Ω(map[int]string{1: "foo", 2: "bar", 3: "baz"}).Should(ConsistOf("foo", "bar", "baz"))
			Ω(map[int]string{1: "foo", 2: "bar", 3: "baz"}).Should(ConsistOf("baz", "bar", "foo"))
			Ω(map[int]string{1: "foo", 2: "bar", 3: "baz"}).ShouldNot(ConsistOf("baz", "bar", "foo", "foo"))
			Ω(map[int]string{1: "foo", 2: "bar", 3: "baz"}).ShouldNot(ConsistOf("baz", "foo"))
		})

	})

	Context("with anything else", func() {
		It("should error", func() {
			failures := InterceptGomegaFailures(func() {
				Ω("foo").Should(ConsistOf("f", "o", "o"))
			})

			Ω(failures).Should(HaveLen(1))
		})
	})

	Context("when passed matchers", func() {
		It("should pass if the matchers pass", func() {
			Ω([]string{"foo", "bar", "baz"}).Should(ConsistOf("foo", MatchRegexp("^ba"), "baz"))
			Ω([]string{"foo", "bar", "baz"}).ShouldNot(ConsistOf("foo", MatchRegexp("^ba")))
			Ω([]string{"foo", "bar", "baz"}).ShouldNot(ConsistOf("foo", MatchRegexp("^ba"), MatchRegexp("foo")))
			Ω([]string{"foo", "bar", "baz"}).Should(ConsistOf("foo", MatchRegexp("^ba"), MatchRegexp("^ba")))
			Ω([]string{"foo", "bar", "baz"}).ShouldNot(ConsistOf("foo", MatchRegexp("^ba"), MatchRegexp("turducken")))
		})

		It("should not depend on the order of the matchers", func() {
			Ω([][]int{[]int{1, 2}, []int{2}}).Should(ConsistOf(ContainElement(1), ContainElement(2)))
			Ω([][]int{[]int{1, 2}, []int{2}}).Should(ConsistOf(ContainElement(2), ContainElement(1)))
		})

		Context("when a matcher errors", func() {
			It("should soldier on", func() {
				Ω([]string{"foo", "bar", "baz"}).ShouldNot(ConsistOf(BeFalse(), "foo", "bar"))
				Ω([]interface{}{"foo", "bar", false}).Should(ConsistOf(BeFalse(), ContainSubstring("foo"), "bar"))
			})
		})
	})

	Context("when passed exactly one argument, and that argument is a slice", func() {
		It("should match against the elements of that argument", func() {
			Ω([]string{"foo", "bar", "baz"}).Should(ConsistOf([]string{"foo", "bar", "baz"}))
		})
	})
})
