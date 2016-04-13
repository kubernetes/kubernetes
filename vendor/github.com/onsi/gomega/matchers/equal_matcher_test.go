package matchers_test

import (
	"errors"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	. "github.com/onsi/gomega/matchers"
)

var _ = Describe("Equal", func() {
	Context("when asserting that nil equals nil", func() {
		It("should error", func() {
			success, err := (&EqualMatcher{Expected: nil}).Match(nil)

			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())
		})
	})

	Context("When asserting equality between objects", func() {
		It("should do the right thing", func() {
			Ω(5).Should(Equal(5))
			Ω(5.0).Should(Equal(5.0))

			Ω(5).ShouldNot(Equal("5"))
			Ω(5).ShouldNot(Equal(5.0))
			Ω(5).ShouldNot(Equal(3))

			Ω("5").Should(Equal("5"))
			Ω([]int{1, 2}).Should(Equal([]int{1, 2}))
			Ω([]int{1, 2}).ShouldNot(Equal([]int{2, 1}))
			Ω(map[string]string{"a": "b", "c": "d"}).Should(Equal(map[string]string{"a": "b", "c": "d"}))
			Ω(map[string]string{"a": "b", "c": "d"}).ShouldNot(Equal(map[string]string{"a": "b", "c": "e"}))
			Ω(errors.New("foo")).Should(Equal(errors.New("foo")))
			Ω(errors.New("foo")).ShouldNot(Equal(errors.New("bar")))

			Ω(myCustomType{s: "foo", n: 3, f: 2.0, arr: []string{"a", "b"}}).Should(Equal(myCustomType{s: "foo", n: 3, f: 2.0, arr: []string{"a", "b"}}))
			Ω(myCustomType{s: "foo", n: 3, f: 2.0, arr: []string{"a", "b"}}).ShouldNot(Equal(myCustomType{s: "bar", n: 3, f: 2.0, arr: []string{"a", "b"}}))
			Ω(myCustomType{s: "foo", n: 3, f: 2.0, arr: []string{"a", "b"}}).ShouldNot(Equal(myCustomType{s: "foo", n: 2, f: 2.0, arr: []string{"a", "b"}}))
			Ω(myCustomType{s: "foo", n: 3, f: 2.0, arr: []string{"a", "b"}}).ShouldNot(Equal(myCustomType{s: "foo", n: 3, f: 3.0, arr: []string{"a", "b"}}))
			Ω(myCustomType{s: "foo", n: 3, f: 2.0, arr: []string{"a", "b"}}).ShouldNot(Equal(myCustomType{s: "foo", n: 3, f: 2.0, arr: []string{"a", "b", "c"}}))
		})
	})
})
