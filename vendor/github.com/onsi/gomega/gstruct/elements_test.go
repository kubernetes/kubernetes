package gstruct_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	. "github.com/onsi/gomega/gstruct"
)

var _ = Describe("Slice", func() {
	allElements := []string{"a", "b"}
	missingElements := []string{"a"}
	extraElements := []string{"a", "b", "c"}
	duplicateElements := []string{"a", "a", "b"}
	empty := []string{}
	var nils []string

	It("should strictly match all elements", func() {
		m := MatchAllElements(id, Elements{
			"b": Equal("b"),
			"a": Equal("a"),
		})
		Ω(allElements).Should(m, "should match all elements")
		Ω(missingElements).ShouldNot(m, "should fail with missing elements")
		Ω(extraElements).ShouldNot(m, "should fail with extra elements")
		Ω(duplicateElements).ShouldNot(m, "should fail with duplicate elements")
		Ω(nils).ShouldNot(m, "should fail with an uninitialized slice")

		m = MatchAllElements(id, Elements{
			"a": Equal("a"),
			"b": Equal("fail"),
		})
		Ω(allElements).ShouldNot(m, "should run nested matchers")

		m = MatchAllElements(id, Elements{})
		Ω(empty).Should(m, "should handle empty slices")
		Ω(allElements).ShouldNot(m, "should handle only empty slices")
		Ω(nils).Should(m, "should handle nil slices")
	})

	It("should ignore extra elements", func() {
		m := MatchElements(id, IgnoreExtras, Elements{
			"b": Equal("b"),
			"a": Equal("a"),
		})
		Ω(allElements).Should(m, "should match all elements")
		Ω(missingElements).ShouldNot(m, "should fail with missing elements")
		Ω(extraElements).Should(m, "should ignore extra elements")
		Ω(duplicateElements).ShouldNot(m, "should fail with duplicate elements")
		Ω(nils).ShouldNot(m, "should fail with an uninitialized slice")
	})

	It("should ignore missing elements", func() {
		m := MatchElements(id, IgnoreMissing, Elements{
			"a": Equal("a"),
			"b": Equal("b"),
		})
		Ω(allElements).Should(m, "should match all elements")
		Ω(missingElements).Should(m, "should ignore missing elements")
		Ω(extraElements).ShouldNot(m, "should fail with extra elements")
		Ω(duplicateElements).ShouldNot(m, "should fail with duplicate elements")
		Ω(nils).Should(m, "should ignore an uninitialized slice")
	})

	It("should ignore missing and extra elements", func() {
		m := MatchElements(id, IgnoreMissing|IgnoreExtras, Elements{
			"a": Equal("a"),
			"b": Equal("b"),
		})
		Ω(allElements).Should(m, "should match all elements")
		Ω(missingElements).Should(m, "should ignore missing elements")
		Ω(extraElements).Should(m, "should ignore extra elements")
		Ω(duplicateElements).ShouldNot(m, "should fail with duplicate elements")
		Ω(nils).Should(m, "should ignore an uninitialized slice")

		m = MatchElements(id, IgnoreExtras|IgnoreMissing, Elements{
			"a": Equal("a"),
			"b": Equal("fail"),
		})
		Ω(allElements).ShouldNot(m, "should run nested matchers")
	})
})

func id(element interface{}) string {
	return element.(string)
}
