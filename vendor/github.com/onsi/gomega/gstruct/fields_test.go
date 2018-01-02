package gstruct_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	. "github.com/onsi/gomega/gstruct"
)

var _ = Describe("Struct", func() {
	allFields := struct{ A, B string }{"a", "b"}
	missingFields := struct{ A string }{"a"}
	extraFields := struct{ A, B, C string }{"a", "b", "c"}
	emptyFields := struct{ A, B string }{}

	It("should strictly match all fields", func() {
		m := MatchAllFields(Fields{
			"B": Equal("b"),
			"A": Equal("a"),
		})
		Ω(allFields).Should(m, "should match all fields")
		Ω(missingFields).ShouldNot(m, "should fail with missing fields")
		Ω(extraFields).ShouldNot(m, "should fail with extra fields")
		Ω(emptyFields).ShouldNot(m, "should fail with empty fields")

		m = MatchAllFields(Fields{
			"A": Equal("a"),
			"B": Equal("fail"),
		})
		Ω(allFields).ShouldNot(m, "should run nested matchers")
	})

	It("should handle empty structs", func() {
		m := MatchAllFields(Fields{})
		Ω(struct{}{}).Should(m, "should handle empty structs")
		Ω(allFields).ShouldNot(m, "should fail with extra fields")
	})

	It("should ignore missing fields", func() {
		m := MatchFields(IgnoreMissing, Fields{
			"B": Equal("b"),
			"A": Equal("a"),
		})
		Ω(allFields).Should(m, "should match all fields")
		Ω(missingFields).Should(m, "should ignore missing fields")
		Ω(extraFields).ShouldNot(m, "should fail with extra fields")
		Ω(emptyFields).ShouldNot(m, "should fail with empty fields")
	})

	It("should ignore extra fields", func() {
		m := MatchFields(IgnoreExtras, Fields{
			"B": Equal("b"),
			"A": Equal("a"),
		})
		Ω(allFields).Should(m, "should match all fields")
		Ω(missingFields).ShouldNot(m, "should fail with missing fields")
		Ω(extraFields).Should(m, "should ignore extra fields")
		Ω(emptyFields).ShouldNot(m, "should fail with empty fields")
	})

	It("should ignore missing and extra fields", func() {
		m := MatchFields(IgnoreMissing|IgnoreExtras, Fields{
			"B": Equal("b"),
			"A": Equal("a"),
		})
		Ω(allFields).Should(m, "should match all fields")
		Ω(missingFields).Should(m, "should ignore missing fields")
		Ω(extraFields).Should(m, "should ignore extra fields")
		Ω(emptyFields).ShouldNot(m, "should fail with empty fields")

		m = MatchFields(IgnoreMissing|IgnoreExtras, Fields{
			"A": Equal("a"),
			"B": Equal("fail"),
		})
		Ω(allFields).ShouldNot(m, "should run nested matchers")
	})
})
