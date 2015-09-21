package matchers_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	. "github.com/onsi/gomega/matchers"
)

var _ = Describe("MatchRegexp", func() {
	Context("when actual is a string", func() {
		It("should match against the string", func() {
			Ω(" a2!bla").Should(MatchRegexp(`\d!`))
			Ω(" a2!bla").ShouldNot(MatchRegexp(`[A-Z]`))
		})
	})

	Context("when actual is a stringer", func() {
		It("should call the stringer and match agains the returned string", func() {
			Ω(&myStringer{a: "Abc3"}).Should(MatchRegexp(`[A-Z][a-z]+\d`))
		})
	})

	Context("when the matcher is called with multiple arguments", func() {
		It("should pass the string and arguments to sprintf", func() {
			Ω(" a23!bla").Should(MatchRegexp(`\d%d!`, 3))
		})
	})

	Context("when actual is neither a string nor a stringer", func() {
		It("should error", func() {
			success, err := (&MatchRegexpMatcher{Regexp: `\d`}).Match(2)
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())
		})
	})

	Context("when the passed in regexp fails to compile", func() {
		It("should error", func() {
			success, err := (&MatchRegexpMatcher{Regexp: "("}).Match("Foo")
			Ω(success).Should(BeFalse())
			Ω(err).Should(HaveOccurred())
		})
	})
})
