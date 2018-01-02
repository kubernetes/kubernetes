package matchers_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	. "github.com/onsi/gomega/matchers"
)

var _ = Describe("MatchYAMLMatcher", func() {
	Context("When passed stringifiables", func() {
		It("should succeed if the YAML matches", func() {
			Expect("---").Should(MatchYAML(""))
			Expect("a: 1").Should(MatchYAML(`{"a":1}`))
			Expect("a: 1\nb: 2").Should(MatchYAML(`{"b":2, "a":1}`))
		})

		It("should explain if the YAML does not match when it should", func() {
			message := (&MatchYAMLMatcher{YAMLToMatch: "a: 1"}).FailureMessage("b: 2")
			Expect(message).To(MatchRegexp(`Expected\s+<string>: b: 2\s+to match YAML of\s+<string>: a: 1`))
		})

		It("should normalise the expected and actual when explaining if the YAML does not match when it should", func() {
			message := (&MatchYAMLMatcher{YAMLToMatch: "a: 'one'"}).FailureMessage("{b: two}")
			Expect(message).To(MatchRegexp(`Expected\s+<string>: b: two\s+to match YAML of\s+<string>: a: one`))
		})

		It("should explain if the YAML matches when it should not", func() {
			message := (&MatchYAMLMatcher{YAMLToMatch: "a: 1"}).NegatedFailureMessage("a: 1")
			Expect(message).To(MatchRegexp(`Expected\s+<string>: a: 1\s+not to match YAML of\s+<string>: a: 1`))
		})

		It("should normalise the expected and actual when explaining if the YAML matches when it should not", func() {
			message := (&MatchYAMLMatcher{YAMLToMatch: "a: 'one'"}).NegatedFailureMessage("{a: one}")
			Expect(message).To(MatchRegexp(`Expected\s+<string>: a: one\s+not to match YAML of\s+<string>: a: one`))
		})

		It("should fail if the YAML does not match", func() {
			Expect("a: 1").ShouldNot(MatchYAML(`{"b":2, "a":1}`))
		})

		It("should work with byte arrays", func() {
			Expect([]byte("a: 1")).Should(MatchYAML([]byte("a: 1")))
			Expect("a: 1").Should(MatchYAML([]byte("a: 1")))
			Expect([]byte("a: 1")).Should(MatchYAML("a: 1"))
		})
	})

	Context("when the expected is not valid YAML", func() {
		It("should error and explain why", func() {
			success, err := (&MatchYAMLMatcher{YAMLToMatch: ""}).Match("good:\nbad")
			Expect(success).Should(BeFalse())
			Expect(err).Should(HaveOccurred())
			Expect(err.Error()).Should(ContainSubstring("Actual 'good:\nbad' should be valid YAML"))
		})
	})

	Context("when the actual is not valid YAML", func() {
		It("should error and explain why", func() {
			success, err := (&MatchYAMLMatcher{YAMLToMatch: "good:\nbad"}).Match("")
			Expect(success).Should(BeFalse())
			Expect(err).Should(HaveOccurred())
			Expect(err.Error()).Should(ContainSubstring("Expected 'good:\nbad' should be valid YAML"))
		})
	})

	Context("when the expected is neither a string nor a stringer nor a byte array", func() {
		It("should error", func() {
			success, err := (&MatchYAMLMatcher{YAMLToMatch: 2}).Match("")
			Expect(success).Should(BeFalse())
			Expect(err).Should(HaveOccurred())
			Expect(err.Error()).Should(ContainSubstring("MatchYAMLMatcher matcher requires a string, stringer, or []byte.  Got expected:\n    <int>: 2"))

			success, err = (&MatchYAMLMatcher{YAMLToMatch: nil}).Match("")
			Expect(success).Should(BeFalse())
			Expect(err).Should(HaveOccurred())
			Expect(err.Error()).Should(ContainSubstring("MatchYAMLMatcher matcher requires a string, stringer, or []byte.  Got expected:\n    <nil>: nil"))
		})
	})

	Context("when the actual is neither a string nor a stringer nor a byte array", func() {
		It("should error", func() {
			success, err := (&MatchYAMLMatcher{YAMLToMatch: ""}).Match(2)
			Expect(success).Should(BeFalse())
			Expect(err).Should(HaveOccurred())
			Expect(err.Error()).Should(ContainSubstring("MatchYAMLMatcher matcher requires a string, stringer, or []byte.  Got actual:\n    <int>: 2"))

			success, err = (&MatchYAMLMatcher{YAMLToMatch: ""}).Match(nil)
			Expect(success).Should(BeFalse())
			Expect(err).Should(HaveOccurred())
			Expect(err.Error()).Should(ContainSubstring("MatchYAMLMatcher matcher requires a string, stringer, or []byte.  Got actual:\n    <nil>: nil"))
		})
	})
})
