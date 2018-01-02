package matchers_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	. "github.com/onsi/gomega/matchers"
	"github.com/onsi/gomega/types"
)

// sample data
var (
	// example input
	input = "hi"
	// some matchers that succeed against the input
	true1 = HaveLen(2)
	true2 = Equal("hi")
	true3 = MatchRegexp("hi")
	// some matchers that fail against the input.
	false1 = HaveLen(1)
	false2 = Equal("hip")
	false3 = MatchRegexp("hope")
)

// verifyFailureMessage expects the matcher to fail with the given input, and verifies the failure message.
func verifyFailureMessage(m types.GomegaMatcher, input string, expectedFailureMsgFragment string) {
	Expect(m.Match(input)).To(BeFalse())
	Expect(m.FailureMessage(input)).To(Equal(
		"Expected\n    <string>: " + input + "\n" + expectedFailureMsgFragment))
}

var _ = Describe("AndMatcher", func() {
	It("works with positive cases", func() {
		Expect(input).To(And())
		Expect(input).To(And(true1))
		Expect(input).To(And(true1, true2))
		Expect(input).To(And(true1, true2, true3))

		// use alias
		Expect(input).To(SatisfyAll(true1, true2, true3))
	})

	It("works with negative cases", func() {
		Expect(input).ToNot(And(false1, false2))
		Expect(input).ToNot(And(true1, true2, false3))
		Expect(input).ToNot(And(true1, false2, false3))
		Expect(input).ToNot(And(false1, true1, true2))
	})

	Context("failure messages", func() {
		Context("when match fails", func() {
			It("gives a descriptive message", func() {
				verifyFailureMessage(And(false1, true1), input, "to have length 1")
				verifyFailureMessage(And(true1, false2), input, "to equal\n    <string>: hip")
				verifyFailureMessage(And(true1, true2, false3), input, "to match regular expression\n    <string>: hope")
			})
		})

		Context("when match succeeds, but expected it to fail", func() {
			It("gives a descriptive message", func() {
				verifyFailureMessage(Not(And(true1, true2)), input,
					`To not satisfy all of these matchers: [%!s(*matchers.HaveLenMatcher=&{2}) %!s(*matchers.EqualMatcher=&{hi})]`)
			})
		})
	})

	Context("MatchMayChangeInTheFuture", func() {
		Context("Match returned false", func() {
			Context("returns value of the failed matcher", func() {
				It("false if failed matcher not going to change", func() {
					// 3 matchers: 1st returns true, 2nd returns false and is not going to change, 3rd is never called
					m := And(Not(BeNil()), Or(), Equal(1))
					Expect(m.Match("hi")).To(BeFalse())
					Expect(m.(*AndMatcher).MatchMayChangeInTheFuture("hi")).To(BeFalse()) // empty Or() indicates not going to change
				})
				It("true if failed matcher indicates it might change", func() {
					// 3 matchers: 1st returns true, 2nd returns false and "might" change, 3rd is never called
					m := And(Not(BeNil()), Equal(5), Equal(1))
					Expect(m.Match("hi")).To(BeFalse())
					Expect(m.(*AndMatcher).MatchMayChangeInTheFuture("hi")).To(BeTrue()) // Equal(5) indicates it might change
				})
			})
		})
		Context("Match returned true", func() {
			It("returns true if any of the matchers could change", func() {
				// 3 matchers, all return true, and all could change
				m := And(Not(BeNil()), Equal("hi"), HaveLen(2))
				Expect(m.Match("hi")).To(BeTrue())
				Expect(m.(*AndMatcher).MatchMayChangeInTheFuture("hi")).To(BeTrue()) // all 3 of these matchers default to 'true'
			})
			It("returns false if none of the matchers could change", func() {
				// empty And() has the property of always matching, and never can change since there are no sub-matchers that could change
				m := And()
				Expect(m.Match("anything")).To(BeTrue())
				Expect(m.(*AndMatcher).MatchMayChangeInTheFuture("anything")).To(BeFalse())

				// And() with 3 sub-matchers that return true, and can't change
				m = And(And(), And(), And())
				Expect(m.Match("hi")).To(BeTrue())
				Expect(m.(*AndMatcher).MatchMayChangeInTheFuture("hi")).To(BeFalse()) // the 3 empty And()'s won't change
			})
		})
	})
})
