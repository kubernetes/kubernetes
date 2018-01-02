package matchers_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	. "github.com/onsi/gomega/matchers"
)

var _ = Describe("OrMatcher", func() {
	It("works with positive cases", func() {
		Expect(input).To(Or(true1))
		Expect(input).To(Or(true1, true2))
		Expect(input).To(Or(true1, false1))
		Expect(input).To(Or(false1, true2))
		Expect(input).To(Or(true1, true2, true3))
		Expect(input).To(Or(true1, true2, false3))
		Expect(input).To(Or(true1, false2, true3))
		Expect(input).To(Or(false1, true2, true3))
		Expect(input).To(Or(true1, false2, false3))
		Expect(input).To(Or(false1, false2, true3))

		// use alias
		Expect(input).To(SatisfyAny(false1, false2, true3))
	})

	It("works with negative cases", func() {
		Expect(input).ToNot(Or())
		Expect(input).ToNot(Or(false1))
		Expect(input).ToNot(Or(false1, false2))
		Expect(input).ToNot(Or(false1, false2, false3))
	})

	Context("failure messages", func() {
		Context("when match fails", func() {
			It("gives a descriptive message", func() {
				verifyFailureMessage(Or(false1, false2), input,
					"To satisfy at least one of these matchers: [%!s(*matchers.HaveLenMatcher=&{1}) %!s(*matchers.EqualMatcher=&{hip})]")
			})
		})

		Context("when match succeeds, but expected it to fail", func() {
			It("gives a descriptive message", func() {
				verifyFailureMessage(Not(Or(true1, true2)), input, `not to have length 2`)
			})
		})
	})

	Context("MatchMayChangeInTheFuture", func() {
		Context("Match returned false", func() {
			It("returns true if any of the matchers could change", func() {
				// 3 matchers, all return false, and all could change
				m := Or(BeNil(), Equal("hip"), HaveLen(1))
				Expect(m.Match("hi")).To(BeFalse())
				Expect(m.(*OrMatcher).MatchMayChangeInTheFuture("hi")).To(BeTrue()) // all 3 of these matchers default to 'true'
			})
			It("returns false if none of the matchers could change", func() {
				// empty Or() has the property of never matching, and never can change since there are no sub-matchers that could change
				m := Or()
				Expect(m.Match("anything")).To(BeFalse())
				Expect(m.(*OrMatcher).MatchMayChangeInTheFuture("anything")).To(BeFalse())

				// Or() with 3 sub-matchers that return false, and can't change
				m = Or(Or(), Or(), Or())
				Expect(m.Match("hi")).To(BeFalse())
				Expect(m.(*OrMatcher).MatchMayChangeInTheFuture("hi")).To(BeFalse()) // the 3 empty Or()'s won't change
			})
		})
		Context("Match returned true", func() {
			Context("returns value of the successful matcher", func() {
				It("false if successful matcher not going to change", func() {
					// 3 matchers: 1st returns false, 2nd returns true and is not going to change, 3rd is never called
					m := Or(BeNil(), And(), Equal(1))
					Expect(m.Match("hi")).To(BeTrue())
					Expect(m.(*OrMatcher).MatchMayChangeInTheFuture("hi")).To(BeFalse())
				})
				It("true if successful matcher indicates it might change", func() {
					// 3 matchers: 1st returns false, 2nd returns true and "might" change, 3rd is never called
					m := Or(Not(BeNil()), Equal("hi"), Equal(1))
					Expect(m.Match("hi")).To(BeTrue())
					Expect(m.(*OrMatcher).MatchMayChangeInTheFuture("hi")).To(BeTrue()) // Equal("hi") indicates it might change
				})
			})
		})
	})
})
