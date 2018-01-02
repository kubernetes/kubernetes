package matchers_test

import (
	"errors"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	. "github.com/onsi/gomega/matchers"
)

var _ = Describe("WithTransformMatcher", func() {

	var plus1 = func(i int) int { return i + 1 }

	Context("Panic if transform function invalid", func() {
		panicsWithTransformer := func(transform interface{}) {
			ExpectWithOffset(1, func() { WithTransform(transform, nil) }).To(Panic())
		}
		It("nil", func() {
			panicsWithTransformer(nil)
		})
		Context("Invalid number of args, but correct return value count", func() {
			It("zero", func() {
				panicsWithTransformer(func() int { return 5 })
			})
			It("two", func() {
				panicsWithTransformer(func(i, j int) int { return 5 })
			})
		})
		Context("Invalid number of return values, but correct number of arguments", func() {
			It("zero", func() {
				panicsWithTransformer(func(i int) {})
			})
			It("two", func() {
				panicsWithTransformer(func(i int) (int, int) { return 5, 6 })
			})
		})
	})

	It("works with positive cases", func() {
		Expect(1).To(WithTransform(plus1, Equal(2)))
		Expect(1).To(WithTransform(plus1, WithTransform(plus1, Equal(3))))
		Expect(1).To(WithTransform(plus1, And(Equal(2), BeNumerically(">", 1))))

		// transform expects custom type
		type S struct {
			A int
			B string
		}
		transformer := func(s S) string { return s.B }
		Expect(S{1, "hi"}).To(WithTransform(transformer, Equal("hi")))

		// transform expects interface
		errString := func(e error) string { return e.Error() }
		Expect(errors.New("abc")).To(WithTransform(errString, Equal("abc")))
	})

	It("works with negative cases", func() {
		Expect(1).ToNot(WithTransform(plus1, Equal(3)))
		Expect(1).ToNot(WithTransform(plus1, WithTransform(plus1, Equal(2))))
	})

	Context("failure messages", func() {
		Context("when match fails", func() {
			It("gives a descriptive message", func() {
				m := WithTransform(plus1, Equal(3))
				Expect(m.Match(1)).To(BeFalse())
				Expect(m.FailureMessage(1)).To(Equal("Expected\n    <int>: 2\nto equal\n    <int>: 3"))
			})
		})

		Context("when match succeeds, but expected it to fail", func() {
			It("gives a descriptive message", func() {
				m := Not(WithTransform(plus1, Equal(3)))
				Expect(m.Match(2)).To(BeFalse())
				Expect(m.FailureMessage(2)).To(Equal("Expected\n    <int>: 3\nnot to equal\n    <int>: 3"))
			})
		})

		Context("actual value is incompatible with transform function's argument type", func() {
			It("gracefully fails if transform cannot be performed", func() {
				m := WithTransform(plus1, Equal(3))
				result, err := m.Match("hi") // give it a string but transform expects int; doesn't panic
				Expect(result).To(BeFalse())
				Expect(err).To(MatchError("Transform function expects 'int' but we have 'string'"))
			})
		})
	})

	Context("MatchMayChangeInTheFuture()", func() {
		It("Propagates value from wrapped matcher on the transformed value", func() {
			m := WithTransform(plus1, Or()) // empty Or() always returns false, and indicates it cannot change
			Expect(m.Match(1)).To(BeFalse())
			Expect(m.(*WithTransformMatcher).MatchMayChangeInTheFuture(1)).To(BeFalse()) // empty Or() indicates cannot change
		})
		It("Defaults to true", func() {
			m := WithTransform(plus1, Equal(2)) // Equal does not have this method
			Expect(m.Match(1)).To(BeTrue())
			Expect(m.(*WithTransformMatcher).MatchMayChangeInTheFuture(1)).To(BeTrue()) // defaults to true
		})
	})
})
