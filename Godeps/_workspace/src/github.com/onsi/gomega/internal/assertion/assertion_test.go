package assertion_test

import (
	"errors"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	. "github.com/onsi/gomega/internal/assertion"
	"github.com/onsi/gomega/internal/fakematcher"
)

var _ = Describe("Assertion", func() {
	var (
		a                 *Assertion
		failureMessage    string
		failureCallerSkip int
		matcher           *fakematcher.FakeMatcher
	)

	input := "The thing I'm testing"

	var fakeFailHandler = func(message string, callerSkip ...int) {
		failureMessage = message
		if len(callerSkip) == 1 {
			failureCallerSkip = callerSkip[0]
		}
	}

	BeforeEach(func() {
		matcher = &fakematcher.FakeMatcher{}
		failureMessage = ""
		failureCallerSkip = 0
		a = New(input, fakeFailHandler, 1)
	})

	Context("when called", func() {
		It("should pass the provided input value to the matcher", func() {
			a.Should(matcher)

			Ω(matcher.ReceivedActual).Should(Equal(input))
			matcher.ReceivedActual = ""

			a.ShouldNot(matcher)

			Ω(matcher.ReceivedActual).Should(Equal(input))
			matcher.ReceivedActual = ""

			a.To(matcher)

			Ω(matcher.ReceivedActual).Should(Equal(input))
			matcher.ReceivedActual = ""

			a.ToNot(matcher)

			Ω(matcher.ReceivedActual).Should(Equal(input))
			matcher.ReceivedActual = ""

			a.NotTo(matcher)

			Ω(matcher.ReceivedActual).Should(Equal(input))
		})
	})

	Context("when the matcher succeeds", func() {
		BeforeEach(func() {
			matcher.MatchesToReturn = true
			matcher.ErrToReturn = nil
		})

		Context("and a positive assertion is being made", func() {
			It("should not call the failure callback", func() {
				a.Should(matcher)
				Ω(failureMessage).Should(Equal(""))
			})

			It("should be true", func() {
				Ω(a.Should(matcher)).Should(BeTrue())
			})
		})

		Context("and a negative assertion is being made", func() {
			It("should call the failure callback", func() {
				a.ShouldNot(matcher)
				Ω(failureMessage).Should(Equal("negative: The thing I'm testing"))
				Ω(failureCallerSkip).Should(Equal(3))
			})

			It("should be false", func() {
				Ω(a.ShouldNot(matcher)).Should(BeFalse())
			})
		})
	})

	Context("when the matcher fails", func() {
		BeforeEach(func() {
			matcher.MatchesToReturn = false
			matcher.ErrToReturn = nil
		})

		Context("and a positive assertion is being made", func() {
			It("should call the failure callback", func() {
				a.Should(matcher)
				Ω(failureMessage).Should(Equal("positive: The thing I'm testing"))
				Ω(failureCallerSkip).Should(Equal(3))
			})

			It("should be false", func() {
				Ω(a.Should(matcher)).Should(BeFalse())
			})
		})

		Context("and a negative assertion is being made", func() {
			It("should not call the failure callback", func() {
				a.ShouldNot(matcher)
				Ω(failureMessage).Should(Equal(""))
			})

			It("should be true", func() {
				Ω(a.ShouldNot(matcher)).Should(BeTrue())
			})
		})
	})

	Context("When reporting a failure", func() {
		BeforeEach(func() {
			matcher.MatchesToReturn = false
			matcher.ErrToReturn = nil
		})

		Context("and there is an optional description", func() {
			It("should append the description to the failure message", func() {
				a.Should(matcher, "A description")
				Ω(failureMessage).Should(Equal("A description\npositive: The thing I'm testing"))
				Ω(failureCallerSkip).Should(Equal(3))
			})
		})

		Context("and there are multiple arguments to the optional description", func() {
			It("should append the formatted description to the failure message", func() {
				a.Should(matcher, "A description of [%d]", 3)
				Ω(failureMessage).Should(Equal("A description of [3]\npositive: The thing I'm testing"))
				Ω(failureCallerSkip).Should(Equal(3))
			})
		})
	})

	Context("When the matcher returns an error", func() {
		BeforeEach(func() {
			matcher.ErrToReturn = errors.New("Kaboom!")
		})

		Context("and a positive assertion is being made", func() {
			It("should call the failure callback", func() {
				matcher.MatchesToReturn = true
				a.Should(matcher)
				Ω(failureMessage).Should(Equal("Kaboom!"))
				Ω(failureCallerSkip).Should(Equal(3))
			})
		})

		Context("and a negative assertion is being made", func() {
			It("should call the failure callback", func() {
				matcher.MatchesToReturn = false
				a.ShouldNot(matcher)
				Ω(failureMessage).Should(Equal("Kaboom!"))
				Ω(failureCallerSkip).Should(Equal(3))
			})
		})

		It("should always be false", func() {
			Ω(a.Should(matcher)).Should(BeFalse())
			Ω(a.ShouldNot(matcher)).Should(BeFalse())
		})
	})

	Context("when there are extra parameters", func() {
		It("(a simple example)", func() {
			Ω(func() (string, int, error) {
				return "foo", 0, nil
			}()).Should(Equal("foo"))
		})

		Context("when the parameters are all nil or zero", func() {
			It("should invoke the matcher", func() {
				matcher.MatchesToReturn = true
				matcher.ErrToReturn = nil

				var typedNil []string
				a = New(input, fakeFailHandler, 1, 0, nil, typedNil)

				result := a.Should(matcher)
				Ω(result).Should(BeTrue())
				Ω(matcher.ReceivedActual).Should(Equal(input))

				Ω(failureMessage).Should(BeZero())
			})
		})

		Context("when any of the parameters are not nil or zero", func() {
			It("should call the failure callback", func() {
				matcher.MatchesToReturn = false
				matcher.ErrToReturn = nil

				a = New(input, fakeFailHandler, 1, errors.New("foo"))
				result := a.Should(matcher)
				Ω(result).Should(BeFalse())
				Ω(matcher.ReceivedActual).Should(BeZero(), "The matcher doesn't even get called")
				Ω(failureMessage).Should(ContainSubstring("foo"))
				failureMessage = ""

				a = New(input, fakeFailHandler, 1, nil, 1)
				result = a.ShouldNot(matcher)
				Ω(result).Should(BeFalse())
				Ω(failureMessage).Should(ContainSubstring("1"))
				failureMessage = ""

				a = New(input, fakeFailHandler, 1, nil, 0, []string{"foo"})
				result = a.To(matcher)
				Ω(result).Should(BeFalse())
				Ω(failureMessage).Should(ContainSubstring("foo"))
				failureMessage = ""

				a = New(input, fakeFailHandler, 1, nil, 0, []string{"foo"})
				result = a.ToNot(matcher)
				Ω(result).Should(BeFalse())
				Ω(failureMessage).Should(ContainSubstring("foo"))
				failureMessage = ""

				a = New(input, fakeFailHandler, 1, nil, 0, []string{"foo"})
				result = a.NotTo(matcher)
				Ω(result).Should(BeFalse())
				Ω(failureMessage).Should(ContainSubstring("foo"))
				Ω(failureCallerSkip).Should(Equal(3))
			})
		})
	})

	Context("Making an assertion without a registered fail handler", func() {
		It("should panic", func() {
			defer func() {
				e := recover()
				RegisterFailHandler(Fail)
				if e == nil {
					Fail("expected a panic to have occured")
				}
			}()

			RegisterFailHandler(nil)
			Ω(true).Should(BeTrue())
		})
	})
})
