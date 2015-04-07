package failer_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/ginkgo/internal/failer"
	. "github.com/onsi/gomega"

	"github.com/onsi/ginkgo/internal/codelocation"
	"github.com/onsi/ginkgo/types"
)

var _ = Describe("Failer", func() {
	var (
		failer        *Failer
		codeLocationA types.CodeLocation
		codeLocationB types.CodeLocation
	)

	BeforeEach(func() {
		codeLocationA = codelocation.New(0)
		codeLocationB = codelocation.New(0)
		failer = New()
	})

	Context("with no failures", func() {
		It("should return success when drained", func() {
			failure, state := failer.Drain(types.SpecComponentTypeIt, 3, codeLocationB)
			Ω(failure).Should(BeZero())
			Ω(state).Should(Equal(types.SpecStatePassed))
		})
	})

	Describe("Fail", func() {
		It("should handle failures", func() {
			failer.Fail("something failed", codeLocationA)
			failure, state := failer.Drain(types.SpecComponentTypeIt, 3, codeLocationB)
			Ω(failure).Should(Equal(types.SpecFailure{
				Message:               "something failed",
				Location:              codeLocationA,
				ForwardedPanic:        "",
				ComponentType:         types.SpecComponentTypeIt,
				ComponentIndex:        3,
				ComponentCodeLocation: codeLocationB,
			}))
			Ω(state).Should(Equal(types.SpecStateFailed))
		})
	})

	Describe("Panic", func() {
		It("should handle panics", func() {
			failer.Panic(codeLocationA, "some forwarded panic")
			failure, state := failer.Drain(types.SpecComponentTypeIt, 3, codeLocationB)
			Ω(failure).Should(Equal(types.SpecFailure{
				Message:               "Test Panicked",
				Location:              codeLocationA,
				ForwardedPanic:        "some forwarded panic",
				ComponentType:         types.SpecComponentTypeIt,
				ComponentIndex:        3,
				ComponentCodeLocation: codeLocationB,
			}))
			Ω(state).Should(Equal(types.SpecStatePanicked))
		})
	})

	Describe("Timeout", func() {
		It("should handle timeouts", func() {
			failer.Timeout(codeLocationA)
			failure, state := failer.Drain(types.SpecComponentTypeIt, 3, codeLocationB)
			Ω(failure).Should(Equal(types.SpecFailure{
				Message:               "Timed out",
				Location:              codeLocationA,
				ForwardedPanic:        "",
				ComponentType:         types.SpecComponentTypeIt,
				ComponentIndex:        3,
				ComponentCodeLocation: codeLocationB,
			}))
			Ω(state).Should(Equal(types.SpecStateTimedOut))
		})
	})

	Context("when multiple failures are registered", func() {
		BeforeEach(func() {
			failer.Fail("something failed", codeLocationA)
			failer.Fail("something else failed", codeLocationA)
		})

		It("should only report the first one when drained", func() {
			failure, state := failer.Drain(types.SpecComponentTypeIt, 3, codeLocationB)

			Ω(failure).Should(Equal(types.SpecFailure{
				Message:               "something failed",
				Location:              codeLocationA,
				ForwardedPanic:        "",
				ComponentType:         types.SpecComponentTypeIt,
				ComponentIndex:        3,
				ComponentCodeLocation: codeLocationB,
			}))
			Ω(state).Should(Equal(types.SpecStateFailed))
		})

		It("should report subsequent failures after being drained", func() {
			failer.Drain(types.SpecComponentTypeIt, 3, codeLocationB)
			failer.Fail("yet another thing failed", codeLocationA)

			failure, state := failer.Drain(types.SpecComponentTypeIt, 3, codeLocationB)

			Ω(failure).Should(Equal(types.SpecFailure{
				Message:               "yet another thing failed",
				Location:              codeLocationA,
				ForwardedPanic:        "",
				ComponentType:         types.SpecComponentTypeIt,
				ComponentIndex:        3,
				ComponentCodeLocation: codeLocationB,
			}))
			Ω(state).Should(Equal(types.SpecStateFailed))
		})

		It("should report sucess on subsequent drains if no errors occur", func() {
			failer.Drain(types.SpecComponentTypeIt, 3, codeLocationB)
			failure, state := failer.Drain(types.SpecComponentTypeIt, 3, codeLocationB)
			Ω(failure).Should(BeZero())
			Ω(state).Should(Equal(types.SpecStatePassed))
		})
	})
})
