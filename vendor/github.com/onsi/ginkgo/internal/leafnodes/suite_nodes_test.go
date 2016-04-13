package leafnodes_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	. "github.com/onsi/ginkgo/internal/leafnodes"

	"time"

	"github.com/onsi/ginkgo/internal/codelocation"
	Failer "github.com/onsi/ginkgo/internal/failer"
	"github.com/onsi/ginkgo/types"
)

var _ = Describe("SuiteNodes", func() {
	Describe("BeforeSuite nodes", func() {
		var befSuite SuiteNode
		var failer *Failer.Failer
		var codeLocation types.CodeLocation
		var innerCodeLocation types.CodeLocation
		var outcome bool

		BeforeEach(func() {
			failer = Failer.New()
			codeLocation = codelocation.New(0)
			innerCodeLocation = codelocation.New(0)
		})

		Context("when the body passes", func() {
			BeforeEach(func() {
				befSuite = NewBeforeSuiteNode(func() {
					time.Sleep(10 * time.Millisecond)
				}, codeLocation, 0, failer)
				outcome = befSuite.Run(0, 0, "")
			})

			It("should return true when run and report as passed", func() {
				Ω(outcome).Should(BeTrue())
				Ω(befSuite.Passed()).Should(BeTrue())
			})

			It("should have the correct summary", func() {
				summary := befSuite.Summary()
				Ω(summary.ComponentType).Should(Equal(types.SpecComponentTypeBeforeSuite))
				Ω(summary.CodeLocation).Should(Equal(codeLocation))
				Ω(summary.State).Should(Equal(types.SpecStatePassed))
				Ω(summary.RunTime).Should(BeNumerically(">=", 10*time.Millisecond))
				Ω(summary.Failure).Should(BeZero())
			})
		})

		Context("when the body fails", func() {
			BeforeEach(func() {
				befSuite = NewBeforeSuiteNode(func() {
					failer.Fail("oops", innerCodeLocation)
				}, codeLocation, 0, failer)
				outcome = befSuite.Run(0, 0, "")
			})

			It("should return false when run and report as failed", func() {
				Ω(outcome).Should(BeFalse())
				Ω(befSuite.Passed()).Should(BeFalse())
			})

			It("should have the correct summary", func() {
				summary := befSuite.Summary()
				Ω(summary.State).Should(Equal(types.SpecStateFailed))
				Ω(summary.Failure.Message).Should(Equal("oops"))
				Ω(summary.Failure.Location).Should(Equal(innerCodeLocation))
				Ω(summary.Failure.ForwardedPanic).Should(BeEmpty())
				Ω(summary.Failure.ComponentIndex).Should(Equal(0))
				Ω(summary.Failure.ComponentType).Should(Equal(types.SpecComponentTypeBeforeSuite))
				Ω(summary.Failure.ComponentCodeLocation).Should(Equal(codeLocation))
			})
		})

		Context("when the body times out", func() {
			BeforeEach(func() {
				befSuite = NewBeforeSuiteNode(func(done Done) {
				}, codeLocation, time.Millisecond, failer)
				outcome = befSuite.Run(0, 0, "")
			})

			It("should return false when run and report as failed", func() {
				Ω(outcome).Should(BeFalse())
				Ω(befSuite.Passed()).Should(BeFalse())
			})

			It("should have the correct summary", func() {
				summary := befSuite.Summary()
				Ω(summary.State).Should(Equal(types.SpecStateTimedOut))
				Ω(summary.Failure.ForwardedPanic).Should(BeEmpty())
				Ω(summary.Failure.ComponentIndex).Should(Equal(0))
				Ω(summary.Failure.ComponentType).Should(Equal(types.SpecComponentTypeBeforeSuite))
				Ω(summary.Failure.ComponentCodeLocation).Should(Equal(codeLocation))
			})
		})

		Context("when the body panics", func() {
			BeforeEach(func() {
				befSuite = NewBeforeSuiteNode(func() {
					panic("bam")
				}, codeLocation, 0, failer)
				outcome = befSuite.Run(0, 0, "")
			})

			It("should return false when run and report as failed", func() {
				Ω(outcome).Should(BeFalse())
				Ω(befSuite.Passed()).Should(BeFalse())
			})

			It("should have the correct summary", func() {
				summary := befSuite.Summary()
				Ω(summary.State).Should(Equal(types.SpecStatePanicked))
				Ω(summary.Failure.ForwardedPanic).Should(Equal("bam"))
				Ω(summary.Failure.ComponentIndex).Should(Equal(0))
				Ω(summary.Failure.ComponentType).Should(Equal(types.SpecComponentTypeBeforeSuite))
				Ω(summary.Failure.ComponentCodeLocation).Should(Equal(codeLocation))
			})
		})
	})

	Describe("AfterSuite nodes", func() {
		var aftSuite SuiteNode
		var failer *Failer.Failer
		var codeLocation types.CodeLocation
		var innerCodeLocation types.CodeLocation
		var outcome bool

		BeforeEach(func() {
			failer = Failer.New()
			codeLocation = codelocation.New(0)
			innerCodeLocation = codelocation.New(0)
		})

		Context("when the body passes", func() {
			BeforeEach(func() {
				aftSuite = NewAfterSuiteNode(func() {
					time.Sleep(10 * time.Millisecond)
				}, codeLocation, 0, failer)
				outcome = aftSuite.Run(0, 0, "")
			})

			It("should return true when run and report as passed", func() {
				Ω(outcome).Should(BeTrue())
				Ω(aftSuite.Passed()).Should(BeTrue())
			})

			It("should have the correct summary", func() {
				summary := aftSuite.Summary()
				Ω(summary.ComponentType).Should(Equal(types.SpecComponentTypeAfterSuite))
				Ω(summary.CodeLocation).Should(Equal(codeLocation))
				Ω(summary.State).Should(Equal(types.SpecStatePassed))
				Ω(summary.RunTime).Should(BeNumerically(">=", 10*time.Millisecond))
				Ω(summary.Failure).Should(BeZero())
			})
		})

		Context("when the body fails", func() {
			BeforeEach(func() {
				aftSuite = NewAfterSuiteNode(func() {
					failer.Fail("oops", innerCodeLocation)
				}, codeLocation, 0, failer)
				outcome = aftSuite.Run(0, 0, "")
			})

			It("should return false when run and report as failed", func() {
				Ω(outcome).Should(BeFalse())
				Ω(aftSuite.Passed()).Should(BeFalse())
			})

			It("should have the correct summary", func() {
				summary := aftSuite.Summary()
				Ω(summary.State).Should(Equal(types.SpecStateFailed))
				Ω(summary.Failure.Message).Should(Equal("oops"))
				Ω(summary.Failure.Location).Should(Equal(innerCodeLocation))
				Ω(summary.Failure.ForwardedPanic).Should(BeEmpty())
				Ω(summary.Failure.ComponentIndex).Should(Equal(0))
				Ω(summary.Failure.ComponentType).Should(Equal(types.SpecComponentTypeAfterSuite))
				Ω(summary.Failure.ComponentCodeLocation).Should(Equal(codeLocation))
			})
		})

		Context("when the body times out", func() {
			BeforeEach(func() {
				aftSuite = NewAfterSuiteNode(func(done Done) {
				}, codeLocation, time.Millisecond, failer)
				outcome = aftSuite.Run(0, 0, "")
			})

			It("should return false when run and report as failed", func() {
				Ω(outcome).Should(BeFalse())
				Ω(aftSuite.Passed()).Should(BeFalse())
			})

			It("should have the correct summary", func() {
				summary := aftSuite.Summary()
				Ω(summary.State).Should(Equal(types.SpecStateTimedOut))
				Ω(summary.Failure.ForwardedPanic).Should(BeEmpty())
				Ω(summary.Failure.ComponentIndex).Should(Equal(0))
				Ω(summary.Failure.ComponentType).Should(Equal(types.SpecComponentTypeAfterSuite))
				Ω(summary.Failure.ComponentCodeLocation).Should(Equal(codeLocation))
			})
		})

		Context("when the body panics", func() {
			BeforeEach(func() {
				aftSuite = NewAfterSuiteNode(func() {
					panic("bam")
				}, codeLocation, 0, failer)
				outcome = aftSuite.Run(0, 0, "")
			})

			It("should return false when run and report as failed", func() {
				Ω(outcome).Should(BeFalse())
				Ω(aftSuite.Passed()).Should(BeFalse())
			})

			It("should have the correct summary", func() {
				summary := aftSuite.Summary()
				Ω(summary.State).Should(Equal(types.SpecStatePanicked))
				Ω(summary.Failure.ForwardedPanic).Should(Equal("bam"))
				Ω(summary.Failure.ComponentIndex).Should(Equal(0))
				Ω(summary.Failure.ComponentType).Should(Equal(types.SpecComponentTypeAfterSuite))
				Ω(summary.Failure.ComponentCodeLocation).Should(Equal(codeLocation))
			})
		})
	})
})
