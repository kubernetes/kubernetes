package types_test

import (
	. "github.com/onsi/ginkgo/types"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var specStates = []SpecState{
	SpecStatePassed,
	SpecStateTimedOut,
	SpecStatePanicked,
	SpecStateFailed,
	SpecStatePending,
	SpecStateSkipped,
}

func verifySpecSummary(caller func(SpecSummary) bool, trueStates ...SpecState) {
	summary := SpecSummary{}
	trueStateLookup := map[SpecState]bool{}
	for _, state := range trueStates {
		trueStateLookup[state] = true
		summary.State = state
		Ω(caller(summary)).Should(BeTrue())
	}

	for _, state := range specStates {
		if trueStateLookup[state] {
			continue
		}
		summary.State = state
		Ω(caller(summary)).Should(BeFalse())
	}
}

var _ = Describe("Types", func() {
	Describe("IsFailureState", func() {
		It("knows when it is in a failure-like state", func() {
			verifySpecSummary(func(summary SpecSummary) bool {
				return summary.State.IsFailure()
			}, SpecStateTimedOut, SpecStatePanicked, SpecStateFailed)
		})
	})

	Describe("SpecSummary", func() {
		It("knows when it is in a failure-like state", func() {
			verifySpecSummary(func(summary SpecSummary) bool {
				return summary.HasFailureState()
			}, SpecStateTimedOut, SpecStatePanicked, SpecStateFailed)
		})

		It("knows when it passed", func() {
			verifySpecSummary(func(summary SpecSummary) bool {
				return summary.Passed()
			}, SpecStatePassed)
		})

		It("knows when it has failed", func() {
			verifySpecSummary(func(summary SpecSummary) bool {
				return summary.Failed()
			}, SpecStateFailed)
		})

		It("knows when it has panicked", func() {
			verifySpecSummary(func(summary SpecSummary) bool {
				return summary.Panicked()
			}, SpecStatePanicked)
		})

		It("knows when it has timed out", func() {
			verifySpecSummary(func(summary SpecSummary) bool {
				return summary.TimedOut()
			}, SpecStateTimedOut)
		})

		It("knows when it is pending", func() {
			verifySpecSummary(func(summary SpecSummary) bool {
				return summary.Pending()
			}, SpecStatePending)
		})

		It("knows when it is skipped", func() {
			verifySpecSummary(func(summary SpecSummary) bool {
				return summary.Skipped()
			}, SpecStateSkipped)
		})
	})

	Describe("SpecMeasurement", func() {
		It("knows how to format values when the precision is 0", func() {
			Ω(SpecMeasurement{}.PrecisionFmt()).Should(Equal("%f"))
		})

		It("knows how to format the values when the precision is 3", func() {
			Ω(SpecMeasurement{Precision: 3}.PrecisionFmt()).Should(Equal("%.3f"))
		})
	})
})
