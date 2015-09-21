package remote_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"github.com/onsi/ginkgo/config"
	. "github.com/onsi/ginkgo/internal/remote"
	st "github.com/onsi/ginkgo/reporters/stenographer"
	"github.com/onsi/ginkgo/types"
	"time"
)

var _ = Describe("Aggregator", func() {
	var (
		aggregator     *Aggregator
		reporterConfig config.DefaultReporterConfigType
		stenographer   *st.FakeStenographer
		result         chan bool

		ginkgoConfig1 config.GinkgoConfigType
		ginkgoConfig2 config.GinkgoConfigType

		suiteSummary1 *types.SuiteSummary
		suiteSummary2 *types.SuiteSummary

		beforeSummary *types.SetupSummary
		afterSummary  *types.SetupSummary
		specSummary   *types.SpecSummary

		suiteDescription string
	)

	BeforeEach(func() {
		reporterConfig = config.DefaultReporterConfigType{
			NoColor:           false,
			SlowSpecThreshold: 0.1,
			NoisyPendings:     true,
			Succinct:          false,
			Verbose:           true,
		}
		stenographer = st.NewFakeStenographer()
		result = make(chan bool, 1)
		aggregator = NewAggregator(2, result, reporterConfig, stenographer)

		//
		// now set up some fixture data
		//

		ginkgoConfig1 = config.GinkgoConfigType{
			RandomSeed:        1138,
			RandomizeAllSpecs: true,
			ParallelNode:      1,
			ParallelTotal:     2,
		}

		ginkgoConfig2 = config.GinkgoConfigType{
			RandomSeed:        1138,
			RandomizeAllSpecs: true,
			ParallelNode:      2,
			ParallelTotal:     2,
		}

		suiteDescription = "My Parallel Suite"

		suiteSummary1 = &types.SuiteSummary{
			SuiteDescription: suiteDescription,

			NumberOfSpecsBeforeParallelization: 30,
			NumberOfTotalSpecs:                 17,
			NumberOfSpecsThatWillBeRun:         15,
			NumberOfPendingSpecs:               1,
			NumberOfSkippedSpecs:               1,
		}

		suiteSummary2 = &types.SuiteSummary{
			SuiteDescription: suiteDescription,

			NumberOfSpecsBeforeParallelization: 30,
			NumberOfTotalSpecs:                 13,
			NumberOfSpecsThatWillBeRun:         8,
			NumberOfPendingSpecs:               2,
			NumberOfSkippedSpecs:               3,
		}

		beforeSummary = &types.SetupSummary{
			State:          types.SpecStatePassed,
			CapturedOutput: "BeforeSuiteOutput",
		}

		afterSummary = &types.SetupSummary{
			State:          types.SpecStatePassed,
			CapturedOutput: "AfterSuiteOutput",
		}

		specSummary = &types.SpecSummary{
			State:          types.SpecStatePassed,
			CapturedOutput: "SpecOutput",
		}
	})

	call := func(method string, args ...interface{}) st.FakeStenographerCall {
		return st.NewFakeStenographerCall(method, args...)
	}

	beginSuite := func() {
		stenographer.Reset()
		aggregator.SpecSuiteWillBegin(ginkgoConfig2, suiteSummary2)
		aggregator.SpecSuiteWillBegin(ginkgoConfig1, suiteSummary1)
		Eventually(func() interface{} {
			return len(stenographer.Calls())
		}).Should(BeNumerically(">=", 3))
	}

	Describe("Announcing the beginning of the suite", func() {
		Context("When one of the parallel-suites starts", func() {
			BeforeEach(func() {
				aggregator.SpecSuiteWillBegin(ginkgoConfig2, suiteSummary2)
			})

			It("should be silent", func() {
				Consistently(func() interface{} { return stenographer.Calls() }).Should(BeEmpty())
			})
		})

		Context("once all of the parallel-suites have started", func() {
			BeforeEach(func() {
				aggregator.SpecSuiteWillBegin(ginkgoConfig2, suiteSummary2)
				aggregator.SpecSuiteWillBegin(ginkgoConfig1, suiteSummary1)
				Eventually(func() interface{} {
					return stenographer.Calls()
				}).Should(HaveLen(3))
			})

			It("should announce the beginning of the suite", func() {
				Ω(stenographer.Calls()).Should(HaveLen(3))
				Ω(stenographer.Calls()[0]).Should(Equal(call("AnnounceSuite", suiteDescription, ginkgoConfig1.RandomSeed, true, false)))
				Ω(stenographer.Calls()[1]).Should(Equal(call("AnnounceNumberOfSpecs", 23, 30, false)))
				Ω(stenographer.Calls()[2]).Should(Equal(call("AnnounceAggregatedParallelRun", 2, false)))
			})
		})
	})

	Describe("Announcing specs and before suites", func() {
		Context("when the parallel-suites have not all started", func() {
			BeforeEach(func() {
				aggregator.BeforeSuiteDidRun(beforeSummary)
				aggregator.AfterSuiteDidRun(afterSummary)
				aggregator.SpecDidComplete(specSummary)
			})

			It("should not announce any specs", func() {
				Consistently(func() interface{} { return stenographer.Calls() }).Should(BeEmpty())
			})

			Context("when the parallel-suites subsequently start", func() {
				BeforeEach(func() {
					beginSuite()
				})

				It("should announce the specs, the before suites and the after suites", func() {
					Eventually(func() interface{} {
						return stenographer.Calls()
					}).Should(ContainElement(call("AnnounceSuccesfulSpec", specSummary)))

					Ω(stenographer.Calls()).Should(ContainElement(call("AnnounceCapturedOutput", beforeSummary.CapturedOutput)))
					Ω(stenographer.Calls()).Should(ContainElement(call("AnnounceCapturedOutput", afterSummary.CapturedOutput)))
				})
			})
		})

		Context("When the parallel-suites have all started", func() {
			BeforeEach(func() {
				beginSuite()
				stenographer.Reset()
			})

			Context("When a spec completes", func() {
				BeforeEach(func() {
					aggregator.BeforeSuiteDidRun(beforeSummary)
					aggregator.SpecDidComplete(specSummary)
					aggregator.AfterSuiteDidRun(afterSummary)
					Eventually(func() interface{} {
						return stenographer.Calls()
					}).Should(HaveLen(5))
				})

				It("should announce the captured output of the BeforeSuite", func() {
					Ω(stenographer.Calls()[0]).Should(Equal(call("AnnounceCapturedOutput", beforeSummary.CapturedOutput)))
				})

				It("should announce that the spec will run (when in verbose mode)", func() {
					Ω(stenographer.Calls()[1]).Should(Equal(call("AnnounceSpecWillRun", specSummary)))
				})

				It("should announce the captured stdout of the spec", func() {
					Ω(stenographer.Calls()[2]).Should(Equal(call("AnnounceCapturedOutput", specSummary.CapturedOutput)))
				})

				It("should announce completion", func() {
					Ω(stenographer.Calls()[3]).Should(Equal(call("AnnounceSuccesfulSpec", specSummary)))
				})

				It("should announce the captured output of the AfterSuite", func() {
					Ω(stenographer.Calls()[4]).Should(Equal(call("AnnounceCapturedOutput", afterSummary.CapturedOutput)))
				})
			})
		})
	})

	Describe("Announcing the end of the suite", func() {
		BeforeEach(func() {
			beginSuite()
			stenographer.Reset()
		})

		Context("When one of the parallel-suites ends", func() {
			BeforeEach(func() {
				aggregator.SpecSuiteDidEnd(suiteSummary2)
			})

			It("should be silent", func() {
				Consistently(func() interface{} { return stenographer.Calls() }).Should(BeEmpty())
			})

			It("should not notify the channel", func() {
				Ω(result).Should(BeEmpty())
			})
		})

		Context("once all of the parallel-suites end", func() {
			BeforeEach(func() {
				time.Sleep(200 * time.Millisecond)

				suiteSummary1.SuiteSucceeded = true
				suiteSummary1.NumberOfPassedSpecs = 15
				suiteSummary1.NumberOfFailedSpecs = 0
				suiteSummary2.SuiteSucceeded = false
				suiteSummary2.NumberOfPassedSpecs = 5
				suiteSummary2.NumberOfFailedSpecs = 3

				aggregator.SpecSuiteDidEnd(suiteSummary2)
				aggregator.SpecSuiteDidEnd(suiteSummary1)
				Eventually(func() interface{} {
					return stenographer.Calls()
				}).Should(HaveLen(2))
			})

			It("should announce the end of the suite", func() {
				compositeSummary := stenographer.Calls()[1].Args[0].(*types.SuiteSummary)

				Ω(compositeSummary.SuiteSucceeded).Should(BeFalse())
				Ω(compositeSummary.NumberOfSpecsThatWillBeRun).Should(Equal(23))
				Ω(compositeSummary.NumberOfTotalSpecs).Should(Equal(30))
				Ω(compositeSummary.NumberOfPassedSpecs).Should(Equal(20))
				Ω(compositeSummary.NumberOfFailedSpecs).Should(Equal(3))
				Ω(compositeSummary.NumberOfPendingSpecs).Should(Equal(3))
				Ω(compositeSummary.NumberOfSkippedSpecs).Should(Equal(4))
				Ω(compositeSummary.RunTime.Seconds()).Should(BeNumerically(">", 0.2))
			})
		})

		Context("when all the parallel-suites pass", func() {
			BeforeEach(func() {
				suiteSummary1.SuiteSucceeded = true
				suiteSummary2.SuiteSucceeded = true

				aggregator.SpecSuiteDidEnd(suiteSummary2)
				aggregator.SpecSuiteDidEnd(suiteSummary1)
				Eventually(func() interface{} {
					return stenographer.Calls()
				}).Should(HaveLen(2))
			})

			It("should report success", func() {
				compositeSummary := stenographer.Calls()[1].Args[0].(*types.SuiteSummary)

				Ω(compositeSummary.SuiteSucceeded).Should(BeTrue())
			})

			It("should notify the channel that it succeded", func(done Done) {
				Ω(<-result).Should(BeTrue())
				close(done)
			})
		})

		Context("when one of the parallel-suites fails", func() {
			BeforeEach(func() {
				suiteSummary1.SuiteSucceeded = true
				suiteSummary2.SuiteSucceeded = false

				aggregator.SpecSuiteDidEnd(suiteSummary2)
				aggregator.SpecSuiteDidEnd(suiteSummary1)
				Eventually(func() interface{} {
					return stenographer.Calls()
				}).Should(HaveLen(2))
			})

			It("should report failure", func() {
				compositeSummary := stenographer.Calls()[1].Args[0].(*types.SuiteSummary)

				Ω(compositeSummary.SuiteSucceeded).Should(BeFalse())
			})

			It("should notify the channel that it failed", func(done Done) {
				Ω(<-result).Should(BeFalse())
				close(done)
			})
		})
	})
})
