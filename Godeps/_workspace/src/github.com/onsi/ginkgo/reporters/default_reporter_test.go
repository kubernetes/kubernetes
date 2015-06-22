package reporters_test

import (
	"time"

	. "github.com/onsi/ginkgo"
	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/reporters"
	st "github.com/onsi/ginkgo/reporters/stenographer"
	"github.com/onsi/ginkgo/types"
	. "github.com/onsi/gomega"
)

var _ = Describe("DefaultReporter", func() {
	var (
		reporter       *reporters.DefaultReporter
		reporterConfig config.DefaultReporterConfigType
		stenographer   *st.FakeStenographer

		ginkgoConfig config.GinkgoConfigType
		suite        *types.SuiteSummary
		spec         *types.SpecSummary
	)

	BeforeEach(func() {
		stenographer = st.NewFakeStenographer()
		reporterConfig = config.DefaultReporterConfigType{
			NoColor:           false,
			SlowSpecThreshold: 0.1,
			NoisyPendings:     false,
			Verbose:           true,
			FullTrace:         true,
		}

		reporter = reporters.NewDefaultReporter(reporterConfig, stenographer)
	})

	call := func(method string, args ...interface{}) st.FakeStenographerCall {
		return st.NewFakeStenographerCall(method, args...)
	}

	Describe("SpecSuiteWillBegin", func() {
		BeforeEach(func() {
			suite = &types.SuiteSummary{
				SuiteDescription:           "A Sweet Suite",
				NumberOfTotalSpecs:         10,
				NumberOfSpecsThatWillBeRun: 8,
			}

			ginkgoConfig = config.GinkgoConfigType{
				RandomSeed:        1138,
				RandomizeAllSpecs: true,
			}
		})

		Context("when a serial (non-parallel) suite begins", func() {
			BeforeEach(func() {
				ginkgoConfig.ParallelTotal = 1

				reporter.SpecSuiteWillBegin(ginkgoConfig, suite)
			})

			It("should announce the suite, then announce the number of specs", func() {
				Ω(stenographer.Calls()).Should(HaveLen(2))
				Ω(stenographer.Calls()[0]).Should(Equal(call("AnnounceSuite", "A Sweet Suite", ginkgoConfig.RandomSeed, true, false)))
				Ω(stenographer.Calls()[1]).Should(Equal(call("AnnounceNumberOfSpecs", 8, 10, false)))
			})
		})

		Context("when a parallel suite begins", func() {
			BeforeEach(func() {
				ginkgoConfig.ParallelTotal = 2
				ginkgoConfig.ParallelNode = 1
				suite.NumberOfSpecsBeforeParallelization = 20

				reporter.SpecSuiteWillBegin(ginkgoConfig, suite)
			})

			It("should announce the suite, announce that it's a parallel run, then announce the number of specs", func() {
				Ω(stenographer.Calls()).Should(HaveLen(3))
				Ω(stenographer.Calls()[0]).Should(Equal(call("AnnounceSuite", "A Sweet Suite", ginkgoConfig.RandomSeed, true, false)))
				Ω(stenographer.Calls()[1]).Should(Equal(call("AnnounceParallelRun", 1, 2, 10, 20, false)))
				Ω(stenographer.Calls()[2]).Should(Equal(call("AnnounceNumberOfSpecs", 8, 10, false)))
			})
		})
	})

	Describe("BeforeSuiteDidRun", func() {
		Context("when the BeforeSuite passes", func() {
			It("should announce nothing", func() {
				reporter.BeforeSuiteDidRun(&types.SetupSummary{
					State: types.SpecStatePassed,
				})

				Ω(stenographer.Calls()).Should(BeEmpty())
			})
		})

		Context("when the BeforeSuite fails", func() {
			It("should announce the failure", func() {
				summary := &types.SetupSummary{
					State: types.SpecStateFailed,
				}
				reporter.BeforeSuiteDidRun(summary)

				Ω(stenographer.Calls()).Should(HaveLen(1))
				Ω(stenographer.Calls()[0]).Should(Equal(call("AnnounceBeforeSuiteFailure", summary, false, true)))
			})
		})
	})

	Describe("AfterSuiteDidRun", func() {
		Context("when the AfterSuite passes", func() {
			It("should announce nothing", func() {
				reporter.AfterSuiteDidRun(&types.SetupSummary{
					State: types.SpecStatePassed,
				})

				Ω(stenographer.Calls()).Should(BeEmpty())
			})
		})

		Context("when the AfterSuite fails", func() {
			It("should announce the failure", func() {
				summary := &types.SetupSummary{
					State: types.SpecStateFailed,
				}
				reporter.AfterSuiteDidRun(summary)

				Ω(stenographer.Calls()).Should(HaveLen(1))
				Ω(stenographer.Calls()[0]).Should(Equal(call("AnnounceAfterSuiteFailure", summary, false, true)))
			})
		})
	})

	Describe("SpecWillRun", func() {
		Context("When running in verbose mode", func() {
			Context("and the spec will run", func() {
				BeforeEach(func() {
					spec = &types.SpecSummary{}
					reporter.SpecWillRun(spec)
				})

				It("should announce that the spec will run", func() {
					Ω(stenographer.Calls()).Should(HaveLen(1))
					Ω(stenographer.Calls()[0]).Should(Equal(call("AnnounceSpecWillRun", spec)))
				})
			})

			Context("and the spec will not run", func() {
				Context("because it is pending", func() {
					BeforeEach(func() {
						spec = &types.SpecSummary{
							State: types.SpecStatePending,
						}
						reporter.SpecWillRun(spec)
					})

					It("should announce nothing", func() {
						Ω(stenographer.Calls()).Should(BeEmpty())
					})
				})

				Context("because it is skipped", func() {
					BeforeEach(func() {
						spec = &types.SpecSummary{
							State: types.SpecStateSkipped,
						}
						reporter.SpecWillRun(spec)
					})

					It("should announce nothing", func() {
						Ω(stenographer.Calls()).Should(BeEmpty())
					})
				})
			})
		})

		Context("When running in verbose & succinct mode", func() {
			BeforeEach(func() {
				reporterConfig.Succinct = true
				reporter = reporters.NewDefaultReporter(reporterConfig, stenographer)
				spec = &types.SpecSummary{}
				reporter.SpecWillRun(spec)
			})

			It("should announce nothing", func() {
				Ω(stenographer.Calls()).Should(BeEmpty())
			})
		})

		Context("When not running in verbose mode", func() {
			BeforeEach(func() {
				reporterConfig.Verbose = false
				reporter = reporters.NewDefaultReporter(reporterConfig, stenographer)
				spec = &types.SpecSummary{}
				reporter.SpecWillRun(spec)
			})

			It("should announce nothing", func() {
				Ω(stenographer.Calls()).Should(BeEmpty())
			})
		})
	})

	Describe("SpecDidComplete", func() {
		JustBeforeEach(func() {
			reporter.SpecDidComplete(spec)
		})

		BeforeEach(func() {
			spec = &types.SpecSummary{}
		})

		Context("When the spec passed", func() {
			BeforeEach(func() {
				spec.State = types.SpecStatePassed
			})

			Context("When the spec was a measurement", func() {
				BeforeEach(func() {
					spec.IsMeasurement = true
				})

				It("should announce the measurement", func() {
					Ω(stenographer.Calls()[0]).Should(Equal(call("AnnounceSuccesfulMeasurement", spec, false)))
				})
			})

			Context("When the spec is slow", func() {
				BeforeEach(func() {
					spec.RunTime = time.Second
				})

				It("should announce that it was slow", func() {
					Ω(stenographer.Calls()[0]).Should(Equal(call("AnnounceSuccesfulSlowSpec", spec, false)))
				})
			})

			Context("Otherwise", func() {
				It("should announce the succesful spec", func() {
					Ω(stenographer.Calls()[0]).Should(Equal(call("AnnounceSuccesfulSpec", spec)))
				})
			})
		})

		Context("When the spec is pending", func() {
			BeforeEach(func() {
				spec.State = types.SpecStatePending
			})

			It("should announce the pending spec, succinctly", func() {
				Ω(stenographer.Calls()[0]).Should(Equal(call("AnnouncePendingSpec", spec, false)))
			})
		})

		Context("When the spec is skipped", func() {
			BeforeEach(func() {
				spec.State = types.SpecStateSkipped
			})

			It("should announce the skipped spec", func() {
				Ω(stenographer.Calls()[0]).Should(Equal(call("AnnounceSkippedSpec", spec, false, true)))
			})
		})

		Context("When the spec timed out", func() {
			BeforeEach(func() {
				spec.State = types.SpecStateTimedOut
			})

			It("should announce the timedout spec", func() {
				Ω(stenographer.Calls()[0]).Should(Equal(call("AnnounceSpecTimedOut", spec, false, true)))
			})
		})

		Context("When the spec panicked", func() {
			BeforeEach(func() {
				spec.State = types.SpecStatePanicked
			})

			It("should announce the panicked spec", func() {
				Ω(stenographer.Calls()[0]).Should(Equal(call("AnnounceSpecPanicked", spec, false, true)))
			})
		})

		Context("When the spec failed", func() {
			BeforeEach(func() {
				spec.State = types.SpecStateFailed
			})

			It("should announce the failed spec", func() {
				Ω(stenographer.Calls()[0]).Should(Equal(call("AnnounceSpecFailed", spec, false, true)))
			})
		})

		Context("in noisy pendings mode", func() {
			BeforeEach(func() {
				reporterConfig.Succinct = false
				reporterConfig.NoisyPendings = true
				reporter = reporters.NewDefaultReporter(reporterConfig, stenographer)
			})

			Context("When the spec is pending", func() {
				BeforeEach(func() {
					spec.State = types.SpecStatePending
				})

				It("should announce the pending spec, noisily", func() {
					Ω(stenographer.Calls()[0]).Should(Equal(call("AnnouncePendingSpec", spec, true)))
				})
			})
		})

		Context("in succinct mode", func() {
			BeforeEach(func() {
				reporterConfig.Succinct = true
				reporter = reporters.NewDefaultReporter(reporterConfig, stenographer)
			})

			Context("When the spec passed", func() {
				BeforeEach(func() {
					spec.State = types.SpecStatePassed
				})

				Context("When the spec was a measurement", func() {
					BeforeEach(func() {
						spec.IsMeasurement = true
					})

					It("should announce the measurement", func() {
						Ω(stenographer.Calls()[0]).Should(Equal(call("AnnounceSuccesfulMeasurement", spec, true)))
					})
				})

				Context("When the spec is slow", func() {
					BeforeEach(func() {
						spec.RunTime = time.Second
					})

					It("should announce that it was slow", func() {
						Ω(stenographer.Calls()[0]).Should(Equal(call("AnnounceSuccesfulSlowSpec", spec, true)))
					})
				})

				Context("Otherwise", func() {
					It("should announce the succesful spec", func() {
						Ω(stenographer.Calls()[0]).Should(Equal(call("AnnounceSuccesfulSpec", spec)))
					})
				})
			})

			Context("When the spec is pending", func() {
				BeforeEach(func() {
					spec.State = types.SpecStatePending
				})

				It("should announce the pending spec, succinctly", func() {
					Ω(stenographer.Calls()[0]).Should(Equal(call("AnnouncePendingSpec", spec, false)))
				})
			})

			Context("When the spec is skipped", func() {
				BeforeEach(func() {
					spec.State = types.SpecStateSkipped
				})

				It("should announce the skipped spec", func() {
					Ω(stenographer.Calls()[0]).Should(Equal(call("AnnounceSkippedSpec", spec, true, true)))
				})
			})

			Context("When the spec timed out", func() {
				BeforeEach(func() {
					spec.State = types.SpecStateTimedOut
				})

				It("should announce the timedout spec", func() {
					Ω(stenographer.Calls()[0]).Should(Equal(call("AnnounceSpecTimedOut", spec, true, true)))
				})
			})

			Context("When the spec panicked", func() {
				BeforeEach(func() {
					spec.State = types.SpecStatePanicked
				})

				It("should announce the panicked spec", func() {
					Ω(stenographer.Calls()[0]).Should(Equal(call("AnnounceSpecPanicked", spec, true, true)))
				})
			})

			Context("When the spec failed", func() {
				BeforeEach(func() {
					spec.State = types.SpecStateFailed
				})

				It("should announce the failed spec", func() {
					Ω(stenographer.Calls()[0]).Should(Equal(call("AnnounceSpecFailed", spec, true, true)))
				})
			})
		})
	})

	Describe("SpecSuiteDidEnd", func() {
		BeforeEach(func() {
			suite = &types.SuiteSummary{}
			reporter.SpecSuiteDidEnd(suite)
		})

		It("should announce the spec run's completion", func() {
			Ω(stenographer.Calls()[1]).Should(Equal(call("AnnounceSpecRunCompletion", suite, false)))
		})
	})
})
