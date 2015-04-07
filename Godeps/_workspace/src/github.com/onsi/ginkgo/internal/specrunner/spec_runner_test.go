package specrunner_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/ginkgo/internal/specrunner"
	"github.com/onsi/ginkgo/types"
	. "github.com/onsi/gomega"

	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/internal/codelocation"
	"github.com/onsi/ginkgo/internal/containernode"
	Failer "github.com/onsi/ginkgo/internal/failer"
	"github.com/onsi/ginkgo/internal/leafnodes"
	"github.com/onsi/ginkgo/internal/spec"
	Writer "github.com/onsi/ginkgo/internal/writer"
	"github.com/onsi/ginkgo/reporters"
)

var noneFlag = types.FlagTypeNone
var focusedFlag = types.FlagTypeFocused
var pendingFlag = types.FlagTypePending

var _ = Describe("Spec Runner", func() {
	var (
		reporter1 *reporters.FakeReporter
		reporter2 *reporters.FakeReporter
		failer    *Failer.Failer
		writer    *Writer.FakeGinkgoWriter

		thingsThatRan []string

		runner *SpecRunner
	)

	newBefSuite := func(text string, fail bool) leafnodes.SuiteNode {
		return leafnodes.NewBeforeSuiteNode(func() {
			writer.AddEvent(text)
			thingsThatRan = append(thingsThatRan, text)
			if fail {
				failer.Fail(text, codelocation.New(0))
			}
		}, codelocation.New(0), 0, failer)
	}

	newAftSuite := func(text string, fail bool) leafnodes.SuiteNode {
		return leafnodes.NewAfterSuiteNode(func() {
			writer.AddEvent(text)
			thingsThatRan = append(thingsThatRan, text)
			if fail {
				failer.Fail(text, codelocation.New(0))
			}
		}, codelocation.New(0), 0, failer)
	}

	newSpec := func(text string, flag types.FlagType, fail bool) *spec.Spec {
		subject := leafnodes.NewItNode(text, func() {
			writer.AddEvent(text)
			thingsThatRan = append(thingsThatRan, text)
			if fail {
				failer.Fail(text, codelocation.New(0))
			}
		}, flag, codelocation.New(0), 0, failer, 0)

		return spec.New(subject, []*containernode.ContainerNode{}, false)
	}

	newSpecWithBody := func(text string, body interface{}) *spec.Spec {
		subject := leafnodes.NewItNode(text, body, noneFlag, codelocation.New(0), 0, failer, 0)

		return spec.New(subject, []*containernode.ContainerNode{}, false)
	}

	newRunner := func(config config.GinkgoConfigType, beforeSuiteNode leafnodes.SuiteNode, afterSuiteNode leafnodes.SuiteNode, specs ...*spec.Spec) *SpecRunner {
		return New("description", beforeSuiteNode, spec.NewSpecs(specs), afterSuiteNode, []reporters.Reporter{reporter1, reporter2}, writer, config)
	}

	BeforeEach(func() {
		reporter1 = reporters.NewFakeReporter()
		reporter2 = reporters.NewFakeReporter()
		writer = Writer.NewFake()
		failer = Failer.New()

		thingsThatRan = []string{}
	})

	Describe("Running and Reporting", func() {
		var specA, pendingSpec, anotherPendingSpec, failedSpec, specB, skippedSpec *spec.Spec
		var willRunCalls, didCompleteCalls []string
		var conf config.GinkgoConfigType

		JustBeforeEach(func() {
			willRunCalls = []string{}
			didCompleteCalls = []string{}
			specA = newSpec("spec A", noneFlag, false)
			pendingSpec = newSpec("pending spec", pendingFlag, false)
			anotherPendingSpec = newSpec("another pending spec", pendingFlag, false)
			failedSpec = newSpec("failed spec", noneFlag, true)
			specB = newSpec("spec B", noneFlag, false)
			skippedSpec = newSpec("skipped spec", noneFlag, false)
			skippedSpec.Skip()

			reporter1.SpecWillRunStub = func(specSummary *types.SpecSummary) {
				willRunCalls = append(willRunCalls, "Reporter1")
			}
			reporter2.SpecWillRunStub = func(specSummary *types.SpecSummary) {
				willRunCalls = append(willRunCalls, "Reporter2")
			}

			reporter1.SpecDidCompleteStub = func(specSummary *types.SpecSummary) {
				didCompleteCalls = append(didCompleteCalls, "Reporter1")
			}
			reporter2.SpecDidCompleteStub = func(specSummary *types.SpecSummary) {
				didCompleteCalls = append(didCompleteCalls, "Reporter2")
			}

			runner = newRunner(conf, newBefSuite("BefSuite", false), newAftSuite("AftSuite", false), specA, pendingSpec, anotherPendingSpec, failedSpec, specB, skippedSpec)
			runner.Run()
		})

		BeforeEach(func() {
			conf = config.GinkgoConfigType{RandomSeed: 17}
		})

		It("should skip skipped/pending tests", func() {
			Ω(thingsThatRan).Should(Equal([]string{"BefSuite", "spec A", "failed spec", "spec B", "AftSuite"}))
		})

		It("should report to any attached reporters", func() {
			Ω(reporter1.Config).Should(Equal(reporter2.Config))
			Ω(reporter1.BeforeSuiteSummary).Should(Equal(reporter2.BeforeSuiteSummary))
			Ω(reporter1.BeginSummary).Should(Equal(reporter2.BeginSummary))
			Ω(reporter1.SpecWillRunSummaries).Should(Equal(reporter2.SpecWillRunSummaries))
			Ω(reporter1.SpecSummaries).Should(Equal(reporter2.SpecSummaries))
			Ω(reporter1.AfterSuiteSummary).Should(Equal(reporter2.AfterSuiteSummary))
			Ω(reporter1.EndSummary).Should(Equal(reporter2.EndSummary))
		})

		It("should report that a spec did end in reverse order", func() {
			Ω(willRunCalls[0:4]).Should(Equal([]string{"Reporter1", "Reporter2", "Reporter1", "Reporter2"}))
			Ω(didCompleteCalls[0:4]).Should(Equal([]string{"Reporter2", "Reporter1", "Reporter2", "Reporter1"}))
		})

		It("should report the passed in config", func() {
			Ω(reporter1.Config.RandomSeed).Should(BeNumerically("==", 17))
		})

		It("should report the beginning of the suite", func() {
			Ω(reporter1.BeginSummary.SuiteDescription).Should(Equal("description"))
			Ω(reporter1.BeginSummary.SuiteID).Should(MatchRegexp("[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}"))
			Ω(reporter1.BeginSummary.NumberOfSpecsBeforeParallelization).Should(Equal(6))
			Ω(reporter1.BeginSummary.NumberOfTotalSpecs).Should(Equal(6))
			Ω(reporter1.BeginSummary.NumberOfSpecsThatWillBeRun).Should(Equal(3))
			Ω(reporter1.BeginSummary.NumberOfPendingSpecs).Should(Equal(2))
			Ω(reporter1.BeginSummary.NumberOfSkippedSpecs).Should(Equal(1))
		})

		It("should report the end of the suite", func() {
			Ω(reporter1.EndSummary.SuiteDescription).Should(Equal("description"))
			Ω(reporter1.EndSummary.SuiteSucceeded).Should(BeFalse())
			Ω(reporter1.EndSummary.SuiteID).Should(MatchRegexp("[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}"))
			Ω(reporter1.EndSummary.NumberOfSpecsBeforeParallelization).Should(Equal(6))
			Ω(reporter1.EndSummary.NumberOfTotalSpecs).Should(Equal(6))
			Ω(reporter1.EndSummary.NumberOfSpecsThatWillBeRun).Should(Equal(3))
			Ω(reporter1.EndSummary.NumberOfPendingSpecs).Should(Equal(2))
			Ω(reporter1.EndSummary.NumberOfSkippedSpecs).Should(Equal(1))
			Ω(reporter1.EndSummary.NumberOfPassedSpecs).Should(Equal(2))
			Ω(reporter1.EndSummary.NumberOfFailedSpecs).Should(Equal(1))
		})

		Context("when told to perform a dry run", func() {
			BeforeEach(func() {
				conf.DryRun = true
			})

			It("should report to the reporters", func() {
				Ω(reporter1.Config).Should(Equal(reporter2.Config))
				Ω(reporter1.BeforeSuiteSummary).Should(Equal(reporter2.BeforeSuiteSummary))
				Ω(reporter1.BeginSummary).Should(Equal(reporter2.BeginSummary))
				Ω(reporter1.SpecWillRunSummaries).Should(Equal(reporter2.SpecWillRunSummaries))
				Ω(reporter1.SpecSummaries).Should(Equal(reporter2.SpecSummaries))
				Ω(reporter1.AfterSuiteSummary).Should(Equal(reporter2.AfterSuiteSummary))
				Ω(reporter1.EndSummary).Should(Equal(reporter2.EndSummary))
			})

			It("should not actually run anything", func() {
				Ω(thingsThatRan).Should(BeEmpty())
			})

			It("report before and after suites as passed", func() {
				Ω(reporter1.BeforeSuiteSummary.State).Should(Equal(types.SpecStatePassed))
				Ω(reporter1.AfterSuiteSummary.State).Should(Equal(types.SpecStatePassed))
			})

			It("should report specs as passed", func() {
				summaries := reporter1.SpecSummaries
				Ω(summaries).Should(HaveLen(6))
				Ω(summaries[0].ComponentTexts).Should(ContainElement("spec A"))
				Ω(summaries[0].State).Should(Equal(types.SpecStatePassed))
				Ω(summaries[1].ComponentTexts).Should(ContainElement("pending spec"))
				Ω(summaries[1].State).Should(Equal(types.SpecStatePending))
				Ω(summaries[2].ComponentTexts).Should(ContainElement("another pending spec"))
				Ω(summaries[2].State).Should(Equal(types.SpecStatePending))
				Ω(summaries[3].ComponentTexts).Should(ContainElement("failed spec"))
				Ω(summaries[3].State).Should(Equal(types.SpecStatePassed))
				Ω(summaries[4].ComponentTexts).Should(ContainElement("spec B"))
				Ω(summaries[4].State).Should(Equal(types.SpecStatePassed))
				Ω(summaries[5].ComponentTexts).Should(ContainElement("skipped spec"))
				Ω(summaries[5].State).Should(Equal(types.SpecStateSkipped))
			})

			It("should report the end of the suite", func() {
				Ω(reporter1.EndSummary.SuiteDescription).Should(Equal("description"))
				Ω(reporter1.EndSummary.SuiteSucceeded).Should(BeTrue())
				Ω(reporter1.EndSummary.SuiteID).Should(MatchRegexp("[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}"))
				Ω(reporter1.EndSummary.NumberOfSpecsBeforeParallelization).Should(Equal(6))
				Ω(reporter1.EndSummary.NumberOfTotalSpecs).Should(Equal(6))
				Ω(reporter1.EndSummary.NumberOfSpecsThatWillBeRun).Should(Equal(3))
				Ω(reporter1.EndSummary.NumberOfPendingSpecs).Should(Equal(2))
				Ω(reporter1.EndSummary.NumberOfSkippedSpecs).Should(Equal(1))
				Ω(reporter1.EndSummary.NumberOfPassedSpecs).Should(Equal(0))
				Ω(reporter1.EndSummary.NumberOfFailedSpecs).Should(Equal(0))
			})
		})
	})

	Describe("reporting on specs", func() {
		var proceed chan bool
		var ready chan bool
		var finished chan bool
		BeforeEach(func() {
			ready = make(chan bool)
			proceed = make(chan bool)
			finished = make(chan bool)
			skippedSpec := newSpec("SKIP", noneFlag, false)
			skippedSpec.Skip()

			runner = newRunner(
				config.GinkgoConfigType{},
				newBefSuite("BefSuite", false),
				newAftSuite("AftSuite", false),
				skippedSpec,
				newSpec("PENDING", pendingFlag, false),
				newSpecWithBody("RUN", func() {
					close(ready)
					<-proceed
				}),
			)
			go func() {
				runner.Run()
				close(finished)
			}()
		})

		It("should report about pending/skipped specs", func() {
			<-ready
			Ω(reporter1.SpecWillRunSummaries).Should(HaveLen(3))

			Ω(reporter1.SpecWillRunSummaries[0].ComponentTexts[0]).Should(Equal("SKIP"))
			Ω(reporter1.SpecWillRunSummaries[1].ComponentTexts[0]).Should(Equal("PENDING"))
			Ω(reporter1.SpecWillRunSummaries[2].ComponentTexts[0]).Should(Equal("RUN"))

			Ω(reporter1.SpecSummaries[0].ComponentTexts[0]).Should(Equal("SKIP"))
			Ω(reporter1.SpecSummaries[1].ComponentTexts[0]).Should(Equal("PENDING"))
			Ω(reporter1.SpecSummaries).Should(HaveLen(2))

			close(proceed)
			<-finished

			Ω(reporter1.SpecSummaries).Should(HaveLen(3))
			Ω(reporter1.SpecSummaries[2].ComponentTexts[0]).Should(Equal("RUN"))
		})
	})

	Describe("Running BeforeSuite & AfterSuite", func() {
		var success bool
		var befSuite leafnodes.SuiteNode
		var aftSuite leafnodes.SuiteNode
		Context("with a nil BeforeSuite & AfterSuite", func() {
			BeforeEach(func() {
				runner = newRunner(
					config.GinkgoConfigType{},
					nil,
					nil,
					newSpec("A", noneFlag, false),
					newSpec("B", noneFlag, false),
				)
				success = runner.Run()
			})

			It("should not report about the BeforeSuite", func() {
				Ω(reporter1.BeforeSuiteSummary).Should(BeNil())
			})

			It("should not report about the AfterSuite", func() {
				Ω(reporter1.AfterSuiteSummary).Should(BeNil())
			})

			It("should run the specs", func() {
				Ω(thingsThatRan).Should(Equal([]string{"A", "B"}))
			})
		})

		Context("when the BeforeSuite & AfterSuite pass", func() {
			BeforeEach(func() {
				befSuite = newBefSuite("BefSuite", false)
				aftSuite = newBefSuite("AftSuite", false)
				runner = newRunner(
					config.GinkgoConfigType{},
					befSuite,
					aftSuite,
					newSpec("A", noneFlag, false),
					newSpec("B", noneFlag, false),
				)
				success = runner.Run()
			})

			It("should run the BeforeSuite, the AfterSuite and the specs", func() {
				Ω(thingsThatRan).Should(Equal([]string{"BefSuite", "A", "B", "AftSuite"}))
			})

			It("should report about the BeforeSuite", func() {
				Ω(reporter1.BeforeSuiteSummary).Should(Equal(befSuite.Summary()))
			})

			It("should report about the AfterSuite", func() {
				Ω(reporter1.AfterSuiteSummary).Should(Equal(aftSuite.Summary()))
			})

			It("should report success", func() {
				Ω(success).Should(BeTrue())
				Ω(reporter1.EndSummary.SuiteSucceeded).Should(BeTrue())
				Ω(reporter1.EndSummary.NumberOfFailedSpecs).Should(Equal(0))
			})

			It("should not dump the writer", func() {
				Ω(writer.EventStream).ShouldNot(ContainElement("DUMP"))
			})
		})

		Context("when the BeforeSuite fails", func() {
			BeforeEach(func() {
				befSuite = newBefSuite("BefSuite", true)
				aftSuite = newBefSuite("AftSuite", false)

				skipped := newSpec("Skipped", noneFlag, false)
				skipped.Skip()

				runner = newRunner(
					config.GinkgoConfigType{},
					befSuite,
					aftSuite,
					newSpec("A", noneFlag, false),
					newSpec("B", noneFlag, false),
					newSpec("Pending", pendingFlag, false),
					skipped,
				)
				success = runner.Run()
			})

			It("should not run the specs, but it should run the AfterSuite", func() {
				Ω(thingsThatRan).Should(Equal([]string{"BefSuite", "AftSuite"}))
			})

			It("should report about the BeforeSuite", func() {
				Ω(reporter1.BeforeSuiteSummary).Should(Equal(befSuite.Summary()))
			})

			It("should report about the AfterSuite", func() {
				Ω(reporter1.AfterSuiteSummary).Should(Equal(aftSuite.Summary()))
			})

			It("should report failure", func() {
				Ω(success).Should(BeFalse())
				Ω(reporter1.EndSummary.SuiteSucceeded).Should(BeFalse())
				Ω(reporter1.EndSummary.NumberOfFailedSpecs).Should(Equal(2))
				Ω(reporter1.EndSummary.NumberOfSpecsThatWillBeRun).Should(Equal(2))
			})

			It("should dump the writer", func() {
				Ω(writer.EventStream).Should(ContainElement("DUMP"))
			})
		})

		Context("when some other test fails", func() {
			BeforeEach(func() {
				aftSuite = newBefSuite("AftSuite", false)

				runner = newRunner(
					config.GinkgoConfigType{},
					nil,
					aftSuite,
					newSpec("A", noneFlag, true),
				)
				success = runner.Run()
			})

			It("should still run the AfterSuite", func() {
				Ω(thingsThatRan).Should(Equal([]string{"A", "AftSuite"}))
			})

			It("should report about the AfterSuite", func() {
				Ω(reporter1.AfterSuiteSummary).Should(Equal(aftSuite.Summary()))
			})

			It("should report failure", func() {
				Ω(success).Should(BeFalse())
				Ω(reporter1.EndSummary.SuiteSucceeded).Should(BeFalse())
				Ω(reporter1.EndSummary.NumberOfFailedSpecs).Should(Equal(1))
				Ω(reporter1.EndSummary.NumberOfSpecsThatWillBeRun).Should(Equal(1))
			})
		})

		Context("when the AfterSuite fails", func() {
			BeforeEach(func() {
				befSuite = newBefSuite("BefSuite", false)
				aftSuite = newBefSuite("AftSuite", true)
				runner = newRunner(
					config.GinkgoConfigType{},
					befSuite,
					aftSuite,
					newSpec("A", noneFlag, false),
					newSpec("B", noneFlag, false),
				)
				success = runner.Run()
			})

			It("should run everything", func() {
				Ω(thingsThatRan).Should(Equal([]string{"BefSuite", "A", "B", "AftSuite"}))
			})

			It("should report about the BeforeSuite", func() {
				Ω(reporter1.BeforeSuiteSummary).Should(Equal(befSuite.Summary()))
			})

			It("should report about the AfterSuite", func() {
				Ω(reporter1.AfterSuiteSummary).Should(Equal(aftSuite.Summary()))
			})

			It("should report failure", func() {
				Ω(success).Should(BeFalse())
				Ω(reporter1.EndSummary.SuiteSucceeded).Should(BeFalse())
				Ω(reporter1.EndSummary.NumberOfFailedSpecs).Should(Equal(0))
			})

			It("should dump the writer", func() {
				Ω(writer.EventStream).Should(ContainElement("DUMP"))
			})
		})
	})

	Describe("When instructed to fail fast", func() {
		BeforeEach(func() {
			conf := config.GinkgoConfigType{
				FailFast: true,
			}
			runner = newRunner(conf, nil, newAftSuite("after-suite", false), newSpec("passing", noneFlag, false), newSpec("failing", noneFlag, true), newSpec("dont-see", noneFlag, true), newSpec("dont-see", noneFlag, true))
		})

		It("should return false, report failure, and not run anything past the failing test", func() {
			Ω(runner.Run()).Should(BeFalse())
			Ω(reporter1.EndSummary.SuiteSucceeded).Should(BeFalse())
			Ω(thingsThatRan).Should(Equal([]string{"passing", "failing", "after-suite"}))
		})

		It("should announce the subsequent specs as skipped", func() {
			runner.Run()
			Ω(reporter1.SpecSummaries).Should(HaveLen(4))
			Ω(reporter1.SpecSummaries[2].State).Should(Equal(types.SpecStateSkipped))
			Ω(reporter1.SpecSummaries[3].State).Should(Equal(types.SpecStateSkipped))
		})

		It("should mark all subsequent specs as skipped", func() {
			runner.Run()
			Ω(reporter1.EndSummary.NumberOfSkippedSpecs).Should(Equal(2))
		})
	})

	Describe("Marking failure and success", func() {
		Context("when all tests pass", func() {
			BeforeEach(func() {
				runner = newRunner(config.GinkgoConfigType{}, nil, nil, newSpec("passing", noneFlag, false), newSpec("pending", pendingFlag, false))
			})

			It("should return true and report success", func() {
				Ω(runner.Run()).Should(BeTrue())
				Ω(reporter1.EndSummary.SuiteSucceeded).Should(BeTrue())
			})
		})

		Context("when a test fails", func() {
			BeforeEach(func() {
				runner = newRunner(config.GinkgoConfigType{}, nil, nil, newSpec("failing", noneFlag, true), newSpec("pending", pendingFlag, false))
			})

			It("should return false and report failure", func() {
				Ω(runner.Run()).Should(BeFalse())
				Ω(reporter1.EndSummary.SuiteSucceeded).Should(BeFalse())
			})
		})

		Context("when there is a pending test, but pendings count as failures", func() {
			BeforeEach(func() {
				runner = newRunner(config.GinkgoConfigType{FailOnPending: true}, nil, nil, newSpec("passing", noneFlag, false), newSpec("pending", pendingFlag, false))
			})

			It("should return false and report failure", func() {
				Ω(runner.Run()).Should(BeFalse())
				Ω(reporter1.EndSummary.SuiteSucceeded).Should(BeFalse())
			})
		})
	})

	Describe("Managing the writer", func() {
		BeforeEach(func() {
			runner = newRunner(
				config.GinkgoConfigType{},
				nil,
				nil,
				newSpec("A", noneFlag, false),
				newSpec("B", noneFlag, true),
				newSpec("C", noneFlag, false),
			)
			reporter1.SpecWillRunStub = func(specSummary *types.SpecSummary) {
				writer.AddEvent("R1.WillRun")
			}
			reporter2.SpecWillRunStub = func(specSummary *types.SpecSummary) {
				writer.AddEvent("R2.WillRun")
			}
			reporter1.SpecDidCompleteStub = func(specSummary *types.SpecSummary) {
				writer.AddEvent("R1.DidComplete")
			}
			reporter2.SpecDidCompleteStub = func(specSummary *types.SpecSummary) {
				writer.AddEvent("R2.DidComplete")
			}
			runner.Run()
		})

		It("should truncate between tests, but only dump if a test fails", func() {
			Ω(writer.EventStream).Should(Equal([]string{
				"TRUNCATE",
				"R1.WillRun",
				"R2.WillRun",
				"A",
				"R2.DidComplete",
				"R1.DidComplete",
				"TRUNCATE",
				"R1.WillRun",
				"R2.WillRun",
				"B",
				"R2.DidComplete",
				"DUMP",
				"R1.DidComplete",
				"TRUNCATE",
				"R1.WillRun",
				"R2.WillRun",
				"C",
				"R2.DidComplete",
				"R1.DidComplete",
			}))
		})
	})

	Describe("CurrentSpecSummary", func() {
		It("should return the spec summary for the currently running spec", func() {
			var summary *types.SpecSummary
			runner = newRunner(
				config.GinkgoConfigType{},
				nil,
				nil,
				newSpec("A", noneFlag, false),
				newSpecWithBody("B", func() {
					var ok bool
					summary, ok = runner.CurrentSpecSummary()
					Ω(ok).Should(BeTrue())
				}),
				newSpec("C", noneFlag, false),
			)
			runner.Run()

			Ω(summary.ComponentTexts).Should(Equal([]string{"B"}))

			summary, ok := runner.CurrentSpecSummary()
			Ω(summary).Should(BeNil())
			Ω(ok).Should(BeFalse())
		})
	})

	Context("When running tests in parallel", func() {
		It("reports the correct number of specs before parallelization", func() {
			specs := spec.NewSpecs([]*spec.Spec{
				newSpec("A", noneFlag, false),
				newSpec("B", pendingFlag, false),
				newSpec("C", noneFlag, false),
			})
			specs.TrimForParallelization(2, 1)
			runner = New("description", nil, specs, nil, []reporters.Reporter{reporter1, reporter2}, writer, config.GinkgoConfigType{})
			runner.Run()

			Ω(reporter1.EndSummary.NumberOfSpecsBeforeParallelization).Should(Equal(3))
			Ω(reporter1.EndSummary.NumberOfTotalSpecs).Should(Equal(2))
			Ω(reporter1.EndSummary.NumberOfSpecsThatWillBeRun).Should(Equal(1))
			Ω(reporter1.EndSummary.NumberOfPendingSpecs).Should(Equal(1))
		})
	})

	Describe("generating a suite id", func() {
		It("should generate an id randomly", func() {
			runnerA := newRunner(config.GinkgoConfigType{}, nil, nil)
			runnerA.Run()
			IDA := reporter1.BeginSummary.SuiteID

			runnerB := newRunner(config.GinkgoConfigType{}, nil, nil)
			runnerB.Run()
			IDB := reporter1.BeginSummary.SuiteID

			IDRegexp := "[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}"
			Ω(IDA).Should(MatchRegexp(IDRegexp))
			Ω(IDB).Should(MatchRegexp(IDRegexp))

			Ω(IDA).ShouldNot(Equal(IDB))
		})
	})
})
