package suite_test

import (
	"bytes"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/ginkgo/internal/suite"
	. "github.com/onsi/gomega"

	"math/rand"
	"time"

	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/internal/codelocation"
	Failer "github.com/onsi/ginkgo/internal/failer"
	Writer "github.com/onsi/ginkgo/internal/writer"
	"github.com/onsi/ginkgo/reporters"
	"github.com/onsi/ginkgo/types"
)

var _ = Describe("Suite", func() {
	var (
		specSuite *Suite
		fakeT     *fakeTestingT
		fakeR     *reporters.FakeReporter
		writer    *Writer.FakeGinkgoWriter
		failer    *Failer.Failer
	)

	BeforeEach(func() {
		writer = Writer.NewFake()
		fakeT = &fakeTestingT{}
		fakeR = reporters.NewFakeReporter()
		failer = Failer.New()
		specSuite = New(failer)
	})

	Describe("running a suite", func() {
		var (
			runOrder             []string
			randomizeAllSpecs    bool
			randomSeed           int64
			focusString          string
			parallelNode         int
			parallelTotal        int
			runResult            bool
			hasProgrammaticFocus bool
		)

		var f = func(runText string) func() {
			return func() {
				runOrder = append(runOrder, runText)
			}
		}

		BeforeEach(func() {
			randomizeAllSpecs = false
			randomSeed = 11
			parallelNode = 1
			parallelTotal = 1
			focusString = ""

			runOrder = make([]string, 0)
			specSuite.SetBeforeSuiteNode(f("BeforeSuite"), codelocation.New(0), 0)
			specSuite.PushBeforeEachNode(f("top BE"), codelocation.New(0), 0)
			specSuite.PushJustBeforeEachNode(f("top JBE"), codelocation.New(0), 0)
			specSuite.PushAfterEachNode(f("top AE"), codelocation.New(0), 0)

			specSuite.PushContainerNode("container", func() {
				specSuite.PushBeforeEachNode(f("BE"), codelocation.New(0), 0)
				specSuite.PushJustBeforeEachNode(f("JBE"), codelocation.New(0), 0)
				specSuite.PushAfterEachNode(f("AE"), codelocation.New(0), 0)
				specSuite.PushItNode("it", f("IT"), types.FlagTypeNone, codelocation.New(0), 0)

				specSuite.PushContainerNode("inner container", func() {
					specSuite.PushItNode("inner it", f("inner IT"), types.FlagTypeNone, codelocation.New(0), 0)
				}, types.FlagTypeNone, codelocation.New(0))
			}, types.FlagTypeNone, codelocation.New(0))

			specSuite.PushContainerNode("container 2", func() {
				specSuite.PushBeforeEachNode(f("BE 2"), codelocation.New(0), 0)
				specSuite.PushItNode("it 2", f("IT 2"), types.FlagTypeNone, codelocation.New(0), 0)
			}, types.FlagTypeNone, codelocation.New(0))

			specSuite.PushItNode("top level it", f("top IT"), types.FlagTypeNone, codelocation.New(0), 0)

			specSuite.SetAfterSuiteNode(f("AfterSuite"), codelocation.New(0), 0)
		})

		JustBeforeEach(func() {
			runResult, hasProgrammaticFocus = specSuite.Run(fakeT, "suite description", []reporters.Reporter{fakeR}, writer, config.GinkgoConfigType{
				RandomSeed:        randomSeed,
				RandomizeAllSpecs: randomizeAllSpecs,
				FocusString:       focusString,
				ParallelNode:      parallelNode,
				ParallelTotal:     parallelTotal,
			})
		})

		It("provides the config and suite description to the reporter", func() {
			Ω(fakeR.Config.RandomSeed).Should(Equal(int64(randomSeed)))
			Ω(fakeR.Config.RandomizeAllSpecs).Should(Equal(randomizeAllSpecs))
			Ω(fakeR.BeginSummary.SuiteDescription).Should(Equal("suite description"))
		})

		It("reports that the BeforeSuite node ran", func() {
			Ω(fakeR.BeforeSuiteSummary).ShouldNot(BeNil())
		})

		It("reports that the AfterSuite node ran", func() {
			Ω(fakeR.AfterSuiteSummary).ShouldNot(BeNil())
		})

		It("provides information about the current test", func() {
			description := CurrentGinkgoTestDescription()
			Ω(description.ComponentTexts).Should(Equal([]string{"Suite", "running a suite", "provides information about the current test"}))
			Ω(description.FullTestText).Should(Equal("Suite running a suite provides information about the current test"))
			Ω(description.TestText).Should(Equal("provides information about the current test"))
			Ω(description.IsMeasurement).Should(BeFalse())
			Ω(description.FileName).Should(ContainSubstring("suite_test.go"))
			Ω(description.LineNumber).Should(BeNumerically(">", 50))
			Ω(description.LineNumber).Should(BeNumerically("<", 150))
			Ω(description.Failed).Should(BeFalse())
		})

		Measure("should run measurements", func(b Benchmarker) {
			r := rand.New(rand.NewSource(time.Now().UnixNano()))

			runtime := b.Time("sleeping", func() {
				sleepTime := time.Duration(r.Float64() * 0.01 * float64(time.Second))
				time.Sleep(sleepTime)
			})
			Ω(runtime.Seconds()).Should(BeNumerically("<=", 1))
			Ω(runtime.Seconds()).Should(BeNumerically(">=", 0))

			randomValue := r.Float64() * 10.0
			b.RecordValue("random value", randomValue)
			Ω(randomValue).Should(BeNumerically("<=", 10.0))
			Ω(randomValue).Should(BeNumerically(">=", 0.0))

			b.RecordValueWithPrecision("specific value", 123.4567, "ms", 2)
			b.RecordValueWithPrecision("specific value", 234.5678, "ms", 2)
		}, 10)

		It("creates a node hierarchy, converts it to a spec collection, and runs it", func() {
			Ω(runOrder).Should(Equal([]string{
				"BeforeSuite",
				"top BE", "BE", "top JBE", "JBE", "IT", "AE", "top AE",
				"top BE", "BE", "top JBE", "JBE", "inner IT", "AE", "top AE",
				"top BE", "BE 2", "top JBE", "IT 2", "top AE",
				"top BE", "top JBE", "top IT", "top AE",
				"AfterSuite",
			}))
		})

		Context("when told to randomize all specs", func() {
			BeforeEach(func() {
				randomizeAllSpecs = true
			})

			It("does", func() {
				Ω(runOrder).Should(Equal([]string{
					"BeforeSuite",
					"top BE", "top JBE", "top IT", "top AE",
					"top BE", "BE", "top JBE", "JBE", "inner IT", "AE", "top AE",
					"top BE", "BE", "top JBE", "JBE", "IT", "AE", "top AE",
					"top BE", "BE 2", "top JBE", "IT 2", "top AE",
					"AfterSuite",
				}))
			})
		})

		Context("when provided with a filter", func() {
			BeforeEach(func() {
				focusString = `inner|\d`
			})

			It("converts the filter to a regular expression and uses it to filter the running specs", func() {
				Ω(runOrder).Should(Equal([]string{
					"BeforeSuite",
					"top BE", "BE", "top JBE", "JBE", "inner IT", "AE", "top AE",
					"top BE", "BE 2", "top JBE", "IT 2", "top AE",
					"AfterSuite",
				}))
			})

			It("should not report a programmatic focus", func() {
				Ω(hasProgrammaticFocus).Should(BeFalse())
			})
		})

		Context("with a programatically focused spec", func() {
			BeforeEach(func() {
				specSuite.PushItNode("focused it", f("focused it"), types.FlagTypeFocused, codelocation.New(0), 0)

				specSuite.PushContainerNode("focused container", func() {
					specSuite.PushItNode("inner focused it", f("inner focused it"), types.FlagTypeFocused, codelocation.New(0), 0)
					specSuite.PushItNode("inner unfocused it", f("inner unfocused it"), types.FlagTypeNone, codelocation.New(0), 0)
				}, types.FlagTypeFocused, codelocation.New(0))

			})

			It("should only run the focused test, applying backpropagation to favor most deeply focused leaf nodes", func() {
				Ω(runOrder).Should(Equal([]string{
					"BeforeSuite",
					"top BE", "top JBE", "focused it", "top AE",
					"top BE", "top JBE", "inner focused it", "top AE",
					"AfterSuite",
				}))
			})

			It("should report a programmatic focus", func() {
				Ω(hasProgrammaticFocus).Should(BeTrue())
			})
		})

		Context("when the specs pass", func() {
			It("doesn't report a failure", func() {
				Ω(fakeT.didFail).Should(BeFalse())
			})

			It("should return true", func() {
				Ω(runResult).Should(BeTrue())
			})
		})

		Context("when a spec fails", func() {
			var location types.CodeLocation
			BeforeEach(func() {
				specSuite.PushItNode("top level it", func() {
					location = codelocation.New(0)
					failer.Fail("oops!", location)
				}, types.FlagTypeNone, codelocation.New(0), 0)
			})

			It("should return false", func() {
				Ω(runResult).Should(BeFalse())
			})

			It("reports a failure", func() {
				Ω(fakeT.didFail).Should(BeTrue())
			})

			It("generates the correct failure data", func() {
				Ω(fakeR.SpecSummaries[0].Failure.Message).Should(Equal("oops!"))
				Ω(fakeR.SpecSummaries[0].Failure.Location).Should(Equal(location))
			})
		})

		Context("when runnable nodes are nested within other runnable nodes", func() {
			Context("when an It is nested", func() {
				BeforeEach(func() {
					specSuite.PushItNode("top level it", func() {
						specSuite.PushItNode("nested it", f("oops"), types.FlagTypeNone, codelocation.New(0), 0)
					}, types.FlagTypeNone, codelocation.New(0), 0)
				})

				It("should fail", func() {
					Ω(fakeT.didFail).Should(BeTrue())
				})
			})

			Context("when a Measure is nested", func() {
				BeforeEach(func() {
					specSuite.PushItNode("top level it", func() {
						specSuite.PushMeasureNode("nested measure", func(Benchmarker) {}, types.FlagTypeNone, codelocation.New(0), 10)
					}, types.FlagTypeNone, codelocation.New(0), 0)
				})

				It("should fail", func() {
					Ω(fakeT.didFail).Should(BeTrue())
				})
			})

			Context("when a BeforeEach is nested", func() {
				BeforeEach(func() {
					specSuite.PushItNode("top level it", func() {
						specSuite.PushBeforeEachNode(f("nested bef"), codelocation.New(0), 0)
					}, types.FlagTypeNone, codelocation.New(0), 0)
				})

				It("should fail", func() {
					Ω(fakeT.didFail).Should(BeTrue())
				})
			})

			Context("when a JustBeforeEach is nested", func() {
				BeforeEach(func() {
					specSuite.PushItNode("top level it", func() {
						specSuite.PushJustBeforeEachNode(f("nested jbef"), codelocation.New(0), 0)
					}, types.FlagTypeNone, codelocation.New(0), 0)
				})

				It("should fail", func() {
					Ω(fakeT.didFail).Should(BeTrue())
				})
			})

			Context("when a AfterEach is nested", func() {
				BeforeEach(func() {
					specSuite.PushItNode("top level it", func() {
						specSuite.PushAfterEachNode(f("nested aft"), codelocation.New(0), 0)
					}, types.FlagTypeNone, codelocation.New(0), 0)
				})

				It("should fail", func() {
					Ω(fakeT.didFail).Should(BeTrue())
				})
			})
		})
	})

	Describe("BeforeSuite", func() {
		Context("when setting BeforeSuite more than once", func() {
			It("should panic", func() {
				specSuite.SetBeforeSuiteNode(func() {}, codelocation.New(0), 0)

				Ω(func() {
					specSuite.SetBeforeSuiteNode(func() {}, codelocation.New(0), 0)
				}).Should(Panic())

			})
		})
	})

	Describe("AfterSuite", func() {
		Context("when setting AfterSuite more than once", func() {
			It("should panic", func() {
				specSuite.SetAfterSuiteNode(func() {}, codelocation.New(0), 0)

				Ω(func() {
					specSuite.SetAfterSuiteNode(func() {}, codelocation.New(0), 0)
				}).Should(Panic())
			})
		})
	})

	Describe("By", func() {
		It("writes to the GinkgoWriter", func() {
			originalGinkgoWriter := GinkgoWriter
			buffer := &bytes.Buffer{}

			GinkgoWriter = buffer
			By("Saying Hello GinkgoWriter")
			GinkgoWriter = originalGinkgoWriter

			Ω(buffer.String()).Should(ContainSubstring("STEP"))
			Ω(buffer.String()).Should(ContainSubstring(": Saying Hello GinkgoWriter\n"))
		})

		It("calls the passed-in callback if present", func() {
			a := 0
			By("calling the callback", func() {
				a = 1
			})
			Ω(a).Should(Equal(1))
		})

		It("panics if there is more than one callback", func() {
			Ω(func() {
				By("registering more than one callback", func() {}, func() {})
			}).Should(Panic())
		})
	})

	Describe("GinkgoRandomSeed", func() {
		It("returns the current config's random seed", func() {
			Ω(GinkgoRandomSeed()).Should(Equal(config.GinkgoConfig.RandomSeed))
		})
	})
})
