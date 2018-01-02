package spec_test

import (
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/onsi/gomega/gbytes"

	. "github.com/onsi/ginkgo/internal/spec"

	"github.com/onsi/ginkgo/internal/codelocation"
	"github.com/onsi/ginkgo/internal/containernode"
	Failer "github.com/onsi/ginkgo/internal/failer"
	"github.com/onsi/ginkgo/internal/leafnodes"
	"github.com/onsi/ginkgo/types"
)

var noneFlag = types.FlagTypeNone
var focusedFlag = types.FlagTypeFocused
var pendingFlag = types.FlagTypePending

var _ = Describe("Spec", func() {
	var (
		failer       *Failer.Failer
		codeLocation types.CodeLocation
		nodesThatRan []string
		spec         *Spec
		buffer       *gbytes.Buffer
	)

	newBody := func(text string, fail bool) func() {
		return func() {
			nodesThatRan = append(nodesThatRan, text)
			if fail {
				failer.Fail(text, codeLocation)
			}
		}
	}

	newIt := func(text string, flag types.FlagType, fail bool) *leafnodes.ItNode {
		return leafnodes.NewItNode(text, newBody(text, fail), flag, codeLocation, 0, failer, 0)
	}

	newItWithBody := func(text string, body interface{}) *leafnodes.ItNode {
		return leafnodes.NewItNode(text, body, noneFlag, codeLocation, 0, failer, 0)
	}

	newMeasure := func(text string, flag types.FlagType, fail bool, samples int) *leafnodes.MeasureNode {
		return leafnodes.NewMeasureNode(text, func(Benchmarker) {
			nodesThatRan = append(nodesThatRan, text)
			if fail {
				failer.Fail(text, codeLocation)
			}
		}, flag, codeLocation, samples, failer, 0)
	}

	newBef := func(text string, fail bool) leafnodes.BasicNode {
		return leafnodes.NewBeforeEachNode(newBody(text, fail), codeLocation, 0, failer, 0)
	}

	newAft := func(text string, fail bool) leafnodes.BasicNode {
		return leafnodes.NewAfterEachNode(newBody(text, fail), codeLocation, 0, failer, 0)
	}

	newJusBef := func(text string, fail bool) leafnodes.BasicNode {
		return leafnodes.NewJustBeforeEachNode(newBody(text, fail), codeLocation, 0, failer, 0)
	}

	newContainer := func(text string, flag types.FlagType, setupNodes ...leafnodes.BasicNode) *containernode.ContainerNode {
		c := containernode.New(text, flag, codeLocation)
		for _, node := range setupNodes {
			c.PushSetupNode(node)
		}
		return c
	}

	containers := func(containers ...*containernode.ContainerNode) []*containernode.ContainerNode {
		return containers
	}

	BeforeEach(func() {
		buffer = gbytes.NewBuffer()
		failer = Failer.New()
		codeLocation = codelocation.New(0)
		nodesThatRan = []string{}
	})

	Describe("marking specs focused and pending", func() {
		It("should satisfy various caes", func() {
			cases := []struct {
				ContainerFlags []types.FlagType
				SubjectFlag    types.FlagType
				Pending        bool
				Focused        bool
			}{
				{[]types.FlagType{}, noneFlag, false, false},
				{[]types.FlagType{}, focusedFlag, false, true},
				{[]types.FlagType{}, pendingFlag, true, false},
				{[]types.FlagType{noneFlag}, noneFlag, false, false},
				{[]types.FlagType{focusedFlag}, noneFlag, false, true},
				{[]types.FlagType{pendingFlag}, noneFlag, true, false},
				{[]types.FlagType{noneFlag}, focusedFlag, false, true},
				{[]types.FlagType{focusedFlag}, focusedFlag, false, true},
				{[]types.FlagType{pendingFlag}, focusedFlag, true, true},
				{[]types.FlagType{noneFlag}, pendingFlag, true, false},
				{[]types.FlagType{focusedFlag}, pendingFlag, true, true},
				{[]types.FlagType{pendingFlag}, pendingFlag, true, false},
				{[]types.FlagType{focusedFlag, noneFlag}, noneFlag, false, true},
				{[]types.FlagType{noneFlag, focusedFlag}, noneFlag, false, true},
				{[]types.FlagType{pendingFlag, noneFlag}, noneFlag, true, false},
				{[]types.FlagType{noneFlag, pendingFlag}, noneFlag, true, false},
				{[]types.FlagType{focusedFlag, pendingFlag}, noneFlag, true, true},
			}

			for i, c := range cases {
				subject := newIt("it node", c.SubjectFlag, false)
				containers := []*containernode.ContainerNode{}
				for _, flag := range c.ContainerFlags {
					containers = append(containers, newContainer("container", flag))
				}

				spec := New(subject, containers, false)
				Ω(spec.Pending()).Should(Equal(c.Pending), "Case %d: %#v", i, c)
				Ω(spec.Focused()).Should(Equal(c.Focused), "Case %d: %#v", i, c)

				if c.Pending {
					Ω(spec.Summary("").State).Should(Equal(types.SpecStatePending))
				}
			}
		})
	})

	Describe("Skip", func() {
		It("should be skipped", func() {
			spec := New(newIt("it node", noneFlag, false), containers(newContainer("container", noneFlag)), false)
			Ω(spec.Skipped()).Should(BeFalse())
			spec.Skip()
			Ω(spec.Skipped()).Should(BeTrue())
			Ω(spec.Summary("").State).Should(Equal(types.SpecStateSkipped))
		})
	})

	Describe("IsMeasurement", func() {
		It("should be true if the subject is a measurement node", func() {
			spec := New(newIt("it node", noneFlag, false), containers(newContainer("container", noneFlag)), false)
			Ω(spec.IsMeasurement()).Should(BeFalse())
			Ω(spec.Summary("").IsMeasurement).Should(BeFalse())
			Ω(spec.Summary("").NumberOfSamples).Should(Equal(1))

			spec = New(newMeasure("measure node", noneFlag, false, 10), containers(newContainer("container", noneFlag)), false)
			Ω(spec.IsMeasurement()).Should(BeTrue())
			Ω(spec.Summary("").IsMeasurement).Should(BeTrue())
			Ω(spec.Summary("").NumberOfSamples).Should(Equal(10))
		})
	})

	Describe("Passed", func() {
		It("should pass when the subject passed", func() {
			spec := New(newIt("it node", noneFlag, false), containers(), false)
			spec.Run(buffer)

			Ω(spec.Passed()).Should(BeTrue())
			Ω(spec.Failed()).Should(BeFalse())
			Ω(spec.Summary("").State).Should(Equal(types.SpecStatePassed))
			Ω(spec.Summary("").Failure).Should(BeZero())
		})
	})

	Describe("Flaked", func() {
		It("should work if Run is called twice and gets different results", func() {
			i := 0
			spec := New(newItWithBody("flaky it", func() {
				i++
				if i == 1 {
					failer.Fail("oops", codeLocation)
				}
			}), containers(), false)
			spec.Run(buffer)
			Ω(spec.Passed()).Should(BeFalse())
			Ω(spec.Failed()).Should(BeTrue())
			Ω(spec.Flaked()).Should(BeFalse())
			Ω(spec.Summary("").State).Should(Equal(types.SpecStateFailed))
			Ω(spec.Summary("").Failure.Message).Should(Equal("oops"))
			spec.Run(buffer)
			Ω(spec.Passed()).Should(BeTrue())
			Ω(spec.Failed()).Should(BeFalse())
			Ω(spec.Flaked()).Should(BeTrue())
			Ω(spec.Summary("").State).Should(Equal(types.SpecStatePassed))
		})
	})

	Describe("Failed", func() {
		It("should be failed if the failure was panic", func() {
			spec := New(newItWithBody("panicky it", func() {
				panic("bam")
			}), containers(), false)
			spec.Run(buffer)
			Ω(spec.Passed()).Should(BeFalse())
			Ω(spec.Failed()).Should(BeTrue())
			Ω(spec.Summary("").State).Should(Equal(types.SpecStatePanicked))
			Ω(spec.Summary("").Failure.Message).Should(Equal("Test Panicked"))
			Ω(spec.Summary("").Failure.ForwardedPanic).Should(Equal("bam"))
		})

		It("should be failed if the failure was a timeout", func() {
			spec := New(newItWithBody("sleepy it", func(done Done) {}), containers(), false)
			spec.Run(buffer)
			Ω(spec.Passed()).Should(BeFalse())
			Ω(spec.Failed()).Should(BeTrue())
			Ω(spec.Summary("").State).Should(Equal(types.SpecStateTimedOut))
			Ω(spec.Summary("").Failure.Message).Should(Equal("Timed out"))
		})

		It("should be failed if the failure was... a failure", func() {
			spec := New(newItWithBody("failing it", func() {
				failer.Fail("bam", codeLocation)
			}), containers(), false)
			spec.Run(buffer)
			Ω(spec.Passed()).Should(BeFalse())
			Ω(spec.Failed()).Should(BeTrue())
			Ω(spec.Summary("").State).Should(Equal(types.SpecStateFailed))
			Ω(spec.Summary("").Failure.Message).Should(Equal("bam"))
		})
	})

	Describe("Concatenated string", func() {
		It("should concatenate the texts of the containers and the subject", func() {
			spec := New(
				newIt("it node", noneFlag, false),
				containers(
					newContainer("outer container", noneFlag),
					newContainer("inner container", noneFlag),
				),
				false,
			)

			Ω(spec.ConcatenatedString()).Should(Equal("outer container inner container it node"))
		})
	})

	Describe("running it specs", func() {
		Context("with just an it", func() {
			Context("that succeeds", func() {
				It("should run the it and report on its success", func() {
					spec := New(newIt("it node", noneFlag, false), containers(), false)
					spec.Run(buffer)
					Ω(spec.Passed()).Should(BeTrue())
					Ω(spec.Failed()).Should(BeFalse())
					Ω(nodesThatRan).Should(Equal([]string{"it node"}))
				})
			})

			Context("that fails", func() {
				It("should run the it and report on its success", func() {
					spec := New(newIt("it node", noneFlag, true), containers(), false)
					spec.Run(buffer)
					Ω(spec.Passed()).Should(BeFalse())
					Ω(spec.Failed()).Should(BeTrue())
					Ω(spec.Summary("").Failure.Message).Should(Equal("it node"))
					Ω(nodesThatRan).Should(Equal([]string{"it node"}))
				})
			})
		})

		Context("with a full set of setup nodes", func() {
			var failingNodes map[string]bool

			BeforeEach(func() {
				failingNodes = map[string]bool{}
			})

			JustBeforeEach(func() {
				spec = New(
					newIt("it node", noneFlag, failingNodes["it node"]),
					containers(
						newContainer("outer container", noneFlag,
							newBef("outer bef A", failingNodes["outer bef A"]),
							newBef("outer bef B", failingNodes["outer bef B"]),
							newJusBef("outer jusbef A", failingNodes["outer jusbef A"]),
							newJusBef("outer jusbef B", failingNodes["outer jusbef B"]),
							newAft("outer aft A", failingNodes["outer aft A"]),
							newAft("outer aft B", failingNodes["outer aft B"]),
						),
						newContainer("inner container", noneFlag,
							newBef("inner bef A", failingNodes["inner bef A"]),
							newBef("inner bef B", failingNodes["inner bef B"]),
							newJusBef("inner jusbef A", failingNodes["inner jusbef A"]),
							newJusBef("inner jusbef B", failingNodes["inner jusbef B"]),
							newAft("inner aft A", failingNodes["inner aft A"]),
							newAft("inner aft B", failingNodes["inner aft B"]),
						),
					),
					false,
				)
				spec.Run(buffer)
			})

			Context("that all pass", func() {
				It("should walk through the nodes in the correct order", func() {
					Ω(spec.Passed()).Should(BeTrue())
					Ω(spec.Failed()).Should(BeFalse())
					Ω(nodesThatRan).Should(Equal([]string{
						"outer bef A",
						"outer bef B",
						"inner bef A",
						"inner bef B",
						"outer jusbef A",
						"outer jusbef B",
						"inner jusbef A",
						"inner jusbef B",
						"it node",
						"inner aft A",
						"inner aft B",
						"outer aft A",
						"outer aft B",
					}))
				})
			})

			Context("when the subject fails", func() {
				BeforeEach(func() {
					failingNodes["it node"] = true
				})

				It("should run the afters", func() {
					Ω(spec.Passed()).Should(BeFalse())
					Ω(spec.Failed()).Should(BeTrue())
					Ω(nodesThatRan).Should(Equal([]string{
						"outer bef A",
						"outer bef B",
						"inner bef A",
						"inner bef B",
						"outer jusbef A",
						"outer jusbef B",
						"inner jusbef A",
						"inner jusbef B",
						"it node",
						"inner aft A",
						"inner aft B",
						"outer aft A",
						"outer aft B",
					}))
					Ω(spec.Summary("").Failure.Message).Should(Equal("it node"))
				})
			})

			Context("when an inner before fails", func() {
				BeforeEach(func() {
					failingNodes["inner bef A"] = true
				})

				It("should not run any other befores, but it should run the subsequent afters", func() {
					Ω(spec.Passed()).Should(BeFalse())
					Ω(spec.Failed()).Should(BeTrue())
					Ω(nodesThatRan).Should(Equal([]string{
						"outer bef A",
						"outer bef B",
						"inner bef A",
						"inner aft A",
						"inner aft B",
						"outer aft A",
						"outer aft B",
					}))
					Ω(spec.Summary("").Failure.Message).Should(Equal("inner bef A"))
				})
			})

			Context("when an outer before fails", func() {
				BeforeEach(func() {
					failingNodes["outer bef B"] = true
				})

				It("should not run any other befores, but it should run the subsequent afters", func() {
					Ω(spec.Passed()).Should(BeFalse())
					Ω(spec.Failed()).Should(BeTrue())
					Ω(nodesThatRan).Should(Equal([]string{
						"outer bef A",
						"outer bef B",
						"outer aft A",
						"outer aft B",
					}))
					Ω(spec.Summary("").Failure.Message).Should(Equal("outer bef B"))
				})
			})

			Context("when an after fails", func() {
				BeforeEach(func() {
					failingNodes["inner aft B"] = true
				})

				It("should run all other afters, but mark the test as failed", func() {
					Ω(spec.Passed()).Should(BeFalse())
					Ω(spec.Failed()).Should(BeTrue())
					Ω(nodesThatRan).Should(Equal([]string{
						"outer bef A",
						"outer bef B",
						"inner bef A",
						"inner bef B",
						"outer jusbef A",
						"outer jusbef B",
						"inner jusbef A",
						"inner jusbef B",
						"it node",
						"inner aft A",
						"inner aft B",
						"outer aft A",
						"outer aft B",
					}))
					Ω(spec.Summary("").Failure.Message).Should(Equal("inner aft B"))
				})
			})

			Context("when a just before each fails", func() {
				BeforeEach(func() {
					failingNodes["outer jusbef B"] = true
				})

				It("should run the afters, but not the subject", func() {
					Ω(spec.Passed()).Should(BeFalse())
					Ω(spec.Failed()).Should(BeTrue())
					Ω(nodesThatRan).Should(Equal([]string{
						"outer bef A",
						"outer bef B",
						"inner bef A",
						"inner bef B",
						"outer jusbef A",
						"outer jusbef B",
						"inner aft A",
						"inner aft B",
						"outer aft A",
						"outer aft B",
					}))
					Ω(spec.Summary("").Failure.Message).Should(Equal("outer jusbef B"))
				})
			})

			Context("when an after fails after an earlier node has failed", func() {
				BeforeEach(func() {
					failingNodes["it node"] = true
					failingNodes["inner aft B"] = true
				})

				It("should record the earlier failure", func() {
					Ω(spec.Passed()).Should(BeFalse())
					Ω(spec.Failed()).Should(BeTrue())
					Ω(nodesThatRan).Should(Equal([]string{
						"outer bef A",
						"outer bef B",
						"inner bef A",
						"inner bef B",
						"outer jusbef A",
						"outer jusbef B",
						"inner jusbef A",
						"inner jusbef B",
						"it node",
						"inner aft A",
						"inner aft B",
						"outer aft A",
						"outer aft B",
					}))
					Ω(spec.Summary("").Failure.Message).Should(Equal("it node"))
				})
			})
		})
	})

	Describe("running measurement specs", func() {
		Context("when the measurement succeeds", func() {
			It("should run N samples", func() {
				spec = New(
					newMeasure("measure node", noneFlag, false, 3),
					containers(
						newContainer("container", noneFlag,
							newBef("bef A", false),
							newJusBef("jusbef A", false),
							newAft("aft A", false),
						),
					),
					false,
				)
				spec.Run(buffer)

				Ω(spec.Passed()).Should(BeTrue())
				Ω(spec.Failed()).Should(BeFalse())
				Ω(nodesThatRan).Should(Equal([]string{
					"bef A",
					"jusbef A",
					"measure node",
					"aft A",
					"bef A",
					"jusbef A",
					"measure node",
					"aft A",
					"bef A",
					"jusbef A",
					"measure node",
					"aft A",
				}))
			})
		})

		Context("when the measurement fails", func() {
			It("should bail after the failure occurs", func() {
				spec = New(
					newMeasure("measure node", noneFlag, true, 3),
					containers(
						newContainer("container", noneFlag,
							newBef("bef A", false),
							newJusBef("jusbef A", false),
							newAft("aft A", false),
						),
					),
					false,
				)
				spec.Run(buffer)

				Ω(spec.Passed()).Should(BeFalse())
				Ω(spec.Failed()).Should(BeTrue())
				Ω(nodesThatRan).Should(Equal([]string{
					"bef A",
					"jusbef A",
					"measure node",
					"aft A",
				}))
			})
		})
	})

	Describe("Summary", func() {
		var (
			subjectCodeLocation        types.CodeLocation
			outerContainerCodeLocation types.CodeLocation
			innerContainerCodeLocation types.CodeLocation
			summary                    *types.SpecSummary
		)

		BeforeEach(func() {
			subjectCodeLocation = codelocation.New(0)
			outerContainerCodeLocation = codelocation.New(0)
			innerContainerCodeLocation = codelocation.New(0)

			spec = New(
				leafnodes.NewItNode("it node", func() {
					time.Sleep(10 * time.Millisecond)
				}, noneFlag, subjectCodeLocation, 0, failer, 0),
				containers(
					containernode.New("outer container", noneFlag, outerContainerCodeLocation),
					containernode.New("inner container", noneFlag, innerContainerCodeLocation),
				),
				false,
			)

			spec.Run(buffer)
			Ω(spec.Passed()).Should(BeTrue())
			summary = spec.Summary("suite id")
		})

		It("should have the suite id", func() {
			Ω(summary.SuiteID).Should(Equal("suite id"))
		})

		It("should have the component texts and code locations", func() {
			Ω(summary.ComponentTexts).Should(Equal([]string{"outer container", "inner container", "it node"}))
			Ω(summary.ComponentCodeLocations).Should(Equal([]types.CodeLocation{outerContainerCodeLocation, innerContainerCodeLocation, subjectCodeLocation}))
		})

		It("should have a runtime", func() {
			Ω(summary.RunTime).Should(BeNumerically(">=", 10*time.Millisecond))
		})

		It("should not be a measurement, or have a measurement summary", func() {
			Ω(summary.IsMeasurement).Should(BeFalse())
			Ω(summary.Measurements).Should(BeEmpty())
		})
	})

	Describe("Summaries for measurements", func() {
		var summary *types.SpecSummary

		BeforeEach(func() {
			spec = New(leafnodes.NewMeasureNode("measure node", func(b Benchmarker) {
				b.RecordValue("a value", 7, "some info")
				b.RecordValueWithPrecision("another value", 8, "ns", 5, "more info")
			}, noneFlag, codeLocation, 4, failer, 0), containers(), false)
			spec.Run(buffer)
			Ω(spec.Passed()).Should(BeTrue())
			summary = spec.Summary("suite id")
		})

		It("should include the number of samples", func() {
			Ω(summary.NumberOfSamples).Should(Equal(4))
		})

		It("should be a measurement", func() {
			Ω(summary.IsMeasurement).Should(BeTrue())
		})

		It("should have the measurements report", func() {
			Ω(summary.Measurements).Should(HaveKey("a value"))
			report := summary.Measurements["a value"]
			Ω(report.Name).Should(Equal("a value"))
			Ω(report.Info).Should(Equal("some info"))
			Ω(report.Results).Should(Equal([]float64{7, 7, 7, 7}))

			Ω(summary.Measurements).Should(HaveKey("another value"))
			report = summary.Measurements["another value"]
			Ω(report.Name).Should(Equal("another value"))
			Ω(report.Info).Should(Equal("more info"))
			Ω(report.Results).Should(Equal([]float64{8, 8, 8, 8}))
			Ω(report.Units).Should(Equal("ns"))
			Ω(report.Precision).Should(Equal(5))
		})
	})

	Describe("When told to emit progress", func() {
		It("should emit progress to the writer as it runs Befores, JustBefores, Afters, and Its", func() {
			spec = New(
				newIt("it node", noneFlag, false),
				containers(
					newContainer("outer container", noneFlag,
						newBef("outer bef A", false),
						newJusBef("outer jusbef A", false),
						newAft("outer aft A", false),
					),
					newContainer("inner container", noneFlag,
						newBef("inner bef A", false),
						newJusBef("inner jusbef A", false),
						newAft("inner aft A", false),
					),
				),
				true,
			)
			spec.Run(buffer)

			Ω(buffer).Should(gbytes.Say(`\[BeforeEach\] outer container`))
			Ω(buffer).Should(gbytes.Say(`\[BeforeEach\] inner container`))
			Ω(buffer).Should(gbytes.Say(`\[JustBeforeEach\] outer container`))
			Ω(buffer).Should(gbytes.Say(`\[JustBeforeEach\] inner container`))
			Ω(buffer).Should(gbytes.Say(`\[It\] it node`))
			Ω(buffer).Should(gbytes.Say(`\[AfterEach\] inner container`))
			Ω(buffer).Should(gbytes.Say(`\[AfterEach\] outer container`))
		})

		It("should emit progress to the writer as it runs Befores, JustBefores, Afters, and Measures", func() {
			spec = New(
				newMeasure("measure node", noneFlag, false, 2),
				containers(),
				true,
			)
			spec.Run(buffer)

			Ω(buffer).Should(gbytes.Say(`\[Measure\] measure node`))
			Ω(buffer).Should(gbytes.Say(`\[Measure\] measure node`))
		})
	})
})
