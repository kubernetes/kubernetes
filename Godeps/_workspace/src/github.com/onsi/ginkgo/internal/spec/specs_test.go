package spec_test

import (
	"math/rand"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/ginkgo/internal/spec"
	. "github.com/onsi/gomega"

	"github.com/onsi/ginkgo/internal/codelocation"
	"github.com/onsi/ginkgo/internal/containernode"
	"github.com/onsi/ginkgo/internal/leafnodes"
	"github.com/onsi/ginkgo/types"
)

var _ = Describe("Specs", func() {
	var specs *Specs

	newSpec := func(text string, flag types.FlagType) *Spec {
		subject := leafnodes.NewItNode(text, func() {}, flag, codelocation.New(0), 0, nil, 0)
		return New(subject, []*containernode.ContainerNode{}, false)
	}

	newMeasureSpec := func(text string, flag types.FlagType) *Spec {
		subject := leafnodes.NewMeasureNode(text, func(Benchmarker) {}, flag, codelocation.New(0), 0, nil, 0)
		return New(subject, []*containernode.ContainerNode{}, false)
	}

	newSpecs := func(args ...interface{}) *Specs {
		specs := []*Spec{}
		for index := 0; index < len(args)-1; index += 2 {
			specs = append(specs, newSpec(args[index].(string), args[index+1].(types.FlagType)))
		}
		return NewSpecs(specs)
	}

	specTexts := func(specs *Specs) []string {
		texts := []string{}
		for _, spec := range specs.Specs() {
			texts = append(texts, spec.ConcatenatedString())
		}
		return texts
	}

	willRunTexts := func(specs *Specs) []string {
		texts := []string{}
		for _, spec := range specs.Specs() {
			if !(spec.Skipped() || spec.Pending()) {
				texts = append(texts, spec.ConcatenatedString())
			}
		}
		return texts
	}

	skippedTexts := func(specs *Specs) []string {
		texts := []string{}
		for _, spec := range specs.Specs() {
			if spec.Skipped() {
				texts = append(texts, spec.ConcatenatedString())
			}
		}
		return texts
	}

	pendingTexts := func(specs *Specs) []string {
		texts := []string{}
		for _, spec := range specs.Specs() {
			if spec.Pending() {
				texts = append(texts, spec.ConcatenatedString())
			}
		}
		return texts
	}

	Describe("Shuffling specs", func() {
		It("should shuffle the specs using the passed in randomizer", func() {
			specs17 := newSpecs("C", noneFlag, "A", noneFlag, "B", noneFlag)
			specs17.Shuffle(rand.New(rand.NewSource(17)))
			texts17 := specTexts(specs17)

			specs17Again := newSpecs("C", noneFlag, "A", noneFlag, "B", noneFlag)
			specs17Again.Shuffle(rand.New(rand.NewSource(17)))
			texts17Again := specTexts(specs17Again)

			specs15 := newSpecs("C", noneFlag, "A", noneFlag, "B", noneFlag)
			specs15.Shuffle(rand.New(rand.NewSource(15)))
			texts15 := specTexts(specs15)

			specsUnshuffled := newSpecs("C", noneFlag, "A", noneFlag, "B", noneFlag)
			textsUnshuffled := specTexts(specsUnshuffled)

			Ω(textsUnshuffled).Should(Equal([]string{"C", "A", "B"}))

			Ω(texts17).Should(Equal(texts17Again))
			Ω(texts17).ShouldNot(Equal(texts15))
			Ω(texts17).ShouldNot(Equal(textsUnshuffled))
			Ω(texts15).ShouldNot(Equal(textsUnshuffled))

			Ω(texts17).Should(HaveLen(3))
			Ω(texts17).Should(ContainElement("A"))
			Ω(texts17).Should(ContainElement("B"))
			Ω(texts17).Should(ContainElement("C"))

			Ω(texts15).Should(HaveLen(3))
			Ω(texts15).Should(ContainElement("A"))
			Ω(texts15).Should(ContainElement("B"))
			Ω(texts15).Should(ContainElement("C"))
		})
	})

	Describe("with no programmatic focus", func() {
		BeforeEach(func() {
			specs = newSpecs("A1", noneFlag, "A2", noneFlag, "B1", noneFlag, "B2", pendingFlag)
			specs.ApplyFocus("", "", "")
		})

		It("should not report as having programmatic specs", func() {
			Ω(specs.HasProgrammaticFocus()).Should(BeFalse())
		})
	})

	Describe("Applying focus/skip", func() {
		var description, focusString, skipString string

		BeforeEach(func() {
			description, focusString, skipString = "", "", ""
		})

		JustBeforeEach(func() {
			specs = newSpecs("A1", focusedFlag, "A2", noneFlag, "B1", focusedFlag, "B2", pendingFlag)
			specs.ApplyFocus(description, focusString, skipString)
		})

		Context("with neither a focus string nor a skip string", func() {
			It("should apply the programmatic focus", func() {
				Ω(willRunTexts(specs)).Should(Equal([]string{"A1", "B1"}))
				Ω(skippedTexts(specs)).Should(Equal([]string{"A2", "B2"}))
				Ω(pendingTexts(specs)).Should(BeEmpty())
			})

			It("should report as having programmatic specs", func() {
				Ω(specs.HasProgrammaticFocus()).Should(BeTrue())
			})
		})

		Context("with a focus regexp", func() {
			BeforeEach(func() {
				focusString = "A"
			})

			It("should override the programmatic focus", func() {
				Ω(willRunTexts(specs)).Should(Equal([]string{"A1", "A2"}))
				Ω(skippedTexts(specs)).Should(Equal([]string{"B1", "B2"}))
				Ω(pendingTexts(specs)).Should(BeEmpty())
			})

			It("should not report as having programmatic specs", func() {
				Ω(specs.HasProgrammaticFocus()).Should(BeFalse())
			})
		})

		Context("with a focus regexp", func() {
			BeforeEach(func() {
				focusString = "B"
			})

			It("should not override any pendings", func() {
				Ω(willRunTexts(specs)).Should(Equal([]string{"B1"}))
				Ω(skippedTexts(specs)).Should(Equal([]string{"A1", "A2"}))
				Ω(pendingTexts(specs)).Should(Equal([]string{"B2"}))
			})
		})

		Context("with a description", func() {
			BeforeEach(func() {
				description = "C"
				focusString = "C"
			})

			It("should include the description in the focus determination", func() {
				Ω(willRunTexts(specs)).Should(Equal([]string{"A1", "A2", "B1"}))
				Ω(skippedTexts(specs)).Should(BeEmpty())
				Ω(pendingTexts(specs)).Should(Equal([]string{"B2"}))
			})
		})

		Context("with a description", func() {
			BeforeEach(func() {
				description = "C"
				skipString = "C"
			})

			It("should include the description in the focus determination", func() {
				Ω(willRunTexts(specs)).Should(BeEmpty())
				Ω(skippedTexts(specs)).Should(Equal([]string{"A1", "A2", "B1", "B2"}))
				Ω(pendingTexts(specs)).Should(BeEmpty())
			})
		})

		Context("with a skip regexp", func() {
			BeforeEach(func() {
				skipString = "A"
			})

			It("should override the programmatic focus", func() {
				Ω(willRunTexts(specs)).Should(Equal([]string{"B1"}))
				Ω(skippedTexts(specs)).Should(Equal([]string{"A1", "A2"}))
				Ω(pendingTexts(specs)).Should(Equal([]string{"B2"}))
			})

			It("should not report as having programmatic specs", func() {
				Ω(specs.HasProgrammaticFocus()).Should(BeFalse())
			})
		})

		Context("with both a focus and a skip regexp", func() {
			BeforeEach(func() {
				focusString = "1"
				skipString = "B"
			})

			It("should AND the two", func() {
				Ω(willRunTexts(specs)).Should(Equal([]string{"A1"}))
				Ω(skippedTexts(specs)).Should(Equal([]string{"A2", "B1", "B2"}))
				Ω(pendingTexts(specs)).Should(BeEmpty())
			})

			It("should not report as having programmatic specs", func() {
				Ω(specs.HasProgrammaticFocus()).Should(BeFalse())
			})
		})
	})

	Describe("With a focused spec within a pending context and a pending spec within a focused context", func() {
		BeforeEach(func() {
			pendingInFocused := New(
				leafnodes.NewItNode("PendingInFocused", func() {}, pendingFlag, codelocation.New(0), 0, nil, 0),
				[]*containernode.ContainerNode{
					containernode.New("", focusedFlag, codelocation.New(0)),
				}, false)

			focusedInPending := New(
				leafnodes.NewItNode("FocusedInPending", func() {}, focusedFlag, codelocation.New(0), 0, nil, 0),
				[]*containernode.ContainerNode{
					containernode.New("", pendingFlag, codelocation.New(0)),
				}, false)

			specs = NewSpecs([]*Spec{
				newSpec("A", noneFlag),
				newSpec("B", noneFlag),
				pendingInFocused,
				focusedInPending,
			})
			specs.ApplyFocus("", "", "")
		})

		It("should not have a programmatic focus and should run all tests", func() {
			Ω(willRunTexts(specs)).Should(Equal([]string{"A", "B"}))
			Ω(skippedTexts(specs)).Should(BeEmpty())
			Ω(pendingTexts(specs)).Should(ConsistOf(ContainSubstring("PendingInFocused"), ContainSubstring("FocusedInPending")))
		})
	})

	Describe("skipping measurements", func() {
		BeforeEach(func() {
			specs = NewSpecs([]*Spec{
				newSpec("A", noneFlag),
				newSpec("B", noneFlag),
				newSpec("C", pendingFlag),
				newMeasureSpec("measurementA", noneFlag),
				newMeasureSpec("measurementB", pendingFlag),
			})
		})

		It("should skip measurements", func() {
			Ω(willRunTexts(specs)).Should(Equal([]string{"A", "B", "measurementA"}))
			Ω(skippedTexts(specs)).Should(BeEmpty())
			Ω(pendingTexts(specs)).Should(Equal([]string{"C", "measurementB"}))

			specs.SkipMeasurements()

			Ω(willRunTexts(specs)).Should(Equal([]string{"A", "B"}))
			Ω(skippedTexts(specs)).Should(Equal([]string{"measurementA", "measurementB"}))
			Ω(pendingTexts(specs)).Should(Equal([]string{"C"}))
		})
	})

	Describe("when running tests in parallel", func() {
		It("should select out a subset of the tests", func() {
			specsNode1 := newSpecs("A", noneFlag, "B", noneFlag, "C", noneFlag, "D", noneFlag, "E", noneFlag)
			specsNode2 := newSpecs("A", noneFlag, "B", noneFlag, "C", noneFlag, "D", noneFlag, "E", noneFlag)
			specsNode3 := newSpecs("A", noneFlag, "B", noneFlag, "C", noneFlag, "D", noneFlag, "E", noneFlag)

			specsNode1.TrimForParallelization(3, 1)
			specsNode2.TrimForParallelization(3, 2)
			specsNode3.TrimForParallelization(3, 3)

			Ω(willRunTexts(specsNode1)).Should(Equal([]string{"A", "B"}))
			Ω(willRunTexts(specsNode2)).Should(Equal([]string{"C", "D"}))
			Ω(willRunTexts(specsNode3)).Should(Equal([]string{"E"}))

			Ω(specsNode1.Specs()).Should(HaveLen(2))
			Ω(specsNode2.Specs()).Should(HaveLen(2))
			Ω(specsNode3.Specs()).Should(HaveLen(1))

			Ω(specsNode1.NumberOfOriginalSpecs()).Should(Equal(5))
			Ω(specsNode2.NumberOfOriginalSpecs()).Should(Equal(5))
			Ω(specsNode3.NumberOfOriginalSpecs()).Should(Equal(5))
		})

		Context("when way too many nodes are used", func() {
			It("should return 0 specs", func() {
				specsNode1 := newSpecs("A", noneFlag, "B", noneFlag)
				specsNode2 := newSpecs("A", noneFlag, "B", noneFlag)
				specsNode3 := newSpecs("A", noneFlag, "B", noneFlag)

				specsNode1.TrimForParallelization(3, 1)
				specsNode2.TrimForParallelization(3, 2)
				specsNode3.TrimForParallelization(3, 3)

				Ω(willRunTexts(specsNode1)).Should(Equal([]string{"A"}))
				Ω(willRunTexts(specsNode2)).Should(Equal([]string{"B"}))
				Ω(willRunTexts(specsNode3)).Should(BeEmpty())

				Ω(specsNode1.Specs()).Should(HaveLen(1))
				Ω(specsNode2.Specs()).Should(HaveLen(1))
				Ω(specsNode3.Specs()).Should(HaveLen(0))

				Ω(specsNode1.NumberOfOriginalSpecs()).Should(Equal(2))
				Ω(specsNode2.NumberOfOriginalSpecs()).Should(Equal(2))
				Ω(specsNode3.NumberOfOriginalSpecs()).Should(Equal(2))
			})
		})
	})
})
