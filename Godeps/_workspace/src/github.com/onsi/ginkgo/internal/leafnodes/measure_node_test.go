package leafnodes_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/ginkgo/internal/leafnodes"
	. "github.com/onsi/gomega"

	"github.com/onsi/ginkgo/internal/codelocation"
	Failer "github.com/onsi/ginkgo/internal/failer"
	"github.com/onsi/ginkgo/types"
	"time"
)

var _ = Describe("Measure Nodes", func() {
	It("should report the correct type, text, flag, and code location", func() {
		codeLocation := codelocation.New(0)
		measure := NewMeasureNode("my measure node", func(b Benchmarker) {}, types.FlagTypeFocused, codeLocation, 10, nil, 3)
		Ω(measure.Type()).Should(Equal(types.SpecComponentTypeMeasure))
		Ω(measure.Flag()).Should(Equal(types.FlagTypeFocused))
		Ω(measure.Text()).Should(Equal("my measure node"))
		Ω(measure.CodeLocation()).Should(Equal(codeLocation))
		Ω(measure.Samples()).Should(Equal(10))
	})

	Describe("benchmarking", func() {
		var measure *MeasureNode

		Describe("Value", func() {
			BeforeEach(func() {
				measure = NewMeasureNode("the measurement", func(b Benchmarker) {
					b.RecordValue("foo", 7, "info!")
					b.RecordValue("foo", 2)
					b.RecordValue("foo", 3)
					b.RecordValue("bar", 0.3)
					b.RecordValue("bar", 0.1)
					b.RecordValue("bar", 0.5)
					b.RecordValue("bar", 0.7)
				}, types.FlagTypeFocused, codelocation.New(0), 1, Failer.New(), 3)
				Ω(measure.Run()).Should(Equal(types.SpecStatePassed))
			})

			It("records passed in values and reports on them", func() {
				report := measure.MeasurementsReport()
				Ω(report).Should(HaveLen(2))
				Ω(report["foo"].Name).Should(Equal("foo"))
				Ω(report["foo"].Info).Should(Equal("info!"))
				Ω(report["foo"].Order).Should(Equal(0))
				Ω(report["foo"].SmallestLabel).Should(Equal("Smallest"))
				Ω(report["foo"].LargestLabel).Should(Equal(" Largest"))
				Ω(report["foo"].AverageLabel).Should(Equal(" Average"))
				Ω(report["foo"].Units).Should(Equal(""))
				Ω(report["foo"].Results).Should(Equal([]float64{7, 2, 3}))
				Ω(report["foo"].Smallest).Should(BeNumerically("==", 2))
				Ω(report["foo"].Largest).Should(BeNumerically("==", 7))
				Ω(report["foo"].Average).Should(BeNumerically("==", 4))
				Ω(report["foo"].StdDeviation).Should(BeNumerically("~", 2.16, 0.01))

				Ω(report["bar"].Name).Should(Equal("bar"))
				Ω(report["bar"].Info).Should(BeNil())
				Ω(report["bar"].SmallestLabel).Should(Equal("Smallest"))
				Ω(report["bar"].Order).Should(Equal(1))
				Ω(report["bar"].LargestLabel).Should(Equal(" Largest"))
				Ω(report["bar"].AverageLabel).Should(Equal(" Average"))
				Ω(report["bar"].Units).Should(Equal(""))
				Ω(report["bar"].Results).Should(Equal([]float64{0.3, 0.1, 0.5, 0.7}))
				Ω(report["bar"].Smallest).Should(BeNumerically("==", 0.1))
				Ω(report["bar"].Largest).Should(BeNumerically("==", 0.7))
				Ω(report["bar"].Average).Should(BeNumerically("==", 0.4))
				Ω(report["bar"].StdDeviation).Should(BeNumerically("~", 0.22, 0.01))
			})
		})

		Describe("Time", func() {
			BeforeEach(func() {
				measure = NewMeasureNode("the measurement", func(b Benchmarker) {
					b.Time("foo", func() {
						time.Sleep(100 * time.Millisecond)
					}, "info!")
					b.Time("foo", func() {
						time.Sleep(200 * time.Millisecond)
					})
					b.Time("foo", func() {
						time.Sleep(170 * time.Millisecond)
					})
				}, types.FlagTypeFocused, codelocation.New(0), 1, Failer.New(), 3)
				Ω(measure.Run()).Should(Equal(types.SpecStatePassed))
			})

			It("records passed in values and reports on them", func() {
				report := measure.MeasurementsReport()
				Ω(report).Should(HaveLen(1))
				Ω(report["foo"].Name).Should(Equal("foo"))
				Ω(report["foo"].Info).Should(Equal("info!"))
				Ω(report["foo"].SmallestLabel).Should(Equal("Fastest Time"))
				Ω(report["foo"].LargestLabel).Should(Equal("Slowest Time"))
				Ω(report["foo"].AverageLabel).Should(Equal("Average Time"))
				Ω(report["foo"].Units).Should(Equal("s"))
				Ω(report["foo"].Results).Should(HaveLen(3))
				Ω(report["foo"].Results[0]).Should(BeNumerically("~", 0.1, 0.01))
				Ω(report["foo"].Results[1]).Should(BeNumerically("~", 0.2, 0.01))
				Ω(report["foo"].Results[2]).Should(BeNumerically("~", 0.17, 0.01))
				Ω(report["foo"].Smallest).Should(BeNumerically("~", 0.1, 0.01))
				Ω(report["foo"].Largest).Should(BeNumerically("~", 0.2, 0.01))
				Ω(report["foo"].Average).Should(BeNumerically("~", 0.16, 0.01))
				Ω(report["foo"].StdDeviation).Should(BeNumerically("~", 0.04, 0.01))
			})
		})
	})
})
