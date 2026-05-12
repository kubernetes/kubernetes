package gmeasure

import (
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/onsi/gomega/gmeasure/table"
)

type MeasurementType uint

const (
	MeasurementTypeInvalid MeasurementType = iota
	MeasurementTypeNote
	MeasurementTypeDuration
	MeasurementTypeValue
)

var letEnumSupport = newEnumSupport(map[uint]string{uint(MeasurementTypeInvalid): "INVALID LOG ENTRY TYPE", uint(MeasurementTypeNote): "Note", uint(MeasurementTypeDuration): "Duration", uint(MeasurementTypeValue): "Value"})

func (s MeasurementType) String() string { return letEnumSupport.String(uint(s)) }
func (s *MeasurementType) UnmarshalJSON(b []byte) error {
	out, err := letEnumSupport.UnmarshJSON(b)
	*s = MeasurementType(out)
	return err
}
func (s MeasurementType) MarshalJSON() ([]byte, error) { return letEnumSupport.MarshJSON(uint(s)) }

/*
Measurement records all captured data for a given measurement.  You generally don't make Measurements directly - but you can fetch them from Experiments using Get().

When using Ginkgo, you can register Measurements as Report Entries via AddReportEntry.  This will emit all the captured data points when Ginkgo generates the report.
*/
type Measurement struct {
	// Type is the MeasurementType - one of MeasurementTypeNote, MeasurementTypeDuration, or MeasurementTypeValue
	Type MeasurementType

	// ExperimentName is the name of the experiment that this Measurement is associated with
	ExperimentName string

	// If Type is MeasurementTypeNote, Note is populated with the note text.
	Note string

	// If Type is MeasurementTypeDuration or MeasurementTypeValue, Name is the name of the recorded measurement
	Name string

	// Style captures the styling information (if any) for this Measurement
	Style string

	// Units capture the units (if any) for this Measurement.  Units is set to "duration" if the Type is MeasurementTypeDuration
	Units string

	// PrecisionBundle captures the precision to use when rendering data for this Measurement.
	// If Type is MeasurementTypeDuration then PrecisionBundle.Duration is used to round any durations before presentation.
	// If Type is MeasurementTypeValue then PrecisionBundle.ValueFormat is used to format any values before presentation
	PrecisionBundle PrecisionBundle

	// If Type is MeasurementTypeDuration, Durations will contain all durations recorded for this measurement
	Durations []time.Duration

	// If Type is MeasurementTypeValue, Values will contain all float64s recorded for this measurement
	Values []float64

	// If Type is MeasurementTypeDuration or MeasurementTypeValue then Annotations will include string annotations for all recorded Durations or Values.
	// If the user does not pass-in an Annotation() decoration for a particular value or duration, the corresponding entry in the Annotations slice will be the empty string ""
	Annotations []string
}

type Measurements []Measurement

func (m Measurements) IdxWithName(name string) int {
	for idx, measurement := range m {
		if measurement.Name == name {
			return idx
		}
	}

	return -1
}

func (m Measurement) report(enableStyling bool) string {
	out := ""
	style := m.Style
	if !enableStyling {
		style = ""
	}
	switch m.Type {
	case MeasurementTypeNote:
		out += fmt.Sprintf("%s - Note\n%s\n", m.ExperimentName, m.Note)
		if style != "" {
			out = style + out + "{{/}}"
		}
		return out
	case MeasurementTypeValue, MeasurementTypeDuration:
		out += fmt.Sprintf("%s - %s", m.ExperimentName, m.Name)
		if m.Units != "" {
			out += " [" + m.Units + "]"
		}
		if style != "" {
			out = style + out + "{{/}}"
		}
		out += "\n"
		out += m.Stats().String() + "\n"
	}
	t := table.NewTable()
	t.TableStyle.EnableTextStyling = enableStyling
	switch m.Type {
	case MeasurementTypeValue:
		t.AppendRow(table.R(table.C("Value", table.AlignTypeCenter), table.C("Annotation", table.AlignTypeCenter), table.Divider("="), style))
		for idx := range m.Values {
			t.AppendRow(table.R(
				table.C(fmt.Sprintf(m.PrecisionBundle.ValueFormat, m.Values[idx]), table.AlignTypeRight),
				table.C(m.Annotations[idx], "{{gray}}", table.AlignTypeLeft),
			))
		}
	case MeasurementTypeDuration:
		t.AppendRow(table.R(table.C("Duration", table.AlignTypeCenter), table.C("Annotation", table.AlignTypeCenter), table.Divider("="), style))
		for idx := range m.Durations {
			t.AppendRow(table.R(
				table.C(m.Durations[idx].Round(m.PrecisionBundle.Duration).String(), style, table.AlignTypeRight),
				table.C(m.Annotations[idx], "{{gray}}", table.AlignTypeLeft),
			))
		}
	}
	out += t.Render()
	return out
}

/*
ColorableString generates a styled report that includes all the data points for this Measurement.
It is called automatically by Ginkgo's reporting infrastructure when the Measurement is registered as a ReportEntry via AddReportEntry.
*/
func (m Measurement) ColorableString() string {
	return m.report(true)
}

/*
String generates an unstyled report that includes all the data points for this Measurement.
*/
func (m Measurement) String() string {
	return m.report(false)
}

/*
Stats returns a Stats struct summarizing the statistic of this measurement
*/
func (m Measurement) Stats() Stats {
	if m.Type == MeasurementTypeInvalid || m.Type == MeasurementTypeNote {
		return Stats{}
	}

	out := Stats{
		ExperimentName:  m.ExperimentName,
		MeasurementName: m.Name,
		Style:           m.Style,
		Units:           m.Units,
		PrecisionBundle: m.PrecisionBundle,
	}

	switch m.Type {
	case MeasurementTypeValue:
		out.Type = StatsTypeValue
		out.N = len(m.Values)
		if out.N == 0 {
			return out
		}
		indices, sum := make([]int, len(m.Values)), 0.0
		for idx, v := range m.Values {
			indices[idx] = idx
			sum += v
		}
		sort.Slice(indices, func(i, j int) bool {
			return m.Values[indices[i]] < m.Values[indices[j]]
		})
		out.ValueBundle = map[Stat]float64{
			StatMin:    m.Values[indices[0]],
			StatMax:    m.Values[indices[out.N-1]],
			StatMean:   sum / float64(out.N),
			StatStdDev: 0.0,
		}
		out.AnnotationBundle = map[Stat]string{
			StatMin: m.Annotations[indices[0]],
			StatMax: m.Annotations[indices[out.N-1]],
		}

		if out.N%2 == 0 {
			out.ValueBundle[StatMedian] = (m.Values[indices[out.N/2]] + m.Values[indices[out.N/2-1]]) / 2.0
		} else {
			out.ValueBundle[StatMedian] = m.Values[indices[(out.N-1)/2]]
		}

		for _, v := range m.Values {
			out.ValueBundle[StatStdDev] += (v - out.ValueBundle[StatMean]) * (v - out.ValueBundle[StatMean])
		}
		out.ValueBundle[StatStdDev] = math.Sqrt(out.ValueBundle[StatStdDev] / float64(out.N))
	case MeasurementTypeDuration:
		out.Type = StatsTypeDuration
		out.N = len(m.Durations)
		if out.N == 0 {
			return out
		}
		indices, sum := make([]int, len(m.Durations)), time.Duration(0)
		for idx, v := range m.Durations {
			indices[idx] = idx
			sum += v
		}
		sort.Slice(indices, func(i, j int) bool {
			return m.Durations[indices[i]] < m.Durations[indices[j]]
		})
		out.DurationBundle = map[Stat]time.Duration{
			StatMin:  m.Durations[indices[0]],
			StatMax:  m.Durations[indices[out.N-1]],
			StatMean: sum / time.Duration(out.N),
		}
		out.AnnotationBundle = map[Stat]string{
			StatMin: m.Annotations[indices[0]],
			StatMax: m.Annotations[indices[out.N-1]],
		}

		if out.N%2 == 0 {
			out.DurationBundle[StatMedian] = (m.Durations[indices[out.N/2]] + m.Durations[indices[out.N/2-1]]) / 2
		} else {
			out.DurationBundle[StatMedian] = m.Durations[indices[(out.N-1)/2]]
		}
		stdDev := 0.0
		for _, v := range m.Durations {
			stdDev += float64(v-out.DurationBundle[StatMean]) * float64(v-out.DurationBundle[StatMean])
		}
		out.DurationBundle[StatStdDev] = time.Duration(math.Sqrt(stdDev / float64(out.N)))
	}

	return out
}
