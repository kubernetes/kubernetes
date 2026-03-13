package gmeasure

import (
	"fmt"
	"time"

	"github.com/onsi/gomega/gmeasure/table"
)

/*
Stat is an enum representing the statistics you can request of a Stats struct
*/
type Stat uint

const (
	StatInvalid Stat = iota
	StatMin
	StatMax
	StatMean
	StatMedian
	StatStdDev
)

var statEnumSupport = newEnumSupport(map[uint]string{uint(StatInvalid): "INVALID STAT", uint(StatMin): "Min", uint(StatMax): "Max", uint(StatMean): "Mean", uint(StatMedian): "Median", uint(StatStdDev): "StdDev"})

func (s Stat) String() string { return statEnumSupport.String(uint(s)) }
func (s *Stat) UnmarshalJSON(b []byte) error {
	out, err := statEnumSupport.UnmarshJSON(b)
	*s = Stat(out)
	return err
}
func (s Stat) MarshalJSON() ([]byte, error) { return statEnumSupport.MarshJSON(uint(s)) }

type StatsType uint

const (
	StatsTypeInvalid StatsType = iota
	StatsTypeValue
	StatsTypeDuration
)

var statsTypeEnumSupport = newEnumSupport(map[uint]string{uint(StatsTypeInvalid): "INVALID STATS TYPE", uint(StatsTypeValue): "StatsTypeValue", uint(StatsTypeDuration): "StatsTypeDuration"})

func (s StatsType) String() string { return statsTypeEnumSupport.String(uint(s)) }
func (s *StatsType) UnmarshalJSON(b []byte) error {
	out, err := statsTypeEnumSupport.UnmarshJSON(b)
	*s = StatsType(out)
	return err
}
func (s StatsType) MarshalJSON() ([]byte, error) { return statsTypeEnumSupport.MarshJSON(uint(s)) }

/*
Stats records the key statistics for a given measurement.  You generally don't make Stats directly - but you can fetch them from Experiments using GetStats() and from Measurements using Stats().

When using Ginkgo, you can register Measurements as Report Entries via AddReportEntry.  This will emit all the captured data points when Ginkgo generates the report.
*/
type Stats struct {
	// Type is the StatType - one of StatTypeDuration or StatTypeValue
	Type StatsType

	// ExperimentName is the name of the Experiment that recorded the Measurement from which this Stat is derived
	ExperimentName string

	// MeasurementName is the name of the Measurement from which this Stat is derived
	MeasurementName string

	// Units captures the Units of the Measurement from which this Stat is derived
	Units string

	// Style captures the Style of the Measurement from which this Stat is derived
	Style string

	// PrecisionBundle captures the precision to use when rendering data for this Measurement.
	// If Type is StatTypeDuration then PrecisionBundle.Duration is used to round any durations before presentation.
	// If Type is StatTypeValue then PrecisionBundle.ValueFormat is used to format any values before presentation
	PrecisionBundle PrecisionBundle

	// N represents the total number of data points in the Measurement from which this Stat is derived
	N int

	// If Type is StatTypeValue, ValueBundle will be populated with float64s representing this Stat's statistics
	ValueBundle map[Stat]float64

	// If Type is StatTypeDuration, DurationBundle will be populated with float64s representing this Stat's statistics
	DurationBundle map[Stat]time.Duration

	// AnnotationBundle is populated with Annotations corresponding to the data points that can be associated with a Stat.
	// For example AnnotationBundle[StatMin] will return the Annotation for the data point that has the minimum value/duration.
	AnnotationBundle map[Stat]string
}

// String returns a minimal summary of the stats of the form "MIN < [MEDIAN] | <MEAN> ±STDDEV < MAX"
func (s Stats) String() string {
	return fmt.Sprintf("%s < [%s] | <%s> ±%s < %s", s.StringFor(StatMin), s.StringFor(StatMedian), s.StringFor(StatMean), s.StringFor(StatStdDev), s.StringFor(StatMax))
}

// ValueFor returns the float64 value for a particular Stat.  You should only use this if the Stats has Type StatsTypeValue
// For example:
//
//	median := experiment.GetStats("length").ValueFor(gmeasure.StatMedian)
//
// will return the median data point for the "length" Measurement.
func (s Stats) ValueFor(stat Stat) float64 {
	return s.ValueBundle[stat]
}

// DurationFor returns the time.Duration for a particular Stat.  You should only use this if the Stats has Type StatsTypeDuration
// For example:
//
//	mean := experiment.GetStats("runtime").ValueFor(gmeasure.StatMean)
//
// will return the mean duration for the "runtime" Measurement.
func (s Stats) DurationFor(stat Stat) time.Duration {
	return s.DurationBundle[stat]
}

// FloatFor returns a float64 representation of the passed-in Stat.
// When Type is StatsTypeValue this is equivalent to s.ValueFor(stat).
// When Type is StatsTypeDuration this is equivalent to float64(s.DurationFor(stat))
func (s Stats) FloatFor(stat Stat) float64 {
	switch s.Type {
	case StatsTypeValue:
		return s.ValueFor(stat)
	case StatsTypeDuration:
		return float64(s.DurationFor(stat))
	}
	return 0
}

// StringFor returns a formatted string representation of the passed-in Stat.
// The formatting honors the precision directives provided in stats.PrecisionBundle
func (s Stats) StringFor(stat Stat) string {
	switch s.Type {
	case StatsTypeValue:
		return fmt.Sprintf(s.PrecisionBundle.ValueFormat, s.ValueFor(stat))
	case StatsTypeDuration:
		return s.DurationFor(stat).Round(s.PrecisionBundle.Duration).String()
	}
	return ""
}

func (s Stats) cells() []table.Cell {
	out := []table.Cell{}
	out = append(out, table.C(fmt.Sprintf("%d", s.N)))
	for _, stat := range []Stat{StatMin, StatMedian, StatMean, StatStdDev, StatMax} {
		content := s.StringFor(stat)
		if s.AnnotationBundle[stat] != "" {
			content += "\n" + s.AnnotationBundle[stat]
		}
		out = append(out, table.C(content))
	}
	return out
}
