/*
Package gomega/gmeasure provides support for benchmarking and measuring code.  It is intended as a more robust replacement for Ginkgo V1's Measure nodes.

gmeasure is organized around the metaphor of an Experiment that can record multiple Measurements.  A Measurement is a named collection of data points and gmeasure supports
measuring Values (of type float64) and Durations (of type time.Duration).

Experiments allows the user to record Measurements directly by passing in Values (i.e. float64) or Durations (i.e. time.Duration)
or to measure measurements by passing in functions to measure.  When measuring functions Experiments take care of timing the duration of functions (for Duration measurements)
and/or recording returned values (for Value measurements).  Experiments also support sampling functions - when told to sample Experiments will run functions repeatedly
and measure and record results.  The sampling behavior is configured by passing in a SamplingConfig that can control the maximum number of samples, the maximum duration for sampling (or both)
and the number of concurrent samples to take.

Measurements can be decorated with additional information.  This is supported by passing in special typed decorators when recording measurements.  These include:

- Units("any string") - to attach units to a Value Measurement (Duration Measurements always have units of "duration")
- Style("any Ginkgo color style string") - to attach styling to a Measurement.  This styling is used when rendering console information about the measurement in reports.  Color style strings are documented at TODO.
- Precision(integer or time.Duration) - to attach precision to a Measurement.  This controls how many decimal places to show for Value Measurements and how to round Duration Measurements when rendering them to screen.

In addition, individual data points in a Measurement can be annotated with an Annotation("any string").  The annotation is associated with the individual data point and is intended to convey additional context about the data point.

Once measurements are complete, an Experiment can generate a comprehensive report by calling its String() or ColorableString() method.

Users can also access and analyze the resulting Measurements directly.  Use Experiment.Get(NAME) to fetch the Measurement named NAME.  This returned struct will have fields containing
all the data points and annotations recorded by the experiment.  You can subsequently fetch the Measurement.Stats() to get a Stats struct that contains basic statistical information about the
Measurement (min, max, median, mean, standard deviation).  You can order these Stats objects using RankStats() to identify best/worst performers across multiple experiments or measurements.

gmeasure also supports caching Experiments via an ExperimentCache.  The cache supports storing and retrieving experiments by name and version.  This allows you to rerun code without
repeating expensive experiments that may not have changed (which can be controlled by the cache version number).  It also enables you to compare new experiment runs with older runs to detect
variations in performance/behavior.

When used with Ginkgo, you can emit experiment reports and encode them in test reports easily using Ginkgo V2's support for Report Entries.
Simply pass your experiment to AddReportEntry to get a report every time the tests run.  You can also use AddReportEntry with Measurements to emit all the captured data
and Rankings to emit measurement summaries in rank order.

Finally, Experiments provide an additional mechanism to measure durations called a Stopwatch.  The Stopwatch makes it easy to pepper code with statements that measure elapsed time across
different sections of code and can be useful when debugging or evaluating bottlenecks in a given codepath.
*/
package gmeasure

import (
	"fmt"
	"math"
	"reflect"
	"sync"
	"time"

	"github.com/onsi/gomega/gmeasure/table"
)

/*
SamplingConfig configures the Sample family of experiment methods.
These methods invoke passed-in functions repeatedly to sample and record a given measurement.
SamplingConfig is used to control the maximum number of samples or time spent sampling (or both).  When both are specified sampling ends as soon as one of the conditions is met.
SamplingConfig can also ensure a minimum interval between samples and can enable concurrent sampling.
*/
type SamplingConfig struct {
	// N - the maximum number of samples to record
	N int
	// Duration - the maximum amount of time to spend recording samples
	Duration time.Duration
	// MinSamplingInterval - the minimum time that must elapse between samplings.  It is an error to specify both MinSamplingInterval and NumParallel.
	MinSamplingInterval time.Duration
	// NumParallel - the number of parallel workers to spin up to record samples.  It is an error to specify both MinSamplingInterval and NumParallel.
	NumParallel int
}

// The Units decorator allows you to specify units (an arbitrary string) when recording values.  It is ignored when recording durations.
//
//	e := gmeasure.NewExperiment("My Experiment")
//	e.RecordValue("length", 3.141, gmeasure.Units("inches"))
//
// Units are only set the first time a value of a given name is recorded.  In the example above any subsequent calls to e.RecordValue("length", X) will maintain the "inches" units even if a new set of Units("UNIT") are passed in later.
type Units string

// The Annotation decorator allows you to attach an annotation to a given recorded data-point:
//
// For example:
//
//	e := gmeasure.NewExperiment("My Experiment")
//	e.RecordValue("length", 3.141, gmeasure.Annotation("bob"))
//	e.RecordValue("length", 2.71, gmeasure.Annotation("jane"))
//
// ...will result in a Measurement named "length" that records two values )[3.141, 2.71]) annotation with (["bob", "jane"])
type Annotation string

// The Style decorator allows you to associate a style with a measurement.  This is used to generate colorful console reports using Ginkgo V2's
// console formatter.  Styles are strings in curly brackets that correspond to a color or style.
//
// For example:
//
//	e := gmeasure.NewExperiment("My Experiment")
//	e.RecordValue("length", 3.141, gmeasure.Style("{{blue}}{{bold}}"))
//	e.RecordValue("length", 2.71)
//	e.RecordDuration("cooking time", 3 * time.Second, gmeasure.Style("{{red}}{{underline}}"))
//	e.RecordDuration("cooking time", 2 * time.Second)
//
// will emit a report with blue bold entries for the length measurement and red underlined entries for the cooking time measurement.
//
// Units are only set the first time a value or duration of a given name is recorded.  In the example above any subsequent calls to e.RecordValue("length", X) will maintain the "{{blue}}{{bold}}" style even if a new Style is passed in later.
type Style string

// The PrecisionBundle decorator controls the rounding of value and duration measurements.  See Precision().
type PrecisionBundle struct {
	Duration    time.Duration
	ValueFormat string
}

// Precision() allows you to specify the precision of a value or duration measurement - this precision is used when rendering the measurement to screen.
//
// To control the precision of Value measurements, pass Precision an integer.  This will denote the number of decimal places to render (equivalen to the format string "%.Nf")
// To control the precision of Duration measurements, pass Precision a time.Duration.  Duration measurements will be rounded oo the nearest time.Duration when rendered.
//
// For example:
//
//	e := gmeasure.NewExperiment("My Experiment")
//	e.RecordValue("length", 3.141, gmeasure.Precision(2))
//	e.RecordValue("length", 2.71)
//	e.RecordDuration("cooking time", 3214 * time.Millisecond, gmeasure.Precision(100*time.Millisecond))
//	e.RecordDuration("cooking time", 2623 * time.Millisecond)
func Precision(p any) PrecisionBundle {
	out := DefaultPrecisionBundle
	switch reflect.TypeOf(p) {
	case reflect.TypeOf(time.Duration(0)):
		out.Duration = p.(time.Duration)
	case reflect.TypeOf(int(0)):
		out.ValueFormat = fmt.Sprintf("%%.%df", p.(int))
	default:
		panic("invalid precision type, must be time.Duration or int")
	}
	return out
}

// DefaultPrecisionBundle captures the default precisions for Vale and Duration measurements.
var DefaultPrecisionBundle = PrecisionBundle{
	Duration:    100 * time.Microsecond,
	ValueFormat: "%.3f",
}

type extractedDecorations struct {
	annotation      Annotation
	units           Units
	precisionBundle PrecisionBundle
	style           Style
}

func extractDecorations(args []any) extractedDecorations {
	var out extractedDecorations
	out.precisionBundle = DefaultPrecisionBundle

	for _, arg := range args {
		switch reflect.TypeOf(arg) {
		case reflect.TypeOf(out.annotation):
			out.annotation = arg.(Annotation)
		case reflect.TypeOf(out.units):
			out.units = arg.(Units)
		case reflect.TypeOf(out.precisionBundle):
			out.precisionBundle = arg.(PrecisionBundle)
		case reflect.TypeOf(out.style):
			out.style = arg.(Style)
		default:
			panic(fmt.Sprintf("unrecognized argument %#v", arg))
		}
	}

	return out
}

/*
Experiment is gmeasure's core data type.  You use experiments to record Measurements and generate reports.
Experiments are thread-safe and all methods can be called from multiple goroutines.
*/
type Experiment struct {
	Name string

	// Measurements includes all Measurements recorded by this experiment.  You should access them by name via Get() and GetStats()
	Measurements Measurements
	lock         *sync.Mutex
}

/*
NexExperiment creates a new experiment with the passed-in name.

When using Ginkgo we recommend immediately registering the experiment as a ReportEntry:

	experiment = NewExperiment("My Experiment")
	AddReportEntry(experiment.Name, experiment)

this will ensure an experiment report is emitted as part of the test output and exported with any test reports.
*/
func NewExperiment(name string) *Experiment {
	experiment := &Experiment{
		Name: name,
		lock: &sync.Mutex{},
	}
	return experiment
}

func (e *Experiment) report(enableStyling bool) string {
	t := table.NewTable()
	t.TableStyle.EnableTextStyling = enableStyling
	t.AppendRow(table.R(
		table.C("Name"), table.C("N"), table.C("Min"), table.C("Median"), table.C("Mean"), table.C("StdDev"), table.C("Max"),
		table.Divider("="),
		"{{bold}}",
	))

	for _, measurement := range e.Measurements {
		r := table.R(measurement.Style)
		t.AppendRow(r)
		switch measurement.Type {
		case MeasurementTypeNote:
			r.AppendCell(table.C(measurement.Note))
		case MeasurementTypeValue, MeasurementTypeDuration:
			name := measurement.Name
			if measurement.Units != "" {
				name += " [" + measurement.Units + "]"
			}
			r.AppendCell(table.C(name))
			r.AppendCell(measurement.Stats().cells()...)
		}
	}

	out := e.Name + "\n"
	if enableStyling {
		out = "{{bold}}" + out + "{{/}}"
	}
	out += t.Render()
	return out
}

/*
ColorableString returns a Ginkgo formatted summary of the experiment and all its Measurements.
It is called automatically by Ginkgo's reporting infrastructure when the Experiment is registered as a ReportEntry via AddReportEntry.
*/
func (e *Experiment) ColorableString() string {
	return e.report(true)
}

/*
ColorableString returns an unformatted summary of the experiment and all its Measurements.
*/
func (e *Experiment) String() string {
	return e.report(false)
}

/*
RecordNote records a Measurement of type MeasurementTypeNote - this is simply a textual note to annotate the experiment.  It will be emitted in any experiment reports.

RecordNote supports the Style() decoration.
*/
func (e *Experiment) RecordNote(note string, args ...any) {
	decorations := extractDecorations(args)

	e.lock.Lock()
	defer e.lock.Unlock()
	e.Measurements = append(e.Measurements, Measurement{
		ExperimentName: e.Name,
		Type:           MeasurementTypeNote,
		Note:           note,
		Style:          string(decorations.style),
	})
}

/*
RecordDuration records the passed-in duration on a Duration Measurement with the passed-in name.  If the Measurement does not exist it is created.

RecordDuration supports the Style(), Precision(), and Annotation() decorations.
*/
func (e *Experiment) RecordDuration(name string, duration time.Duration, args ...any) {
	decorations := extractDecorations(args)
	e.recordDuration(name, duration, decorations)
}

/*
MeasureDuration runs the passed-in callback and times how long it takes to complete.  The resulting duration is recorded on a Duration Measurement with the passed-in name.  If the Measurement does not exist it is created.

MeasureDuration supports the Style(), Precision(), and Annotation() decorations.
*/
func (e *Experiment) MeasureDuration(name string, callback func(), args ...any) time.Duration {
	t := time.Now()
	callback()
	duration := time.Since(t)
	e.RecordDuration(name, duration, args...)
	return duration
}

/*
SampleDuration samples the passed-in callback and times how long it takes to complete each sample.
The resulting durations are recorded on a Duration Measurement with the passed-in name.  If the Measurement does not exist it is created.

The callback is given a zero-based index that increments by one between samples.  The Sampling is configured via the passed-in SamplingConfig

SampleDuration supports the Style(), Precision(), and Annotation() decorations.  When passed an Annotation() the same annotation is applied to all sample measurements.
*/
func (e *Experiment) SampleDuration(name string, callback func(idx int), samplingConfig SamplingConfig, args ...any) {
	decorations := extractDecorations(args)
	e.Sample(func(idx int) {
		t := time.Now()
		callback(idx)
		duration := time.Since(t)
		e.recordDuration(name, duration, decorations)
	}, samplingConfig)
}

/*
SampleDuration samples the passed-in callback and times how long it takes to complete each sample.
The resulting durations are recorded on a Duration Measurement with the passed-in name.  If the Measurement does not exist it is created.

The callback is given a zero-based index that increments by one between samples.  The callback must return an Annotation - this annotation is attached to the measured duration.

# The Sampling is configured via the passed-in SamplingConfig

SampleAnnotatedDuration supports the Style() and Precision() decorations.
*/
func (e *Experiment) SampleAnnotatedDuration(name string, callback func(idx int) Annotation, samplingConfig SamplingConfig, args ...any) {
	decorations := extractDecorations(args)
	e.Sample(func(idx int) {
		t := time.Now()
		decorations.annotation = callback(idx)
		duration := time.Since(t)
		e.recordDuration(name, duration, decorations)
	}, samplingConfig)
}

func (e *Experiment) recordDuration(name string, duration time.Duration, decorations extractedDecorations) {
	e.lock.Lock()
	defer e.lock.Unlock()
	idx := e.Measurements.IdxWithName(name)
	if idx == -1 {
		measurement := Measurement{
			ExperimentName:  e.Name,
			Type:            MeasurementTypeDuration,
			Name:            name,
			Units:           "duration",
			Durations:       []time.Duration{duration},
			PrecisionBundle: decorations.precisionBundle,
			Style:           string(decorations.style),
			Annotations:     []string{string(decorations.annotation)},
		}
		e.Measurements = append(e.Measurements, measurement)
	} else {
		if e.Measurements[idx].Type != MeasurementTypeDuration {
			panic(fmt.Sprintf("attempting to record duration with name '%s'.  That name is already in-use for recording values.", name))
		}
		e.Measurements[idx].Durations = append(e.Measurements[idx].Durations, duration)
		e.Measurements[idx].Annotations = append(e.Measurements[idx].Annotations, string(decorations.annotation))
	}
}

/*
NewStopwatch() returns a stopwatch configured to record duration measurements with this experiment.
*/
func (e *Experiment) NewStopwatch() *Stopwatch {
	return newStopwatch(e)
}

/*
RecordValue records the passed-in value on a Value Measurement with the passed-in name.  If the Measurement does not exist it is created.

RecordValue supports the Style(), Units(), Precision(), and Annotation() decorations.
*/
func (e *Experiment) RecordValue(name string, value float64, args ...any) {
	decorations := extractDecorations(args)
	e.recordValue(name, value, decorations)
}

/*
MeasureValue runs the passed-in callback and records the return value on a Value Measurement with the passed-in name.  If the Measurement does not exist it is created.

MeasureValue supports the Style(), Units(), Precision(), and Annotation() decorations.
*/
func (e *Experiment) MeasureValue(name string, callback func() float64, args ...any) float64 {
	value := callback()
	e.RecordValue(name, value, args...)
	return value
}

/*
SampleValue samples the passed-in callback and records the return value on a Value Measurement with the passed-in name. If the Measurement does not exist it is created.

The callback is given a zero-based index that increments by one between samples.  The callback must return a float64.  The Sampling is configured via the passed-in SamplingConfig

SampleValue supports the Style(), Units(), Precision(), and Annotation() decorations.  When passed an Annotation() the same annotation is applied to all sample measurements.
*/
func (e *Experiment) SampleValue(name string, callback func(idx int) float64, samplingConfig SamplingConfig, args ...any) {
	decorations := extractDecorations(args)
	e.Sample(func(idx int) {
		value := callback(idx)
		e.recordValue(name, value, decorations)
	}, samplingConfig)
}

/*
SampleAnnotatedValue samples the passed-in callback and records the return value on a Value Measurement with the passed-in name. If the Measurement does not exist it is created.

The callback is given a zero-based index that increments by one between samples.  The callback must return a float64 and an Annotation - the annotation is attached to the recorded value.

# The Sampling is configured via the passed-in SamplingConfig

SampleValue supports the Style(), Units(), and Precision() decorations.
*/
func (e *Experiment) SampleAnnotatedValue(name string, callback func(idx int) (float64, Annotation), samplingConfig SamplingConfig, args ...any) {
	decorations := extractDecorations(args)
	e.Sample(func(idx int) {
		var value float64
		value, decorations.annotation = callback(idx)
		e.recordValue(name, value, decorations)
	}, samplingConfig)
}

func (e *Experiment) recordValue(name string, value float64, decorations extractedDecorations) {
	e.lock.Lock()
	defer e.lock.Unlock()
	idx := e.Measurements.IdxWithName(name)
	if idx == -1 {
		measurement := Measurement{
			ExperimentName:  e.Name,
			Type:            MeasurementTypeValue,
			Name:            name,
			Style:           string(decorations.style),
			Units:           string(decorations.units),
			PrecisionBundle: decorations.precisionBundle,
			Values:          []float64{value},
			Annotations:     []string{string(decorations.annotation)},
		}
		e.Measurements = append(e.Measurements, measurement)
	} else {
		if e.Measurements[idx].Type != MeasurementTypeValue {
			panic(fmt.Sprintf("attempting to record value with name '%s'.  That name is already in-use for recording durations.", name))
		}
		e.Measurements[idx].Values = append(e.Measurements[idx].Values, value)
		e.Measurements[idx].Annotations = append(e.Measurements[idx].Annotations, string(decorations.annotation))
	}
}

/*
Sample samples the passed-in callback repeatedly.  The sampling is governed by the passed in SamplingConfig.

The SamplingConfig can limit the total number of samples and/or the total time spent sampling the callback.
The SamplingConfig can also instruct Sample to run with multiple concurrent workers.

The callback is called with a zero-based index that incerements by one between samples.
*/
func (e *Experiment) Sample(callback func(idx int), samplingConfig SamplingConfig) {
	if samplingConfig.N == 0 && samplingConfig.Duration == 0 {
		panic("you must specify at least one of SamplingConfig.N and SamplingConfig.Duration")
	}
	if samplingConfig.MinSamplingInterval > 0 && samplingConfig.NumParallel > 1 {
		panic("you cannot specify both SamplingConfig.MinSamplingInterval and SamplingConfig.NumParallel")
	}
	maxTime := time.Now().Add(100000 * time.Hour)
	if samplingConfig.Duration > 0 {
		maxTime = time.Now().Add(samplingConfig.Duration)
	}
	maxN := math.MaxInt32
	if samplingConfig.N > 0 {
		maxN = samplingConfig.N
	}
	numParallel := max(samplingConfig.NumParallel, 1)
	minSamplingInterval := samplingConfig.MinSamplingInterval

	work := make(chan int)
	var wg sync.WaitGroup
	defer func() {
		close(work)
		wg.Wait()
	}()
	if numParallel > 1 {
		wg.Add(numParallel)
		for worker := 0; worker < numParallel; worker++ {
			go func() {
				for idx := range work {
					callback(idx)
				}
				wg.Done()
			}()
		}
	}

	idx := 0
	var avgDt time.Duration
	for {
		t := time.Now()
		if numParallel > 1 {
			work <- idx
		} else {
			callback(idx)
		}
		dt := time.Since(t)
		if numParallel == 1 && dt < minSamplingInterval {
			time.Sleep(minSamplingInterval - dt)
			dt = time.Since(t)
		}
		if idx >= numParallel {
			avgDt = (avgDt*time.Duration(idx-numParallel) + dt) / time.Duration(idx-numParallel+1)
		}
		idx += 1
		if idx >= maxN {
			return
		}
		if time.Now().Add(avgDt).After(maxTime) {
			return
		}
	}
}

/*
Get returns the Measurement with the associated name.  If no Measurement is found a zero Measurement{} is returned.
*/
func (e *Experiment) Get(name string) Measurement {
	e.lock.Lock()
	defer e.lock.Unlock()
	idx := e.Measurements.IdxWithName(name)
	if idx == -1 {
		return Measurement{}
	}
	return e.Measurements[idx]
}

/*
GetStats returns the Stats for the Measurement with the associated name.  If no Measurement is found a zero Stats{} is returned.

experiment.GetStats(name) is equivalent to experiment.Get(name).Stats()
*/
func (e *Experiment) GetStats(name string) Stats {
	measurement := e.Get(name)
	e.lock.Lock()
	defer e.lock.Unlock()
	return measurement.Stats()
}
