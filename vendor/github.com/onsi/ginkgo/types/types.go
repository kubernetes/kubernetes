package types

import (
	"strconv"
	"time"
)

const GINKGO_FOCUS_EXIT_CODE = 197

/*
SuiteSummary represents the a summary of the test suite and is passed to both
Reporter.SpecSuiteWillBegin
Reporter.SpecSuiteDidEnd

this is unfortunate as these two methods should receive different objects.  When running in parallel
each node does not deterministically know how many specs it will end up running.

Unfortunately making such a change would break backward compatibility.

Until Ginkgo 2.0 comes out we will continue to reuse this struct but populate unkown fields
with -1.
*/
type SuiteSummary struct {
	SuiteDescription string
	SuiteSucceeded   bool
	SuiteID          string

	NumberOfSpecsBeforeParallelization int
	NumberOfTotalSpecs                 int
	NumberOfSpecsThatWillBeRun         int
	NumberOfPendingSpecs               int
	NumberOfSkippedSpecs               int
	NumberOfPassedSpecs                int
	NumberOfFailedSpecs                int
	// Flaked specs are those that failed initially, but then passed on a
	// subsequent try.
	NumberOfFlakedSpecs int
	RunTime             time.Duration
}

type SpecSummary struct {
	ComponentTexts         []string
	ComponentCodeLocations []CodeLocation

	State           SpecState
	RunTime         time.Duration
	Failure         SpecFailure
	IsMeasurement   bool
	NumberOfSamples int
	Measurements    map[string]*SpecMeasurement

	CapturedOutput string
	SuiteID        string
}

func (s SpecSummary) HasFailureState() bool {
	return s.State.IsFailure()
}

func (s SpecSummary) TimedOut() bool {
	return s.State == SpecStateTimedOut
}

func (s SpecSummary) Panicked() bool {
	return s.State == SpecStatePanicked
}

func (s SpecSummary) Failed() bool {
	return s.State == SpecStateFailed
}

func (s SpecSummary) Passed() bool {
	return s.State == SpecStatePassed
}

func (s SpecSummary) Skipped() bool {
	return s.State == SpecStateSkipped
}

func (s SpecSummary) Pending() bool {
	return s.State == SpecStatePending
}

type SetupSummary struct {
	ComponentType SpecComponentType
	CodeLocation  CodeLocation

	State   SpecState
	RunTime time.Duration
	Failure SpecFailure

	CapturedOutput string
	SuiteID        string
}

type SpecFailure struct {
	Message        string
	Location       CodeLocation
	ForwardedPanic string

	ComponentIndex        int
	ComponentType         SpecComponentType
	ComponentCodeLocation CodeLocation
}

type SpecMeasurement struct {
	Name  string
	Info  interface{}
	Order int

	Results []float64

	Smallest     float64
	Largest      float64
	Average      float64
	StdDeviation float64

	SmallestLabel string
	LargestLabel  string
	AverageLabel  string
	Units         string
	Precision     int
}

func (s SpecMeasurement) PrecisionFmt() string {
	if s.Precision == 0 {
		return "%f"
	}

	str := strconv.Itoa(s.Precision)

	return "%." + str + "f"
}

type SpecState uint

const (
	SpecStateInvalid SpecState = iota

	SpecStatePending
	SpecStateSkipped
	SpecStatePassed
	SpecStateFailed
	SpecStatePanicked
	SpecStateTimedOut
)

func (state SpecState) IsFailure() bool {
	return state == SpecStateTimedOut || state == SpecStatePanicked || state == SpecStateFailed
}

type SpecComponentType uint

const (
	SpecComponentTypeInvalid SpecComponentType = iota

	SpecComponentTypeContainer
	SpecComponentTypeBeforeSuite
	SpecComponentTypeAfterSuite
	SpecComponentTypeBeforeEach
	SpecComponentTypeJustBeforeEach
	SpecComponentTypeAfterEach
	SpecComponentTypeIt
	SpecComponentTypeMeasure
)

type FlagType uint

const (
	FlagTypeNone FlagType = iota
	FlagTypeFocused
	FlagTypePending
)
