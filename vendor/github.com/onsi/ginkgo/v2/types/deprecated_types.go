package types

import (
	"strconv"
	"time"
)

/*
	A set of deprecations to make the transition from v1 to v2 easier for users who have written custom reporters.
*/

type SuiteSummary = DeprecatedSuiteSummary
type SetupSummary = DeprecatedSetupSummary
type SpecSummary = DeprecatedSpecSummary
type SpecMeasurement = DeprecatedSpecMeasurement
type SpecComponentType = NodeType
type SpecFailure = DeprecatedSpecFailure

var (
	SpecComponentTypeInvalid                 = NodeTypeInvalid
	SpecComponentTypeContainer               = NodeTypeContainer
	SpecComponentTypeIt                      = NodeTypeIt
	SpecComponentTypeBeforeEach              = NodeTypeBeforeEach
	SpecComponentTypeJustBeforeEach          = NodeTypeJustBeforeEach
	SpecComponentTypeAfterEach               = NodeTypeAfterEach
	SpecComponentTypeJustAfterEach           = NodeTypeJustAfterEach
	SpecComponentTypeBeforeSuite             = NodeTypeBeforeSuite
	SpecComponentTypeSynchronizedBeforeSuite = NodeTypeSynchronizedBeforeSuite
	SpecComponentTypeAfterSuite              = NodeTypeAfterSuite
	SpecComponentTypeSynchronizedAfterSuite  = NodeTypeSynchronizedAfterSuite
)

type DeprecatedSuiteSummary struct {
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
	NumberOfFlakedSpecs                int
	RunTime                            time.Duration
}

type DeprecatedSetupSummary struct {
	ComponentType SpecComponentType
	CodeLocation  CodeLocation

	State   SpecState
	RunTime time.Duration
	Failure SpecFailure

	CapturedOutput string
	SuiteID        string
}

type DeprecatedSpecSummary struct {
	ComponentTexts         []string
	ComponentCodeLocations []CodeLocation

	State           SpecState
	RunTime         time.Duration
	Failure         SpecFailure
	IsMeasurement   bool
	NumberOfSamples int
	Measurements    map[string]*DeprecatedSpecMeasurement

	CapturedOutput string
	SuiteID        string
}

func (s DeprecatedSpecSummary) HasFailureState() bool {
	return s.State.Is(SpecStateFailureStates)
}

func (s DeprecatedSpecSummary) TimedOut() bool {
	return false
}

func (s DeprecatedSpecSummary) Panicked() bool {
	return s.State == SpecStatePanicked
}

func (s DeprecatedSpecSummary) Failed() bool {
	return s.State == SpecStateFailed
}

func (s DeprecatedSpecSummary) Passed() bool {
	return s.State == SpecStatePassed
}

func (s DeprecatedSpecSummary) Skipped() bool {
	return s.State == SpecStateSkipped
}

func (s DeprecatedSpecSummary) Pending() bool {
	return s.State == SpecStatePending
}

type DeprecatedSpecFailure struct {
	Message        string
	Location       CodeLocation
	ForwardedPanic string

	ComponentIndex        int
	ComponentType         SpecComponentType
	ComponentCodeLocation CodeLocation
}

type DeprecatedSpecMeasurement struct {
	Name  string
	Info  any
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

func (s DeprecatedSpecMeasurement) PrecisionFmt() string {
	if s.Precision == 0 {
		return "%f"
	}

	str := strconv.Itoa(s.Precision)

	return "%." + str + "f"
}
