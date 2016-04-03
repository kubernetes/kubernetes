package reporters_test

import (
	"bytes"
	"fmt"
	. "github.com/onsi/ginkgo"
	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/internal/codelocation"
	"github.com/onsi/ginkgo/reporters"
	"github.com/onsi/ginkgo/types"
	. "github.com/onsi/gomega"
	"time"
)

var _ = Describe("TeamCity Reporter", func() {
	var (
		buffer   bytes.Buffer
		reporter Reporter
	)

	BeforeEach(func() {
		buffer.Truncate(0)
		reporter = reporters.NewTeamCityReporter(&buffer)
		reporter.SpecSuiteWillBegin(config.GinkgoConfigType{}, &types.SuiteSummary{
			SuiteDescription:           "Foo's test suite",
			NumberOfSpecsThatWillBeRun: 1,
		})
	})

	Describe("a passing test", func() {
		BeforeEach(func() {
			beforeSuite := &types.SetupSummary{
				State: types.SpecStatePassed,
			}
			reporter.BeforeSuiteDidRun(beforeSuite)

			afterSuite := &types.SetupSummary{
				State: types.SpecStatePassed,
			}
			reporter.AfterSuiteDidRun(afterSuite)

			spec := &types.SpecSummary{
				ComponentTexts: []string{"[Top Level]", "A", "B", "C"},
				State:          types.SpecStatePassed,
				RunTime:        5 * time.Second,
			}
			reporter.SpecWillRun(spec)
			reporter.SpecDidComplete(spec)

			reporter.SpecSuiteDidEnd(&types.SuiteSummary{
				NumberOfSpecsThatWillBeRun: 1,
				NumberOfFailedSpecs:        0,
				RunTime:                    10 * time.Second,
			})
		})

		It("should record the test as passing", func() {
			actual := buffer.String()
			expected :=
				"##teamcity[testSuiteStarted name='Foo|'s test suite']" +
					"##teamcity[testStarted name='A B C']" +
					"##teamcity[testFinished name='A B C' duration='5000']" +
					"##teamcity[testSuiteFinished name='Foo|'s test suite']"
			Ω(actual).Should(Equal(expected))
		})
	})

	Describe("when the BeforeSuite fails", func() {
		var beforeSuite *types.SetupSummary

		BeforeEach(func() {
			beforeSuite = &types.SetupSummary{
				State:   types.SpecStateFailed,
				RunTime: 3 * time.Second,
				Failure: types.SpecFailure{
					Message:               "failed to setup\n",
					ComponentCodeLocation: codelocation.New(0),
				},
			}
			reporter.BeforeSuiteDidRun(beforeSuite)

			reporter.SpecSuiteDidEnd(&types.SuiteSummary{
				NumberOfSpecsThatWillBeRun: 1,
				NumberOfFailedSpecs:        1,
				RunTime:                    10 * time.Second,
			})
		})

		It("should record the test as having failed", func() {
			actual := buffer.String()
			expected := fmt.Sprintf(
				"##teamcity[testSuiteStarted name='Foo|'s test suite']"+
					"##teamcity[testStarted name='BeforeSuite']"+
					"##teamcity[testFailed name='BeforeSuite' message='%s' details='failed to setup|n']"+
					"##teamcity[testFinished name='BeforeSuite' duration='3000']"+
					"##teamcity[testSuiteFinished name='Foo|'s test suite']", beforeSuite.Failure.ComponentCodeLocation.String(),
			)
			Ω(actual).Should(Equal(expected))
		})
	})

	Describe("when the AfterSuite fails", func() {
		var afterSuite *types.SetupSummary

		BeforeEach(func() {
			afterSuite = &types.SetupSummary{
				State:   types.SpecStateFailed,
				RunTime: 3 * time.Second,
				Failure: types.SpecFailure{
					Message:               "failed to setup\n",
					ComponentCodeLocation: codelocation.New(0),
				},
			}
			reporter.AfterSuiteDidRun(afterSuite)

			reporter.SpecSuiteDidEnd(&types.SuiteSummary{
				NumberOfSpecsThatWillBeRun: 1,
				NumberOfFailedSpecs:        1,
				RunTime:                    10 * time.Second,
			})
		})

		It("should record the test as having failed", func() {
			actual := buffer.String()
			expected := fmt.Sprintf(
				"##teamcity[testSuiteStarted name='Foo|'s test suite']"+
					"##teamcity[testStarted name='AfterSuite']"+
					"##teamcity[testFailed name='AfterSuite' message='%s' details='failed to setup|n']"+
					"##teamcity[testFinished name='AfterSuite' duration='3000']"+
					"##teamcity[testSuiteFinished name='Foo|'s test suite']", afterSuite.Failure.ComponentCodeLocation.String(),
			)
			Ω(actual).Should(Equal(expected))
		})
	})
	specStateCases := []struct {
		state   types.SpecState
		message string
	}{
		{types.SpecStateFailed, "Failure"},
		{types.SpecStateTimedOut, "Timeout"},
		{types.SpecStatePanicked, "Panic"},
	}

	for _, specStateCase := range specStateCases {
		specStateCase := specStateCase
		Describe("a failing test", func() {
			var spec *types.SpecSummary
			BeforeEach(func() {
				spec = &types.SpecSummary{
					ComponentTexts: []string{"[Top Level]", "A", "B", "C"},
					State:          specStateCase.state,
					RunTime:        5 * time.Second,
					Failure: types.SpecFailure{
						ComponentCodeLocation: codelocation.New(0),
						Message:               "I failed",
					},
				}
				reporter.SpecWillRun(spec)
				reporter.SpecDidComplete(spec)

				reporter.SpecSuiteDidEnd(&types.SuiteSummary{
					NumberOfSpecsThatWillBeRun: 1,
					NumberOfFailedSpecs:        1,
					RunTime:                    10 * time.Second,
				})
			})

			It("should record test as failing", func() {
				actual := buffer.String()
				expected :=
					fmt.Sprintf("##teamcity[testSuiteStarted name='Foo|'s test suite']"+
						"##teamcity[testStarted name='A B C']"+
						"##teamcity[testFailed name='A B C' message='%s' details='I failed']"+
						"##teamcity[testFinished name='A B C' duration='5000']"+
						"##teamcity[testSuiteFinished name='Foo|'s test suite']", spec.Failure.ComponentCodeLocation.String())
				Ω(actual).Should(Equal(expected))
			})
		})
	}

	for _, specStateCase := range []types.SpecState{types.SpecStatePending, types.SpecStateSkipped} {
		specStateCase := specStateCase
		Describe("a skipped test", func() {
			var spec *types.SpecSummary
			BeforeEach(func() {
				spec = &types.SpecSummary{
					ComponentTexts: []string{"[Top Level]", "A", "B", "C"},
					State:          specStateCase,
					RunTime:        5 * time.Second,
				}
				reporter.SpecWillRun(spec)
				reporter.SpecDidComplete(spec)

				reporter.SpecSuiteDidEnd(&types.SuiteSummary{
					NumberOfSpecsThatWillBeRun: 1,
					NumberOfFailedSpecs:        0,
					RunTime:                    10 * time.Second,
				})
			})

			It("should record test as ignored", func() {
				actual := buffer.String()
				expected :=
					"##teamcity[testSuiteStarted name='Foo|'s test suite']" +
						"##teamcity[testStarted name='A B C']" +
						"##teamcity[testIgnored name='A B C']" +
						"##teamcity[testFinished name='A B C' duration='5000']" +
						"##teamcity[testSuiteFinished name='Foo|'s test suite']"
				Ω(actual).Should(Equal(expected))
			})
		})
	}
})
