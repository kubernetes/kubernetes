package ginkgo

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/ginkgo/v2/types"
	"github.com/onsi/gomega"

	"github.com/openshift-eng/openshift-tests-extension/pkg/util/sets"

	ext "github.com/openshift-eng/openshift-tests-extension/pkg/extension/extensiontests"
)

func configureGinkgo() (*types.SuiteConfig, *types.ReporterConfig, error) {
	if !ginkgo.GetSuite().InPhaseBuildTree() {
		if err := ginkgo.GetSuite().BuildTree(); err != nil {
			return nil, nil, fmt.Errorf("couldn't build ginkgo tree: %w", err)
		}
	}

	// Ginkgo initialization
	ginkgo.GetSuite().ClearBeforeAndAfterSuiteNodes()
	suiteConfig, reporterConfig := ginkgo.GinkgoConfiguration()
	suiteConfig.RandomizeAllSpecs = true
	suiteConfig.Timeout = 24 * time.Hour
	reporterConfig.NoColor = true
	reporterConfig.Verbose = true
	ginkgo.SetReporterConfig(reporterConfig)

	// Write output to Stderr
	ginkgo.GinkgoWriter = ginkgo.NewWriter(os.Stderr)
	ginkgo.GinkgoLogr = GinkgoLogrFunc(ginkgo.GinkgoWriter)

	gomega.RegisterFailHandler(ginkgo.Fail)

	return &suiteConfig, &reporterConfig, nil
}

// BuildExtensionTestSpecsFromOpenShiftGinkgoSuite generates OTE specs for Gingko tests. While OTE isn't limited to
// calling ginkgo tests, anything that implements the ExtensionTestSpec interface can be used, it's the most common
// course of action.  The typical use case is to omit selectFns, but if provided, these will filter the returned list
// of specs, applied in the order provided.
func BuildExtensionTestSpecsFromOpenShiftGinkgoSuite(selectFns ...ext.SelectFunction) (ext.ExtensionTestSpecs, error) {
	var specs ext.ExtensionTestSpecs
	var enforceSerialExecutionForGinkgo sync.Mutex // in-process parallelization for ginkgo is impossible so far

	if _, _, err := configureGinkgo(); err != nil {
		return nil, err
	}

	cwd, err := os.Getwd()
	if err != nil {
		return nil, fmt.Errorf("couldn't get current working directory: %w", err)
	}

	ginkgo.GetSuite().WalkTests(func(name string, spec types.TestSpec) {
		var codeLocations []string
		for _, cl := range spec.CodeLocations() {
			codeLocations = append(codeLocations, cl.String())
		}

		testCase := &ext.ExtensionTestSpec{
			Name:          spec.Text(),
			Labels:        sets.New[string](spec.Labels()...),
			CodeLocations: codeLocations,
			Lifecycle:     GetLifecycle(spec.Labels()),
			Run: func(ctx context.Context) *ext.ExtensionTestResult {
				enforceSerialExecutionForGinkgo.Lock()
				defer enforceSerialExecutionForGinkgo.Unlock()

				suiteConfig, reporterConfig, _ := configureGinkgo()

				result := &ext.ExtensionTestResult{
					Name: spec.Text(),
				}

				ginkgo.GetSuite().RunSpec(spec, ginkgo.Labels{}, "", cwd, ginkgo.GetFailer(), ginkgo.GetWriter(), *suiteConfig,
					*reporterConfig)
				summary := findSpecReport(ginkgo.GetSuite().GetReport().SpecReports)

				result.Output = summary.CapturedGinkgoWriterOutput
				result.Error = summary.CapturedStdOutErr

				switch summary.State {
				case types.SpecStatePassed:
					result.Result = ext.ResultPassed
				case types.SpecStateSkipped, types.SpecStatePending:
					result.Result = ext.ResultSkipped
					if len(summary.Failure.Message) > 0 {
						result.Output = fmt.Sprintf(
							"%s\n skip [%s:%d]: %s\n",
							result.Output,
							lastFilenameSegment(summary.Failure.Location.FileName),
							summary.Failure.Location.LineNumber,
							summary.Failure.Message,
						)
					} else if len(summary.Failure.ForwardedPanic) > 0 {
						result.Output = fmt.Sprintf(
							"%s\n skip [%s:%d]: %s\n",
							result.Output,
							lastFilenameSegment(summary.Failure.Location.FileName),
							summary.Failure.Location.LineNumber,
							summary.Failure.ForwardedPanic,
						)
					}
				case types.SpecStateFailed, types.SpecStatePanicked, types.SpecStateInterrupted, types.SpecStateAborted:
					result.Result = ext.ResultFailed
					var errors []string
					if len(summary.Failure.ForwardedPanic) > 0 {
						if len(summary.Failure.Location.FullStackTrace) > 0 {
							errors = append(errors, fmt.Sprintf("\n%s\n", summary.Failure.Location.FullStackTrace))
						}
						errors = append(errors, fmt.Sprintf("fail [%s:%d]: Test Panicked: %s", lastFilenameSegment(summary.Failure.Location.FileName), summary.Failure.Location.LineNumber, summary.Failure.ForwardedPanic))
					}
					errors = append(errors, fmt.Sprintf("fail [%s:%d]: %s", lastFilenameSegment(summary.Failure.Location.FileName), summary.Failure.Location.LineNumber, summary.Failure.Message))
					result.Error = strings.Join(errors, "\n")
				case types.SpecStateTimedout:
					result.Result = ext.ResultFailed
					var errors []string
					for _, additionalFailure := range summary.AdditionalFailures {
						collectAdditionalFailures(&errors, "  ", additionalFailure.Failure)
					}
					if summary.Failure.AdditionalFailure != nil {
						collectAdditionalFailures(&errors, "  ", summary.Failure.AdditionalFailure.Failure)
					}
					errors = append(errors, fmt.Sprintf("fail [%s:%d]: %s", lastFilenameSegment(summary.Failure.Location.FileName), summary.Failure.Location.LineNumber, summary.Failure.Message))
					result.Error = strings.Join(errors, "\n")
				case types.SpecStateInvalid:
					result.Result = ext.ResultFailed
					result.Error = fmt.Sprintf("test produced no spec report; this is a bug in the test framework: %#v", summary)
				default:
					result.Result = ext.ResultFailed
					result.Error = fmt.Sprintf("test produced unknown outcome: %#v", summary)
				}

				return result
			},
		}
		testCase.RunParallel = func(ctx context.Context) *ext.ExtensionTestResult {
			timeout := 90 * time.Minute
			if testCase.Timeout > 0 {
				timeout = testCase.Timeout
			}
			return SpawnProcessToRunTest(ctx, name, timeout)
		}
		specs = append(specs, testCase)
	})

	// Default select function is to exclude vendored specs.  When relying on Kubernetes test framework for its helpers,
	// it also unfortunately ends up importing *all* Gingko specs.  This is unsafe: it would potentially override the
	// kube specs already present in origin.  The best course of action is enforce this behavior on everyone.  If for
	// some reason, you must include vendored specs, you can opt-in directly by supplying your own SelectFunctions or using
	// AllTestsIncludedVendored().
	if len(selectFns) == 0 {
		selectFns = []ext.SelectFunction{ext.ModuleTestsOnly()}
	}

	for _, selectFn := range selectFns {
		specs = specs.Select(selectFn)
	}

	return specs, nil
}

func Informing() ginkgo.Labels {
	return ginkgo.Label(fmt.Sprintf("Lifecycle:%s", ext.LifecycleInforming))
}

func Slow() ginkgo.Labels {
	return ginkgo.Label("SLOW")
}

func Blocking() ginkgo.Labels {
	return ginkgo.Label(fmt.Sprintf("Lifecycle:%s", ext.LifecycleBlocking))
}

func GetLifecycle(labels ginkgo.Labels) ext.Lifecycle {
	for _, label := range labels {
		res := strings.Split(label, ":")
		if len(res) != 2 || !strings.EqualFold(res[0], "lifecycle") {
			continue
		}
		return MustLifecycle(res[1]) // this panics if unsupported lifecycle is used
	}

	return ext.LifecycleBlocking
}

func MustLifecycle(l string) ext.Lifecycle {
	switch ext.Lifecycle(l) {
	case ext.LifecycleInforming, ext.LifecycleBlocking:
		return ext.Lifecycle(l)
	default:
		panic(fmt.Sprintf("unknown test lifecycle: %s", l))
	}
}

func lastFilenameSegment(filename string) string {
	if parts := strings.Split(filename, "/vendor/"); len(parts) > 1 {
		return parts[len(parts)-1]
	}
	if parts := strings.Split(filename, "/src/"); len(parts) > 1 {
		return parts[len(parts)-1]
	}
	return filename
}

// findSpecReport selects the best matching spec report from the list of reports
// produced by RunSpec. It first looks for a report that was actually attempted
// (NumAttempts > 0), which covers passed/failed/panicked specs. If none is found
// (as happens with Pending or Skipped specs where Ginkgo never enters the
// execution loop), it falls back to the last report in the list.
func findSpecReport(reports types.SpecReports) types.SpecReport {
	var summary types.SpecReport
	for _, report := range reports {
		if report.NumAttempts > 0 {
			summary = report
		}
	}
	// Pending/Skipped specs have NumAttempts==0; fall back to the last report
	if summary.State == types.SpecStateInvalid && len(reports) > 0 {
		summary = reports[len(reports)-1]
	}
	return summary
}

func collectAdditionalFailures(errors *[]string, suffix string, failure types.Failure) {
	if failure.IsZero() {
		return
	}

	if len(failure.ForwardedPanic) > 0 {
		if len(failure.Location.FullStackTrace) > 0 {
			*errors = append(*errors, fmt.Sprintf("\n%s\n", failure.Location.FullStackTrace))
		}
		*errors = append(*errors, fmt.Sprintf("fail [%s:%d]: Test Panicked: %s%s", lastFilenameSegment(failure.Location.FileName), failure.Location.LineNumber, failure.ForwardedPanic, suffix))
	}
	*errors = append(*errors, fmt.Sprintf("fail [%s:%d] %s%s", lastFilenameSegment(failure.Location.FileName), failure.Location.LineNumber, failure.Message, suffix))

	if failure.AdditionalFailure != nil {
		collectAdditionalFailures(errors, "  ", failure.AdditionalFailure.Failure)
	}
}
