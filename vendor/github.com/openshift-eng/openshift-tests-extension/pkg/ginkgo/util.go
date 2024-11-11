package ginkgo

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/ginkgo/v2/types"
	"github.com/onsi/gomega"
	"github.com/pkg/errors"
	"k8s.io/apimachinery/pkg/util/sets"

	ext "github.com/openshift-eng/openshift-tests-extension/pkg/extension/extensiontests"
)

func configureGinkgo() (*types.SuiteConfig, *types.ReporterConfig, error) {
	if !ginkgo.GetSuite().InPhaseBuildTree() {
		if err := ginkgo.GetSuite().BuildTree(); err != nil {
			return nil, nil, errors.Wrapf(err, "couldn't build ginkgo tree")
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

	gomega.RegisterFailHandler(ginkgo.Fail)

	return &suiteConfig, &reporterConfig, nil
}

func BuildExtensionTestSpecsFromOpenShiftGinkgoSuite() (ext.ExtensionTestSpecs, error) {
	var tests []*ext.ExtensionTestSpec
	var enforceSerialExecutionForGinkgo sync.Mutex // in-process parallelization for ginkgo is impossible so far

	if _, _, err := configureGinkgo(); err != nil {
		return nil, err
	}

	cwd, err := os.Getwd()
	if err != nil {
		return nil, errors.Wrap(err, "couldn't get current working directory")
	}

	ginkgo.GetSuite().WalkTests(func(name string, spec types.TestSpec) {
		testCase := &ext.ExtensionTestSpec{
			Name:      spec.Text(),
			Labels:    sets.New[string](spec.Labels()...),
			Lifecycle: GetLifecycle(spec.Labels()),
			Run: func() *ext.ExtensionTestResult {
				enforceSerialExecutionForGinkgo.Lock()
				defer enforceSerialExecutionForGinkgo.Unlock()

				suiteConfig, reporterConfig, _ := configureGinkgo()

				result := &ext.ExtensionTestResult{
					Name: spec.Text(),
				}

				var summary types.SpecReport
				ginkgo.GetSuite().RunSpec(spec, ginkgo.Labels{}, "", cwd, ginkgo.GetFailer(), ginkgo.GetWriter(), *suiteConfig,
					*reporterConfig)
				for _, report := range ginkgo.GetSuite().GetReport().SpecReports {
					if report.NumAttempts > 0 {
						summary = report
					}
				}

				result.Output = summary.CapturedGinkgoWriterOutput
				result.Error = summary.CapturedStdOutErr

				switch {
				case summary.State == types.SpecStatePassed:
					result.Result = ext.ResultPassed
				case summary.State == types.SpecStateSkipped:
					result.Result = ext.ResultSkipped
				case summary.State == types.SpecStateFailed, summary.State == types.SpecStatePanicked, summary.State == types.SpecStateInterrupted:
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
				default:
					panic(fmt.Sprintf("test produced unknown outcome: %#v", summary))
				}

				return result
			},
		}
		tests = append(tests, testCase)
	})

	return tests, nil
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
