/*
Copyright 2023 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package benchmark

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"maps"
	"os"
	"path"
	"regexp"
	"slices"
	"strings"
	"testing"
	"time"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/version"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/logs"
	logsapi "k8s.io/component-base/logs/api/v1"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/scheme"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/validation"
	schedframework "k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/kubernetes/test/utils/ktesting/initoption"
	"sigs.k8s.io/yaml"
)

type operationCode string

const (
	allocResourceClaimsOpcode    operationCode = "allocResourceClaims"
	createAnyOpcode              operationCode = "createAny"
	createNodesOpcode            operationCode = "createNodes"
	createNamespacesOpcode       operationCode = "createNamespaces"
	createPodsOpcode             operationCode = "createPods"
	createPodSetsOpcode          operationCode = "createPodSets"
	deletePodsOpcode             operationCode = "deletePods"
	createResourceClaimsOpcode   operationCode = "createResourceClaims"
	createResourceDriverOpcode   operationCode = "createResourceDriver"
	churnOpcode                  operationCode = "churn"
	updateAnyOpcode              operationCode = "updateAny"
	barrierOpcode                operationCode = "barrier"
	sleepOpcode                  operationCode = "sleep"
	startCollectingMetricsOpcode operationCode = "startCollectingMetrics"
	stopCollectingMetricsOpcode  operationCode = "stopCollectingMetrics"
)

const (
	// Two modes supported in "churn" operator.

	// Create continuously create API objects without deleting them.
	Create = "create"
	// Recreate creates a number of API objects and then delete them, and repeat the iteration.
	Recreate = "recreate"
)

const (
	extensionPointsLabelName = "extension_point"
	resultLabelName          = "result"
	pluginLabelName          = "plugin"
	eventLabelName           = "event"
)

// Run with -v=2, this is the default log level in production.
//
// In a PR this can be bumped up temporarily to run pull-kubernetes-scheduler-perf
// with more log output.
const DefaultLoggingVerbosity = 2

var LoggingFeatureGate FeatureGateFlag
var LoggingConfig *logsapi.LoggingConfiguration

type FeatureGateFlag interface {
	featuregate.FeatureGate
	flag.Value
}

func init() {
	f := featuregate.NewFeatureGate()
	utilruntime.Must(logsapi.AddFeatureGates(f))
	LoggingFeatureGate = f

	LoggingConfig = logsapi.NewLoggingConfiguration()
	LoggingConfig.Verbosity = DefaultLoggingVerbosity
}

var (
	defaultMetricsCollectorConfig = metricsCollectorConfig{
		Metrics: map[string][]*labelValues{
			"scheduler_framework_extension_point_duration_seconds": {
				{
					Label:  extensionPointsLabelName,
					Values: metrics.ExtensionPoints,
				},
			},
			"scheduler_scheduling_attempt_duration_seconds": {
				{
					Label:  resultLabelName,
					Values: []string{metrics.ScheduledResult, metrics.UnschedulableResult, metrics.ErrorResult},
				},
			},
			"scheduler_pod_scheduling_duration_seconds": nil,
			"scheduler_plugin_execution_duration_seconds": {
				{
					Label:  pluginLabelName,
					Values: PluginNames,
				},
				{
					Label:  extensionPointsLabelName,
					Values: metrics.ExtensionPoints,
				},
			},
		},
	}

	qHintMetrics = map[string][]*labelValues{
		"scheduler_queueing_hint_execution_duration_seconds": {
			{
				Label:  pluginLabelName,
				Values: PluginNames,
			},
			{
				Label:  eventLabelName,
				Values: schedframework.AllClusterEventLabels(),
			},
		},
		"scheduler_event_handling_duration_seconds": {
			{
				Label:  eventLabelName,
				Values: schedframework.AllClusterEventLabels(),
			},
		},
	}

	// PluginNames is the names of the plugins that scheduler_perf collects metrics for.
	// We export this variable because people outside k/k may want to put their custom plugins.
	PluginNames = []string{
		names.PrioritySort,
		names.DefaultBinder,
		names.DefaultPreemption,
		names.DynamicResources,
		names.ImageLocality,
		names.InterPodAffinity,
		names.NodeAffinity,
		names.NodeName,
		names.NodePorts,
		names.NodeResourcesBalancedAllocation,
		names.NodeResourcesFit,
		names.NodeUnschedulable,
		names.NodeVolumeLimits,
		names.PodTopologySpread,
		names.SchedulingGates,
		names.TaintToleration,
		names.VolumeBinding,
		names.VolumeRestrictions,
		names.VolumeZone,
	}
)

var UseTestingLog bool
var PerfSchedulingLabelFilter string
var TestSchedulingLabelFilter string

// InitTests should be called in a TestMain in each config subdirectory.
func InitTests() error {
	// Run with -v=2, this is the default log level in production.
	ktesting.SetDefaultVerbosity(DefaultLoggingVerbosity)

	// test/integration/framework/flags.go unconditionally initializes the
	// logging flags. That's correct for most tests, but in the
	// scheduler_perf test we want more control over the flags, therefore
	// here strip them out.
	var fs flag.FlagSet
	flag.CommandLine.VisitAll(func(f *flag.Flag) {
		switch f.Name {
		case "log-flush-frequency", "v", "vmodule":
			// These will be added below ourselves, don't copy.
		default:
			fs.Var(f.Value, f.Name, f.Usage)
		}
	})
	flag.CommandLine = &fs

	flag.Var(LoggingFeatureGate, "feature-gate",
		"A set of key=value pairs that describe feature gates for alpha/experimental features. "+
			"Options are:\n"+strings.Join(LoggingFeatureGate.KnownFeatures(), "\n"))

	flag.BoolVar(&UseTestingLog, "use-testing-log", false, "Write log entries with testing.TB.Log. This is more suitable for unit testing and debugging, but less realistic in real benchmarks.")
	flag.StringVar(&PerfSchedulingLabelFilter, "perf-scheduling-label-filter", "performance", "comma-separated list of labels which a testcase must have (no prefix or +) or must not have (-), used by BenchmarkPerfScheduling")
	flag.StringVar(&TestSchedulingLabelFilter, "test-scheduling-label-filter", "integration-test,-performance", "comma-separated list of labels which a testcase must have (no prefix or +) or must not have (-), used by TestScheduling")

	// This would fail if we hadn't removed the logging flags above.
	logsapi.AddGoFlags(LoggingConfig, flag.CommandLine)

	flag.Parse()

	logs.InitLogs()
	return logsapi.ValidateAndApply(LoggingConfig, LoggingFeatureGate)
}

func registerQHintMetrics() {
	for k, v := range qHintMetrics {
		defaultMetricsCollectorConfig.Metrics[k] = v
	}
}

func unregisterQHintMetrics() {
	for k := range qHintMetrics {
		delete(defaultMetricsCollectorConfig.Metrics, k)
	}
}

// testCase defines a set of test cases that intends to test the performance of
// similar workloads of varying sizes with shared overall settings such as
// feature gates and metrics collected.
type testCase struct {
	// Name of the testCase.
	Name string
	// Feature gates to set before running the test.
	// Optional
	FeatureGates map[featuregate.Feature]bool
	// List of metrics to collect. Defaults to
	// defaultMetricsCollectorConfig if unspecified.
	// Optional
	MetricsCollectorConfig *metricsCollectorConfig
	// Template for sequence of ops that each workload must follow. Each op will
	// be executed serially one after another.
	WorkloadTemplate []op
	// List of workloads to run under this testCase.
	Workloads []*workload
	// SchedulerConfigPath is the path of scheduler configuration
	// Optional
	SchedulerConfigPath string
	// Default path to spec file describing the pods to create.
	// This path can be overridden in createPodsOp by setting PodTemplatePath .
	// Optional
	DefaultPodTemplatePath *string
	// Labels can be used to enable or disable workloads inside this test case.
	Labels []string
	// DefaultThresholdMetricSelector defines default metric used for threshold comparison.
	// It is only populated to workloads without their ThresholdMetricSelector set.
	// If nil, the default metric is set to "SchedulingThroughput" with "Average" data bucket.
	// Optional
	DefaultThresholdMetricSelector *thresholdMetricSelector
}

func (tc *testCase) collectsMetrics() bool {
	for _, op := range tc.WorkloadTemplate {
		if op.realOp.collectsMetrics() {
			return true
		}
	}
	return false
}

func (tc *testCase) workloadNamesUnique() error {
	workloadUniqueNames := map[string]bool{}
	for _, w := range tc.Workloads {
		if workloadUniqueNames[w.Name] {
			return fmt.Errorf("%s: workload name %s is not unique", tc.Name, w.Name)
		}
		workloadUniqueNames[w.Name] = true
	}
	return nil
}

// workload is a subtest under a testCase that tests the scheduler performance
// for a certain ordering of ops. The set of nodes created and pods scheduled
// in a workload may be heterogeneous.
type workload struct {
	// Name of the workload.
	Name string
	// Values of parameters used in the workloadTemplate.
	Params params
	// Labels can be used to enable or disable a workload.
	Labels []string
	// Threshold is compared to average value of metric specified using thresholdMetricSelector.
	// The comparison is performed for op with CollectMetrics set to true.
	// If the measured value is below the threshold, the workload's test case will fail.
	// If set to zero, the threshold check is disabled.
	//
	// May contain a single value or map of topic name to value.
	// The single value is used if there is no entry in the map for the topic name.
	// Topic names are passed to RunBenchmarkPerfScheduling. This approach
	// makes it possible to reuse the same test cases in different configurations.
	//
	// Optional
	Threshold thresholds
	// ThresholdMetricSelector defines to what metric the Threshold should be compared.
	// If nil, the metric is set to DefaultThresholdMetricSelector of the testCase.
	// If DefaultThresholdMetricSelector is nil, the metric is set to "SchedulingThroughput".
	// Optional
	ThresholdMetricSelector *thresholdMetricSelector
	// Feature gates to set before running the workload.
	// Explicitly setting a feature in this map overrides the test case settings.
	// Optional
	FeatureGates map[featuregate.Feature]bool
}

func (w *workload) isValid(mcc *metricsCollectorConfig) error {
	if w.Threshold.value < 0 {
		return fmt.Errorf("invalid Threshold=%f; should be non-negative", w.Threshold.value)
	}
	for topicName, value := range w.Threshold.valuesByTopic {
		if value < 0 {
			return fmt.Errorf("invalid Threshold=%f for topic %q; should be non-negative", value, topicName)
		}
	}

	return w.ThresholdMetricSelector.isValid(mcc)
}

func (w *workload) setDefaults(testCaseThresholdMetricSelector *thresholdMetricSelector) {
	if w.ThresholdMetricSelector != nil {
		return
	}
	if testCaseThresholdMetricSelector != nil {
		w.ThresholdMetricSelector = testCaseThresholdMetricSelector
		return
	}
	// By default, SchedulingThroughput Average should be compared with the threshold.
	w.ThresholdMetricSelector = &thresholdMetricSelector{
		Name:       "SchedulingThroughput",
		DataBucket: "Average",
	}
}

type thresholds struct {
	value         float64
	valuesByTopic map[string]float64
}

func (t *thresholds) UnmarshalJSON(text []byte) error {
	if errFloat64 := json.Unmarshal(text, &t.value); errFloat64 != nil {
		// Not a plain number. Let's try as map.
		if errMap := json.Unmarshal(text, &t.valuesByTopic); errMap != nil {
			return fmt.Errorf("expected either float64 or topic name -> float64 map: %w, %w", errFloat64, errMap)
		}
	}
	return nil
}

func (t *thresholds) Get(topicName string) float64 {
	if value, ok := t.valuesByTopic[topicName]; ok {
		return value
	}
	return t.value
}

// thresholdMetricSelector defines the name and labels of metric to compare with threshold.
type thresholdMetricSelector struct {
	// Name of the metric is compared to "Metric" field in DataItem labels.
	Name string
	// Labels of the metric. All of them needs to match the metric's labels to assume equality.
	Labels map[string]string
	// DataBucket specifies which data bucket should be compared against the threshold.
	DataBucket string
	// ExpectLower defines whether the threshold should denote the maximum allowable value of the metric.
	// If false, the threshold defines minimum allowable value.
	// Optional
	ExpectLower bool
}

func (ms thresholdMetricSelector) isValid(mcc *metricsCollectorConfig) error {
	if ms.DataBucket == "" {
		return fmt.Errorf("dataBucket should be set for metric %v", ms.Name)
	}
	if ms.Name == "SchedulingThroughput" {
		return nil
	}

	if mcc == nil {
		mcc = &defaultMetricsCollectorConfig
	}

	labels, ok := mcc.Metrics[ms.Name]
	if !ok {
		return fmt.Errorf("the metric %v is targeted, but it's not collected during the test. Make sure the MetricsCollectorConfig is valid", ms.Name)
	}

	for _, labelsComb := range uniqueLVCombos(labels) {
		if labelsMatch(labelsComb, ms.Labels) {
			return nil
		}
	}
	return fmt.Errorf("no matching labels found for metric %v", ms.Name)
}

type params struct {
	params map[string]any
	// isUsed field records whether params is used or not.
	isUsed map[string]bool
}

// UnmarshalJSON is a custom unmarshaler for params.
//
// from(json):
//
//	{
//		"initNodes": 500,
//		"initPods": 50
//	}
//
// to:
//
//	params{
//		params: map[string]any{
//			"intNodes": 500,
//			"initPods": 50,
//		},
//		isUsed: map[string]bool{}, // empty map
//	}
func (p *params) UnmarshalJSON(b []byte) error {
	aux := map[string]any{}

	if err := json.Unmarshal(b, &aux); err != nil {
		return err
	}

	p.params = aux
	p.isUsed = map[string]bool{}
	return nil
}

// get retrieves the parameter as an integer
func (p params) get(key string) (int, error) {
	// JSON unmarshal integer constants in an "any" field as float.
	f, err := getParam[float64](p, key)
	if err != nil {
		return 0, err
	}
	return int(f), nil
}

// getParam retrieves the parameter as specific type. There is no conversion,
// so in practice this means that only types that JSON unmarshalling uses
// (float64, string, bool) work.
func getParam[T float64 | string | bool](p params, key string) (T, error) {
	p.isUsed[key] = true
	param, ok := p.params[key]
	var t T
	if !ok {
		return t, fmt.Errorf("parameter %s is undefined", key)
	}
	t, ok = param.(T)
	if !ok {
		return t, fmt.Errorf("parameter %s has the wrong type %T", key, param)
	}
	return t, nil
}

// unusedParams returns the names of unusedParams
func (w workload) unusedParams() []string {
	var ret []string
	for name := range w.Params.params {
		if !w.Params.isUsed[name] {
			ret = append(ret, name)
		}
	}
	return ret
}

// op is a dummy struct which stores the real op in itself.
type op struct {
	realOp realOp
}

// UnmarshalJSON is a custom unmarshaler for the op struct since we don't know
// which op we're decoding at runtime.
func (op *op) UnmarshalJSON(b []byte) error {
	possibleOps := map[operationCode]realOp{
		allocResourceClaimsOpcode:    &allocResourceClaimsOp{},
		createAnyOpcode:              &createAny{},
		createNodesOpcode:            &createNodesOp{},
		createNamespacesOpcode:       &createNamespacesOp{},
		createPodsOpcode:             &createPodsOp{},
		createPodSetsOpcode:          &createPodSetsOp{},
		deletePodsOpcode:             &deletePodsOp{},
		createResourceClaimsOpcode:   &createResourceClaimsOp{},
		createResourceDriverOpcode:   &createResourceDriverOp{},
		churnOpcode:                  &churnOp{},
		updateAnyOpcode:              &updateAny{},
		barrierOpcode:                &barrierOp{},
		sleepOpcode:                  &sleepOp{},
		startCollectingMetricsOpcode: &startCollectingMetricsOp{},
		stopCollectingMetricsOpcode:  &stopCollectingMetricsOp{},
		// TODO(#94601): add a delete nodes op to simulate scaling behaviour?
	}
	// First determine the opcode using lenient decoding (= ignore extra fields).
	var possibleOp struct {
		Opcode operationCode
	}
	if err := json.Unmarshal(b, &possibleOp); err != nil {
		return fmt.Errorf("decoding opcode from %s: %w", string(b), err)
	}
	realOp, ok := possibleOps[possibleOp.Opcode]
	if !ok {
		return fmt.Errorf("unknown opcode %q in %s", possibleOp.Opcode, string(b))
	}
	decoder := json.NewDecoder(bytes.NewReader(b))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(realOp); err != nil {
		return fmt.Errorf("decoding %s into %T: %w", string(b), realOp, err)
	}
	if err := realOp.isValid(true); err != nil {
		return fmt.Errorf("%s not valid for %T: %w", string(b), realOp, err)
	}
	op.realOp = realOp
	return nil
}

// realOp is an interface that is implemented by different structs. To evaluate
// the validity of ops at parse-time, a isValid function must be implemented.
type realOp interface {
	// isValid verifies the validity of the op args such as node/pod count. Note
	// that we don't catch undefined parameters at this stage.
	//
	// This returns errInvalidOp if the configured operation does not match.
	isValid(allowParameterization bool) error
	// collectsMetrics checks if the op collects metrics.
	collectsMetrics() bool
	// patchParams returns a patched realOp of the same type after substituting
	// parameterizable values with workload-specific values. One should implement
	// this method on the value receiver base type, not a pointer receiver base
	// type, even though calls will be made from with a *realOp. This is because
	// callers don't want the receiver to inadvertently modify the realOp
	// (instead, it's returned as a return value).
	patchParams(w *workload) (realOp, error)
}

// runnableOp is an interface implemented by some operations. It makes it possible
// to execute the operation without having to add separate code into runWorkload.
type runnableOp interface {
	realOp

	// requiredNamespaces returns all namespaces that runWorkload must create
	// before running the operation.
	requiredNamespaces() []string
	// run executes the steps provided by the operation.
	run(ktesting.TContext)
}

func isValidParameterizable(val string) bool {
	return strings.HasPrefix(val, "$")
}

func isValidCount(allowParameterization bool, count int, countParam string) bool {
	if !allowParameterization || countParam == "" {
		// Ignore parameter. The value itself must be okay.
		return count >= 0
	}
	return isValidParameterizable(countParam)
}

func initTestOutput(tb testing.TB) io.Writer {
	var output io.Writer
	if UseTestingLog {
		output = framework.NewTBWriter(tb)
	} else {
		tmpDir := tb.TempDir()
		logfileName := path.Join(tmpDir, "output.log")
		fileOutput, err := os.Create(logfileName)
		if err != nil {
			tb.Fatalf("create log file: %v", err)
		}
		output = fileOutput

		tb.Cleanup(func() {
			// Dump the log output when the test is done.  The user
			// can decide how much of it will be visible in case of
			// success: then "go test" truncates, "go test -v"
			// doesn't. All of it will be shown for a failure.
			if err := fileOutput.Close(); err != nil {
				tb.Fatalf("close log file: %v", err)
			}
			log, err := os.ReadFile(logfileName)
			if err != nil {
				tb.Fatalf("read log file: %v", err)
			}
			tb.Logf("full log output:\n%s", string(log))
		})
	}
	return output
}

var specialFilenameChars = regexp.MustCompile(`[^a-zA-Z0-9-_]`)

func setupTestCase(t testing.TB, tc *testCase, featureGates map[featuregate.Feature]bool, outOfTreePluginRegistry frameworkruntime.Registry) (*scheduler.Scheduler, informers.SharedInformerFactory, ktesting.TContext) {
	tCtx := ktesting.Init(t, initoption.PerTestOutput(UseTestingLog))
	artifacts, doArtifacts := os.LookupEnv("ARTIFACTS")
	if !UseTestingLog && doArtifacts {
		// Reconfigure logging so that it goes to a separate file per
		// test instead of stderr. If the test passes, the file gets
		// deleted. The overall output can be very large (> 200 MB for
		// ci-benchmark-scheduler-perf-master). With this approach, we
		// have log output for failures without having to store large
		// amounts of data that no-one is looking at. The performance
		// is the same as writing to stderr.
		if err := logsapi.ResetForTest(LoggingFeatureGate); err != nil {
			t.Fatalf("Failed to reset the logging configuration: %v", err)
		}
		logfileName := path.Join(artifacts, specialFilenameChars.ReplaceAllString(t.Name(), "_")+".log")
		out, err := os.Create(logfileName)
		if err != nil {
			t.Fatalf("Failed to create per-test log output file: %v", err)
		}
		t.Cleanup(func() {
			// Everything should have stopped by now, checked below
			// by GoleakCheck (which runs first during test
			// shutdown!). Therefore we can clean up. Errors get logged
			// and fail the test, but cleanup tries to continue.
			//
			// Note that the race detector will flag any goroutine
			// as causing a race if there is no explicit wait for
			// that goroutine to stop.  We know that they must have
			// stopped (GoLeakCheck!) but the race detector
			// doesn't.
			//
			// This is a major issue because many Kubernetes goroutines get
			// started without waiting for them to stop :-(
			//
			// In practice, klog's own flushing got called out by the race detector.
			// As we know about that one, we can force it to stop explicitly to
			// satisfy the race detector.
			klog.StopFlushDaemon()
			if err := logsapi.ResetForTest(LoggingFeatureGate); err != nil {
				t.Errorf("Failed to reset the logging configuration: %v", err)
			}
			if err := out.Close(); err != nil {
				t.Errorf("Failed to close the per-test log output file: %s: %v", logfileName, err)
			}
			if !t.Failed() {
				if err := os.Remove(logfileName); err != nil {
					t.Errorf("Failed to remove the per-test log output file: %v", err)
				}
			}
		})
		opts := &logsapi.LoggingOptions{
			ErrorStream: out,
			InfoStream:  out,
		}
		if err := logsapi.ValidateAndApplyWithOptions(LoggingConfig, opts, LoggingFeatureGate); err != nil {
			t.Fatalf("Failed to apply the per-test logging configuration: %v", err)
		}
	}

	// Ensure that there are no leaked
	// goroutines.  They could influence
	// performance of the next benchmark.
	// This must *after* RedirectKlog
	// because then during cleanup, the
	// test will wait for goroutines to
	// quit *before* restoring klog settings.
	framework.GoleakCheck(t)

	// Now that we are ready to run, start
	// a brand new etcd.
	logger := tCtx.Logger()
	if !UseTestingLog {
		// Associate output going to the global log with the current test.
		logger = logger.WithName(tCtx.Name())
	}
	framework.StartEtcd(logger, t, true)

	// We need to set emulation version for QueueingHints feature gate, which is locked at 1.34.
	// Only emulate v1.33 when QueueingHints is explicitly disabled.
	if qhEnabled, exists := featureGates[features.SchedulerQueueingHints]; exists && !qhEnabled {
		featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse("1.33"))
	} else if _, found := featureGates[features.OpportunisticBatching]; !found {
		if featureGates == nil {
			featureGates = map[featuregate.Feature]bool{}
		}
		featureGates[features.OpportunisticBatching] = false
	}
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featureGates)

	// 30 minutes should be plenty enough even for the 5000-node tests.
	timeout := 30 * time.Minute
	tCtx = tCtx.WithTimeout(timeout, fmt.Sprintf("timed out after the %s per-test timeout", timeout))

	if utilfeature.DefaultFeatureGate.Enabled(features.SchedulerQueueingHints) {
		registerQHintMetrics()
		t.Cleanup(func() {
			unregisterQHintMetrics()
		})
	}

	return setupClusterForWorkload(tCtx, tc.SchedulerConfigPath, featureGates, outOfTreePluginRegistry)
}

func featureGatesMerge(src map[featuregate.Feature]bool, overrides map[featuregate.Feature]bool) map[featuregate.Feature]bool {
	result := make(map[featuregate.Feature]bool)
	maps.Copy(result, src)
	maps.Copy(result, overrides)
	return result
}

// fixJSONOutput works around Go not emitting a "pass" action for
// sub-benchmarks
// (https://github.com/golang/go/issues/66825#issuecomment-2343229005), which
// causes gotestsum to report a successful benchmark run as failed
// (https://github.com/gotestyourself/gotestsum/issues/413#issuecomment-2343206787).
//
// It does this by printing the missing "PASS" output line that test2json
// then converts into the "pass" action.
func fixJSONOutput(b *testing.B) {
	if !slices.Contains(os.Args, "-test.v=test2json") {
		// Not printing JSON.
		return
	}

	start := time.Now()
	b.Cleanup(func() {
		if b.Failed() {
			// Really has failed, do nothing.
			return
		}
		// SYN gets injected when using -test.v=test2json, see
		// https://cs.opensource.google/go/go/+/refs/tags/go1.23.3:src/testing/testing.go;drc=87ec2c959c73e62bfae230ef7efca11ec2a90804;l=527
		fmt.Fprintf(os.Stderr, "%c--- PASS: %s (%.2fs)\n", 22 /* SYN */, b.Name(), time.Since(start).Seconds())
	})
}

// RunBenchmarkPerfScheduling runs the scheduler performance benchmark tests.
//
// You can pass your own scheduler plugins via outOfTreePluginRegistry.
// Also, you may want to put your plugins in PluginNames variable in this package
// to collect metrics for them.
func RunBenchmarkPerfScheduling(b *testing.B, configFile string, topicName string, outOfTreePluginRegistry frameworkruntime.Registry, options ...SchedulerPerfOption) {
	opts := &schedulerPerfOptions{}

	for _, option := range options {
		option(opts)
	}

	testCases, err := getTestCases(configFile)
	if err != nil {
		b.Fatal(err)
	}
	if err = validateTestCases(testCases); err != nil {
		b.Fatal(err)
	}
	fixJSONOutput(b)

	if testing.Short() {
		PerfSchedulingLabelFilter += ",+short"
	}
	testcaseLabelSelectors := strings.Split(PerfSchedulingLabelFilter, ",")

	output := initTestOutput(b)

	// Because we run sequentially, it is possible to change the global
	// klog logger and redirect log output. Quite a lot of code still uses
	// it instead of supporting contextual logging.
	//
	// Because we leak one goroutine which calls klog, we cannot restore
	// the previous state.
	_ = framework.RedirectKlog(b, output)

	dataItems := DataItems{Version: "v1"}
	for _, tc := range testCases {
		b.Run(tc.Name, func(b *testing.B) {
			fixJSONOutput(b)
			for _, w := range tc.Workloads {
				b.Run(w.Name, func(b *testing.B) {
					if !enabled(testcaseLabelSelectors, append(tc.Labels, w.Labels...)...) {
						b.Skipf("disabled by label filter %q", PerfSchedulingLabelFilter)
					}
					fixJSONOutput(b)

					featureGates := featureGatesMerge(tc.FeatureGates, w.FeatureGates)
					scheduler, informerFactory, tCtx := setupTestCase(b, tc, featureGates, outOfTreePluginRegistry)

					err := w.isValid(tc.MetricsCollectorConfig)
					if err != nil {
						b.Fatalf("workload %s is not valid: %v", w.Name, err)
					}

					if opts.prepareFn != nil {
						err = opts.prepareFn(tCtx)
						if err != nil {
							b.Fatalf("failed to run prepareFn: %v", err)
						}
					}

					results, err := runWorkload(tCtx, tc, w, topicName, scheduler, informerFactory)
					if err != nil {
						tCtx.Fatalf("Error running workload %s: %s", w.Name, err)
					}
					dataItems.DataItems = append(dataItems.DataItems, results...)

					if len(results) > 0 {
						// The default ns/op is not
						// useful because it includes
						// the time spent on
						// initialization and shutdown. Here we suppress it.
						b.ReportMetric(0, "ns/op")

						// Instead, report the same
						// results that also get stored
						// in the JSON file.
						for _, result := range results {
							// For some metrics like
							// scheduler_framework_extension_point_duration_seconds
							// the actual value has some
							// other unit. We patch the key
							// to make it look right.
							metric := strings.ReplaceAll(result.Labels["Metric"], "_seconds", "_"+result.Unit)
							for key, value := range result.Data {
								b.ReportMetric(value, metric+"/"+key)
							}
						}
					}

					if featureGates[features.SchedulerQueueingHints] {
						// In any case, we should make sure InFlightEvents is empty after running the scenario.
						if err = checkEmptyInFlightEvents(); err != nil {
							tCtx.Errorf("%s: %s", w.Name, err)
						}
					}

					// Reset metrics to prevent metrics generated in current workload gets
					// carried over to the next workload.
					legacyregistry.Reset()

					// Exactly one result is expected to contain the progress information.
					for _, item := range results {
						if len(item.progress) == 0 {
							continue
						}

						destFile, err := dataFilename(strings.ReplaceAll(fmt.Sprintf("%s_%s_%s_%s.dat", tc.Name, w.Name, topicName, runID), "/", "_"))
						if err != nil {
							b.Fatalf("prepare data file: %v", err)
						}
						f, err := os.Create(destFile)
						if err != nil {
							b.Fatalf("create data file: %v", err)
						}

						// Print progress over time.
						for _, sample := range item.progress {
							fmt.Fprintf(f, "%.1fs %d %d %d %f\n", sample.ts.Sub(item.start).Seconds(), sample.completed, sample.attempts, sample.observedTotal, sample.observedRate)
						}
						if err := f.Close(); err != nil {
							b.Fatalf("closing data file: %v", err)
						}
					}
				})
			}
		})
	}
	// Different top-level BenchmarkPerfScheduling* tests are supported as long as they use unique topic names.
	// The final JSON file then is always called BenchmarkPerfScheduling_benchmark_<topic name>_<date+time>.json
	// because that is what perf-dash is configured to read:
	// https://github.com/kubernetes/perf-tests/blob/581139e45e79cf04b9c2777b82677957f1e7f90b/perfdash/config.go#L520-L525
	namePrefix := b.Name()
	namePrefix = regexp.MustCompile(`^BenchmarkPerfScheduling[^/]*`).ReplaceAllString(namePrefix, "BenchmarkPerfScheduling")
	namePrefix = strings.ReplaceAll(namePrefix, "/", "_")
	namePrefix += "_benchmark_" + topicName
	if err := dataItems2JSONFile(dataItems, namePrefix); err != nil {
		b.Fatalf("unable to write measured data %+v: %v", dataItems, err)
	}
}

// RunIntegrationPerfScheduling runs the scheduler performance integration tests.
func RunIntegrationPerfScheduling(t *testing.T, configFile string) {
	testCases, err := getTestCases(configFile)
	if err != nil {
		t.Fatal(err)
	}
	if err = validateTestCases(testCases); err != nil {
		t.Fatal(err)
	}

	if testing.Short() {
		TestSchedulingLabelFilter += ",+short"
	}
	testcaseLabelSelectors := strings.Split(TestSchedulingLabelFilter, ",")

	for _, tc := range testCases {
		t.Run(tc.Name, func(t *testing.T) {
			for _, w := range tc.Workloads {
				t.Run(w.Name, func(t *testing.T) {
					if !enabled(testcaseLabelSelectors, append(tc.Labels, w.Labels...)...) {
						t.Skipf("disabled by label filter %q", TestSchedulingLabelFilter)
					}
					featureGates := featureGatesMerge(tc.FeatureGates, w.FeatureGates)
					scheduler, informerFactory, tCtx := setupTestCase(t, tc, featureGates, nil)
					err := w.isValid(tc.MetricsCollectorConfig)
					if err != nil {
						t.Fatalf("workload %s is not valid: %v", w.Name, err)
					}

					_, err = runWorkload(tCtx, tc, w, "" /* topic name not relevant */, scheduler, informerFactory)
					if err != nil {
						tCtx.Fatalf("Error running workload %s: %s", w.Name, err)
					}

					if featureGates[features.SchedulerQueueingHints] {
						// In any case, we should make sure InFlightEvents is empty after running the scenario.
						if err = checkEmptyInFlightEvents(); err != nil {
							tCtx.Errorf("%s: %s", w.Name, err)
						}
					}

					// Reset metrics to prevent metrics generated in current workload gets
					// carried over to the next workload.
					legacyregistry.Reset()
				})
			}
		})
	}
}

func loadSchedulerConfig(file string) (*config.KubeSchedulerConfiguration, error) {
	data, err := os.ReadFile(file)
	if err != nil {
		return nil, err
	}
	// The UniversalDecoder runs defaulting and returns the internal type by default.
	obj, gvk, err := scheme.Codecs.UniversalDecoder().Decode(data, nil, nil)
	if err != nil {
		return nil, err
	}
	if cfgObj, ok := obj.(*config.KubeSchedulerConfiguration); ok {
		return cfgObj, nil
	}
	return nil, fmt.Errorf("couldn't decode as KubeSchedulerConfiguration, got %s: ", gvk)
}

func unrollWorkloadTemplate(tb ktesting.TB, wt []op, w *workload) []op {
	var unrolled []op
	for opIndex, o := range wt {
		realOp, err := o.realOp.patchParams(w)
		if err != nil {
			tb.Fatalf("op %d: %v", opIndex, err)
		}
		switch concreteOp := realOp.(type) {
		case *createPodSetsOp:
			tb.Logf("Creating %d pod sets %s", concreteOp.Count, concreteOp.CountParam)
			for i := 0; i < concreteOp.Count; i++ {
				copy := concreteOp.CreatePodsOp
				ns := fmt.Sprintf("%s-%d", concreteOp.NamespacePrefix, i)
				copy.Namespace = &ns
				unrolled = append(unrolled, op{realOp: &copy})
			}
		default:
			unrolled = append(unrolled, o)
		}
	}
	return unrolled
}

func setupClusterForWorkload(tCtx ktesting.TContext, configPath string, featureGates map[featuregate.Feature]bool, outOfTreePluginRegistry frameworkruntime.Registry) (*scheduler.Scheduler, informers.SharedInformerFactory, ktesting.TContext) {
	var cfg *config.KubeSchedulerConfiguration
	var err error
	if configPath != "" {
		cfg, err = loadSchedulerConfig(configPath)
		if err != nil {
			tCtx.Fatalf("error loading scheduler config file: %v", err)
		}
		if err = validation.ValidateKubeSchedulerConfiguration(cfg); err != nil {
			tCtx.Fatalf("validate scheduler config file failed: %v", err)
		}
	}
	return mustSetupCluster(tCtx, cfg, featureGates, outOfTreePluginRegistry)
}

func labelsMatch(actualLabels, requiredLabels map[string]string) bool {
	for requiredLabel, requiredValue := range requiredLabels {
		actualValue, ok := actualLabels[requiredLabel]
		if !ok || requiredValue != actualValue {
			return false
		}
	}
	return true
}

func valueWithinThreshold(value, threshold float64, expectLower bool) bool {
	if expectLower {
		return value < threshold
	}
	return value > threshold
}

// applyThreshold adds the threshold to data item with metric specified via metricSelector and verifies that
// this metrics value is within threshold.
func applyThreshold(items []DataItem, threshold float64, metricSelector thresholdMetricSelector) error {
	if threshold == 0 {
		return nil
	}
	dataBucket := metricSelector.DataBucket
	var errs []error
	for _, item := range items {
		if item.Labels["Metric"] != metricSelector.Name || !labelsMatch(item.Labels, metricSelector.Labels) {
			continue
		}
		thresholdItemName := dataBucket + "Threshold"
		item.Data[thresholdItemName] = threshold
		dataItem, ok := item.Data[dataBucket]
		if !ok {
			errs = append(errs, fmt.Errorf("%s: no data present for %q metric %q bucket", item.Labels["Name"], metricSelector.Name, dataBucket))
			continue
		}
		if !valueWithinThreshold(dataItem, threshold, metricSelector.ExpectLower) {
			if metricSelector.ExpectLower {
				errs = append(errs, fmt.Errorf("%s: expected %q %q to be lower: got %f, want %f", item.Labels["Name"], metricSelector.Name, dataBucket, dataItem, threshold))
			} else {
				errs = append(errs, fmt.Errorf("%s: expected %q %q to be higher: got %f, want %f", item.Labels["Name"], metricSelector.Name, dataBucket, dataItem, threshold))
			}
		}
	}
	return errors.Join(errs...)
}

func checkEmptyInFlightEvents() error {
	labels := append(schedframework.AllClusterEventLabels(), metrics.PodPoppedInFlightEvent)
	for _, label := range labels {
		value, err := testutil.GetGaugeMetricValue(metrics.InFlightEvents.WithLabelValues(label))
		if err != nil {
			return fmt.Errorf("failed to get InFlightEvents metric for label %s", label)
		}
		if value > 0 {
			return fmt.Errorf("InFlightEvents for label %s should be empty, but has %v items", label, value)
		}
	}
	return nil
}

func runWorkload(tCtx ktesting.TContext, tc *testCase, w *workload, topicName string, scheduler *scheduler.Scheduler, informerFactory informers.SharedInformerFactory) ([]DataItem, error) {
	b, benchmarking := tCtx.TB().(*testing.B)
	if benchmarking {
		start := time.Now()
		b.Cleanup(func() {
			duration := time.Since(start)
			// This includes startup and shutdown time and thus does not
			// reflect scheduling performance. It's useful to get a feeling
			// for how long each workload runs overall.
			b.ReportMetric(duration.Seconds(), "runtime_seconds")
		})
	}

	// Disable error checking of the sampling interval length in the
	// throughput collector by default. When running benchmarks, report
	// it as test failure when samples are not taken regularly.
	var throughputErrorMargin float64
	if benchmarking {
		// TODO: To prevent the perf-test failure, we increased the error margin, if still not enough
		// one day, we should think of another approach to avoid this trick.
		throughputErrorMargin = 30
	}

	// Additional informers needed for testing. The pod informer was
	// already created before (scheduler.NewInformerFactory) and the
	// factory was started for it (mustSetupCluster), therefore we don't
	// need to start again.
	podInformer := informerFactory.Core().V1().Pods()

	eventInformer := informerFactory.Core().V1().Events()
	// Trigger registration so Start() launches it
	_ = eventInformer.Informer()
	informerFactory.Start(tCtx.Done())
	informerFactory.WaitForCacheSync(tCtx.Done())


	// Everything else started by this function gets stopped before it returns.
	tCtx = tCtx.WithCancel()

	executor := WorkloadExecutor{
		tCtx:                         tCtx,
		scheduler:                    scheduler,
		numPodsScheduledPerNamespace: make(map[string]int),
		podInformer:                  podInformer,
		workloadLister:               informerFactory.Scheduling().V1alpha1().Workloads().Lister(),
		eventInformer:                eventInformer,
		throughputErrorMargin:        throughputErrorMargin,
		testCase:                     tc,
		workload:                     w,
		topicName:                    topicName,
	}

	tCtx.TB().Cleanup(func() {
		tCtx.Cancel("workload is done")
		executor.wait()
	})

	for opIndex, op := range unrollWorkloadTemplate(tCtx, tc.WorkloadTemplate, w) {
		realOp, err := op.realOp.patchParams(w)
		if err != nil {
			return nil, fmt.Errorf("op %d: %w", opIndex, err)
		}
		select {
		case <-tCtx.Done():
			return nil, fmt.Errorf("op %d: %w", opIndex, context.Cause(tCtx))
		default:
		}
		err = executor.runOp(realOp, opIndex)
		if err != nil {
			return nil, fmt.Errorf("op %d: %w", opIndex, err)
		}
	}

	// check unused params and inform users
	unusedParams := w.unusedParams()
	if len(unusedParams) != 0 {
		return nil, fmt.Errorf("the parameters %v are defined on workload %s, but unused.\nPlease make sure there are no typos", unusedParams, w.Name)
	}

	// Some tests have unschedulable pods. Do not add an implicit barrier at the
	// end as we do not want to wait for them.
	return executor.dataItems, nil
}

func getSpecFromFile(path *string, spec interface{}) error {
	bytes, err := os.ReadFile(*path)
	if err != nil {
		return err
	}
	return yaml.UnmarshalStrict(bytes, spec)
}

func getTestCases(path string) ([]*testCase, error) {
	testCases := make([]*testCase, 0)
	if err := getSpecFromFile(&path, &testCases); err != nil {
		return nil, fmt.Errorf("parsing test cases error: %w", err)
	}
	for _, tc := range testCases {
		for _, w := range tc.Workloads {
			w.setDefaults(tc.DefaultThresholdMetricSelector)
		}
	}
	return testCases, nil
}

func validateTestCases(testCases []*testCase) error {
	if len(testCases) == 0 {
		return fmt.Errorf("no test cases defined")
	}
	testCaseUniqueNames := map[string]bool{}
	for _, tc := range testCases {
		if testCaseUniqueNames[tc.Name] {
			return fmt.Errorf("%s: name is not unique", tc.Name)
		}
		testCaseUniqueNames[tc.Name] = true
		if len(tc.Workloads) == 0 {
			return fmt.Errorf("%s: no workloads defined", tc.Name)
		}
		if err := tc.workloadNamesUnique(); err != nil {
			return err
		}
		if len(tc.WorkloadTemplate) == 0 {
			return fmt.Errorf("%s: no ops defined", tc.Name)
		}
		// Make sure there's at least one CreatePods op with collectMetrics set to
		// true, or a startCollectingMetricsOp together with stopCollectingMetricsOp
		// in each workload. What's the point of running a performance
		// benchmark if no statistics are collected for reporting?
		if !tc.collectsMetrics() {
			return fmt.Errorf("%s: no op in the workload template collects metrics", tc.Name)
		}
	}
	return nil
}
