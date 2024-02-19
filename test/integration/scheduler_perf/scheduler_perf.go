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
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	cacheddiscovery "k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/restmapper"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/scheme"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/validation"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/kubernetes/test/integration/framework"
	testutils "k8s.io/kubernetes/test/utils"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/kubernetes/test/utils/ktesting/initoption"
	"sigs.k8s.io/yaml"
)

type operationCode string

const (
	createAnyOpcode            operationCode = "createAny"
	createNodesOpcode          operationCode = "createNodes"
	createNamespacesOpcode     operationCode = "createNamespaces"
	createPodsOpcode           operationCode = "createPods"
	createPodSetsOpcode        operationCode = "createPodSets"
	createResourceClaimsOpcode operationCode = "createResourceClaims"
	createResourceDriverOpcode operationCode = "createResourceDriver"
	churnOpcode                operationCode = "churn"
	barrierOpcode              operationCode = "barrier"
	sleepOpcode                operationCode = "sleep"
)

const (
	// Two modes supported in "churn" operator.

	// Create continuously create API objects without deleting them.
	Create = "create"
	// Recreate creates a number of API objects and then delete them, and repeat the iteration.
	Recreate = "recreate"
)

const (
	configFile               = "config/performance-config.yaml"
	extensionPointsLabelName = "extension_point"
	resultLabelName          = "result"
)

var (
	defaultMetricsCollectorConfig = metricsCollectorConfig{
		Metrics: map[string]*labelValues{
			"scheduler_framework_extension_point_duration_seconds": {
				label:  extensionPointsLabelName,
				values: []string{"Filter", "Score"},
			},
			"scheduler_scheduling_attempt_duration_seconds": {
				label:  resultLabelName,
				values: []string{metrics.ScheduledResult, metrics.UnschedulableResult, metrics.ErrorResult},
			},
			"scheduler_pod_scheduling_duration_seconds":     nil,
			"scheduler_pod_scheduling_sli_duration_seconds": nil,
		},
	}
)

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
	// be executed serially one after another. Each element of the list must be
	// createNodesOp, createPodsOp, or barrierOp.
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
}

type params struct {
	params map[string]int
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
//		params: map[string]int{
//			"intNodes": 500,
//			"initPods": 50,
//		},
//		isUsed: map[string]bool{}, // empty map
//	}
func (p *params) UnmarshalJSON(b []byte) error {
	aux := map[string]int{}

	if err := json.Unmarshal(b, &aux); err != nil {
		return err
	}

	p.params = aux
	p.isUsed = map[string]bool{}
	return nil
}

// get returns param.
func (p params) get(key string) (int, error) {
	p.isUsed[key] = true
	param, ok := p.params[key]
	if ok {
		return param, nil
	}
	return 0, fmt.Errorf("parameter %s is undefined", key)
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
	possibleOps := []realOp{
		&createAny{},
		&createNodesOp{},
		&createNamespacesOp{},
		&createPodsOp{},
		&createPodSetsOp{},
		&createResourceClaimsOp{},
		&createResourceDriverOp{},
		&churnOp{},
		&barrierOp{},
		&sleepOp{},
		// TODO(#94601): add a delete nodes op to simulate scaling behaviour?
	}
	var firstError error
	for _, possibleOp := range possibleOps {
		if err := json.Unmarshal(b, possibleOp); err == nil {
			if err2 := possibleOp.isValid(true); err2 == nil {
				op.realOp = possibleOp
				return nil
			} else if firstError == nil {
				// Don't return an error yet. Even though this op is invalid, it may
				// still match other possible ops.
				firstError = err2
			}
		}
	}
	return fmt.Errorf("cannot unmarshal %s into any known op type: %w", string(b), firstError)
}

// realOp is an interface that is implemented by different structs. To evaluate
// the validity of ops at parse-time, a isValid function must be implemented.
type realOp interface {
	// isValid verifies the validity of the op args such as node/pod count. Note
	// that we don't catch undefined parameters at this stage.
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

// runnableOp is an interface implemented by some operations. It makes it posssible
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

// createNodesOp defines an op where nodes are created as a part of a workload.
type createNodesOp struct {
	// Must be "createNodes".
	Opcode operationCode
	// Number of nodes to create. Parameterizable through CountParam.
	Count int
	// Template parameter for Count.
	CountParam string
	// Path to spec file describing the nodes to create.
	// Optional
	NodeTemplatePath *string
	// At most one of the following strategies can be defined. Defaults
	// to TrivialNodePrepareStrategy if unspecified.
	// Optional
	NodeAllocatableStrategy  *testutils.NodeAllocatableStrategy
	LabelNodePrepareStrategy *testutils.LabelNodePrepareStrategy
	UniqueNodeLabelStrategy  *testutils.UniqueNodeLabelStrategy
}

func (cno *createNodesOp) isValid(allowParameterization bool) error {
	if cno.Opcode != createNodesOpcode {
		return fmt.Errorf("invalid opcode %q", cno.Opcode)
	}
	if !isValidCount(allowParameterization, cno.Count, cno.CountParam) {
		return fmt.Errorf("invalid Count=%d / CountParam=%q", cno.Count, cno.CountParam)
	}
	return nil
}

func (*createNodesOp) collectsMetrics() bool {
	return false
}

func (cno createNodesOp) patchParams(w *workload) (realOp, error) {
	if cno.CountParam != "" {
		var err error
		cno.Count, err = w.Params.get(cno.CountParam[1:])
		if err != nil {
			return nil, err
		}
	}
	return &cno, (&cno).isValid(false)
}

// createNamespacesOp defines an op for creating namespaces
type createNamespacesOp struct {
	// Must be "createNamespaces".
	Opcode operationCode
	// Name prefix of the Namespace. The format is "<prefix>-<number>", where number is
	// between 0 and count-1.
	Prefix string
	// Number of namespaces to create. Parameterizable through CountParam.
	Count int
	// Template parameter for Count. Takes precedence over Count if both set.
	CountParam string
	// Path to spec file describing the Namespaces to create.
	// Optional
	NamespaceTemplatePath *string
}

func (cmo *createNamespacesOp) isValid(allowParameterization bool) error {
	if cmo.Opcode != createNamespacesOpcode {
		return fmt.Errorf("invalid opcode %q", cmo.Opcode)
	}
	if !isValidCount(allowParameterization, cmo.Count, cmo.CountParam) {
		return fmt.Errorf("invalid Count=%d / CountParam=%q", cmo.Count, cmo.CountParam)
	}
	return nil
}

func (*createNamespacesOp) collectsMetrics() bool {
	return false
}

func (cmo createNamespacesOp) patchParams(w *workload) (realOp, error) {
	if cmo.CountParam != "" {
		var err error
		cmo.Count, err = w.Params.get(cmo.CountParam[1:])
		if err != nil {
			return nil, err
		}
	}
	return &cmo, (&cmo).isValid(false)
}

// createPodsOp defines an op where pods are scheduled as a part of a workload.
// The test can block on the completion of this op before moving forward or
// continue asynchronously.
type createPodsOp struct {
	// Must be "createPods".
	Opcode operationCode
	// Number of pods to schedule. Parameterizable through CountParam.
	Count int
	// Template parameter for Count.
	CountParam string
	// Whether or not to enable metrics collection for this createPodsOp.
	// Optional. Both CollectMetrics and SkipWaitToCompletion cannot be true at
	// the same time for a particular createPodsOp.
	CollectMetrics bool
	// Namespace the pods should be created in. Defaults to a unique
	// namespace of the format "namespace-<number>".
	// Optional
	Namespace *string
	// Path to spec file describing the pods to schedule.
	// If nil, DefaultPodTemplatePath will be used.
	// Optional
	PodTemplatePath *string
	// Whether or not to wait for all pods in this op to get scheduled.
	// Defaults to false if not specified.
	// Optional
	SkipWaitToCompletion bool
	// Persistent volume settings for the pods to be scheduled.
	// Optional
	PersistentVolumeTemplatePath      *string
	PersistentVolumeClaimTemplatePath *string
}

func (cpo *createPodsOp) isValid(allowParameterization bool) error {
	if cpo.Opcode != createPodsOpcode {
		return fmt.Errorf("invalid opcode %q; expected %q", cpo.Opcode, createPodsOpcode)
	}
	if !isValidCount(allowParameterization, cpo.Count, cpo.CountParam) {
		return fmt.Errorf("invalid Count=%d / CountParam=%q", cpo.Count, cpo.CountParam)
	}
	if cpo.CollectMetrics && cpo.SkipWaitToCompletion {
		// While it's technically possible to achieve this, the additional
		// complexity is not worth it, especially given that we don't have any
		// use-cases right now.
		return fmt.Errorf("collectMetrics and skipWaitToCompletion cannot be true at the same time")
	}
	return nil
}

func (cpo *createPodsOp) collectsMetrics() bool {
	return cpo.CollectMetrics
}

func (cpo createPodsOp) patchParams(w *workload) (realOp, error) {
	if cpo.CountParam != "" {
		var err error
		cpo.Count, err = w.Params.get(cpo.CountParam[1:])
		if err != nil {
			return nil, err
		}
	}
	return &cpo, (&cpo).isValid(false)
}

// createPodSetsOp defines an op where a set of createPodsOps is created in each unique namespace.
type createPodSetsOp struct {
	// Must be "createPodSets".
	Opcode operationCode
	// Number of sets to create.
	Count int
	// Template parameter for Count.
	CountParam string
	// Each set of pods will be created in a namespace of the form namespacePrefix-<number>,
	// where number is from 0 to count-1
	NamespacePrefix string
	// The template of a createPodsOp.
	CreatePodsOp createPodsOp
}

func (cpso *createPodSetsOp) isValid(allowParameterization bool) error {
	if cpso.Opcode != createPodSetsOpcode {
		return fmt.Errorf("invalid opcode %q; expected %q", cpso.Opcode, createPodSetsOpcode)
	}
	if !isValidCount(allowParameterization, cpso.Count, cpso.CountParam) {
		return fmt.Errorf("invalid Count=%d / CountParam=%q", cpso.Count, cpso.CountParam)
	}
	return cpso.CreatePodsOp.isValid(allowParameterization)
}

func (cpso *createPodSetsOp) collectsMetrics() bool {
	return cpso.CreatePodsOp.CollectMetrics
}

func (cpso createPodSetsOp) patchParams(w *workload) (realOp, error) {
	if cpso.CountParam != "" {
		var err error
		cpso.Count, err = w.Params.get(cpso.CountParam[1:])
		if err != nil {
			return nil, err
		}
	}
	return &cpso, (&cpso).isValid(true)
}

// churnOp defines an op where services are created as a part of a workload.
type churnOp struct {
	// Must be "churnOp".
	Opcode operationCode
	// Value must be one of the followings:
	// - recreate. In this mode, API objects will be created for N cycles, and then
	//   deleted in the next N cycles. N is specified by the "Number" field.
	// - create. In this mode, API objects will be created (without deletion) until
	//   reaching a threshold - which is specified by the "Number" field.
	Mode string
	// Maximum number of API objects to be created.
	// Defaults to 0, which means unlimited.
	Number int
	// Intervals of churning. Defaults to 500 millisecond.
	IntervalMilliseconds int64
	// Namespace the churning objects should be created in. Defaults to a unique
	// namespace of the format "namespace-<number>".
	// Optional
	Namespace *string
	// Path of API spec files.
	TemplatePaths []string
}

func (co *churnOp) isValid(_ bool) error {
	if co.Opcode != churnOpcode {
		return fmt.Errorf("invalid opcode %q", co.Opcode)
	}
	if co.Mode != Recreate && co.Mode != Create {
		return fmt.Errorf("invalid mode: %v. must be one of %v", co.Mode, []string{Recreate, Create})
	}
	if co.Number < 0 {
		return fmt.Errorf("number (%v) cannot be negative", co.Number)
	}
	if co.Mode == Recreate && co.Number == 0 {
		return fmt.Errorf("number cannot be 0 when mode is %v", Recreate)
	}
	if len(co.TemplatePaths) == 0 {
		return fmt.Errorf("at least one template spec file needs to be specified")
	}
	return nil
}

func (*churnOp) collectsMetrics() bool {
	return false
}

func (co churnOp) patchParams(w *workload) (realOp, error) {
	return &co, nil
}

// barrierOp defines an op that can be used to wait until all scheduled pods of
// one or many namespaces have been bound to nodes. This is useful when pods
// were scheduled with SkipWaitToCompletion set to true.
type barrierOp struct {
	// Must be "barrier".
	Opcode operationCode
	// Namespaces to block on. Empty array or not specifying this field signifies
	// that the barrier should block on all namespaces.
	Namespaces []string
}

func (bo *barrierOp) isValid(allowParameterization bool) error {
	if bo.Opcode != barrierOpcode {
		return fmt.Errorf("invalid opcode %q", bo.Opcode)
	}
	return nil
}

func (*barrierOp) collectsMetrics() bool {
	return false
}

func (bo barrierOp) patchParams(w *workload) (realOp, error) {
	return &bo, nil
}

// sleepOp defines an op that can be used to sleep for a specified amount of time.
// This is useful in simulating workloads that require some sort of time-based synchronisation.
type sleepOp struct {
	// Must be "sleep".
	Opcode operationCode
	// duration of sleep.
	Duration time.Duration
}

func (so *sleepOp) UnmarshalJSON(data []byte) (err error) {
	var tmp struct {
		Opcode   operationCode
		Duration string
	}
	if err = json.Unmarshal(data, &tmp); err != nil {
		return err
	}

	so.Opcode = tmp.Opcode
	so.Duration, err = time.ParseDuration(tmp.Duration)
	return err
}

func (so *sleepOp) isValid(_ bool) error {
	if so.Opcode != sleepOpcode {
		return fmt.Errorf("invalid opcode %q; expected %q", so.Opcode, sleepOpcode)
	}
	return nil
}

func (so *sleepOp) collectsMetrics() bool {
	return false
}

func (so sleepOp) patchParams(_ *workload) (realOp, error) {
	return &so, nil
}

var useTestingLog = flag.Bool("use-testing-log", false, "Write log entries with testing.TB.Log. This is more suitable for unit testing and debugging, but less realistic in real benchmarks.")

func initTestOutput(tb testing.TB) io.Writer {
	var output io.Writer
	if *useTestingLog {
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

var perfSchedulingLabelFilter = flag.String("perf-scheduling-label-filter", "performance", "comma-separated list of labels which a testcase must have (no prefix or +) or must not have (-), used by BenchmarkPerfScheduling")

// RunBenchmarkPerfScheduling runs the scheduler performance tests.
// Optionally, you can pass your own scheduler plugin via outOfTreePluginRegistry.
func RunBenchmarkPerfScheduling(b *testing.B, outOfTreePluginRegistry frameworkruntime.Registry) {
	testCases, err := getTestCases(configFile)
	if err != nil {
		b.Fatal(err)
	}
	if err = validateTestCases(testCases); err != nil {
		b.Fatal(err)
	}

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
			for _, w := range tc.Workloads {
				b.Run(w.Name, func(b *testing.B) {
					if !enabled(*perfSchedulingLabelFilter, append(tc.Labels, w.Labels...)...) {
						b.Skipf("disabled by label filter %q", *perfSchedulingLabelFilter)
					}
					tCtx := ktesting.Init(b, initoption.PerTestOutput(*useTestingLog))

					// Ensure that there are no leaked
					// goroutines.  They could influence
					// performance of the next benchmark.
					// This must *after* RedirectKlog
					// because then during cleanup, the
					// test will wait for goroutines to
					// quit *before* restoring klog settings.
					framework.GoleakCheck(b)

					// Now that we are ready to run, start
					// etcd.
					framework.StartEtcd(b, output)

					// 30 minutes should be plenty enough even for the 5000-node tests.
					timeout := 30 * time.Minute
					tCtx = ktesting.WithTimeout(tCtx, timeout, fmt.Sprintf("timed out after the %s per-test timeout", timeout))

					for feature, flag := range tc.FeatureGates {
						defer featuregatetesting.SetFeatureGateDuringTest(b, utilfeature.DefaultFeatureGate, feature, flag)()
					}
					informerFactory, tCtx := setupClusterForWorkload(tCtx, tc.SchedulerConfigPath, tc.FeatureGates, outOfTreePluginRegistry)
					results := runWorkload(tCtx, tc, w, informerFactory, false)
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

					// Reset metrics to prevent metrics generated in current workload gets
					// carried over to the next workload.
					legacyregistry.Reset()
				})
			}
		})
	}
	if err := dataItems2JSONFile(dataItems, b.Name()+"_benchmark"); err != nil {
		b.Fatalf("unable to write measured data %+v: %v", dataItems, err)
	}
}

var testSchedulingLabelFilter = flag.String("test-scheduling-label-filter", "integration-test", "comma-separated list of labels which a testcase must have (no prefix or +) or must not have (-), used by TestScheduling")

type schedulerConfig struct {
	schedulerConfigPath string
	featureGates        map[featuregate.Feature]bool
}

func (c schedulerConfig) equals(tc *testCase) bool {
	return c.schedulerConfigPath == tc.SchedulerConfigPath &&
		cmp.Equal(c.featureGates, tc.FeatureGates)
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

func setupClusterForWorkload(tCtx ktesting.TContext, configPath string, featureGates map[featuregate.Feature]bool, outOfTreePluginRegistry frameworkruntime.Registry) (informers.SharedInformerFactory, ktesting.TContext) {
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

func runWorkload(tCtx ktesting.TContext, tc *testCase, w *workload, informerFactory informers.SharedInformerFactory, cleanup bool) []DataItem {
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

	// Everything else started by this function gets stopped before it returns.
	tCtx = ktesting.WithCancel(tCtx)
	var wg sync.WaitGroup
	defer wg.Wait()
	defer tCtx.Cancel("workload is done")

	var mu sync.Mutex
	var dataItems []DataItem
	nextNodeIndex := 0
	// numPodsScheduledPerNamespace has all namespaces created in workload and the number of pods they (will) have.
	// All namespaces listed in numPodsScheduledPerNamespace will be cleaned up.
	numPodsScheduledPerNamespace := make(map[string]int)

	if cleanup {
		// This must run before controllers get shut down.
		defer cleanupWorkload(tCtx, tc, numPodsScheduledPerNamespace)
	}

	for opIndex, op := range unrollWorkloadTemplate(tCtx, tc.WorkloadTemplate, w) {
		realOp, err := op.realOp.patchParams(w)
		if err != nil {
			tCtx.Fatalf("op %d: %v", opIndex, err)
		}
		select {
		case <-tCtx.Done():
			tCtx.Fatalf("op %d: %v", opIndex, context.Cause(tCtx))
		default:
		}
		switch concreteOp := realOp.(type) {
		case *createNodesOp:
			nodePreparer, err := getNodePreparer(fmt.Sprintf("node-%d-", opIndex), concreteOp, tCtx.Client())
			if err != nil {
				tCtx.Fatalf("op %d: %v", opIndex, err)
			}
			if err := nodePreparer.PrepareNodes(tCtx, nextNodeIndex); err != nil {
				tCtx.Fatalf("op %d: %v", opIndex, err)
			}
			if cleanup {
				defer func() {
					if err := nodePreparer.CleanupNodes(tCtx); err != nil {
						tCtx.Fatalf("failed to clean up nodes, error: %v", err)
					}
				}()
			}
			nextNodeIndex += concreteOp.Count

		case *createNamespacesOp:
			nsPreparer, err := newNamespacePreparer(tCtx, concreteOp)
			if err != nil {
				tCtx.Fatalf("op %d: %v", opIndex, err)
			}
			if err := nsPreparer.prepare(tCtx); err != nil {
				err2 := nsPreparer.cleanup(tCtx)
				if err2 != nil {
					err = fmt.Errorf("prepare: %v; cleanup: %v", err, err2)
				}
				tCtx.Fatalf("op %d: %v", opIndex, err)
			}
			for _, n := range nsPreparer.namespaces() {
				if _, ok := numPodsScheduledPerNamespace[n]; ok {
					// this namespace has been already created.
					continue
				}
				numPodsScheduledPerNamespace[n] = 0
			}

		case *createPodsOp:
			var namespace string
			// define Pod's namespace automatically, and create that namespace.
			namespace = fmt.Sprintf("namespace-%d", opIndex)
			if concreteOp.Namespace != nil {
				namespace = *concreteOp.Namespace
			}
			createNamespaceIfNotPresent(tCtx, namespace, &numPodsScheduledPerNamespace)
			if concreteOp.PodTemplatePath == nil {
				concreteOp.PodTemplatePath = tc.DefaultPodTemplatePath
			}
			var collectors []testDataCollector
			// This needs a separate context and wait group because
			// the code below needs to be sure that the goroutines
			// are stopped.
			var collectorCtx ktesting.TContext
			var collectorWG sync.WaitGroup
			defer collectorWG.Wait()

			if concreteOp.CollectMetrics {
				collectorCtx = ktesting.WithCancel(tCtx)
				defer collectorCtx.Cancel("cleaning up")
				name := tCtx.Name()
				// The first part is the same for each work load, therefore we can strip it.
				name = name[strings.Index(name, "/")+1:]
				collectors = getTestDataCollectors(collectorCtx, podInformer, fmt.Sprintf("%s/%s", name, namespace), namespace, tc.MetricsCollectorConfig, throughputErrorMargin)
				for _, collector := range collectors {
					// Need loop-local variable for function below.
					collector := collector
					collectorWG.Add(1)
					go func() {
						defer collectorWG.Done()
						collector.run(collectorCtx)
					}()
				}
			}
			if err := createPods(tCtx, namespace, concreteOp); err != nil {
				tCtx.Fatalf("op %d: %v", opIndex, err)
			}
			if concreteOp.SkipWaitToCompletion {
				// Only record those namespaces that may potentially require barriers
				// in the future.
				numPodsScheduledPerNamespace[namespace] += concreteOp.Count
			} else {
				if err := waitUntilPodsScheduledInNamespace(tCtx, podInformer, namespace, concreteOp.Count); err != nil {
					tCtx.Fatalf("op %d: error in waiting for pods to get scheduled: %v", opIndex, err)
				}
			}
			if concreteOp.CollectMetrics {
				// CollectMetrics and SkipWaitToCompletion can never be true at the
				// same time, so if we're here, it means that all pods have been
				// scheduled.
				collectorCtx.Cancel("collecting metrix, collector must stop first")
				collectorWG.Wait()
				mu.Lock()
				for _, collector := range collectors {
					dataItems = append(dataItems, collector.collect()...)
				}
				mu.Unlock()
			}

			if !concreteOp.SkipWaitToCompletion {
				// SkipWaitToCompletion=false indicates this step has waited for the Pods to be scheduled.
				// So we reset the metrics in global registry; otherwise metrics gathered in this step
				// will be carried over to next step.
				legacyregistry.Reset()
			}

		case *churnOp:
			var namespace string
			if concreteOp.Namespace != nil {
				namespace = *concreteOp.Namespace
			} else {
				namespace = fmt.Sprintf("namespace-%d", opIndex)
			}
			restMapper := restmapper.NewDeferredDiscoveryRESTMapper(cacheddiscovery.NewMemCacheClient(tCtx.Client().Discovery()))
			// Ensure the namespace exists.
			nsObj := &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: namespace}}
			if _, err := tCtx.Client().CoreV1().Namespaces().Create(tCtx, nsObj, metav1.CreateOptions{}); err != nil && !apierrors.IsAlreadyExists(err) {
				tCtx.Fatalf("op %d: unable to create namespace %v: %v", opIndex, namespace, err)
			}

			var churnFns []func(name string) string

			for i, path := range concreteOp.TemplatePaths {
				unstructuredObj, gvk, err := getUnstructuredFromFile(path)
				if err != nil {
					tCtx.Fatalf("op %d: unable to parse the %v-th template path: %v", opIndex, i, err)
				}
				// Obtain GVR.
				mapping, err := restMapper.RESTMapping(gvk.GroupKind(), gvk.Version)
				if err != nil {
					tCtx.Fatalf("op %d: unable to find GVR for %v: %v", opIndex, gvk, err)
				}
				gvr := mapping.Resource
				// Distinguish cluster-scoped with namespaced API objects.
				var dynRes dynamic.ResourceInterface
				if mapping.Scope.Name() == meta.RESTScopeNameNamespace {
					dynRes = tCtx.Dynamic().Resource(gvr).Namespace(namespace)
				} else {
					dynRes = tCtx.Dynamic().Resource(gvr)
				}

				churnFns = append(churnFns, func(name string) string {
					if name != "" {
						if err := dynRes.Delete(tCtx, name, metav1.DeleteOptions{}); err != nil {
							tCtx.Errorf("op %d: unable to delete %v: %v", opIndex, name, err)
						}
						return ""
					}

					live, err := dynRes.Create(tCtx, unstructuredObj, metav1.CreateOptions{})
					if err != nil {
						return ""
					}
					return live.GetName()
				})
			}

			var interval int64 = 500
			if concreteOp.IntervalMilliseconds != 0 {
				interval = concreteOp.IntervalMilliseconds
			}
			ticker := time.NewTicker(time.Duration(interval) * time.Millisecond)
			defer ticker.Stop()

			switch concreteOp.Mode {
			case Create:
				wg.Add(1)
				go func() {
					defer wg.Done()
					count, threshold := 0, concreteOp.Number
					if threshold == 0 {
						threshold = math.MaxInt32
					}
					for count < threshold {
						select {
						case <-ticker.C:
							for i := range churnFns {
								churnFns[i]("")
							}
							count++
						case <-tCtx.Done():
							return
						}
					}
				}()
			case Recreate:
				wg.Add(1)
				go func() {
					defer wg.Done()
					retVals := make([][]string, len(churnFns))
					// For each churn function, instantiate a slice of strings with length "concreteOp.Number".
					for i := range retVals {
						retVals[i] = make([]string, concreteOp.Number)
					}

					count := 0
					for {
						select {
						case <-ticker.C:
							for i := range churnFns {
								retVals[i][count%concreteOp.Number] = churnFns[i](retVals[i][count%concreteOp.Number])
							}
							count++
						case <-tCtx.Done():
							return
						}
					}
				}()
			}

		case *barrierOp:
			for _, namespace := range concreteOp.Namespaces {
				if _, ok := numPodsScheduledPerNamespace[namespace]; !ok {
					tCtx.Fatalf("op %d: unknown namespace %s", opIndex, namespace)
				}
			}
			if err := waitUntilPodsScheduled(tCtx, podInformer, concreteOp.Namespaces, numPodsScheduledPerNamespace); err != nil {
				tCtx.Fatalf("op %d: %v", opIndex, err)
			}
			// At the end of the barrier, we can be sure that there are no pods
			// pending scheduling in the namespaces that we just blocked on.
			if len(concreteOp.Namespaces) == 0 {
				numPodsScheduledPerNamespace = make(map[string]int)
			} else {
				for _, namespace := range concreteOp.Namespaces {
					delete(numPodsScheduledPerNamespace, namespace)
				}
			}

		case *sleepOp:
			select {
			case <-tCtx.Done():
			case <-time.After(concreteOp.Duration):
			}
		default:
			runable, ok := concreteOp.(runnableOp)
			if !ok {
				tCtx.Fatalf("op %d: invalid op %v", opIndex, concreteOp)
			}
			for _, namespace := range runable.requiredNamespaces() {
				createNamespaceIfNotPresent(tCtx, namespace, &numPodsScheduledPerNamespace)
			}
			runable.run(tCtx)
		}
	}

	// check unused params and inform users
	unusedParams := w.unusedParams()
	if len(unusedParams) != 0 {
		tCtx.Fatalf("the parameters %v are defined on workload %s, but unused.\nPlease make sure there are no typos.", unusedParams, w.Name)
	}

	// Some tests have unschedulable pods. Do not add an implicit barrier at the
	// end as we do not want to wait for them.
	return dataItems
}

// cleanupWorkload ensures that everything is removed from the API server that
// might have been created by runWorkload. This must be done before starting
// the next workload because otherwise it might stumble over previously created
// objects. For example, the namespaces are the same in different workloads, so
// not deleting them would cause the next one to fail with "cannot create
// namespace: already exists".
//
// Calling cleanupWorkload can be skipped if it is known that the next workload
// will run with a fresh etcd instance.
func cleanupWorkload(tCtx ktesting.TContext, tc *testCase, numPodsScheduledPerNamespace map[string]int) {
	deleteNow := *metav1.NewDeleteOptions(0)
	for namespace := range numPodsScheduledPerNamespace {
		// Pods have to be deleted explicitly, with no grace period. Normally
		// kubelet will set the DeletionGracePeriodSeconds to zero when it's okay
		// to remove a deleted pod, but we don't run kubelet...
		if err := tCtx.Client().CoreV1().Pods(namespace).DeleteCollection(tCtx, deleteNow, metav1.ListOptions{}); err != nil {
			tCtx.Fatalf("failed to delete pods in namespace %q: %v", namespace, err)
		}
		if err := tCtx.Client().CoreV1().Namespaces().Delete(tCtx, namespace, deleteNow); err != nil {
			tCtx.Fatalf("Deleting Namespace %q in numPodsScheduledPerNamespace: %v", namespace, err)
		}
	}

	// We need to wait here because even with deletion timestamp set,
	// actually removing a namespace can take some time (garbage collecting
	// other generated object like secrets, etc.) and we don't want to
	// start the next workloads while that cleanup is still going on.
	if err := wait.PollUntilContextTimeout(tCtx, time.Second, 5*time.Minute, false, func(ctx context.Context) (bool, error) {
		namespaces, err := tCtx.Client().CoreV1().Namespaces().List(ctx, metav1.ListOptions{})
		if err != nil {
			return false, err
		}
		for _, namespace := range namespaces.Items {
			if _, ok := numPodsScheduledPerNamespace[namespace.Name]; ok {
				// A namespace created by the workload, need to wait.
				return false, nil
			}
		}
		// All namespaces gone.
		return true, nil
	}); err != nil {
		tCtx.Fatalf("failed while waiting for namespace removal: %v", err)
	}
}

func createNamespaceIfNotPresent(tCtx ktesting.TContext, namespace string, podsPerNamespace *map[string]int) {
	if _, ok := (*podsPerNamespace)[namespace]; !ok {
		// The namespace has not created yet.
		// So, create that and register it.
		_, err := tCtx.Client().CoreV1().Namespaces().Create(tCtx, &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: namespace}}, metav1.CreateOptions{})
		if err != nil {
			tCtx.Fatalf("failed to create namespace for Pod: %v", namespace)
		}
		(*podsPerNamespace)[namespace] = 0
	}
}

type testDataCollector interface {
	run(tCtx ktesting.TContext)
	collect() []DataItem
}

func getTestDataCollectors(tCtx ktesting.TContext, podInformer coreinformers.PodInformer, name, namespace string, mcc *metricsCollectorConfig, throughputErrorMargin float64) []testDataCollector {
	if mcc == nil {
		mcc = &defaultMetricsCollectorConfig
	}
	return []testDataCollector{
		newThroughputCollector(tCtx, podInformer, map[string]string{"Name": name}, []string{namespace}, throughputErrorMargin),
		newMetricsCollector(mcc, map[string]string{"Name": name}),
	}
}

func getNodePreparer(prefix string, cno *createNodesOp, clientset clientset.Interface) (testutils.TestNodePreparer, error) {
	var nodeStrategy testutils.PrepareNodeStrategy = &testutils.TrivialNodePrepareStrategy{}
	if cno.NodeAllocatableStrategy != nil {
		nodeStrategy = cno.NodeAllocatableStrategy
	} else if cno.LabelNodePrepareStrategy != nil {
		nodeStrategy = cno.LabelNodePrepareStrategy
	} else if cno.UniqueNodeLabelStrategy != nil {
		nodeStrategy = cno.UniqueNodeLabelStrategy
	}

	if cno.NodeTemplatePath != nil {
		node, err := getNodeSpecFromFile(cno.NodeTemplatePath)
		if err != nil {
			return nil, err
		}
		return framework.NewIntegrationTestNodePreparerWithNodeSpec(
			clientset,
			[]testutils.CountToStrategy{{Count: cno.Count, Strategy: nodeStrategy}},
			node,
		), nil
	}
	return framework.NewIntegrationTestNodePreparer(
		clientset,
		[]testutils.CountToStrategy{{Count: cno.Count, Strategy: nodeStrategy}},
		prefix,
	), nil
}

func createPods(tCtx ktesting.TContext, namespace string, cpo *createPodsOp) error {
	strategy, err := getPodStrategy(cpo)
	if err != nil {
		return err
	}
	tCtx.Logf("creating %d pods in namespace %q", cpo.Count, namespace)
	config := testutils.NewTestPodCreatorConfig()
	config.AddStrategy(namespace, cpo.Count, strategy)
	podCreator := testutils.NewTestPodCreator(tCtx.Client(), config)
	return podCreator.CreatePods(tCtx)
}

// waitUntilPodsScheduledInNamespace blocks until all pods in the given
// namespace are scheduled. Times out after 10 minutes because even at the
// lowest observed QPS of ~10 pods/sec, a 5000-node test should complete.
func waitUntilPodsScheduledInNamespace(tCtx ktesting.TContext, podInformer coreinformers.PodInformer, namespace string, wantCount int) error {
	var pendingPod *v1.Pod

	err := wait.PollUntilContextTimeout(tCtx, 1*time.Second, 10*time.Minute, true, func(ctx context.Context) (bool, error) {
		select {
		case <-ctx.Done():
			return true, ctx.Err()
		default:
		}
		scheduled, unscheduled, err := getScheduledPods(podInformer, namespace)
		if err != nil {
			return false, err
		}
		if len(scheduled) >= wantCount {
			tCtx.Logf("scheduling succeed")
			return true, nil
		}
		tCtx.Logf("namespace: %s, pods: want %d, got %d", namespace, wantCount, len(scheduled))
		if len(unscheduled) > 0 {
			pendingPod = unscheduled[0]
		} else {
			pendingPod = nil
		}
		return false, nil
	})

	if err != nil && pendingPod != nil {
		err = fmt.Errorf("at least pod %s is not scheduled: %v", klog.KObj(pendingPod), err)
	}
	return err
}

// waitUntilPodsScheduled blocks until the all pods in the given namespaces are
// scheduled.
func waitUntilPodsScheduled(tCtx ktesting.TContext, podInformer coreinformers.PodInformer, namespaces []string, numPodsScheduledPerNamespace map[string]int) error {
	// If unspecified, default to all known namespaces.
	if len(namespaces) == 0 {
		for namespace := range numPodsScheduledPerNamespace {
			namespaces = append(namespaces, namespace)
		}
	}
	for _, namespace := range namespaces {
		select {
		case <-tCtx.Done():
			return context.Cause(tCtx)
		default:
		}
		wantCount, ok := numPodsScheduledPerNamespace[namespace]
		if !ok {
			return fmt.Errorf("unknown namespace %s", namespace)
		}
		if err := waitUntilPodsScheduledInNamespace(tCtx, podInformer, namespace, wantCount); err != nil {
			return fmt.Errorf("error waiting for pods in namespace %q: %w", namespace, err)
		}
	}
	return nil
}

func getSpecFromFile(path *string, spec interface{}) error {
	bytes, err := os.ReadFile(*path)
	if err != nil {
		return err
	}
	return yaml.UnmarshalStrict(bytes, spec)
}

func getUnstructuredFromFile(path string) (*unstructured.Unstructured, *schema.GroupVersionKind, error) {
	bytes, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}

	bytes, err = yaml.YAMLToJSONStrict(bytes)
	if err != nil {
		return nil, nil, fmt.Errorf("cannot covert YAML to JSON: %v", err)
	}

	obj, gvk, err := unstructured.UnstructuredJSONScheme.Decode(bytes, nil, nil)
	if err != nil {
		return nil, nil, err
	}
	unstructuredObj, ok := obj.(*unstructured.Unstructured)
	if !ok {
		return nil, nil, fmt.Errorf("cannot convert spec file in %v to an unstructured obj", path)
	}
	return unstructuredObj, gvk, nil
}

func getTestCases(path string) ([]*testCase, error) {
	testCases := make([]*testCase, 0)
	if err := getSpecFromFile(&path, &testCases); err != nil {
		return nil, fmt.Errorf("parsing test cases error: %w", err)
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
		// true in each workload. What's the point of running a performance
		// benchmark if no statistics are collected for reporting?
		if !tc.collectsMetrics() {
			return fmt.Errorf("%s: no op in the workload template collects metrics", tc.Name)
		}
		// TODO(#93795): make sure each workload within a test case has a unique
		// name? The name is used to identify the stats in benchmark reports.
		// TODO(#94404): check for unused template parameters? Probably a typo.
	}
	return nil
}

func getPodStrategy(cpo *createPodsOp) (testutils.TestPodCreateStrategy, error) {
	basePod := makeBasePod()
	if cpo.PodTemplatePath != nil {
		var err error
		basePod, err = getPodSpecFromFile(cpo.PodTemplatePath)
		if err != nil {
			return nil, err
		}
	}
	if cpo.PersistentVolumeClaimTemplatePath == nil {
		return testutils.NewCustomCreatePodStrategy(basePod), nil
	}

	pvTemplate, err := getPersistentVolumeSpecFromFile(cpo.PersistentVolumeTemplatePath)
	if err != nil {
		return nil, err
	}
	pvcTemplate, err := getPersistentVolumeClaimSpecFromFile(cpo.PersistentVolumeClaimTemplatePath)
	if err != nil {
		return nil, err
	}
	return testutils.NewCreatePodWithPersistentVolumeStrategy(pvcTemplate, getCustomVolumeFactory(pvTemplate), basePod), nil
}

func getNodeSpecFromFile(path *string) (*v1.Node, error) {
	nodeSpec := &v1.Node{}
	if err := getSpecFromFile(path, nodeSpec); err != nil {
		return nil, fmt.Errorf("parsing Node: %w", err)
	}
	return nodeSpec, nil
}

func getPodSpecFromFile(path *string) (*v1.Pod, error) {
	podSpec := &v1.Pod{}
	if err := getSpecFromFile(path, podSpec); err != nil {
		return nil, fmt.Errorf("parsing Pod: %w", err)
	}
	return podSpec, nil
}

func getPersistentVolumeSpecFromFile(path *string) (*v1.PersistentVolume, error) {
	persistentVolumeSpec := &v1.PersistentVolume{}
	if err := getSpecFromFile(path, persistentVolumeSpec); err != nil {
		return nil, fmt.Errorf("parsing PersistentVolume: %w", err)
	}
	return persistentVolumeSpec, nil
}

func getPersistentVolumeClaimSpecFromFile(path *string) (*v1.PersistentVolumeClaim, error) {
	persistentVolumeClaimSpec := &v1.PersistentVolumeClaim{}
	if err := getSpecFromFile(path, persistentVolumeClaimSpec); err != nil {
		return nil, fmt.Errorf("parsing PersistentVolumeClaim: %w", err)
	}
	return persistentVolumeClaimSpec, nil
}

func getCustomVolumeFactory(pvTemplate *v1.PersistentVolume) func(id int) *v1.PersistentVolume {
	return func(id int) *v1.PersistentVolume {
		pv := pvTemplate.DeepCopy()
		volumeID := fmt.Sprintf("vol-%d", id)
		pv.ObjectMeta.Name = volumeID
		pvs := pv.Spec.PersistentVolumeSource
		if pvs.CSI != nil {
			pvs.CSI.VolumeHandle = volumeID
		} else if pvs.AWSElasticBlockStore != nil {
			pvs.AWSElasticBlockStore.VolumeID = volumeID
		}
		return pv
	}
}

// namespacePreparer holds configuration information for the test namespace preparer.
type namespacePreparer struct {
	count  int
	prefix string
	spec   *v1.Namespace
}

func newNamespacePreparer(tCtx ktesting.TContext, cno *createNamespacesOp) (*namespacePreparer, error) {
	ns := &v1.Namespace{}
	if cno.NamespaceTemplatePath != nil {
		if err := getSpecFromFile(cno.NamespaceTemplatePath, ns); err != nil {
			return nil, fmt.Errorf("parsing NamespaceTemplate: %w", err)
		}
	}

	return &namespacePreparer{
		count:  cno.Count,
		prefix: cno.Prefix,
		spec:   ns,
	}, nil
}

// namespaces returns namespace names have been (or will be) created by this namespacePreparer
func (p *namespacePreparer) namespaces() []string {
	namespaces := make([]string, p.count)
	for i := 0; i < p.count; i++ {
		namespaces[i] = fmt.Sprintf("%s-%d", p.prefix, i)
	}
	return namespaces
}

// prepare creates the namespaces.
func (p *namespacePreparer) prepare(tCtx ktesting.TContext) error {
	base := &v1.Namespace{}
	if p.spec != nil {
		base = p.spec
	}
	tCtx.Logf("Making %d namespaces with prefix %q and template %v", p.count, p.prefix, *base)
	for i := 0; i < p.count; i++ {
		n := base.DeepCopy()
		n.Name = fmt.Sprintf("%s-%d", p.prefix, i)
		if err := testutils.RetryWithExponentialBackOff(func() (bool, error) {
			_, err := tCtx.Client().CoreV1().Namespaces().Create(tCtx, n, metav1.CreateOptions{})
			return err == nil || apierrors.IsAlreadyExists(err), nil
		}); err != nil {
			return err
		}
	}
	return nil
}

// cleanup deletes existing test namespaces.
func (p *namespacePreparer) cleanup(tCtx ktesting.TContext) error {
	var errRet error
	for i := 0; i < p.count; i++ {
		n := fmt.Sprintf("%s-%d", p.prefix, i)
		if err := tCtx.Client().CoreV1().Namespaces().Delete(tCtx, n, metav1.DeleteOptions{}); err != nil {
			tCtx.Errorf("Deleting Namespace: %v", err)
			errRet = err
		}
	}
	return errRet
}
