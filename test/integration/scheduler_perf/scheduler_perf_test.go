/*
Copyright 2019 The Kubernetes Authors.

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
	"fmt"
	"math"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	cacheddiscovery "k8s.io/client-go/discovery/cached"
	"k8s.io/client-go/dynamic"
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
	"k8s.io/kubernetes/test/integration/framework"
	testutils "k8s.io/kubernetes/test/utils"
	"sigs.k8s.io/yaml"
)

const (
	configFile               = "config/performance-config.yaml"
	createNodesOpcode        = "createNodes"
	createNamespacesOpcode   = "createNamespaces"
	createPodsOpcode         = "createPods"
	createPodSetsOpcode      = "createPodSets"
	churnOpcode              = "churn"
	barrierOpcode            = "barrier"
	sleepOpcode              = "sleep"
	extensionPointsLabelName = "extension_point"

	// Two modes supported in "churn" operator.

	// Recreate creates a number of API objects and then delete them, and repeat the iteration.
	Recreate = "recreate"
	// Create continuously create API objects without deleting them.
	Create = "create"
)

var (
	defaultMetricsCollectorConfig = metricsCollectorConfig{
		Metrics: map[string]*labelValues{
			"scheduler_framework_extension_point_duration_seconds": {
				label:  extensionPointsLabelName,
				values: []string{"Filter", "Score"},
			},
			"scheduler_scheduling_attempt_duration_seconds": nil,
			"scheduler_pod_scheduling_duration_seconds":     nil,
		},
	}
)

// testCase defines a set of test cases that intend to test the performance of
// similar workloads of varying sizes with shared overall settings such as
// feature gates and metrics collected.
type testCase struct {
	// Name of the testCase.
	Name string
	// Feature gates to set before running the test. Optional.
	FeatureGates map[featuregate.Feature]bool
	// List of metrics to collect. Optional, defaults to
	// defaultMetricsCollectorConfig if unspecified.
	MetricsCollectorConfig *metricsCollectorConfig
	// Template for sequence of ops that each workload must follow. Each op will
	// be executed serially one after another. Each element of the list must be
	// createNodesOp, createPodsOp, or barrierOp.
	WorkloadTemplate []op
	// List of workloads to run under this testCase.
	Workloads []*workload
	// SchedulerConfigFile is the path of scheduler configuration
	SchedulerConfigFile string
	// Default path to spec file describing the pods to create. Optional.
	// This path can be overridden in createPodsOp by setting PodTemplatePath .
	DefaultPodTemplatePath *string
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
}

type params struct {
	params map[string]int
	// isUsed field records whether params is used or not.
	isUsed map[string]bool
}

// UnmarshalJSON is a custom unmarshaler for params.
//
// from(json):
// 	{
// 		"initNodes": 500,
// 		"initPods": 50
// 	}
//
// to:
//	params{
//		params: map[string]int{
//			"intNodes": 500,
//			"initPods": 50,
//		},
//		isUsed: map[string]bool{}, // empty map
//	}
//
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
		&createNodesOp{},
		&createNamespacesOp{},
		&createPodsOp{},
		&createPodSetsOp{},
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

func isValidParameterizable(val string) bool {
	return strings.HasPrefix(val, "$")
}

// createNodesOp defines an op where nodes are created as a part of a workload.
type createNodesOp struct {
	// Must be "createNodes".
	Opcode string
	// Number of nodes to create. Parameterizable through CountParam.
	Count int
	// Template parameter for Count.
	CountParam string
	// Path to spec file describing the nodes to create. Optional.
	NodeTemplatePath *string
	// At most one of the following strategies can be defined. Optional, defaults
	// to TrivialNodePrepareStrategy if unspecified.
	NodeAllocatableStrategy  *testutils.NodeAllocatableStrategy
	LabelNodePrepareStrategy *testutils.LabelNodePrepareStrategy
	UniqueNodeLabelStrategy  *testutils.UniqueNodeLabelStrategy
}

func (cno *createNodesOp) isValid(allowParameterization bool) error {
	if cno.Opcode != createNodesOpcode {
		return fmt.Errorf("invalid opcode %q", cno.Opcode)
	}
	ok := cno.Count > 0 ||
		(cno.CountParam != "" && allowParameterization && isValidParameterizable(cno.CountParam))
	if !ok {
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
	Opcode string
	// Name prefix of the Namespace. The format is "<prefix>-<number>", where number is
	// between 0 and count-1.
	Prefix string
	// Number of namespaces to create. Parameterizable through CountParam.
	Count int
	// Template parameter for Count. Takes precedence over Count if both set.
	CountParam string
	// Path to spec file describing the Namespaces to create. Optional.
	NamespaceTemplatePath *string
}

func (cmo *createNamespacesOp) isValid(allowParameterization bool) error {
	if cmo.Opcode != createNamespacesOpcode {
		return fmt.Errorf("invalid opcode %q", cmo.Opcode)
	}
	ok := cmo.Count > 0 ||
		(cmo.CountParam != "" && allowParameterization && isValidParameterizable(cmo.CountParam))
	if !ok {
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
	Opcode string
	// Number of pods to schedule. Parameterizable through CountParam.
	Count int
	// Template parameter for Count.
	CountParam string
	// Whether or not to enable metrics collection for this createPodsOp.
	// Optional. Both CollectMetrics and SkipWaitToCompletion cannot be true at
	// the same time for a particular createPodsOp.
	CollectMetrics bool
	// Namespace the pods should be created in. Optional, defaults to a unique
	// namespace of the format "namespace-<number>".
	Namespace *string
	// Path to spec file describing the pods to schedule. Optional.
	// If nil, DefaultPodTemplatePath will be used.
	PodTemplatePath *string
	// Whether or not to wait for all pods in this op to get scheduled. Optional,
	// defaults to false.
	SkipWaitToCompletion bool
	// Persistent volume settings for the pods to be scheduled. Optional.
	PersistentVolumeTemplatePath      *string
	PersistentVolumeClaimTemplatePath *string
}

func (cpo *createPodsOp) isValid(allowParameterization bool) error {
	if cpo.Opcode != createPodsOpcode {
		return fmt.Errorf("invalid opcode %q; expected %q", cpo.Opcode, createPodsOpcode)
	}
	ok := cpo.Count > 0 ||
		(cpo.CountParam != "" && allowParameterization && isValidParameterizable(cpo.CountParam))
	if !ok {
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

// createPodSetsOp defines an op where a set of createPodsOp is created each in a unique namespace.
type createPodSetsOp struct {
	// Must be "createPodSets".
	Opcode string
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
	ok := cpso.Count > 0 ||
		(cpso.CountParam != "" && allowParameterization && isValidParameterizable(cpso.CountParam))
	if !ok {
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
	Opcode string
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
	// Namespace the churning objects should be created in. Optional, defaults to a unique
	// namespace of the format "namespace-<number>".
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
	Opcode string
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
	Opcode string
	// duration of sleep.
	Duration time.Duration
}

func (so *sleepOp) UnmarshalJSON(data []byte) (err error) {
	var tmp struct {
		Opcode   string
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

func BenchmarkPerfScheduling(b *testing.B) {
	testCases, err := getTestCases(configFile)
	if err != nil {
		b.Fatal(err)
	}
	if err = validateTestCases(testCases); err != nil {
		b.Fatal(err)
	}

	dataItems := DataItems{Version: "v1"}
	for _, tc := range testCases {
		b.Run(tc.Name, func(b *testing.B) {
			for _, w := range tc.Workloads {
				b.Run(w.Name, func(b *testing.B) {
					for feature, flag := range tc.FeatureGates {
						defer featuregatetesting.SetFeatureGateDuringTest(b, utilfeature.DefaultFeatureGate, feature, flag)()
					}
					dataItems.DataItems = append(dataItems.DataItems, runWorkload(b, tc, w)...)
					// Reset metrics to prevent metrics generated in current workload gets
					// carried over to the next workload.
					legacyregistry.Reset()
				})
			}
		})
	}
	if err := dataItems2JSONFile(dataItems, b.Name()); err != nil {
		klog.Fatalf("%v: unable to write measured data %+v: %v", b.Name(), dataItems, err)
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

func unrollWorkloadTemplate(b *testing.B, wt []op, w *workload) []op {
	var unrolled []op
	for opIndex, o := range wt {
		realOp, err := o.realOp.patchParams(w)
		if err != nil {
			b.Fatalf("op %d: %v", opIndex, err)
		}
		switch concreteOp := realOp.(type) {
		case *createPodSetsOp:
			klog.Infof("Creating %d pod sets %s", concreteOp.Count, concreteOp.CountParam)
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

func runWorkload(b *testing.B, tc *testCase, w *workload) []DataItem {
	// 30 minutes should be plenty enough even for the 5000-node tests.
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()
	var cfg *config.KubeSchedulerConfiguration
	var err error
	if len(tc.SchedulerConfigFile) != 0 {
		cfg, err = loadSchedulerConfig(tc.SchedulerConfigFile)
		if err != nil {
			b.Fatalf("error loading scheduler config file: %v", err)
		}
		if err = validation.ValidateKubeSchedulerConfiguration(cfg); err != nil {
			b.Fatalf("validate scheduler config file failed: %v", err)
		}
	}
	finalFunc, podInformer, client, dynClient := mustSetupScheduler(cfg)
	b.Cleanup(finalFunc)

	var mu sync.Mutex
	var dataItems []DataItem
	nextNodeIndex := 0
	// numPodsScheduledPerNamespace has all namespaces created in workload and the number of pods they (will) have.
	// All namespaces listed in numPodsScheduledPerNamespace will be cleaned up.
	numPodsScheduledPerNamespace := make(map[string]int)
	b.Cleanup(func() {
		for namespace := range numPodsScheduledPerNamespace {
			if err := client.CoreV1().Namespaces().Delete(context.Background(), namespace, metav1.DeleteOptions{}); err != nil {
				b.Errorf("Deleting Namespace in numPodsScheduledPerNamespace: %v", err)
			}
		}
	})

	for opIndex, op := range unrollWorkloadTemplate(b, tc.WorkloadTemplate, w) {
		realOp, err := op.realOp.patchParams(w)
		if err != nil {
			b.Fatalf("op %d: %v", opIndex, err)
		}
		select {
		case <-ctx.Done():
			b.Fatalf("op %d: %v", opIndex, ctx.Err())
		default:
		}
		switch concreteOp := realOp.(type) {
		case *createNodesOp:
			nodePreparer, err := getNodePreparer(fmt.Sprintf("node-%d-", opIndex), concreteOp, client)
			if err != nil {
				b.Fatalf("op %d: %v", opIndex, err)
			}
			if err := nodePreparer.PrepareNodes(nextNodeIndex); err != nil {
				b.Fatalf("op %d: %v", opIndex, err)
			}
			b.Cleanup(func() {
				nodePreparer.CleanupNodes()
			})
			nextNodeIndex += concreteOp.Count

		case *createNamespacesOp:
			nsPreparer, err := newNamespacePreparer(concreteOp, client)
			if err != nil {
				b.Fatalf("op %d: %v", opIndex, err)
			}
			if err := nsPreparer.prepare(); err != nil {
				nsPreparer.cleanup()
				b.Fatalf("op %d: %v", opIndex, err)
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
			if _, ok := numPodsScheduledPerNamespace[namespace]; !ok {
				// The namespace has not created yet.
				// So, creat that and register it to numPodsScheduledPerNamespace.
				_, err := client.CoreV1().Namespaces().Create(ctx, &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: namespace}}, metav1.CreateOptions{})
				if err != nil {
					b.Fatalf("failed to create namespace for Pod: %v", namespace)
				}
				numPodsScheduledPerNamespace[namespace] = 0
			}
			if concreteOp.PodTemplatePath == nil {
				concreteOp.PodTemplatePath = tc.DefaultPodTemplatePath
			}
			var collectors []testDataCollector
			var collectorCtx context.Context
			var collectorCancel func()
			if concreteOp.CollectMetrics {
				collectorCtx, collectorCancel = context.WithCancel(ctx)
				defer collectorCancel()
				collectors = getTestDataCollectors(podInformer, fmt.Sprintf("%s/%s", b.Name(), namespace), namespace, tc.MetricsCollectorConfig)
				for _, collector := range collectors {
					go collector.run(collectorCtx)
				}
			}
			if err := createPods(namespace, concreteOp, client); err != nil {
				b.Fatalf("op %d: %v", opIndex, err)
			}
			if concreteOp.SkipWaitToCompletion {
				// Only record those namespaces that may potentially require barriers
				// in the future.
				if _, ok := numPodsScheduledPerNamespace[namespace]; ok {
					numPodsScheduledPerNamespace[namespace] += concreteOp.Count
				} else {
					numPodsScheduledPerNamespace[namespace] = concreteOp.Count
				}
			} else {
				if err := waitUntilPodsScheduledInNamespace(ctx, podInformer, b.Name(), namespace, concreteOp.Count); err != nil {
					b.Fatalf("op %d: error in waiting for pods to get scheduled: %v", opIndex, err)
				}
			}
			if concreteOp.CollectMetrics {
				// CollectMetrics and SkipWaitToCompletion can never be true at the
				// same time, so if we're here, it means that all pods have been
				// scheduled.
				collectorCancel()
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
			restMapper := restmapper.NewDeferredDiscoveryRESTMapper(cacheddiscovery.NewMemCacheClient(client.Discovery()))
			// Ensure the namespace exists.
			nsObj := &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: namespace}}
			if _, err := client.CoreV1().Namespaces().Create(ctx, nsObj, metav1.CreateOptions{}); err != nil && !apierrors.IsAlreadyExists(err) {
				b.Fatalf("op %d: unable to create namespace %v: %v", opIndex, namespace, err)
			}

			var churnFns []func(name string) string

			for i, path := range concreteOp.TemplatePaths {
				unstructuredObj, gvk, err := getUnstructuredFromFile(path)
				if err != nil {
					b.Fatalf("op %d: unable to parse the %v-th template path: %v", opIndex, i, err)
				}
				// Obtain GVR.
				mapping, err := restMapper.RESTMapping(gvk.GroupKind(), gvk.Version)
				if err != nil {
					b.Fatalf("op %d: unable to find GVR for %v: %v", opIndex, gvk, err)
				}
				gvr := mapping.Resource
				// Distinguish cluster-scoped with namespaced API objects.
				var dynRes dynamic.ResourceInterface
				if mapping.Scope.Name() == meta.RESTScopeNameNamespace {
					dynRes = dynClient.Resource(gvr).Namespace(namespace)
				} else {
					dynRes = dynClient.Resource(gvr)
				}

				churnFns = append(churnFns, func(name string) string {
					if name != "" {
						dynRes.Delete(ctx, name, metav1.DeleteOptions{})
						return ""
					}

					live, err := dynRes.Create(ctx, unstructuredObj, metav1.CreateOptions{})
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

			if concreteOp.Mode == Recreate {
				go func() {
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
						case <-ctx.Done():
							return
						}
					}
				}()
			} else if concreteOp.Mode == Create {
				go func() {
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
						case <-ctx.Done():
							return
						}
					}
				}()
			}

		case *barrierOp:
			for _, namespace := range concreteOp.Namespaces {
				if _, ok := numPodsScheduledPerNamespace[namespace]; !ok {
					b.Fatalf("op %d: unknown namespace %s", opIndex, namespace)
				}
			}
			if err := waitUntilPodsScheduled(ctx, podInformer, b.Name(), concreteOp.Namespaces, numPodsScheduledPerNamespace); err != nil {
				b.Fatalf("op %d: %v", opIndex, err)
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
			case <-ctx.Done():
			case <-time.After(concreteOp.Duration):
			}
		default:
			b.Fatalf("op %d: invalid op %v", opIndex, concreteOp)
		}
	}

	// check unused params and inform users
	unusedParams := w.unusedParams()
	if len(unusedParams) != 0 {
		b.Fatalf("the parameters %v are defined on workload %s, but unused.\nPlease make sure there are no typos.", unusedParams, w.Name)
	}

	// Some tests have unschedulable pods. Do not add an implicit barrier at the
	// end as we do not want to wait for them.
	return dataItems
}

type testDataCollector interface {
	run(ctx context.Context)
	collect() []DataItem
}

func getTestDataCollectors(podInformer coreinformers.PodInformer, name, namespace string, mcc *metricsCollectorConfig) []testDataCollector {
	if mcc == nil {
		mcc = &defaultMetricsCollectorConfig
	}
	return []testDataCollector{
		newThroughputCollector(podInformer, map[string]string{"Name": name}, []string{namespace}),
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

func createPods(namespace string, cpo *createPodsOp, clientset clientset.Interface) error {
	strategy, err := getPodStrategy(cpo)
	if err != nil {
		return err
	}
	klog.Infof("Creating %d pods in namespace %q", cpo.Count, namespace)
	config := testutils.NewTestPodCreatorConfig()
	config.AddStrategy(namespace, cpo.Count, strategy)
	podCreator := testutils.NewTestPodCreator(clientset, config)
	return podCreator.CreatePods()
}

// waitUntilPodsScheduledInNamespace blocks until all pods in the given
// namespace are scheduled. Times out after 10 minutes because even at the
// lowest observed QPS of ~10 pods/sec, a 5000-node test should complete.
func waitUntilPodsScheduledInNamespace(ctx context.Context, podInformer coreinformers.PodInformer, name string, namespace string, wantCount int) error {
	return wait.PollImmediate(1*time.Second, 10*time.Minute, func() (bool, error) {
		select {
		case <-ctx.Done():
			return true, ctx.Err()
		default:
		}
		scheduled, err := getScheduledPods(podInformer, namespace)
		if err != nil {
			return false, err
		}
		if len(scheduled) >= wantCount {
			return true, nil
		}
		klog.Infof("%s: namespace %s: got %d pods, want %d", name, namespace, len(scheduled), wantCount)
		return false, nil
	})
}

// waitUntilPodsScheduled blocks until the all pods in the given namespaces are
// scheduled.
func waitUntilPodsScheduled(ctx context.Context, podInformer coreinformers.PodInformer, name string, namespaces []string, numPodsScheduledPerNamespace map[string]int) error {
	// If unspecified, default to all known namespaces.
	if len(namespaces) == 0 {
		for namespace := range numPodsScheduledPerNamespace {
			namespaces = append(namespaces, namespace)
		}
	}
	for _, namespace := range namespaces {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		wantCount, ok := numPodsScheduledPerNamespace[namespace]
		if !ok {
			return fmt.Errorf("unknown namespace %s", namespace)
		}
		if err := waitUntilPodsScheduledInNamespace(ctx, podInformer, name, namespace, wantCount); err != nil {
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
		return nil, fmt.Errorf("parsing test cases: %w", err)
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
	client clientset.Interface
	count  int
	prefix string
	spec   *v1.Namespace
}

func newNamespacePreparer(cno *createNamespacesOp, clientset clientset.Interface) (*namespacePreparer, error) {
	ns := &v1.Namespace{}
	if cno.NamespaceTemplatePath != nil {
		if err := getSpecFromFile(cno.NamespaceTemplatePath, ns); err != nil {
			return nil, fmt.Errorf("parsing NamespaceTemplate: %w", err)
		}
	}

	return &namespacePreparer{
		client: clientset,
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
func (p *namespacePreparer) prepare() error {
	base := &v1.Namespace{}
	if p.spec != nil {
		base = p.spec
	}
	klog.Infof("Making %d namespaces with prefix %q and template %v", p.count, p.prefix, *base)
	for i := 0; i < p.count; i++ {
		n := base.DeepCopy()
		n.Name = fmt.Sprintf("%s-%d", p.prefix, i)
		if err := testutils.RetryWithExponentialBackOff(func() (bool, error) {
			_, err := p.client.CoreV1().Namespaces().Create(context.Background(), n, metav1.CreateOptions{})
			return err == nil || apierrors.IsAlreadyExists(err), nil
		}); err != nil {
			return err
		}
	}
	return nil
}

// cleanup deletes existing test namespaces.
func (p *namespacePreparer) cleanup() error {
	var errRet error
	for i := 0; i < p.count; i++ {
		n := fmt.Sprintf("%s-%d", p.prefix, i)
		if err := p.client.CoreV1().Namespaces().Delete(context.Background(), n, metav1.DeleteOptions{}); err != nil {
			klog.Errorf("Deleting Namespace: %v", err)
			errRet = err
		}
	}
	return errRet
}
