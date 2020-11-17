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
	"io/ioutil"
	"strings"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/integration/framework"
	testutils "k8s.io/kubernetes/test/utils"
	"sigs.k8s.io/yaml"
)

const (
	configFile        = "config/performance-config.yaml"
	createNodesOpcode = "createNodes"
	createPodsOpcode  = "createPods"
	barrierOpcode     = "barrier"
)

var (
	defaultMetricsCollectorConfig = metricsCollectorConfig{
		Metrics: []string{
			"scheduler_scheduling_algorithm_predicate_evaluation_seconds",
			"scheduler_scheduling_algorithm_priority_evaluation_seconds",
			"scheduler_binding_duration_seconds",
			"scheduler_e2e_scheduling_duration_seconds",
			"scheduler_scheduling_algorithm_preemption_evaluation_seconds",
			"scheduler_pod_scheduling_duration_seconds",
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
	// TODO(#93792): reduce config toil by having a default pod and node spec per
	// testCase? CreatePods and CreateNodes ops will inherit these unless
	// manually overridden.
}

func (tc *testCase) collectsMetrics() bool {
	for _, op := range tc.WorkloadTemplate {
		if op.realOp.collectsMetrics() {
			return true
		}
	}
	return false
}

// workload is a subtest under a testCase that tests the scheduler performance
// for a certain ordering of ops. The set of nodes created and pods scheduled
// in a workload may be heterogenous.
type workload struct {
	// Name of the workload.
	Name string
	// Values of parameters used in the workloadTemplate.
	Params map[string]int
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
		&createPodsOp{},
		&barrierOp{},
		// TODO(#93793): add a sleep timer op to simulate waiting?
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
		return fmt.Errorf("invalid opcode")
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
		var ok bool
		if cno.Count, ok = w.Params[cno.CountParam[1:]]; !ok {
			return nil, fmt.Errorf("parameter %s is undefined", cno.CountParam)
		}
	}
	return &cno, (&cno).isValid(false)
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
		return fmt.Errorf("invalid opcode")
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
		var ok bool
		if cpo.Count, ok = w.Params[cpo.CountParam[1:]]; !ok {
			return nil, fmt.Errorf("parameter %s is undefined", cpo.CountParam)
		}
	}
	return &cpo, (&cpo).isValid(false)
}

// barrierOp defines an op that can be used to wait until all scheduled pods of
// one or many namespaces have been bound to nodes. This is useful when pods
// were scheduled with SkipWaitToCompletion set to true. A barrierOp is added
// at the end of each each workload automatically.
type barrierOp struct {
	// Must be "barrier".
	Opcode string
	// Namespaces to block on. Empty array or not specifying this field signifies
	// that the barrier should block on all namespaces.
	Namespaces []string
}

func (bo *barrierOp) isValid(allowParameterization bool) error {
	if bo.Opcode != barrierOpcode {
		return fmt.Errorf("invalid opcode")
	}
	return nil
}

func (*barrierOp) collectsMetrics() bool {
	return false
}

func (bo barrierOp) patchParams(w *workload) (realOp, error) {
	return &bo, nil
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
				})
			}
		})
	}
	if err := dataItems2JSONFile(dataItems, b.Name()); err != nil {
		klog.Fatalf("%v: unable to write measured data: %v", b.Name(), err)
	}
}

func runWorkload(b *testing.B, tc *testCase, w *workload) []DataItem {
	// 30 minutes should be plenty enough even for the 5000-node tests.
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()
	finalFunc, podInformer, clientset := mustSetupScheduler()
	b.Cleanup(finalFunc)

	var mu sync.Mutex
	var dataItems []DataItem
	numPodsScheduledPerNamespace := make(map[string]int)
	nextNodeIndex := 0

	for opIndex, op := range tc.WorkloadTemplate {
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
			nodePreparer, err := getNodePreparer(fmt.Sprintf("node-%d-", opIndex), concreteOp, clientset)
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

		case *createPodsOp:
			var namespace string
			if concreteOp.Namespace != nil {
				namespace = *concreteOp.Namespace
			} else {
				namespace = fmt.Sprintf("namespace-%d", opIndex)
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
			if err := createPods(namespace, concreteOp, clientset); err != nil {
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

		default:
			b.Fatalf("op %d: invalid op %v", opIndex, concreteOp)
		}
	}
	if err := waitUntilPodsScheduled(ctx, podInformer, b.Name(), nil, numPodsScheduledPerNamespace); err != nil {
		// Any pending pods must be scheduled before this test can be considered to
		// be complete.
		b.Fatal(err)
	}
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
	bytes, err := ioutil.ReadFile(*path)
	if err != nil {
		return err
	}
	return yaml.UnmarshalStrict(bytes, spec)
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
	for _, tc := range testCases {
		if len(tc.Workloads) == 0 {
			return fmt.Errorf("%s: no workloads defined", tc.Name)
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
