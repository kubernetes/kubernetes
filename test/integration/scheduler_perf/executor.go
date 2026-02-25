/*
Copyright 2026 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"maps"
	"math"
	"os"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	cacheddiscovery "k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/dynamic"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	schedulinglisters "k8s.io/client-go/listers/scheduling/v1alpha1"
	"k8s.io/client-go/restmapper"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
	testutils "k8s.io/kubernetes/test/utils"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/yaml"
)

type WorkloadExecutor struct {
	tCtx                         ktesting.TContext
	scheduler                    *scheduler.Scheduler
	wg                           sync.WaitGroup
	collectorCtx                 ktesting.TContext
	collectorWG                  sync.WaitGroup
	collectors                   []testDataCollector
	dataItems                    []DataItem
	numPodsScheduledPerNamespace map[string]int
	podInformer                  coreinformers.PodInformer
	workloadLister               schedulinglisters.WorkloadLister
	eventInformer                coreinformers.EventInformer
	throughputErrorMargin        float64
	testCase                     *testCase
	workload                     *workload
	topicName                    string
	nextNodeIndex                int
}

func (e *WorkloadExecutor) wait() {
	e.collectorWG.Wait()
	e.wg.Wait()
}

func (e *WorkloadExecutor) runOp(op realOp, opIndex int) error {
	switch concreteOp := op.(type) {
	case *createNodesOp:
		return e.runCreateNodesOp(opIndex, concreteOp)
	case *createNamespacesOp:
		return e.runCreateNamespaceOp(opIndex, concreteOp)
	case *createPodsOp:
		return e.runCreatePodsOp(opIndex, concreteOp)
	case *deletePodsOp:
		return e.runDeletePodsOp(opIndex, concreteOp)
	case *churnOp:
		return e.runChurnOp(opIndex, concreteOp)
	case *barrierOp:
		return e.runBarrierOp(opIndex, concreteOp)
	case *sleepOp:
		return e.runSleepOp(concreteOp)
	case *startCollectingMetricsOp:
		return e.runStartCollectingMetricsOp(opIndex, concreteOp)
	case *stopCollectingMetricsOp:
		return e.runStopCollectingMetrics(opIndex)
	case *createResourceDriverOp:
		concreteOp.run(e.tCtx, e.scheduler.Profiles["default-scheduler"].SharedDRAManager())
		return nil
	default:
		return e.runDefaultOp(opIndex, concreteOp)
	}
}

func (e *WorkloadExecutor) runCreateNodesOp(opIndex int, op *createNodesOp) error {
	nodePreparer, err := getNodePreparer(fmt.Sprintf("node-%d-", opIndex), op, e.tCtx.Client())
	if err != nil {
		return err
	}
	if err := nodePreparer.PrepareNodes(e.tCtx, e.nextNodeIndex); err != nil {
		return err
	}
	e.nextNodeIndex += op.Count
	return nil
}

func (e *WorkloadExecutor) runCreateNamespaceOp(opIndex int, op *createNamespacesOp) error {
	nsPreparer, err := newNamespacePreparer(e.tCtx, op)
	if err != nil {
		return err
	}
	if err := nsPreparer.prepare(e.tCtx); err != nil {
		err2 := nsPreparer.cleanup(e.tCtx)
		if err2 != nil {
			err = fmt.Errorf("prepare: %w; cleanup: %w", err, err2)
		}
		return err
	}
	for _, n := range nsPreparer.namespaces() {
		if _, ok := e.numPodsScheduledPerNamespace[n]; ok {
			// this namespace has been already created.
			continue
		}
		e.numPodsScheduledPerNamespace[n] = 0
	}
	return nil
}

func (e *WorkloadExecutor) runBarrierOp(opIndex int, op *barrierOp) error {
	for _, namespace := range op.Namespaces {
		if _, ok := e.numPodsScheduledPerNamespace[namespace]; !ok {
			return fmt.Errorf("unknown namespace %s", namespace)
		}
	}
	switch op.StageRequirement {
	case Attempted:
		if err := waitUntilPodsAttempted(e.tCtx, e.podInformer, op.LabelSelector, op.Namespaces, e.numPodsScheduledPerNamespace); err != nil {
			return err
		}
	case Scheduled:
		// Default should be treated like "Scheduled", so handling both in the same way.
		fallthrough
	default:
		if err := waitUntilPodsScheduled(e.tCtx, e.podInformer, op.LabelSelector, op.Namespaces, e.numPodsScheduledPerNamespace); err != nil {
			return err
		}
		// At the end of the barrier, we can be sure that there are no pods
		// pending scheduling in the namespaces that we just blocked on.
		if len(op.Namespaces) == 0 {
			e.numPodsScheduledPerNamespace = make(map[string]int)
		} else {
			for _, namespace := range op.Namespaces {
				delete(e.numPodsScheduledPerNamespace, namespace)
			}
		}
	}
	return nil
}

func (e *WorkloadExecutor) runSleepOp(op *sleepOp) error {
	select {
	case <-e.tCtx.Done():
	case <-time.After(op.Duration.Duration):
	}
	return nil
}

func (e *WorkloadExecutor) runStopCollectingMetrics(opIndex int) error {
	items, err := stopCollectingMetrics(e.tCtx, e.collectorCtx, &e.collectorWG, e.workload.Threshold.Get(e.topicName), *e.workload.ThresholdMetricSelector, opIndex, e.collectors)
	if err != nil {
		return err
	}
	e.dataItems = append(e.dataItems, items...)
	e.collectorCtx = nil
	return nil
}

func (e *WorkloadExecutor) runCreatePodsOp(opIndex int, op *createPodsOp) error {
	// define Pod's namespace automatically, and create that namespace.
	namespace := fmt.Sprintf("namespace-%d", opIndex)
	if op.Namespace != nil {
		namespace = *op.Namespace
	}
	err := createNamespaceIfNotPresent(e.tCtx, namespace, &e.numPodsScheduledPerNamespace)
	if err != nil {
		return err
	}
	if op.PodTemplatePath == nil {
		op.PodTemplatePath = e.testCase.DefaultPodTemplatePath
	}

	if op.CollectMetrics {
		if e.collectorCtx != nil {
			return fmt.Errorf("metrics collection is overlapping. Probably second collector was started before stopping a previous one")
		}
		var err error
		e.collectorCtx, e.collectors, err = startCollectingMetrics(e.tCtx, &e.collectorWG, e.podInformer, e.workloadLister, e.eventInformer, e.testCase.MetricsCollectorConfig, e.throughputErrorMargin, opIndex, namespace, []string{namespace}, nil, nil)
		if err != nil {
			return err
		}
	}
	if err := createPodsRapidly(e.tCtx, namespace, op); err != nil {
		return err
	}
	switch {
	case op.SkipWaitToCompletion:
		// Only record those namespaces that may potentially require barriers
		// in the future.
		e.numPodsScheduledPerNamespace[namespace] += op.Count
	case op.SteadyState:
		if err := createPodsSteadily(e.tCtx, namespace, e.podInformer, op); err != nil {
			return err
		}
	default:
		if err := waitUntilPodsScheduledInNamespace(e.tCtx, e.podInformer, nil, namespace, op.Count); err != nil {
			return fmt.Errorf("error in waiting for pods to get scheduled: %w", err)
		}
	}
	if op.CollectMetrics {
		// CollectMetrics and SkipWaitToCompletion can never be true at the
		// same time, so if we're here, it means that all pods have been
		// scheduled.
		items, err := stopCollectingMetrics(e.tCtx, e.collectorCtx, &e.collectorWG, e.workload.Threshold.Get(e.topicName), *e.workload.ThresholdMetricSelector, opIndex, e.collectors)
		if err != nil {
			return err
		}
		e.dataItems = append(e.dataItems, items...)
		e.collectorCtx = nil
	}
	return nil
}

func (e *WorkloadExecutor) runDeletePodsOp(opIndex int, op *deletePodsOp) error {
	labelSelector := labels.ValidatedSetSelector(op.LabelSelector)

	podsToDelete, err := e.podInformer.Lister().Pods(op.Namespace).List(labelSelector)
	if err != nil {
		return fmt.Errorf("error in listing pods in the namespace %s: %w", op.Namespace, err)
	}

	deletePods := func(opIndex int) {
		if op.DeletePodsPerSecond > 0 {
			ticker := time.NewTicker(time.Second / time.Duration(op.DeletePodsPerSecond))
			defer ticker.Stop()

			for i := range podsToDelete {
				select {
				case <-ticker.C:
					if err := e.tCtx.Client().CoreV1().Pods(op.Namespace).Delete(e.tCtx, podsToDelete[i].Name, metav1.DeleteOptions{}); err != nil {
						if errors.Is(err, context.Canceled) {
							return
						}
						e.tCtx.Errorf("op %d: unable to delete pod %v: %v", opIndex, podsToDelete[i].Name, err)
					}
				case <-e.tCtx.Done():
					return
				}
			}
			return
		}
		listOpts := metav1.ListOptions{
			LabelSelector: labelSelector.String(),
		}
		if err := e.tCtx.Client().CoreV1().Pods(op.Namespace).DeleteCollection(e.tCtx, metav1.DeleteOptions{}, listOpts); err != nil {
			if errors.Is(err, context.Canceled) {
				return
			}
			e.tCtx.Errorf("op %d: unable to delete pods in namespace %v: %v", opIndex, op.Namespace, err)
		}
	}

	if op.SkipWaitToCompletion {
		e.wg.Add(1)
		go func(opIndex int) {
			defer e.wg.Done()
			deletePods(opIndex)
		}(opIndex)
	} else {
		deletePods(opIndex)
	}
	return nil
}

func (e *WorkloadExecutor) runChurnOp(opIndex int, op *churnOp) error {
	var namespace string
	if op.Namespace != nil {
		namespace = *op.Namespace
	} else {
		namespace = fmt.Sprintf("namespace-%d", opIndex)
	}
	restMapper := restmapper.NewDeferredDiscoveryRESTMapper(cacheddiscovery.NewMemCacheClient(e.tCtx.Client().Discovery()))
	// Ensure the namespace exists.
	nsObj := &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: namespace}}
	if _, err := e.tCtx.Client().CoreV1().Namespaces().Create(e.tCtx, nsObj, metav1.CreateOptions{}); err != nil && !apierrors.IsAlreadyExists(err) {
		return fmt.Errorf("unable to create namespace %v: %w", namespace, err)
	}

	var churnFns []func(name string) string

	for i, path := range op.TemplatePaths {
		unstructuredObj, gvk, err := getUnstructuredFromFile(path)
		if err != nil {
			return fmt.Errorf("unable to parse the %v-th template path: %w", i, err)
		}
		// Obtain GVR.
		mapping, err := restMapper.RESTMapping(gvk.GroupKind(), gvk.Version)
		if err != nil {
			return fmt.Errorf("unable to find GVR for %v: %w", gvk, err)
		}
		gvr := mapping.Resource
		// Distinguish cluster-scoped with namespaced API objects.
		var dynRes dynamic.ResourceInterface
		if mapping.Scope.Name() == meta.RESTScopeNameNamespace {
			dynRes = e.tCtx.Dynamic().Resource(gvr).Namespace(namespace)
		} else {
			dynRes = e.tCtx.Dynamic().Resource(gvr)
		}

		churnFns = append(churnFns, func(name string) string {
			if name != "" {
				if err := dynRes.Delete(e.tCtx, name, metav1.DeleteOptions{}); err != nil && !errors.Is(err, context.Canceled) {
					e.tCtx.Errorf("op %d: unable to delete %v: %v", opIndex, name, err)
				}
				return ""
			}

			live, err := dynRes.Create(e.tCtx, unstructuredObj, metav1.CreateOptions{})
			if err != nil {
				return ""
			}
			return live.GetName()
		})
	}

	var interval int64 = 500
	if op.IntervalMilliseconds != 0 {
		interval = op.IntervalMilliseconds
	}
	ticker := time.NewTicker(time.Duration(interval) * time.Millisecond)

	switch op.Mode {
	case Create:
		e.wg.Add(1)
		go func() {
			defer e.wg.Done()
			defer ticker.Stop()
			count, threshold := 0, op.Number
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
				case <-e.tCtx.Done():
					return
				}
			}
		}()
	case Recreate:
		e.wg.Add(1)
		go func() {
			defer e.wg.Done()
			defer ticker.Stop()
			retVals := make([][]string, len(churnFns))
			// For each churn function, instantiate a slice of strings with length "op.Number".
			for i := range retVals {
				retVals[i] = make([]string, op.Number)
			}

			count := 0
			for {
				select {
				case <-ticker.C:
					for i := range churnFns {
						retVals[i][count%op.Number] = churnFns[i](retVals[i][count%op.Number])
					}
					count++
				case <-e.tCtx.Done():
					return
				}
			}
		}()
	}
	return nil
}

func (e *WorkloadExecutor) runDefaultOp(opIndex int, op realOp) error {
	runnable, ok := op.(runnableOp)
	if !ok {
		return fmt.Errorf("invalid op %v", op)
	}
	for _, namespace := range runnable.requiredNamespaces() {
		err := createNamespaceIfNotPresent(e.tCtx, namespace, &e.numPodsScheduledPerNamespace)
		if err != nil {
			return err
		}
	}
	runnable.run(e.tCtx)
	return nil
}

func (e *WorkloadExecutor) runStartCollectingMetricsOp(opIndex int, op *startCollectingMetricsOp) error {
	if e.collectorCtx != nil {
		return fmt.Errorf("metrics collection is overlapping. Probably second collector was started before stopping a previous one")
	}
	var err error
	e.collectorCtx, e.collectors, err = startCollectingMetrics(e.tCtx, &e.collectorWG, e.podInformer, e.workloadLister, e.eventInformer, e.testCase.MetricsCollectorConfig, e.throughputErrorMargin, opIndex, op.Name, op.Namespaces, op.LabelSelector, op.Collectors)
	if err != nil {
		return err
	}
	return nil
}

func startCollectingMetrics(tCtx ktesting.TContext, collectorWG *sync.WaitGroup, podInformer coreinformers.PodInformer, workloadLister schedulinglisters.WorkloadLister, eventInformer coreinformers.EventInformer, mcc *metricsCollectorConfig, throughputErrorMargin float64, opIndex int, name string, namespaces []string, labelSelector map[string]string, collectors []string) (ktesting.TContext, []testDataCollector, error) {
	collectorCtx := tCtx.WithCancel()
	workloadName := tCtx.Name()

	// Clean up memory usage from the initial setup phase.
	runtime.GC()

	// The first part is the same for each workload, therefore we can strip it.
	workloadName = workloadName[strings.Index(name, "/")+1:]
	collectorsList := getTestDataCollectors(podInformer, workloadLister, eventInformer, fmt.Sprintf("%s/%s", workloadName, name), namespaces, labelSelector, mcc, throughputErrorMargin, collectors)
	for _, collector := range collectorsList {
		// Need loop-local variable for function below.
		err := collector.init()
		if err != nil {
			return nil, nil, fmt.Errorf("failed to initialize data collector: %w", err)
		}
		tCtx.TB().Cleanup(func() {
			collectorCtx.Cancel("cleaning up")
		})
		collectorWG.Add(1)
		go func() {
			defer collectorWG.Done()
			collector.run(collectorCtx)
		}()
	}
	if b, ok := tCtx.TB().(*testing.B); ok {
		b.ResetTimer()
	}
	tCtx.Log("Started metrics collection")
	return collectorCtx, collectorsList, nil
}

func stopCollectingMetrics(tCtx ktesting.TContext, collectorCtx ktesting.TContext, collectorWG *sync.WaitGroup, threshold float64, tms thresholdMetricSelector, opIndex int, collectors []testDataCollector) ([]DataItem, error) {
	if b, ok := tCtx.TB().(*testing.B); ok {
		b.StopTimer()
	}
	if collectorCtx == nil {
		return nil, fmt.Errorf("missing startCollectingMetrics operation before stopping")
	}
	collectorCtx.Cancel("collecting metrics, collector must stop first")
	collectorWG.Wait()
	var dataItems []DataItem
	for _, collector := range collectors {
		items := collector.collect(tCtx)
		dataItems = append(dataItems, items...)
		err := applyThreshold(items, threshold, tms)
		if err != nil {
			tCtx.Errorf("op %d: %s", opIndex, err)
		}
	}
	tCtx.Log("Stopped metrics collection")
	return dataItems, nil
}

type testDataCollector interface {
	init() error
	run(tCtx ktesting.TContext)
	collect(tCtx ktesting.TContext) []DataItem
}

// var for mocking in tests.
var getTestDataCollectors = func(podInformer coreinformers.PodInformer, workloadLister schedulinglisters.WorkloadLister, eventInformer coreinformers.EventInformer, name string, namespaces []string, labelSelector map[string]string, mcc *metricsCollectorConfig, throughputErrorMargin float64, collectors []string) []testDataCollector {
	if len(collectors) == 0 {
		return getDefaultTestDataCollectors(podInformer, name, namespaces, labelSelector, mcc, throughputErrorMargin)
	}
	var testDataCollectors []testDataCollector
	if mcc == nil {
		mcc = &defaultMetricsCollectorConfig
	}
	for _, collector := range collectors {
		switch collector {
		case "Throughput":
			testDataCollectors = append(testDataCollectors, newThroughputCollector(podInformer, map[string]string{"Name": name}, labelSelector, namespaces, throughputErrorMargin))
		case "Metrics":
			testDataCollectors = append(testDataCollectors, newMetricsCollector(mcc, map[string]string{"Name": name}))
		case "Memory":
			testDataCollectors = append(testDataCollectors, newMemoryCollector(map[string]string{"Name": name}, 500*time.Millisecond))
		case "SchedulingDuration":
			testDataCollectors = append(testDataCollectors, newSchedulingDurationCollector(map[string]string{"Name": name}))
		case "PodGroupLatency":
			testDataCollectors = append(testDataCollectors, newPodGroupLatencyCollector(podInformer, eventInformer, workloadLister))
		default:
			panic(fmt.Sprintf("unknown collector: %s", collector))
		}
	}
	return testDataCollectors
}

func getDefaultTestDataCollectors(podInformer coreinformers.PodInformer, name string, namespaces []string, labelSelector map[string]string, mcc *metricsCollectorConfig, throughputErrorMargin float64) []testDataCollector {
	if mcc == nil {
		mcc = &defaultMetricsCollectorConfig
	}
	return []testDataCollector{
		newThroughputCollector(podInformer, map[string]string{"Name": name}, labelSelector, namespaces, throughputErrorMargin),
		newMetricsCollector(mcc, map[string]string{"Name": name}),
		newMemoryCollector(map[string]string{"Name": name}, 500*time.Millisecond),
		newSchedulingDurationCollector(map[string]string{"Name": name}),
	}
}

func createNamespaceIfNotPresent(tCtx ktesting.TContext, namespace string, podsPerNamespace *map[string]int) error {
	if _, ok := (*podsPerNamespace)[namespace]; !ok {
		// The namespace has not created yet.
		// So, create that and register it.
		_, err := tCtx.Client().CoreV1().Namespaces().Create(tCtx, &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: namespace}}, metav1.CreateOptions{})
		if err != nil {
			return fmt.Errorf("failed to create namespace for Pod: %v", namespace)
		}
		(*podsPerNamespace)[namespace] = 0
	}
	return nil
}

// createPodsRapidly implements the "create pods rapidly" mode of [createPodsOp].
// It's a nop when cpo.SteadyState is true.
func createPodsRapidly(tCtx ktesting.TContext, namespace string, cpo *createPodsOp) error {
	if cpo.SteadyState {
		return nil
	}
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

// createPodsSteadily implements the "create pods and delete pods" mode of [createPodsOp].
// It's a nop when cpo.SteadyState is false.
func createPodsSteadily(tCtx ktesting.TContext, namespace string, podInformer coreinformers.PodInformer, cpo *createPodsOp) error {
	if !cpo.SteadyState {
		return nil
	}
	strategy, err := getPodStrategy(cpo)
	if err != nil {
		return err
	}
	tCtx.Logf("creating pods in namespace %q for %s", namespace, cpo.Duration)
	tCtx = tCtx.WithTimeout(cpo.Duration.Duration, fmt.Sprintf("the operation ran for the configured %s", cpo.Duration.Duration))

	// Start watching pods in the namespace. Any pod which is seen as being scheduled
	// gets deleted.
	scheduledPods := make(chan *v1.Pod, cpo.Count)
	scheduledPodsClosed := false
	var mutex sync.Mutex
	defer func() {
		mutex.Lock()
		defer mutex.Unlock()
		close(scheduledPods)
		scheduledPodsClosed = true
	}()

	existingPods := 0
	runningPods := 0
	onPodChange := func(oldObj, newObj any) {
		oldPod, newPod, err := schedutil.As[*v1.Pod](oldObj, newObj)
		if err != nil {
			tCtx.Errorf("unexpected pod events: %v", err)
			return
		}

		mutex.Lock()
		defer mutex.Unlock()
		if oldPod == nil {
			existingPods++
		}
		if (oldPod == nil || oldPod.Spec.NodeName == "") && newPod.Spec.NodeName != "" {
			// Got scheduled.
			runningPods++

			// Only ask for deletion in our namespace.
			if newPod.Namespace != namespace {
				return
			}
			if !scheduledPodsClosed {
				select {
				case <-tCtx.Done():
				case scheduledPods <- newPod:
				}
			}
		}
	}
	handle, err := podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj any) {
			onPodChange(nil, obj)
		},
		UpdateFunc: func(oldObj, newObj any) {
			onPodChange(oldObj, newObj)
		},
		DeleteFunc: func(obj any) {
			pod, _, err := schedutil.As[*v1.Pod](obj, nil)
			if err != nil {
				tCtx.Errorf("unexpected pod events: %v", err)
				return
			}

			mutex.Lock()
			defer mutex.Unlock()
			existingPods--
			if pod.Spec.NodeName != "" {
				runningPods--
			}
		},
	})
	if err != nil {
		return fmt.Errorf("register event handler: %w", err)
	}
	defer func() {
		tCtx.ExpectNoError(podInformer.Informer().RemoveEventHandler(handle), "remove event handler")
	}()

	// Seed the namespace with the initial number of pods.
	if err := strategy(tCtx, tCtx.Client(), namespace, cpo.Count); err != nil {
		return fmt.Errorf("create initial %d pods: %w", cpo.Count, err)
	}

	// Now loop until we are done. Report periodically how many pods were scheduled.
	countScheduledPods := 0
	lastCountScheduledPods := 0
	logPeriod := time.Second
	ticker := time.NewTicker(logPeriod)
	defer ticker.Stop()
	for {
		select {
		case <-tCtx.Done():
			tCtx.Logf("Completed after seeing %d scheduled pod: %v", countScheduledPods, context.Cause(tCtx))

			// Sanity check: at least one pod should have been scheduled,
			// giving us a non-zero average.
			//
			// This is important because otherwise "no pods scheduled because of constant
			// failure" would not get detected in unit tests. For benchmarks
			// it's less important, but indicates that the time period
			// might have been too small.
			//
			// The collector logs "Failed to measure SchedulingThroughput ... Increase pods and/or nodes to make scheduling take longer"
			// but that is hard to spot.
			//
			// The non-steady case blocks until all pods have been scheduled
			// and doesn't need this check.
			if countScheduledPods == 0 {
				return errors.New("no pod at all got scheduled, either because of a problem or because the test interval was too small")
			}
			return nil
		case <-scheduledPods:
			countScheduledPods++
			if countScheduledPods%cpo.Count == 0 {
				// All scheduled. Start over with a new batch.
				err := tCtx.Client().CoreV1().Pods(namespace).DeleteCollection(tCtx, metav1.DeleteOptions{
					GracePeriodSeconds: ptr.To(int64(0)),
					PropagationPolicy:  ptr.To(metav1.DeletePropagationBackground), // Foreground will block.
				}, metav1.ListOptions{})
				// Ignore errors when the time is up. errors.Is(context.Canceled) would
				// be more precise, but doesn't work because client-go doesn't reliably
				// propagate it.
				if tCtx.Err() != nil {
					continue
				}
				if err != nil {
					// Worse, sometimes rate limiting gives up *before* the context deadline is reached.
					// Then we get here with this error:
					//   client rate limiter Wait returned an error: rate: Wait(n=1) would exceed context deadline
					//
					// This also can be ignored. We'll retry if the test is not done yet.
					if strings.Contains(err.Error(), "would exceed context deadline") {
						continue
					}
					return fmt.Errorf("delete scheduled pods: %w", err)
				}
				err = strategy(tCtx, tCtx.Client(), namespace, cpo.Count)
				if tCtx.Err() != nil {
					continue
				}
				if err != nil {
					return fmt.Errorf("create next batch of pods: %w", err)
				}
			}
		case <-ticker.C:
			delta := countScheduledPods - lastCountScheduledPods
			lastCountScheduledPods = countScheduledPods
			func() {
				mutex.Lock()
				defer mutex.Unlock()

				tCtx.Logf("%d pods got scheduled in total in namespace %q, overall %d out of %d pods scheduled: %f pods/s in last interval",
					countScheduledPods, namespace,
					runningPods, existingPods,
					float64(delta)/logPeriod.Seconds(),
				)
			}()
		}
	}
}

// waitUntilPodsScheduled blocks until the all pods in the given namespaces are
// scheduled.
func waitUntilPodsScheduled(tCtx ktesting.TContext, podInformer coreinformers.PodInformer, labelSelector map[string]string, namespaces []string, numPodsScheduledPerNamespace map[string]int) error {
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
		if err := waitUntilPodsScheduledInNamespace(tCtx, podInformer, labelSelector, namespace, wantCount); err != nil {
			return fmt.Errorf("error waiting for pods in namespace %q: %w", namespace, err)
		}
	}
	return nil
}

// waitUntilPodsAttempted blocks until the all pods in the given namespaces are
// attempted (at least once went through a scheduling cycle).
func waitUntilPodsAttempted(tCtx ktesting.TContext, podInformer coreinformers.PodInformer, labelSelector map[string]string, namespaces []string, numPodsScheduledPerNamespace map[string]int) error {
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
		if err := waitUntilPodsAttemptedInNamespace(tCtx, podInformer, labelSelector, namespace, wantCount); err != nil {
			return fmt.Errorf("error waiting for pods in namespace %q: %w", namespace, err)
		}
	}
	return nil
}

// waitUntilPodsAttemptedInNamespace blocks until all pods in the given
// namespace at least once went through a scheduling cycle.
// Times out after 10 minutes similarly to waitUntilPodsScheduledInNamespace.
func waitUntilPodsAttemptedInNamespace(tCtx ktesting.TContext, podInformer coreinformers.PodInformer, labelSelector map[string]string, namespace string, wantCount int) error {
	var pendingPod *v1.Pod

	err := wait.PollUntilContextTimeout(tCtx, 1*time.Second, 10*time.Minute, true, func(ctx context.Context) (bool, error) {
		select {
		case <-ctx.Done():
			return true, ctx.Err()
		default:
		}
		scheduled, attempted, unattempted, err := getScheduledPods(podInformer, labelSelector, namespace)
		if err != nil {
			return false, err
		}
		if len(scheduled)+len(attempted) >= wantCount {
			tCtx.Logf("all pods attempted to be scheduled")
			return true, nil
		}
		tCtx.Logf("namespace: %s, attempted pods: want %d, got %d", namespace, wantCount, len(scheduled)+len(attempted))
		if len(unattempted) > 0 {
			pendingPod = unattempted[0]
		} else {
			pendingPod = nil
		}
		return false, nil
	})

	if err != nil && pendingPod != nil {
		err = fmt.Errorf("at least pod %s is not attempted: %w", klog.KObj(pendingPod), err)
	}
	return err
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

	nodeTemplate := StaticNodeTemplate(makeBaseNode(prefix))
	if cno.NodeTemplatePath != nil {
		nodeTemplate = nodeTemplateWithParams{path: *cno.NodeTemplatePath, params: cno.TemplateParams}
	}

	return NewIntegrationTestNodePreparer(
		clientset,
		[]testutils.CountToStrategy{{Count: cno.Count, Strategy: nodeStrategy}},
		nodeTemplate,
	), nil
}

// waitUntilPodsScheduledInNamespace blocks until all pods in the given
// namespace are scheduled. Times out after 10 minutes because even at the
// lowest observed QPS of ~10 pods/sec, a 5000-node test should complete.
func waitUntilPodsScheduledInNamespace(tCtx ktesting.TContext, podInformer coreinformers.PodInformer, labelSelector map[string]string, namespace string, wantCount int) error {
	var pendingPod *v1.Pod

	err := wait.PollUntilContextTimeout(tCtx, 1*time.Second, 10*time.Minute, true, func(ctx context.Context) (bool, error) {
		select {
		case <-ctx.Done():
			return true, ctx.Err()
		default:
		}
		scheduled, attempted, unattempted, err := getScheduledPods(podInformer, labelSelector, namespace)
		if err != nil {
			return false, err
		}
		if len(scheduled) >= wantCount {
			tCtx.Logf("scheduling succeed")
			return true, nil
		}
		tCtx.Logf("namespace: %s, pods: want %d, got %d", namespace, wantCount, len(scheduled))
		if len(attempted) > 0 {
			pendingPod = attempted[0]
		} else if len(unattempted) > 0 {
			pendingPod = unattempted[0]
		} else {
			pendingPod = nil
		}
		return false, nil
	})

	if err != nil && pendingPod != nil {
		err = fmt.Errorf("at least pod %s is not scheduled: %w", klog.KObj(pendingPod), err)
	}
	return err
}

func getPodStrategy(cpo *createPodsOp) (testutils.TestPodCreateStrategy, error) {
	podTemplate := testutils.StaticPodTemplate(makeBasePod())
	if cpo.PodTemplatePath != nil {
		podTemplate = podTemplateWithParams{path: *cpo.PodTemplatePath, params: cpo.TemplateParams}
	}
	if cpo.PersistentVolumeClaimTemplatePath == nil {
		return testutils.NewCustomCreatePodStrategy(podTemplate), nil
	}

	pvTemplate, err := getPersistentVolumeSpecFromFile(cpo.PersistentVolumeTemplatePath)
	if err != nil {
		return nil, err
	}
	pvcTemplate, err := getPersistentVolumeClaimSpecFromFile(cpo.PersistentVolumeClaimTemplatePath)
	if err != nil {
		return nil, err
	}
	return testutils.NewCreatePodWithPersistentVolumeStrategy(pvcTemplate, getCustomVolumeFactory(pvTemplate), podTemplate), nil
}

type nodeTemplateWithParams struct {
	path   string
	params map[string]any
}

func (n nodeTemplateWithParams) GetNodeTemplate(index, count int) (*v1.Node, error) {
	env := make(map[string]any)
	maps.Copy(env, n.params)
	env["Index"] = index
	env["Count"] = count
	nodeSpec := &v1.Node{}
	if err := getSpecFromTextTemplateFile(n.path, env, nodeSpec); err != nil {
		return nil, fmt.Errorf("parsing Node: %w", err)
	}
	return nodeSpec, nil
}

type podTemplateWithParams struct {
	path   string
	params map[string]any
}

func (p podTemplateWithParams) GetPodTemplate(index, count int) (*v1.Pod, error) {
	env := make(map[string]any)
	maps.Copy(env, p.params)
	env["Index"] = index
	env["Count"] = count
	podSpec := &v1.Pod{}
	if err := getSpecFromTextTemplateFile(p.path, env, podSpec); err != nil {
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
