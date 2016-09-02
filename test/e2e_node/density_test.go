// +build linux

/*
Copyright 2015 The Kubernetes Authors.

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

package e2e_node

import (
	"fmt"
	"sort"
	"strconv"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/cache"
	controllerframework "k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/stats"
	kubemetrics "k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/metrics"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	kubeletAddr = "localhost:10255"
)

var _ = framework.KubeDescribe("Density [Serial] [Slow]", func() {
	const (
		// The data collection time of resource collector and the standalone cadvisor
		// is not synchronizated, so resource collector may miss data or
		// collect duplicated data
		containerStatsPollingPeriod = 500 * time.Millisecond
	)

	var (
		ns       string
		nodeName string
		rc       *ResourceCollector
	)

	f := framework.NewDefaultFramework("density-test")

	BeforeEach(func() {
		ns = f.Namespace.Name
		nodeName = framework.TestContext.NodeName
		// Start a standalone cadvisor pod using 'createSync', the pod is running when it returns
		f.PodClient().CreateSync(getCadvisorPod())
		// Resource collector monitors fine-grain CPU/memory usage by a standalone Cadvisor with
		// 1s housingkeeping interval
		rc = NewResourceCollector(containerStatsPollingPeriod)
	})

	Context("create a batch of pods", func() {
		// TODO(coufon): the values are generous, set more precise limits with benchmark data
		// and add more tests
		dTests := []densityTest{
			{
				podsNr:   10,
				interval: 0 * time.Millisecond,
				cpuLimits: framework.ContainersCPUSummary{
					stats.SystemContainerKubelet: {0.50: 0.30, 0.95: 0.50},
					stats.SystemContainerRuntime: {0.50: 0.40, 0.95: 0.60},
				},
				memLimits: framework.ResourceUsagePerContainer{
					stats.SystemContainerKubelet: &framework.ContainerResourceUsage{MemoryRSSInBytes: 100 * 1024 * 1024},
					stats.SystemContainerRuntime: &framework.ContainerResourceUsage{MemoryRSSInBytes: 500 * 1024 * 1024},
				},
				// percentile limit of single pod startup latency
				podStartupLimits: framework.LatencyMetric{
					Perc50: 16 * time.Second,
					Perc90: 18 * time.Second,
					Perc99: 20 * time.Second,
				},
				// upbound of startup latency of a batch of pods
				podBatchStartupLimit: 25 * time.Second,
			},
		}

		for _, testArg := range dTests {
			itArg := testArg
			It(fmt.Sprintf("latency/resource should be within limit when create %d pods with %v interval",
				itArg.podsNr, itArg.interval), func() {
				itArg.createMethod = "batch"
				testName := itArg.getTestName()
				batchLag, e2eLags := runDensityBatchTest(f, rc, itArg, false)

				By("Verifying latency")
				logAndVerifyLatency(batchLag, e2eLags, itArg.podStartupLimits, itArg.podBatchStartupLimit, testName, true)

				By("Verifying resource")
				logAndVerifyResource(f, rc, itArg.cpuLimits, itArg.memLimits, testName, true)
			})
		}
	})

	Context("create a batch of pods", func() {
		dTests := []densityTest{
			{
				podsNr:   10,
				interval: 0 * time.Millisecond,
			},
			{
				podsNr:   35,
				interval: 0 * time.Millisecond,
			},
			{
				podsNr:   105,
				interval: 0 * time.Millisecond,
			},
			{
				podsNr:   10,
				interval: 100 * time.Millisecond,
			},
			{
				podsNr:   35,
				interval: 100 * time.Millisecond,
			},
			{
				podsNr:   105,
				interval: 100 * time.Millisecond,
			},
			{
				podsNr:   10,
				interval: 300 * time.Millisecond,
			},
			{
				podsNr:   35,
				interval: 300 * time.Millisecond,
			},
			{
				podsNr:   105,
				interval: 300 * time.Millisecond,
			},
		}

		for _, testArg := range dTests {
			itArg := testArg
			It(fmt.Sprintf("latency/resource should be within limit when create %d pods with %v interval [Benchmark]",
				itArg.podsNr, itArg.interval), func() {
				itArg.createMethod = "batch"
				testName := itArg.getTestName()
				batchLag, e2eLags := runDensityBatchTest(f, rc, itArg, true)

				By("Verifying latency")
				logAndVerifyLatency(batchLag, e2eLags, itArg.podStartupLimits, itArg.podBatchStartupLimit, testName, false)

				By("Verifying resource")
				logAndVerifyResource(f, rc, itArg.cpuLimits, itArg.memLimits, testName, false)
			})
		}
	})

	Context("create a sequence of pods", func() {
		dTests := []densityTest{
			{
				podsNr:   10,
				bgPodsNr: 50,
				cpuLimits: framework.ContainersCPUSummary{
					stats.SystemContainerKubelet: {0.50: 0.30, 0.95: 0.50},
					stats.SystemContainerRuntime: {0.50: 0.40, 0.95: 0.60},
				},
				memLimits: framework.ResourceUsagePerContainer{
					stats.SystemContainerKubelet: &framework.ContainerResourceUsage{MemoryRSSInBytes: 100 * 1024 * 1024},
					stats.SystemContainerRuntime: &framework.ContainerResourceUsage{MemoryRSSInBytes: 500 * 1024 * 1024},
				},
				podStartupLimits: framework.LatencyMetric{
					Perc50: 5000 * time.Millisecond,
					Perc90: 9000 * time.Millisecond,
					Perc99: 10000 * time.Millisecond,
				},
			},
		}

		for _, testArg := range dTests {
			itArg := testArg
			It(fmt.Sprintf("latency/resource should be within limit when create %d pods with %d background pods",
				itArg.podsNr, itArg.bgPodsNr), func() {
				itArg.createMethod = "sequence"
				testName := itArg.getTestName()
				batchlag, e2eLags := runDensitySeqTest(f, rc, itArg)

				By("Verifying latency")
				logAndVerifyLatency(batchlag, e2eLags, itArg.podStartupLimits, itArg.podBatchStartupLimit, testName, true)

				By("Verifying resource")
				logAndVerifyResource(f, rc, itArg.cpuLimits, itArg.memLimits, testName, true)
			})
		}
	})

	Context("create a sequence of pods", func() {
		dTests := []densityTest{
			{
				podsNr:   10,
				bgPodsNr: 50,
			},
			{
				podsNr:   30,
				bgPodsNr: 50,
			},
			{
				podsNr:   50,
				bgPodsNr: 50,
			},
		}

		for _, testArg := range dTests {
			itArg := testArg
			It(fmt.Sprintf("latency/resource should be within limit when create %d pods with %d background pods [Benchmark]",
				itArg.podsNr, itArg.bgPodsNr), func() {
				itArg.createMethod = "sequence"
				testName := itArg.getTestName()
				batchlag, e2eLags := runDensitySeqTest(f, rc, itArg)

				By("Verifying latency")
				logAndVerifyLatency(batchlag, e2eLags, itArg.podStartupLimits, itArg.podBatchStartupLimit, testName, false)

				By("Verifying resource")
				logAndVerifyResource(f, rc, itArg.cpuLimits, itArg.memLimits, testName, false)
			})
		}
	})
})

type densityTest struct {
	// number of pods
	podsNr int
	// number of background pods
	bgPodsNr int
	// interval between creating pod (rate control)
	interval time.Duration
	// create pods in 'batch' or 'sequence'
	createMethod string
	// performance limits
	cpuLimits            framework.ContainersCPUSummary
	memLimits            framework.ResourceUsagePerContainer
	podStartupLimits     framework.LatencyMetric
	podBatchStartupLimit time.Duration
}

func (dt *densityTest) getTestName() string {
	return fmt.Sprintf("density_create_%s_%d_%d_%d", dt.createMethod, dt.podsNr, dt.bgPodsNr, dt.interval.Nanoseconds()/1000000)
}

// runDensityBatchTest runs the density batch pod creation test
func runDensityBatchTest(f *framework.Framework, rc *ResourceCollector, testArg densityTest,
	isLogTimeSeries bool) (time.Duration, []framework.PodLatencyData) {
	const (
		podType               = "density_test_pod"
		sleepBeforeCreatePods = 30 * time.Second
	)
	var (
		mutex      = &sync.Mutex{}
		watchTimes = make(map[string]unversioned.Time, 0)
		stopCh     = make(chan struct{})
	)

	// create test pod data structure
	pods := newTestPods(testArg.podsNr, ImageRegistry[pauseImage], podType)

	// the controller watches the change of pod status
	controller := newInformerWatchPod(f, mutex, watchTimes, podType)
	go controller.Run(stopCh)
	defer close(stopCh)

	// TODO(coufon): in the test we found kubelet starts while it is busy on something, as a result 'syncLoop'
	// does not response to pod creation immediately. Creating the first pod has a delay around 5s.
	// The node status has already been 'ready' so `wait and check node being ready does not help here.
	// Now wait here for a grace period to let 'syncLoop' be ready
	time.Sleep(sleepBeforeCreatePods)

	rc.Start()
	// Explicitly delete pods to prevent namespace controller cleanning up timeout
	defer deletePodsSync(f, append(pods, getCadvisorPod()))
	defer rc.Stop()

	By("Creating a batch of pods")
	// It returns a map['pod name']'creation time' containing the creation timestamps
	createTimes := createBatchPodWithRateControl(f, pods, testArg.interval)

	By("Waiting for all Pods to be observed by the watch...")

	Eventually(func() bool {
		return len(watchTimes) == testArg.podsNr
	}, 10*time.Minute, 10*time.Second).Should(BeTrue())

	if len(watchTimes) < testArg.podsNr {
		framework.Failf("Timeout reached waiting for all Pods to be observed by the watch.")
	}

	// Analyze results
	var (
		firstCreate unversioned.Time
		lastRunning unversioned.Time
		init        = true
		e2eLags     = make([]framework.PodLatencyData, 0)
	)

	for name, create := range createTimes {
		watch, ok := watchTimes[name]
		Expect(ok).To(Equal(true))

		e2eLags = append(e2eLags,
			framework.PodLatencyData{Name: name, Latency: watch.Time.Sub(create.Time)})

		if !init {
			if firstCreate.Time.After(create.Time) {
				firstCreate = create
			}
			if lastRunning.Time.Before(watch.Time) {
				lastRunning = watch
			}
		} else {
			init = false
			firstCreate, lastRunning = create, watch
		}
	}

	sort.Sort(framework.LatencySlice(e2eLags))
	batchLag := lastRunning.Time.Sub(firstCreate.Time)

	testName := testArg.getTestName()
	// Log time series data.
	if isLogTimeSeries {
		logDensityTimeSeries(rc, createTimes, watchTimes, testName)
	}
	// Log throughput data.
	logPodCreateThroughput(batchLag, e2eLags, testArg.podsNr, testName)

	return batchLag, e2eLags
}

// runDensitySeqTest runs the density sequential pod creation test
func runDensitySeqTest(f *framework.Framework, rc *ResourceCollector, testArg densityTest) (time.Duration, []framework.PodLatencyData) {
	const (
		podType               = "density_test_pod"
		sleepBeforeCreatePods = 30 * time.Second
	)
	bgPods := newTestPods(testArg.bgPodsNr, ImageRegistry[pauseImage], "background_pod")
	testPods := newTestPods(testArg.podsNr, ImageRegistry[pauseImage], podType)

	By("Creating a batch of background pods")

	// CreatBatch is synchronized, all pods are running when it returns
	f.PodClient().CreateBatch(bgPods)

	time.Sleep(sleepBeforeCreatePods)

	rc.Start()
	// Explicitly delete pods to prevent namespace controller cleanning up timeout
	defer deletePodsSync(f, append(bgPods, append(testPods, getCadvisorPod())...))
	defer rc.Stop()

	// Create pods sequentially (back-to-back). e2eLags have been sorted.
	batchlag, e2eLags := createBatchPodSequential(f, testPods)

	// Log throughput data.
	logPodCreateThroughput(batchlag, e2eLags, testArg.podsNr, testArg.getTestName())

	return batchlag, e2eLags
}

// createBatchPodWithRateControl creates a batch of pods concurrently, uses one goroutine for each creation.
// between creations there is an interval for throughput control
func createBatchPodWithRateControl(f *framework.Framework, pods []*api.Pod, interval time.Duration) map[string]unversioned.Time {
	createTimes := make(map[string]unversioned.Time)
	for _, pod := range pods {
		createTimes[pod.ObjectMeta.Name] = unversioned.Now()
		go f.PodClient().Create(pod)
		time.Sleep(interval)
	}
	return createTimes
}

// getPodStartLatency gets prometheus metric 'pod start latency' from kubelet
func getPodStartLatency(node string) (framework.KubeletLatencyMetrics, error) {
	latencyMetrics := framework.KubeletLatencyMetrics{}
	ms, err := metrics.GrabKubeletMetricsWithoutProxy(node)
	Expect(err).NotTo(HaveOccurred())

	for _, samples := range ms {
		for _, sample := range samples {
			if sample.Metric["__name__"] == kubemetrics.KubeletSubsystem+"_"+kubemetrics.PodStartLatencyKey {
				quantile, _ := strconv.ParseFloat(string(sample.Metric["quantile"]), 64)
				latencyMetrics = append(latencyMetrics,
					framework.KubeletLatencyMetric{
						Quantile: quantile,
						Method:   kubemetrics.PodStartLatencyKey,
						Latency:  time.Duration(int(sample.Value)) * time.Microsecond})
			}
		}
	}
	return latencyMetrics, nil
}

// verifyPodStartupLatency verifies whether 50, 90 and 99th percentiles of PodStartupLatency are
// within the threshold.
func verifyPodStartupLatency(expect, actual framework.LatencyMetric) error {
	if actual.Perc50 > expect.Perc50 {
		return fmt.Errorf("too high pod startup latency 50th percentile: %v", actual.Perc50)
	}
	if actual.Perc90 > expect.Perc90 {
		return fmt.Errorf("too high pod startup latency 90th percentile: %v", actual.Perc90)
	}
	if actual.Perc99 > actual.Perc99 {
		return fmt.Errorf("too high pod startup latency 99th percentil: %v", actual.Perc99)
	}
	return nil
}

// newInformerWatchPod creates an informer to check whether all pods are running.
func newInformerWatchPod(f *framework.Framework, mutex *sync.Mutex, watchTimes map[string]unversioned.Time,
	podType string) *controllerframework.Controller {
	ns := f.Namespace.Name
	checkPodRunning := func(p *api.Pod) {
		mutex.Lock()
		defer mutex.Unlock()
		defer GinkgoRecover()

		if p.Status.Phase == api.PodRunning {
			if _, found := watchTimes[p.Name]; !found {
				watchTimes[p.Name] = unversioned.Now()
			}
		}
	}

	_, controller := controllerframework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				options.LabelSelector = labels.SelectorFromSet(labels.Set{"type": podType})
				return f.Client.Pods(ns).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				options.LabelSelector = labels.SelectorFromSet(labels.Set{"type": podType})
				return f.Client.Pods(ns).Watch(options)
			},
		},
		&api.Pod{},
		0,
		controllerframework.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				p, ok := obj.(*api.Pod)
				Expect(ok).To(Equal(true))
				go checkPodRunning(p)
			},
			UpdateFunc: func(oldObj, newObj interface{}) {
				p, ok := newObj.(*api.Pod)
				Expect(ok).To(Equal(true))
				go checkPodRunning(p)
			},
		},
	)
	return controller
}

// createBatchPodSequential creats pods back-to-back in sequence.
func createBatchPodSequential(f *framework.Framework, pods []*api.Pod) (time.Duration, []framework.PodLatencyData) {
	batchStartTime := unversioned.Now()
	e2eLags := make([]framework.PodLatencyData, 0)
	for _, pod := range pods {
		create := unversioned.Now()
		f.PodClient().CreateSync(pod)
		e2eLags = append(e2eLags,
			framework.PodLatencyData{Name: pod.Name, Latency: unversioned.Now().Time.Sub(create.Time)})
	}
	batchLag := unversioned.Now().Time.Sub(batchStartTime.Time)
	sort.Sort(framework.LatencySlice(e2eLags))
	return batchLag, e2eLags
}

// logAndVerifyLatency verifies that whether pod creation latency satisfies the limit.
func logAndVerifyLatency(batchLag time.Duration, e2eLags []framework.PodLatencyData, podStartupLimits framework.LatencyMetric,
	podBatchStartupLimit time.Duration, testName string, isVerify bool) {
	framework.PrintLatencies(e2eLags, "worst client e2e total latencies")

	// TODO(coufon): do not trust 'kubelet' metrics since they are not reset!
	latencyMetrics, _ := getPodStartLatency(kubeletAddr)
	framework.Logf("Kubelet Prometheus metrics (not reset):\n%s", framework.PrettyPrintJSON(latencyMetrics))

	podCreateLatency := framework.PodStartupLatency{Latency: framework.ExtractLatencyMetrics(e2eLags)}

	// log latency perf data
	framework.PrintPerfData(getLatencyPerfData(podCreateLatency.Latency, testName))

	if isVerify {
		// check whether e2e pod startup time is acceptable.
		framework.ExpectNoError(verifyPodStartupLatency(podStartupLimits, podCreateLatency.Latency))

		// check bactch pod creation latency
		if podBatchStartupLimit > 0 {
			Expect(batchLag <= podBatchStartupLimit).To(Equal(true), "Batch creation startup time %v exceed limit %v",
				batchLag, podBatchStartupLimit)
		}
	}
}

// logThroughput calculates and logs pod creation throughput.
func logPodCreateThroughput(batchLag time.Duration, e2eLags []framework.PodLatencyData, podsNr int, testName string) {
	framework.PrintPerfData(getThroughputPerfData(batchLag, e2eLags, podsNr, testName))
}
