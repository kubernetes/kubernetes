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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/api/v1"
	stats "k8s.io/kubernetes/pkg/kubelet/apis/stats/v1alpha1"
	kubemetrics "k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/metrics"
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
		rc *ResourceCollector
	)

	f := framework.NewDefaultFramework("density-test")

	BeforeEach(func() {
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
			desc := fmt.Sprintf("latency/resource should be within limit when create %d pods with %v interval", itArg.podsNr, itArg.interval)
			It(desc, func() {
				itArg.createMethod = "batch"
				testInfo := getTestNodeInfo(f, itArg.getTestName(), desc)

				batchLag, e2eLags := runDensityBatchTest(f, rc, itArg, testInfo, false)

				By("Verifying latency")
				logAndVerifyLatency(batchLag, e2eLags, itArg.podStartupLimits, itArg.podBatchStartupLimit, testInfo, true)

				By("Verifying resource")
				logAndVerifyResource(f, rc, itArg.cpuLimits, itArg.memLimits, testInfo, true)
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
			desc := fmt.Sprintf("latency/resource should be within limit when create %d pods with %v interval [Benchmark]", itArg.podsNr, itArg.interval)
			It(desc, func() {
				itArg.createMethod = "batch"
				testInfo := getTestNodeInfo(f, itArg.getTestName(), desc)

				batchLag, e2eLags := runDensityBatchTest(f, rc, itArg, testInfo, true)

				By("Verifying latency")
				logAndVerifyLatency(batchLag, e2eLags, itArg.podStartupLimits, itArg.podBatchStartupLimit, testInfo, false)

				By("Verifying resource")
				logAndVerifyResource(f, rc, itArg.cpuLimits, itArg.memLimits, testInfo, false)
			})
		}
	})

	Context("create a batch of pods with higher API QPS", func() {
		dTests := []densityTest{
			{
				podsNr:      105,
				interval:    0 * time.Millisecond,
				APIQPSLimit: 60,
			},
			{
				podsNr:      105,
				interval:    100 * time.Millisecond,
				APIQPSLimit: 60,
			},
			{
				podsNr:      105,
				interval:    300 * time.Millisecond,
				APIQPSLimit: 60,
			},
		}

		for _, testArg := range dTests {
			itArg := testArg
			desc := fmt.Sprintf("latency/resource should be within limit when create %d pods with %v interval (QPS %d) [Benchmark]", itArg.podsNr, itArg.interval, itArg.APIQPSLimit)
			It(desc, func() {
				itArg.createMethod = "batch"
				testInfo := getTestNodeInfo(f, itArg.getTestName(), desc)
				// The latency caused by API QPS limit takes a large portion (up to ~33%) of e2e latency.
				// It makes the pod startup latency of Kubelet (creation throughput as well) under-estimated.
				// Here we set API QPS limit from default 5 to 60 in order to test real Kubelet performance.
				// Note that it will cause higher resource usage.
				setKubeletAPIQPSLimit(f, int32(itArg.APIQPSLimit))
				batchLag, e2eLags := runDensityBatchTest(f, rc, itArg, testInfo, true)

				By("Verifying latency")
				logAndVerifyLatency(batchLag, e2eLags, itArg.podStartupLimits, itArg.podBatchStartupLimit, testInfo, false)

				By("Verifying resource")
				logAndVerifyResource(f, rc, itArg.cpuLimits, itArg.memLimits, testInfo, false)
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
			desc := fmt.Sprintf("latency/resource should be within limit when create %d pods with %d background pods", itArg.podsNr, itArg.bgPodsNr)
			It(desc, func() {
				itArg.createMethod = "sequence"
				testInfo := getTestNodeInfo(f, itArg.getTestName(), desc)
				batchlag, e2eLags := runDensitySeqTest(f, rc, itArg, testInfo)

				By("Verifying latency")
				logAndVerifyLatency(batchlag, e2eLags, itArg.podStartupLimits, itArg.podBatchStartupLimit, testInfo, true)

				By("Verifying resource")
				logAndVerifyResource(f, rc, itArg.cpuLimits, itArg.memLimits, testInfo, true)
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
			desc := fmt.Sprintf("latency/resource should be within limit when create %d pods with %d background pods [Benchmark]", itArg.podsNr, itArg.bgPodsNr)
			It(desc, func() {
				itArg.createMethod = "sequence"
				testInfo := getTestNodeInfo(f, itArg.getTestName(), desc)
				batchlag, e2eLags := runDensitySeqTest(f, rc, itArg, testInfo)

				By("Verifying latency")
				logAndVerifyLatency(batchlag, e2eLags, itArg.podStartupLimits, itArg.podBatchStartupLimit, testInfo, false)

				By("Verifying resource")
				logAndVerifyResource(f, rc, itArg.cpuLimits, itArg.memLimits, testInfo, false)
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
	// API QPS limit
	APIQPSLimit int
	// performance limits
	cpuLimits            framework.ContainersCPUSummary
	memLimits            framework.ResourceUsagePerContainer
	podStartupLimits     framework.LatencyMetric
	podBatchStartupLimit time.Duration
}

func (dt *densityTest) getTestName() string {
	// The current default API QPS limit is 5
	// TODO(coufon): is there any way to not hard code this?
	APIQPSLimit := 5
	if dt.APIQPSLimit > 0 {
		APIQPSLimit = dt.APIQPSLimit
	}
	return fmt.Sprintf("density_create_%s_%d_%d_%d_%d", dt.createMethod, dt.podsNr, dt.bgPodsNr,
		dt.interval.Nanoseconds()/1000000, APIQPSLimit)
}

// runDensityBatchTest runs the density batch pod creation test
func runDensityBatchTest(f *framework.Framework, rc *ResourceCollector, testArg densityTest, testInfo map[string]string,
	isLogTimeSeries bool) (time.Duration, []framework.PodLatencyData) {
	const (
		podType               = "density_test_pod"
		sleepBeforeCreatePods = 30 * time.Second
	)
	var (
		mutex      = &sync.Mutex{}
		watchTimes = make(map[string]metav1.Time, 0)
		stopCh     = make(chan struct{})
	)

	// create test pod data structure
	pods := newTestPods(testArg.podsNr, true, framework.GetPauseImageNameForHostArch(), podType)

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
		firstCreate metav1.Time
		lastRunning metav1.Time
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

	rc.Stop()
	deletePodsSync(f, pods)

	// Log time series data.
	if isLogTimeSeries {
		logDensityTimeSeries(rc, createTimes, watchTimes, testInfo)
	}
	// Log throughput data.
	logPodCreateThroughput(batchLag, e2eLags, testArg.podsNr, testInfo)

	deletePodsSync(f, []*v1.Pod{getCadvisorPod()})

	return batchLag, e2eLags
}

// runDensitySeqTest runs the density sequential pod creation test
func runDensitySeqTest(f *framework.Framework, rc *ResourceCollector, testArg densityTest, testInfo map[string]string) (time.Duration, []framework.PodLatencyData) {
	const (
		podType               = "density_test_pod"
		sleepBeforeCreatePods = 30 * time.Second
	)
	bgPods := newTestPods(testArg.bgPodsNr, true, framework.GetPauseImageNameForHostArch(), "background_pod")
	testPods := newTestPods(testArg.podsNr, true, framework.GetPauseImageNameForHostArch(), podType)

	By("Creating a batch of background pods")

	// CreatBatch is synchronized, all pods are running when it returns
	f.PodClient().CreateBatch(bgPods)

	time.Sleep(sleepBeforeCreatePods)

	rc.Start()

	// Create pods sequentially (back-to-back). e2eLags have been sorted.
	batchlag, e2eLags := createBatchPodSequential(f, testPods)

	rc.Stop()
	deletePodsSync(f, append(bgPods, testPods...))

	// Log throughput data.
	logPodCreateThroughput(batchlag, e2eLags, testArg.podsNr, testInfo)

	deletePodsSync(f, []*v1.Pod{getCadvisorPod()})

	return batchlag, e2eLags
}

// createBatchPodWithRateControl creates a batch of pods concurrently, uses one goroutine for each creation.
// between creations there is an interval for throughput control
func createBatchPodWithRateControl(f *framework.Framework, pods []*v1.Pod, interval time.Duration) map[string]metav1.Time {
	createTimes := make(map[string]metav1.Time)
	for _, pod := range pods {
		createTimes[pod.ObjectMeta.Name] = metav1.Now()
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
	if actual.Perc99 > expect.Perc99 {
		return fmt.Errorf("too high pod startup latency 99th percentile: %v", actual.Perc99)
	}
	return nil
}

// newInformerWatchPod creates an informer to check whether all pods are running.
func newInformerWatchPod(f *framework.Framework, mutex *sync.Mutex, watchTimes map[string]metav1.Time, podType string) cache.Controller {
	ns := f.Namespace.Name
	checkPodRunning := func(p *v1.Pod) {
		mutex.Lock()
		defer mutex.Unlock()
		defer GinkgoRecover()

		if p.Status.Phase == v1.PodRunning {
			if _, found := watchTimes[p.Name]; !found {
				watchTimes[p.Name] = metav1.Now()
			}
		}
	}

	_, controller := cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				options.LabelSelector = labels.SelectorFromSet(labels.Set{"type": podType}).String()
				obj, err := f.ClientSet.Core().Pods(ns).List(options)
				return runtime.Object(obj), err
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				options.LabelSelector = labels.SelectorFromSet(labels.Set{"type": podType}).String()
				return f.ClientSet.Core().Pods(ns).Watch(options)
			},
		},
		&v1.Pod{},
		0,
		cache.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				p, ok := obj.(*v1.Pod)
				Expect(ok).To(Equal(true))
				go checkPodRunning(p)
			},
			UpdateFunc: func(oldObj, newObj interface{}) {
				p, ok := newObj.(*v1.Pod)
				Expect(ok).To(Equal(true))
				go checkPodRunning(p)
			},
		},
	)
	return controller
}

// createBatchPodSequential creats pods back-to-back in sequence.
func createBatchPodSequential(f *framework.Framework, pods []*v1.Pod) (time.Duration, []framework.PodLatencyData) {
	batchStartTime := metav1.Now()
	e2eLags := make([]framework.PodLatencyData, 0)
	for _, pod := range pods {
		create := metav1.Now()
		f.PodClient().CreateSync(pod)
		e2eLags = append(e2eLags,
			framework.PodLatencyData{Name: pod.Name, Latency: metav1.Now().Time.Sub(create.Time)})
	}
	batchLag := metav1.Now().Time.Sub(batchStartTime.Time)
	sort.Sort(framework.LatencySlice(e2eLags))
	return batchLag, e2eLags
}

// logAndVerifyLatency verifies that whether pod creation latency satisfies the limit.
func logAndVerifyLatency(batchLag time.Duration, e2eLags []framework.PodLatencyData, podStartupLimits framework.LatencyMetric,
	podBatchStartupLimit time.Duration, testInfo map[string]string, isVerify bool) {
	framework.PrintLatencies(e2eLags, "worst client e2e total latencies")

	// TODO(coufon): do not trust 'kubelet' metrics since they are not reset!
	latencyMetrics, _ := getPodStartLatency(kubeletAddr)
	framework.Logf("Kubelet Prometheus metrics (not reset):\n%s", framework.PrettyPrintJSON(latencyMetrics))

	podCreateLatency := framework.PodStartupLatency{Latency: framework.ExtractLatencyMetrics(e2eLags)}

	// log latency perf data
	logPerfData(getLatencyPerfData(podCreateLatency.Latency, testInfo), "latency")

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
func logPodCreateThroughput(batchLag time.Duration, e2eLags []framework.PodLatencyData, podsNr int, testInfo map[string]string) {
	logPerfData(getThroughputPerfData(batchLag, e2eLags, podsNr, testInfo), "throughput")
}

// increaseKubeletAPIQPSLimit sets Kubelet API QPS via ConfigMap. Kubelet will restart with the new QPS.
func setKubeletAPIQPSLimit(f *framework.Framework, newAPIQPS int32) {
	const restartGap = 40 * time.Second

	resp := pollConfigz(2*time.Minute, 5*time.Second)
	kubeCfg, err := decodeConfigz(resp)
	framework.ExpectNoError(err)
	framework.Logf("Old QPS limit is: %d\n", kubeCfg.KubeAPIQPS)

	// Set new API QPS limit
	kubeCfg.KubeAPIQPS = newAPIQPS
	// TODO(coufon): createConfigMap should firstly check whether configmap already exists, if so, use updateConfigMap.
	// Calling createConfigMap twice will result in error. It is fine for benchmark test because we only run one test on a new node.
	_, err = createConfigMap(f, kubeCfg)
	framework.ExpectNoError(err)

	// Wait for Kubelet to restart
	time.Sleep(restartGap)

	// Check new QPS has been set
	resp = pollConfigz(2*time.Minute, 5*time.Second)
	kubeCfg, err = decodeConfigz(resp)
	framework.ExpectNoError(err)
	framework.Logf("New QPS limit is: %d\n", kubeCfg.KubeAPIQPS)

	// TODO(coufon): check test result to see if we need to retry here
	if kubeCfg.KubeAPIQPS != newAPIQPS {
		framework.Failf("Fail to set new kubelet API QPS limit.")
	}
}
