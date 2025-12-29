//go:build linux

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

package e2enode

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strconv"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
	kubeletstatsv1alpha1 "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubemetrics "k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubelet "k8s.io/kubernetes/test/e2e/framework/kubelet"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	kubeletAddr = "localhost:10255"
)

var _ = SIGDescribe("Density", framework.WithSerial(), framework.WithSlow(), func() {
	const (
		// The data collection time of resource collector and the standalone cadvisor
		// is not synchronized, so resource collector may miss data or
		// collect duplicated data
		containerStatsPollingPeriod = 500 * time.Millisecond
	)

	var (
		rc *ResourceCollector
	)

	f := framework.NewDefaultFramework("density-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.BeforeEach(func(ctx context.Context) {
		// Start a standalone cadvisor pod using 'createSync', the pod is running when it returns
		e2epod.NewPodClient(f).CreateSync(ctx, getCadvisorPod())
		// Resource collector monitors fine-grain CPU/memory usage by a standalone Cadvisor with
		// 1s housingkeeping interval
		rc = NewResourceCollector(containerStatsPollingPeriod)
	})

	f.Context("create a batch of pods", framework.WithFlaky(), func() {
		// If this test case fails with am error similar to
		// "container "runtime": expected 95th% usage < 0.900; got 0.941",
		// it is likely that cpuLimits or memLimits need to be increased.
		// Actual resource usage values can be found in the test output, e.g.:
		// I1029 11:28:15.671913 1005 resource_usage_test.go:206] CPU usage of containers:
		// container 50th% 90th% 95th% 99th% 100th%
		// "runtime" 0.004 0.941 0.941 0.941 0.976
		// "kubelet" 0.009 0.082 0.082 0.082 0.101
		dTests := []densityTest{
			{
				podsNr:   10,
				interval: 0 * time.Millisecond,
				cpuLimits: e2ekubelet.ContainersCPUSummary{
					kubeletstatsv1alpha1.SystemContainerKubelet: {0.50: 0.1, 0.95: 0.20},
					kubeletstatsv1alpha1.SystemContainerRuntime: {0.50: 0.1, 0.95: 1.5},
				},
				memLimits: e2ekubelet.ResourceUsagePerContainer{
					kubeletstatsv1alpha1.SystemContainerKubelet: &e2ekubelet.ContainerResourceUsage{MemoryRSSInBytes: 50 * 1024 * 1024},
					kubeletstatsv1alpha1.SystemContainerRuntime: &e2ekubelet.ContainerResourceUsage{MemoryRSSInBytes: 500 * 1024 * 1024},
				},
				// percentile limit of single pod startup latency
				podStartupLimits: e2emetrics.LatencyMetric{
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
			ginkgo.It(desc, func(ctx context.Context) {
				itArg.createMethod = "batch"
				testInfo := getTestNodeInfo(f, itArg.getTestName(), desc)

				batchLag, e2eLags := runDensityBatchTest(ctx, f, rc, itArg, testInfo, false)

				ginkgo.By("Verifying latency")
				logAndVerifyLatency(ctx, batchLag, e2eLags, itArg.podStartupLimits, itArg.podBatchStartupLimit, testInfo, true)

				ginkgo.By("Verifying resource")
				logAndVerifyResource(ctx, f, rc, itArg.cpuLimits, itArg.memLimits, testInfo, true)
			})
		}
	})

	ginkgo.Context("create a batch of pods", func() {
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
				podsNr:   90,
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
				podsNr:   90,
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
				podsNr:   90,
				interval: 300 * time.Millisecond,
			},
		}

		for _, testArg := range dTests {
			itArg := testArg
			desc := fmt.Sprintf("latency/resource should be within limit when create %d pods with %v interval [Benchmark]", itArg.podsNr, itArg.interval)
			ginkgo.It(desc, func(ctx context.Context) {
				itArg.createMethod = "batch"
				testInfo := getTestNodeInfo(f, itArg.getTestName(), desc)

				batchLag, e2eLags := runDensityBatchTest(ctx, f, rc, itArg, testInfo, true)

				ginkgo.By("Verifying latency")
				logAndVerifyLatency(ctx, batchLag, e2eLags, itArg.podStartupLimits, itArg.podBatchStartupLimit, testInfo, false)

				ginkgo.By("Verifying resource")
				logAndVerifyResource(ctx, f, rc, itArg.cpuLimits, itArg.memLimits, testInfo, false)
			})
		}
	})

	ginkgo.Context("create a batch of pods with higher API QPS", func() {
		dTests := []densityTest{
			{
				podsNr:      90,
				interval:    0 * time.Millisecond,
				APIQPSLimit: 60,
			},
			{
				podsNr:      90,
				interval:    100 * time.Millisecond,
				APIQPSLimit: 60,
			},
			{
				podsNr:      90,
				interval:    300 * time.Millisecond,
				APIQPSLimit: 60,
			},
		}

		for _, testArg := range dTests {
			itArg := testArg
			ginkgo.Context("", func() {
				desc := fmt.Sprintf("latency/resource should be within limit when create %d pods with %v interval (QPS %d) [Benchmark]", itArg.podsNr, itArg.interval, itArg.APIQPSLimit)
				// The latency caused by API QPS limit takes a large portion (up to ~33%) of e2e latency.
				// It makes the pod startup latency of Kubelet (creation throughput as well) under-estimated.
				// Here we set API QPS limit from default 5 to 60 in order to test real Kubelet performance.
				// Note that it will cause higher resource usage.
				tempSetCurrentKubeletConfig(f, func(ctx context.Context, cfg *kubeletconfig.KubeletConfiguration) {
					framework.Logf("Old QPS limit is: %d", cfg.KubeAPIQPS)
					// Set new API QPS limit
					cfg.KubeAPIQPS = int32(itArg.APIQPSLimit)
				})
				ginkgo.It(desc, func(ctx context.Context) {
					itArg.createMethod = "batch"
					testInfo := getTestNodeInfo(f, itArg.getTestName(), desc)
					batchLag, e2eLags := runDensityBatchTest(ctx, f, rc, itArg, testInfo, true)

					ginkgo.By("Verifying latency")
					logAndVerifyLatency(ctx, batchLag, e2eLags, itArg.podStartupLimits, itArg.podBatchStartupLimit, testInfo, false)

					ginkgo.By("Verifying resource")
					logAndVerifyResource(ctx, f, rc, itArg.cpuLimits, itArg.memLimits, testInfo, false)
				})
			})
		}
	})

	f.Context("create a sequence of pods", framework.WithFlaky(), func() {
		dTests := []densityTest{
			{
				podsNr:   10,
				bgPodsNr: 50,
				cpuLimits: e2ekubelet.ContainersCPUSummary{
					kubeletstatsv1alpha1.SystemContainerKubelet: {0.50: 0.30, 0.95: 0.50},
					kubeletstatsv1alpha1.SystemContainerRuntime: {0.50: 0.40, 0.95: 0.60},
				},
				memLimits: e2ekubelet.ResourceUsagePerContainer{
					kubeletstatsv1alpha1.SystemContainerKubelet: &e2ekubelet.ContainerResourceUsage{MemoryRSSInBytes: 100 * 1024 * 1024},
					kubeletstatsv1alpha1.SystemContainerRuntime: &e2ekubelet.ContainerResourceUsage{MemoryRSSInBytes: 500 * 1024 * 1024},
				},
				podStartupLimits: e2emetrics.LatencyMetric{
					Perc50: 5000 * time.Millisecond,
					Perc90: 9000 * time.Millisecond,
					Perc99: 10000 * time.Millisecond,
				},
			},
		}

		for _, testArg := range dTests {
			itArg := testArg
			desc := fmt.Sprintf("latency/resource should be within limit when create %d pods with %d background pods", itArg.podsNr, itArg.bgPodsNr)
			ginkgo.It(desc, func(ctx context.Context) {
				itArg.createMethod = "sequence"
				testInfo := getTestNodeInfo(f, itArg.getTestName(), desc)
				batchlag, e2eLags := runDensitySeqTest(ctx, f, rc, itArg, testInfo)

				ginkgo.By("Verifying latency")
				logAndVerifyLatency(ctx, batchlag, e2eLags, itArg.podStartupLimits, itArg.podBatchStartupLimit, testInfo, true)

				ginkgo.By("Verifying resource")
				logAndVerifyResource(ctx, f, rc, itArg.cpuLimits, itArg.memLimits, testInfo, true)
			})
		}
	})

	ginkgo.Context("create a sequence of pods", func() {
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
			ginkgo.It(desc, func(ctx context.Context) {
				itArg.createMethod = "sequence"
				testInfo := getTestNodeInfo(f, itArg.getTestName(), desc)
				batchlag, e2eLags := runDensitySeqTest(ctx, f, rc, itArg, testInfo)

				ginkgo.By("Verifying latency")
				logAndVerifyLatency(ctx, batchlag, e2eLags, itArg.podStartupLimits, itArg.podBatchStartupLimit, testInfo, false)

				ginkgo.By("Verifying resource")
				logAndVerifyResource(ctx, f, rc, itArg.cpuLimits, itArg.memLimits, testInfo, false)
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
	cpuLimits            e2ekubelet.ContainersCPUSummary
	memLimits            e2ekubelet.ResourceUsagePerContainer
	podStartupLimits     e2emetrics.LatencyMetric
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
func runDensityBatchTest(ctx context.Context, f *framework.Framework, rc *ResourceCollector, testArg densityTest, testInfo map[string]string,
	isLogTimeSeries bool) (time.Duration, []e2emetrics.PodLatencyData) {
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
	pods := newTestPods(testArg.podsNr, true, imageutils.GetPauseImageName(), podType)

	// the controller watches the change of pod status
	controller := newInformerWatchPod(ctx, f, mutex, watchTimes, podType)
	go controller.Run(stopCh)
	defer close(stopCh)

	// TODO(coufon): in the test we found kubelet starts while it is busy on something, as a result 'syncLoop'
	// does not response to pod creation immediately. Creating the first pod has a delay around 5s.
	// The node status has already been 'ready' so `wait and check node being ready does not help here.
	// Now wait here for a grace period to let 'syncLoop' be ready
	time.Sleep(sleepBeforeCreatePods)

	rc.Start()
	ginkgo.DeferCleanup(rc.Stop)

	ginkgo.By("Creating a batch of pods")
	// It returns a map['pod name']'creation time' containing the creation timestamps
	createTimes := createBatchPodWithRateControl(ctx, f, pods, testArg.interval)
	ginkgo.DeferCleanup(deletePodsSync, f, pods)
	ginkgo.DeferCleanup(deletePodsSync, f, []*v1.Pod{getCadvisorPod()})

	ginkgo.By("Waiting for all Pods to be observed by the watch...")

	gomega.Eventually(ctx, func() bool {
		return len(watchTimes) == testArg.podsNr
	}, 10*time.Minute, 10*time.Second).Should(gomega.BeTrueBecause("All pods should be observed by the watch."))

	if len(watchTimes) < testArg.podsNr {
		framework.Failf("Timeout reached waiting for all Pods to be observed by the watch.")
	}

	// Analyze results
	var (
		firstCreate metav1.Time
		lastRunning metav1.Time
		init        = true
		e2eLags     = make([]e2emetrics.PodLatencyData, 0)
	)

	for name, create := range createTimes {
		watch := watchTimes[name]
		gomega.Expect(watchTimes).To(gomega.HaveKey(name))

		e2eLags = append(e2eLags,
			e2emetrics.PodLatencyData{Name: name, Latency: watch.Time.Sub(create.Time)})

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

	sort.Sort(e2emetrics.LatencySlice(e2eLags))
	batchLag := lastRunning.Time.Sub(firstCreate.Time)

	// Log time series data.
	if isLogTimeSeries {
		logDensityTimeSeries(rc, createTimes, watchTimes, testInfo)
	}
	// Log throughput data.
	logPodCreateThroughput(batchLag, e2eLags, testArg.podsNr, testInfo)

	return batchLag, e2eLags
}

// runDensitySeqTest runs the density sequential pod creation test
func runDensitySeqTest(ctx context.Context, f *framework.Framework, rc *ResourceCollector, testArg densityTest, testInfo map[string]string) (time.Duration, []e2emetrics.PodLatencyData) {
	const (
		podType               = "density_test_pod"
		sleepBeforeCreatePods = 30 * time.Second
	)
	bgPods := newTestPods(testArg.bgPodsNr, true, imageutils.GetPauseImageName(), "background_pod")
	testPods := newTestPods(testArg.podsNr, true, imageutils.GetPauseImageName(), podType)

	ginkgo.By("Creating a batch of background pods")

	// CreatBatch is synchronized, all pods are running when it returns
	e2epod.NewPodClient(f).CreateBatch(ctx, bgPods)
	ginkgo.DeferCleanup(deletePodsSync, f, bgPods)
	ginkgo.DeferCleanup(deletePodsSync, f, []*v1.Pod{getCadvisorPod()})

	time.Sleep(sleepBeforeCreatePods)

	rc.Start()
	ginkgo.DeferCleanup(rc.Stop)

	// Create pods sequentially (back-to-back). e2eLags have been sorted.
	batchlag, e2eLags := createBatchPodSequential(ctx, f, testPods, podType)
	ginkgo.DeferCleanup(deletePodsSync, f, testPods)

	// Log throughput data.
	logPodCreateThroughput(batchlag, e2eLags, testArg.podsNr, testInfo)

	return batchlag, e2eLags
}

// createBatchPodWithRateControl creates a batch of pods concurrently, uses one goroutine for each creation.
// between creations there is an interval for throughput control
func createBatchPodWithRateControl(ctx context.Context, f *framework.Framework, pods []*v1.Pod, interval time.Duration) map[string]metav1.Time {
	createTimes := make(map[string]metav1.Time)
	for i := range pods {
		pod := pods[i]
		createTimes[pod.ObjectMeta.Name] = metav1.Now()
		go e2epod.NewPodClient(f).Create(ctx, pod)
		time.Sleep(interval)
	}
	return createTimes
}

// getPodStartLatency gets prometheus metric 'pod start latency' from kubelet
func getPodStartLatency(ctx context.Context, node string) (e2emetrics.KubeletLatencyMetrics, error) {
	latencyMetrics := e2emetrics.KubeletLatencyMetrics{}
	ms, err := e2emetrics.GrabKubeletMetricsWithoutProxy(ctx, node, "/metrics")
	framework.ExpectNoError(err, "Failed to get kubelet metrics without proxy in node %s", node)

	for _, samples := range ms {
		for _, sample := range samples {
			if sample.Metric["__name__"] == kubemetrics.KubeletSubsystem+"_"+kubemetrics.PodStartDurationKey {
				quantile, _ := strconv.ParseFloat(string(sample.Metric["quantile"]), 64)
				latencyMetrics = append(latencyMetrics,
					e2emetrics.KubeletLatencyMetric{
						Quantile: quantile,
						Method:   kubemetrics.PodStartDurationKey,
						Latency:  time.Duration(int(sample.Value)) * time.Microsecond})
			}
		}
	}
	return latencyMetrics, nil
}

// newInformerWatchPod creates an informer to check whether all pods are running.
func newInformerWatchPod(ctx context.Context, f *framework.Framework, mutex *sync.Mutex, watchTimes map[string]metav1.Time, podType string) cache.Controller {
	ns := f.Namespace.Name
	checkPodRunning := func(p *v1.Pod) {
		mutex.Lock()
		defer mutex.Unlock()
		defer ginkgo.GinkgoRecover()

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
				obj, err := f.ClientSet.CoreV1().Pods(ns).List(ctx, options)
				return runtime.Object(obj), err
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				options.LabelSelector = labels.SelectorFromSet(labels.Set{"type": podType}).String()
				return f.ClientSet.CoreV1().Pods(ns).Watch(ctx, options)
			},
		},
		&v1.Pod{},
		0,
		cache.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				p, ok := obj.(*v1.Pod)
				if !ok {
					framework.Failf("Failed to cast object %T to Pod", obj)
				}
				go checkPodRunning(p)
			},
			UpdateFunc: func(oldObj, newObj interface{}) {
				p, ok := newObj.(*v1.Pod)
				if !ok {
					framework.Failf("Failed to cast object %T to Pod", newObj)
				}
				go checkPodRunning(p)
			},
		},
	)
	return controller
}

// createBatchPodSequential creates pods back-to-back in sequence.
func createBatchPodSequential(ctx context.Context, f *framework.Framework, pods []*v1.Pod, podType string) (time.Duration, []e2emetrics.PodLatencyData) {
	var (
		mutex       = &sync.Mutex{}
		watchTimes  = make(map[string]metav1.Time, 0)
		stopCh      = make(chan struct{})
		firstCreate metav1.Time
		lastRunning metav1.Time
		init        = true
	)
	// the controller watches the change of pod status
	controller := newInformerWatchPod(ctx, f, mutex, watchTimes, podType)
	go controller.Run(stopCh)
	defer close(stopCh)

	batchStartTime := metav1.Now()
	e2eLags := make([]e2emetrics.PodLatencyData, 0)
	createTimes := make(map[string]metav1.Time)
	for _, pod := range pods {
		create := metav1.Now()
		createTimes[pod.Name] = create
		p := e2epod.NewPodClient(f).Create(ctx, pod)
		framework.ExpectNoError(wait.PollUntilContextTimeout(ctx, 2*time.Second, framework.PodStartTimeout, true, podWatchedRunning(watchTimes, p.Name)))
		e2eLags = append(e2eLags,
			e2emetrics.PodLatencyData{Name: pod.Name, Latency: watchTimes[pod.Name].Time.Sub(create.Time)})
	}

	for name, create := range createTimes {
		watch := watchTimes[name]
		gomega.Expect(watchTimes).To(gomega.HaveKey(name))
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
	batchLag := lastRunning.Time.Sub(batchStartTime.Time)
	sort.Sort(e2emetrics.LatencySlice(e2eLags))
	return batchLag, e2eLags
}

// podWatchedRunning verifies whether the pod becomes Running, as the watchTime was set by informer
func podWatchedRunning(watchTimes map[string]metav1.Time, podName string) wait.ConditionWithContextFunc {
	return func(ctx context.Context) (done bool, err error) {
		if _, found := watchTimes[podName]; found {
			return true, nil
		}
		return false, nil
	}
}

// verifyLatencyWithinThreshold verifies whether 50, 90 and 99th percentiles of a latency metric are
// within the expected threshold.
func verifyLatencyWithinThreshold(threshold, actual e2emetrics.LatencyMetric, metricName string) error {
	if actual.Perc50 > threshold.Perc50 {
		return fmt.Errorf("too high %v latency 50th percentile: %v", metricName, actual.Perc50)
	}
	if actual.Perc90 > threshold.Perc90 {
		return fmt.Errorf("too high %v latency 90th percentile: %v", metricName, actual.Perc90)
	}
	if actual.Perc99 > threshold.Perc99 {
		return fmt.Errorf("too high %v latency 99th percentile: %v", metricName, actual.Perc99)
	}
	return nil
}

// extractLatencyMetrics returns latency metrics for each percentile(50th, 90th and 99th).
func extractLatencyMetrics(latencies []e2emetrics.PodLatencyData) e2emetrics.LatencyMetric {
	length := len(latencies)
	perc50 := latencies[int(math.Ceil(float64(length*50)/100))-1].Latency
	perc90 := latencies[int(math.Ceil(float64(length*90)/100))-1].Latency
	perc99 := latencies[int(math.Ceil(float64(length*99)/100))-1].Latency
	perc100 := latencies[length-1].Latency
	return e2emetrics.LatencyMetric{Perc50: perc50, Perc90: perc90, Perc99: perc99, Perc100: perc100}
}

// printLatencies outputs latencies to log with readable format.
func printLatencies(latencies []e2emetrics.PodLatencyData, header string) {
	metrics := extractLatencyMetrics(latencies)
	framework.Logf("10%% %s: %v", header, latencies[(len(latencies)*9)/10:])
	framework.Logf("perc50: %v, perc90: %v, perc99: %v", metrics.Perc50, metrics.Perc90, metrics.Perc99)
}

// logAndVerifyLatency verifies that whether pod creation latency satisfies the limit.
func logAndVerifyLatency(ctx context.Context, batchLag time.Duration, e2eLags []e2emetrics.PodLatencyData, podStartupLimits e2emetrics.LatencyMetric,
	podBatchStartupLimit time.Duration, testInfo map[string]string, isVerify bool) {
	printLatencies(e2eLags, "worst client e2e total latencies")

	// TODO(coufon): do not trust 'kubelet' metrics since they are not reset!
	latencyMetrics, _ := getPodStartLatency(ctx, kubeletAddr)
	framework.Logf("Kubelet Prometheus metrics (not reset):\n%s", framework.PrettyPrintJSON(latencyMetrics))

	podStartupLatency := extractLatencyMetrics(e2eLags)

	// log latency perf data
	logPerfData(getLatencyPerfData(podStartupLatency, testInfo), "latency")

	if isVerify {
		// check whether e2e pod startup time is acceptable.
		framework.ExpectNoError(verifyLatencyWithinThreshold(podStartupLimits, podStartupLatency, "pod startup"))

		// check bactch pod creation latency
		if podBatchStartupLimit > 0 {
			if batchLag > podBatchStartupLimit {
				framework.Failf("Batch creation startup time %v exceed limit %v", batchLag, podBatchStartupLimit)
			}
		}
	}
}

// logThroughput calculates and logs pod creation throughput.
func logPodCreateThroughput(batchLag time.Duration, e2eLags []e2emetrics.PodLatencyData, podsNr int, testInfo map[string]string) {
	logPerfData(getThroughputPerfData(batchLag, e2eLags, podsNr, testInfo), "throughput")
}
