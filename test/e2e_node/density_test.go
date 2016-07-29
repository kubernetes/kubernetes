/*
Copyright 2016 The Kubernetes Authors.
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
	"errors"
	"fmt"
	"sort"
	"strconv"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
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

var _ = framework.KubeDescribe("Density", func() {
	const (
		// the data collection time of `resource collector' and the standalone cadvisor
		// is not synchronizated. Therefore `resource collector' may miss data or
		// collect duplicated data
		monitoringInterval    = 500 * time.Millisecond
		sleepBeforeEach       = 30 * time.Second
		sleepBeforeCreatePods = 30 * time.Second
		sleepAfterDeletePods  = 60 * time.Second
	)

	var (
		ns       string
		nodeName string
	)

	f := framework.NewDefaultFramework("density-test")
	podType := "density_test_pod"

	BeforeEach(func() {
		ns = f.Namespace.Name
		nodeName = framework.TestContext.NodeName
	})

	AfterEach(func() {
		time.Sleep(sleepAfterDeletePods)
	})

	Context("create a batch of pods", func() {
		densityTests := []DensityTest{
			{
				podsNr:   10,
				interval: 0 * time.Millisecond,
				cpuLimits: framework.ContainersCPUSummary{
					stats.SystemContainerKubelet: {0.50: 0.10, 0.95: 0.20},
					stats.SystemContainerRuntime: {0.50: 0.10, 0.95: 0.50},
				},
				memLimits: framework.ResourceUsagePerContainer{
					stats.SystemContainerKubelet: &framework.ContainerResourceUsage{MemoryRSSInBytes: 40 * 1024 * 1024},
					stats.SystemContainerRuntime: &framework.ContainerResourceUsage{MemoryRSSInBytes: 250 * 1024 * 1024},
				},
				// percentile limit of single pod startup latency
				podStartupLimits: framework.LatencyMetric{
					Perc50: 7 * time.Second,
					Perc90: 10 * time.Second,
					Perc99: 15 * time.Second,
				},
				// upbound of startup latency of a batch of pods
				podBatchStartupLimit: 20 * time.Second,
			},
			{
				podsNr:   30,
				interval: 0 * time.Millisecond,
				cpuLimits: framework.ContainersCPUSummary{
					stats.SystemContainerKubelet: {0.50: 0.10, 0.95: 0.35},
					stats.SystemContainerRuntime: {0.50: 0.10, 0.95: 0.70},
				},
				memLimits: framework.ResourceUsagePerContainer{
					stats.SystemContainerKubelet: &framework.ContainerResourceUsage{MemoryRSSInBytes: 40 * 1024 * 1024},
					stats.SystemContainerRuntime: &framework.ContainerResourceUsage{MemoryRSSInBytes: 300 * 1024 * 1024},
				},
				// percentile limit of single pod startup latency
				podStartupLimits: framework.LatencyMetric{
					Perc50: 30 * time.Second,
					Perc90: 35 * time.Second,
					Perc99: 40 * time.Second,
				},
				// upbound of startup latency of a batch of pods
				podBatchStartupLimit: 90 * time.Second,
			},
		}

		for _, testArg := range densityTests {
			itArg := testArg
			It(fmt.Sprintf("latency/resource should be within limit when create %d pods with %v interval",
				itArg.podsNr, itArg.interval), func() {
				var (
					mutex      = &sync.Mutex{}
					watchTimes = make(map[string]unversioned.Time, 0)
					stopCh     = make(chan struct{})
				)

				// create specifications of the test pods
				pods := newTestPods(itArg.podsNr, ImageRegistry[pauseImage], podType)

				// start a standalone cadvisor pod
				// it uses `createSync', so the pod is running when it returns
				createCadvisorPod(f)

				// `resource collector' monitoring fine-grain CPU/memory usage by a standalone Cadvisor with
				// 1s housingkeeping interval
				rm := NewResourceCollector(monitoringInterval)

				// the controller watches the change of pod status
				controller := newInformerWatchPod(f, mutex, watchTimes, podType)
				go controller.Run(stopCh)

				// Zhou: In test we see kubelet starts while it is busy on sth, as a result `syncLoop'
				// does not response to pod creation immediately. Creating the first pod has a delay around 5s.
				// The node status has been `ready' so `wait and check node being ready' does not help here.
				// Now wait here for a grace period to have `syncLoop' be ready
				time.Sleep(sleepBeforeCreatePods)

				// the density test only monitors the overhead of creating pod
				// or start earliest and call `rm.Reset()' here to clear the buffer
				rm.Start()

				By("Creating a batch of pods")
				// it returns a map[`pod name']`creation time' as the creation timestamps
				createTimes := createBatchPodWithRateControl(f, pods, itArg.interval)

				By("Waiting for all Pods begin observed by the watch...")
				// checks every 10s util all pods are running. it timeouts ater 10min
				Eventually(func() bool {
					return len(watchTimes) == itArg.podsNr
				}, 10*time.Minute, 10*time.Second).Should(BeTrue())

				if len(watchTimes) < itArg.podsNr {
					framework.Failf("Timeout reached waiting for all Pods being observed by the watch.")
				}

				// stop the watching controller, and the resource collector
				close(stopCh)
				rm.Stop()

				// data analyis
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

				// verify latency
				By("Verifying latency")
				verifyLatency(lastRunning.Time.Sub(firstCreate.Time), e2eLags, itArg)

				// verify resource
				By("Verifying resource")
				verifyResource(f, testArg, rm)

				// delete pods
				By("Deleting a batch of pods")
				deleteBatchPod(f, pods)

				// tear down cadvisor
				Expect(f.Client.Pods(ns).Delete(cadvisorPodName, api.NewDeleteOptions(30))).
					NotTo(HaveOccurred())

				Eventually(func() error {
					return checkPodDeleted(f, cadvisorPodName)
				}, 10*time.Minute, time.Second*3).Should(BeNil())
			})
		}
	})

	Context("create a sequence of pods", func() {
		densityTests := []DensityTest{
			{
				podsNr:   10,
				bgPodsNr: 10,
				cpuLimits: framework.ContainersCPUSummary{
					stats.SystemContainerKubelet: {0.50: 0.10, 0.95: 0.12},
					stats.SystemContainerRuntime: {0.50: 0.16, 0.95: 0.20},
				},
				memLimits: framework.ResourceUsagePerContainer{
					stats.SystemContainerKubelet: &framework.ContainerResourceUsage{MemoryRSSInBytes: 40 * 1024 * 1024},
					stats.SystemContainerRuntime: &framework.ContainerResourceUsage{MemoryRSSInBytes: 300 * 1024 * 1024},
				},
				podStartupLimits: framework.LatencyMetric{
					Perc50: 1500 * time.Millisecond,
					Perc90: 2500 * time.Millisecond,
					Perc99: 3500 * time.Millisecond,
				},
			},
			{
				podsNr:   10,
				bgPodsNr: 30,
				cpuLimits: framework.ContainersCPUSummary{
					stats.SystemContainerKubelet: {0.50: 0.12, 0.95: 0.15},
					stats.SystemContainerRuntime: {0.50: 0.22, 0.95: 0.27},
				},
				memLimits: framework.ResourceUsagePerContainer{
					stats.SystemContainerKubelet: &framework.ContainerResourceUsage{MemoryRSSInBytes: 40 * 1024 * 1024},
					stats.SystemContainerRuntime: &framework.ContainerResourceUsage{MemoryRSSInBytes: 300 * 1024 * 1024},
				},
				podStartupLimits: framework.LatencyMetric{
					Perc50: 1500 * time.Millisecond,
					Perc90: 2500 * time.Millisecond,
					Perc99: 3500 * time.Millisecond,
				},
			},
		}

		for _, testArg := range densityTests {
			itArg := testArg
			It(fmt.Sprintf("latency/resource should be within limit when create %d pods with %d background pods",
				itArg.podsNr, itArg.bgPodsNr), func() {
				bgPods := newTestPods(itArg.bgPodsNr, ImageRegistry[pauseImage], "background_pod")
				testPods := newTestPods(itArg.podsNr, ImageRegistry[pauseImage], podType)

				createCadvisorPod(f)
				rm := NewResourceCollector(monitoringInterval)

				By("Creating a batch of background pods")
				// creatBatch is synchronized
				// all pods are running when it returns
				f.PodClient().CreateBatch(bgPods)

				//time.Sleep(sleepBeforeCreatePods)

				// starting resource monitoring
				rm.Start()

				// do a sequential creation of pod (back to back)
				batchlag, e2eLags := createBatchPodSequential(f, testPods)

				rm.Stop()

				// verify latency
				By("Verifying latency")
				verifyLatency(batchlag, e2eLags, itArg)

				// verify resource
				By("Verifying resource")
				verifyResource(f, testArg, rm)

				// delete pods
				By("Deleting a batch of pods")
				deleteBatchPod(f, append(bgPods, testPods...))

				// tear down cadvisor
				Expect(f.Client.Pods(ns).Delete(cadvisorPodName, api.NewDeleteOptions(30))).
					NotTo(HaveOccurred())

				Eventually(func() error {
					return checkPodDeleted(f, cadvisorPodName)
				}, 10*time.Minute, time.Second*3).Should(BeNil())
			})
		}
	})
})

type DensityTest struct {
	// number of pods
	podsNr   int
	bgPodsNr int
	// interval between creating pod (rate control)
	interval time.Duration
	// resource bound
	cpuLimits            framework.ContainersCPUSummary
	memLimits            framework.ResourceUsagePerContainer
	podStartupLimits     framework.LatencyMetric
	podBatchStartupLimit time.Duration
}

// it creates a batch of pods concurrently, uses one goroutine for each creation.
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

func checkPodDeleted(f *framework.Framework, podName string) error {
	ns := f.Namespace.Name
	_, err := f.Client.Pods(ns).Get(podName)
	if apierrors.IsNotFound(err) {
		return nil
	}
	return errors.New("Pod Not Deleted")
}

// get prometheus metric `pod start latency' from kubelet
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

// Verifies whether 50, 90 and 99th percentiles of PodStartupLatency are
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

func verifyLatency(batchLag time.Duration, e2eLags []framework.PodLatencyData, testArg DensityTest) {
	framework.PrintLatencies(e2eLags, "worst client e2e total latencies")

	// Zhou: do not trust `kubelet' metrics since they are not reset!
	latencyMetrics, _ := getPodStartLatency(kubeletAddr)
	framework.Logf("Kubelet Prometheus metrics (not reset):\n%s", framework.PrettyPrintJSON(latencyMetrics))

	// check whether e2e pod startup time is acceptable.
	podCreateLatency := framework.PodStartupLatency{Latency: framework.ExtractLatencyMetrics(e2eLags)}
	framework.Logf("Pod create latency: %s", framework.PrettyPrintJSON(podCreateLatency))
	framework.ExpectNoError(verifyPodStartupLatency(testArg.podStartupLimits, podCreateLatency.Latency))

	// check bactch pod creation latency
	if testArg.podBatchStartupLimit > 0 {
		Expect(batchLag <= testArg.podBatchStartupLimit).To(Equal(true), "Batch creation startup time %v exceed limit %v",
			batchLag, testArg.podBatchStartupLimit)
	}

	// calculate and log throughput
	throughputBatch := float64(testArg.podsNr) / batchLag.Minutes()
	framework.Logf("Batch creation throughput is %.1f pods/min", throughputBatch)
	throughputSequential := 1.0 / e2eLags[len(e2eLags)-1].Latency.Minutes()
	framework.Logf("Sequential creation throughput is %.1f pods/min", throughputSequential)
}

func verifyResource(f *framework.Framework, testArg DensityTest, rm *ResourceCollector) {
	nodeName := framework.TestContext.NodeName

	// verify and log memory
	usagePerContainer, err := rm.GetLatest()
	Expect(err).NotTo(HaveOccurred())
	framework.Logf("%s", formatResourceUsageStats(usagePerContainer))

	usagePerNode := make(framework.ResourceUsagePerNode)
	usagePerNode[nodeName] = usagePerContainer

	memPerfData := framework.ResourceUsageToPerfData(usagePerNode)
	framework.PrintPerfData(memPerfData)

	verifyMemoryLimits(f.Client, testArg.memLimits, usagePerNode)

	// verify and log cpu
	cpuSummary := rm.GetCPUSummary()
	framework.Logf("%s", formatCPUSummary(cpuSummary))

	cpuSummaryPerNode := make(framework.NodesCPUSummary)
	cpuSummaryPerNode[nodeName] = cpuSummary

	cpuPerfData := framework.CPUUsageToPerfData(cpuSummaryPerNode)
	framework.PrintPerfData(cpuPerfData)

	verifyCPULimits(testArg.cpuLimits, cpuSummaryPerNode)
}

func createBatchPodSequential(f *framework.Framework, pods []*api.Pod) (time.Duration, []framework.PodLatencyData) {
	batchStartTime := unversioned.Now()
	e2eLags := make([]framework.PodLatencyData, 0)
	for _, pod := range pods {
		create := unversioned.Now()
		f.PodClient().CreateSync(pod)
		e2eLags = append(e2eLags,
			framework.PodLatencyData{Name: pod.ObjectMeta.Name, Latency: unversioned.Now().Time.Sub(create.Time)})
	}
	batchLag := unversioned.Now().Time.Sub(batchStartTime.Time)
	sort.Sort(framework.LatencySlice(e2eLags))
	return batchLag, e2eLags
}
