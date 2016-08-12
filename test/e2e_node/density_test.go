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

var _ = framework.KubeDescribe("Density [Serial] [Slow]", func() {
	const (
		// the data collection time of `resource collector' and the standalone cadvisor
		// is not synchronizated. Therefore `resource collector' may miss data or
		// collect duplicated data
		monitoringInterval    = 500 * time.Millisecond
		sleepBeforeCreatePods = 30 * time.Second
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
	})

	Context("create a batch of pods", func() {
		// TODO(coufon): add more tests and the values are generous, set more precise limits after benchmark
		dTests := []densityTest{
			{
				podsNr:   10,
				interval: 0 * time.Millisecond,
				cpuLimits: framework.ContainersCPUSummary{
					stats.SystemContainerKubelet: {0.50: 0.20, 0.95: 0.30},
					stats.SystemContainerRuntime: {0.50: 0.40, 0.95: 0.60},
				},
				memLimits: framework.ResourceUsagePerContainer{
					stats.SystemContainerKubelet: &framework.ContainerResourceUsage{MemoryRSSInBytes: 100 * 1024 * 1024},
					stats.SystemContainerRuntime: &framework.ContainerResourceUsage{MemoryRSSInBytes: 400 * 1024 * 1024},
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
				rc := NewResourceCollector(monitoringInterval)

				// the controller watches the change of pod status
				controller := newInformerWatchPod(f, mutex, watchTimes, podType)
				go controller.Run(stopCh)

				// Zhou: In test we see kubelet starts while it is busy on something, as a result `syncLoop'
				// does not response to pod creation immediately. Creating the first pod has a delay around 5s.
				// The node status has been `ready' so `wait and check node being ready' does not help here.
				// Now wait here for a grace period to have `syncLoop' be ready
				time.Sleep(sleepBeforeCreatePods)

				// the density test only monitors the overhead of creating pod
				// or start earliest and call `rc.Reset()' here to clear the buffer
				rc.Start()

				By("Creating a batch of pods")
				// it returns a map[`pod name']`creation time' as the creation timestamps
				createTimes := createBatchPodWithRateControl(f, pods, itArg.interval)

				By("Waiting for all Pods to be observed by the watch...")
				// checks every 10s util all pods are running. it times out ater 10min
				Eventually(func() bool {
					return len(watchTimes) == itArg.podsNr
				}, 10*time.Minute, 10*time.Second).Should(BeTrue())

				if len(watchTimes) < itArg.podsNr {
					framework.Failf("Timeout reached waiting for all Pods to be observed by the watch.")
				}

				// stop the watching controller, and the resource collector
				close(stopCh)
				rc.Stop()

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
				verifyResource(f, itArg.cpuLimits, itArg.memLimits, rc)
			})
		}
	})

	Context("create a sequence of pods", func() {
		// TODO(coufon): add more tests and the values are generous, set more precise limits after benchmark
		dTests := []densityTest{
			{
				podsNr:   10,
				bgPodsNr: 10,
				cpuLimits: framework.ContainersCPUSummary{
					stats.SystemContainerKubelet: {0.50: 0.20, 0.95: 0.25},
					stats.SystemContainerRuntime: {0.50: 0.40, 0.95: 0.60},
				},
				memLimits: framework.ResourceUsagePerContainer{
					stats.SystemContainerKubelet: &framework.ContainerResourceUsage{MemoryRSSInBytes: 100 * 1024 * 1024},
					stats.SystemContainerRuntime: &framework.ContainerResourceUsage{MemoryRSSInBytes: 400 * 1024 * 1024},
				},
				podStartupLimits: framework.LatencyMetric{
					Perc50: 3000 * time.Millisecond,
					Perc90: 4000 * time.Millisecond,
					Perc99: 5000 * time.Millisecond,
				},
			},
		}

		for _, testArg := range dTests {
			itArg := testArg
			It(fmt.Sprintf("latency/resource should be within limit when create %d pods with %d background pods",
				itArg.podsNr, itArg.bgPodsNr), func() {
				bgPods := newTestPods(itArg.bgPodsNr, ImageRegistry[pauseImage], "background_pod")
				testPods := newTestPods(itArg.podsNr, ImageRegistry[pauseImage], podType)

				createCadvisorPod(f)
				rc := NewResourceCollector(monitoringInterval)

				By("Creating a batch of background pods")
				// creatBatch is synchronized
				// all pods are running when it returns
				f.PodClient().CreateBatch(bgPods)

				time.Sleep(sleepBeforeCreatePods)

				// starting resource monitoring
				rc.Start()

				// do a sequential creation of pod (back to back)
				batchlag, e2eLags := createBatchPodSequential(f, testPods)

				rc.Stop()

				// verify latency
				By("Verifying latency")
				verifyLatency(batchlag, e2eLags, itArg)

				// verify resource
				By("Verifying resource")
				verifyResource(f, itArg.cpuLimits, itArg.memLimits, rc)
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

// checkPodDeleted checks whether a pod has been successfully deleted
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

// verifyLatency verifies that whether pod creation latency satisfies the limit.
func verifyLatency(batchLag time.Duration, e2eLags []framework.PodLatencyData, testArg densityTest) {
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

// createBatchPodSequential creats pods back-to-back in sequence.
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
