/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package e2e

import (
	"fmt"
	"os"
	"sort"
	"strconv"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/watch"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("[Performance Suite] Latency", func() {
	var c *client.Client
	var nodeCount int
	var additionalPodsPrefix string
	var ns string
	var uuid string

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		expectNoError(err)
		nodes, err := c.Nodes().List(labels.Everything(), fields.Everything())
		expectNoError(err)
		nodeCount = len(nodes.Items)
		Expect(nodeCount).NotTo(BeZero())

		// Terminating a namespace (deleting the remaining objects from it - which
		// generally means events) can affect the current run. Thus we wait for all
		// terminating namespace to be finally deleted before starting this test.
		err = deleteTestingNS(c)
		expectNoError(err)

		nsForTesting, err := createTestingNS("latency", c)
		ns = nsForTesting.Name
		expectNoError(err)
		uuid = string(util.NewUUID())

		expectNoError(resetMetrics(c))
		expectNoError(os.Mkdir(fmt.Sprintf(testContext.OutputDir+"/%s", uuid), 0777))
		expectNoError(writePerfData(c, fmt.Sprintf(testContext.OutputDir+"/%s", uuid), "before"))
		gcloudListNodes()
	})

	AfterEach(func() {
		By("Removing additional pods if any")
		for i := 1; i <= nodeCount; i++ {
			name := additionalPodsPrefix + "-" + strconv.Itoa(i)
			c.Pods(ns).Delete(name, nil)
		}

		By(fmt.Sprintf("Destroying namespace for this suite %v", ns))
		if err := c.Namespaces().Delete(ns); err != nil {
			Failf("Couldn't delete ns %s", err)
		}

		expectNoError(writePerfData(c, fmt.Sprintf(testContext.OutputDir+"/%s", uuid), "after"))

		// Verify latency metrics
		highLatencyRequests, err := HighLatencyRequests(c, 3*time.Second, sets.NewString("events"))
		expectNoError(err)
		Expect(highLatencyRequests).NotTo(BeNumerically(">", 0), "There should be no high-latency requests")
	})

	It("pod start latency should be acceptable", func() {
		runLatencyTest(nodeCount, c, ns)
	})
})

func runLatencyTest(nodeCount int, c *client.Client, ns string) {
	var (
		nodes              = make(map[string]string, 0)    // pod name -> node name
		createTimestamps   = make(map[string]util.Time, 0) // pod name -> create time
		scheduleTimestamps = make(map[string]util.Time, 0) // pod name -> schedule time
		startTimestamps    = make(map[string]util.Time, 0) // pod name -> time to run
		watchTimestamps    = make(map[string]util.Time, 0) // pod name -> time to read from informer

		additionalPodsPrefix = "latency-pod-" + string(util.NewUUID())
	)

	var mutex sync.Mutex
	readPodInfo := func(p *api.Pod) {
		mutex.Lock()
		defer mutex.Unlock()
		defer GinkgoRecover()

		if p.Status.Phase == api.PodRunning {
			if _, found := watchTimestamps[p.Name]; !found {
				watchTimestamps[p.Name] = util.Now()
				createTimestamps[p.Name] = p.CreationTimestamp
				nodes[p.Name] = p.Spec.NodeName
				var startTimestamp util.Time
				for _, cs := range p.Status.ContainerStatuses {
					if cs.State.Running != nil {
						if startTimestamp.Before(cs.State.Running.StartedAt) {
							startTimestamp = cs.State.Running.StartedAt
						}
					}
				}
				if startTimestamp != util.NewTime(time.Time{}) {
					startTimestamps[p.Name] = startTimestamp
				} else {
					Failf("Pod %v is reported to be running, but none of its containers are", p.Name)
				}
			}
		}
	}

	// Create a informer to read timestamps for each pod
	stopCh := make(chan struct{})
	_, informer := framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return c.Pods(ns).List(labels.SelectorFromSet(labels.Set{"name": additionalPodsPrefix}), fields.Everything())
			},
			WatchFunc: func(rv string) (watch.Interface, error) {
				return c.Pods(ns).Watch(labels.SelectorFromSet(labels.Set{"name": additionalPodsPrefix}), fields.Everything(), rv)
			},
		},
		&api.Pod{},
		0,
		framework.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				p, ok := obj.(*api.Pod)
				Expect(ok).To(Equal(true))
				go readPodInfo(p)
			},
			UpdateFunc: func(oldObj, newObj interface{}) {
				p, ok := newObj.(*api.Pod)
				Expect(ok).To(Equal(true))
				go readPodInfo(p)
			},
		},
	)
	go informer.Run(stopCh)

	// Create  additional pods with throughput ~5 pods/sec.
	var wg sync.WaitGroup
	wg.Add(nodeCount)
	podLabels := map[string]string{
		"name": additionalPodsPrefix,
	}
	for i := 1; i <= nodeCount; i++ {
		name := additionalPodsPrefix + "-" + strconv.Itoa(i)
		go createRunningPod(&wg, c, name, ns, "gcr.io/google_containers/pause:go", podLabels)
		time.Sleep(200 * time.Millisecond)
	}
	wg.Wait()

	Logf("Waiting for all Pods begin observed by the watch...")
	for start := time.Now(); len(watchTimestamps) < nodeCount && time.Since(start) < timeout; time.Sleep(10 * time.Second) {
	}
	close(stopCh)

	// Read the schedule timestamp by checking the scheduler event for each pod
	schedEvents, err := c.Events(ns).List(
		labels.Everything(),
		fields.Set{
			"involvedObject.kind":      "Pod",
			"involvedObject.namespace": ns,
			"source":                   "scheduler",
		}.AsSelector())
	expectNoError(err)
	for k := range createTimestamps {
		for _, event := range schedEvents.Items {
			if event.InvolvedObject.Name == k {
				scheduleTimestamps[k] = event.FirstTimestamp
				break
			}
		}
	}

	var (
		scheduleLatencies        = make([]podLatencyData, 0)
		startLatencies           = make([]podLatencyData, 0)
		watchLatencies           = make([]podLatencyData, 0)
		scheduleToWatchLatencies = make([]podLatencyData, 0)
		e2eLatencies             = make([]podLatencyData, 0)
	)

	for name, podNode := range nodes {
		createTs, ok := createTimestamps[name]
		Expect(ok).To(Equal(true))
		scheduleTs, ok := scheduleTimestamps[name]
		Expect(ok).To(Equal(true))
		runTs, ok := startTimestamps[name]
		Expect(ok).To(Equal(true))
		watchTs, ok := watchTimestamps[name]
		Expect(ok).To(Equal(true))

		var (
			scheduleLatency        = podLatencyData{name, podNode, scheduleTs.Time.Sub(createTs.Time)}
			startLatency           = podLatencyData{name, podNode, runTs.Time.Sub(scheduleTs.Time)}
			watchLatency           = podLatencyData{name, podNode, watchTs.Time.Sub(runTs.Time)}
			scheduleToWatchLatency = podLatencyData{name, podNode, watchTs.Time.Sub(scheduleTs.Time)}
			e2eLatency             = podLatencyData{name, podNode, watchTs.Time.Sub(createTs.Time)}
		)

		scheduleLatencies = append(scheduleLatencies, scheduleLatency)
		startLatencies = append(startLatencies, startLatency)
		watchLatencies = append(watchLatencies, watchLatency)
		scheduleToWatchLatencies = append(scheduleToWatchLatencies, scheduleToWatchLatency)
		e2eLatencies = append(e2eLatencies, e2eLatency)
	}

	sort.Sort(latencySlice(scheduleLatencies))
	sort.Sort(latencySlice(startLatencies))
	sort.Sort(latencySlice(watchLatencies))
	sort.Sort(latencySlice(scheduleToWatchLatencies))
	sort.Sort(latencySlice(e2eLatencies))

	printLatencies(scheduleLatencies, "worst schedule latencies")
	printLatencies(startLatencies, "worst run-after-schedule latencies")
	printLatencies(watchLatencies, "worst watch latencies")
	printLatencies(scheduleToWatchLatencies, "worst scheduled-to-end total latencies")
	printLatencies(e2eLatencies, "worst e2e total latencies")

	// Test whether e2e pod startup time is acceptable.
	// TODO: Switch it to 5 seconds once we are sure our tests are passing.
	podStartupThreshold := 8 * time.Second
	e2ePodStartupTime50perc := startLatencies[len(startLatencies)/2].Latency
	e2ePodStartupTime90perc := startLatencies[len(startLatencies)*9/10].Latency
	e2ePodStartupTime99perc := startLatencies[len(startLatencies)*99/100].Latency
	Expect(e2ePodStartupTime50perc).To(BeNumerically("<", podStartupThreshold), "Too high pod startup time 50th percentile")
	Expect(e2ePodStartupTime90perc).To(BeNumerically("<", podStartupThreshold), "Too high pod startup time 90th percentile")
	Expect(e2ePodStartupTime99perc).To(BeNumerically("<", podStartupThreshold), "Too high pod startup time 99th percentile")

	// Log suspicious latency metrics/docker errors from all nodes that had slow startup times
	for _, l := range startLatencies {
		if l.Latency > NodeStartupThreshold {
			HighLatencyKubeletOperations(c, 1*time.Second, l.Node)
		}
	}

	Logf("Approx throughput: %v pods/min",
		float64(nodeCount)/(startLatencies[len(startLatencies)-1].Latency.Minutes()))
}
