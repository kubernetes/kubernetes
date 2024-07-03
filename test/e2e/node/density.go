/*
Copyright 2024 The Kubernetes Authors.

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

package node

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/test/e2e/framework"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("Density", framework.WithSerial(), framework.WithSlow(), func() {
	options := framework.Options{
		ClientQPS:   50,
		ClientBurst: 100,
	}
	f := framework.NewFramework("density-test", options, nil)
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.Context("create a batch of pods", func() {
		dTests := []densityTest{
			{
				podsNr: 10,
			},
			{
				podsNr: 40,
			},
			{
				podsNr: 90,
			},
		}

		for _, testArg := range dTests {
			desc := fmt.Sprintf("latency should not exceed 10s when create %d pods [Benchmark]", testArg.podsNr)
			ginkgo.It(desc, func(ctx context.Context) {
				batchLag, e2eLags := runDensityBatchTest(ctx, f, testArg)
				framework.Logf("creating %d pods in batches reaches Running state in %v.", testArg.podsNr, batchLag)
				printLatencies(e2eLags, "worst client e2e total latencies")
			})
		}
	})
})

type densityTest struct {
	// number of pods
	podsNr int
}

// runDensityBatchTest runs the density batch pod creation test
func runDensityBatchTest(ctx context.Context, f *framework.Framework, testArg densityTest) (time.Duration, []e2emetrics.PodLatencyData) {
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
	pods := newTestPods(testArg.podsNr, imageutils.GetPauseImageName(), podType)

	// the controller watches the change of pod status
	controller := newInformerWatchPod(ctx, f, mutex, watchTimes, podType)
	go controller.Run(stopCh)
	defer close(stopCh)

	time.Sleep(sleepBeforeCreatePods)

	ginkgo.By("Creating a batch of pods")
	// It returns a map['pod name']'creation time' containing the creation timestamps
	createTimes := createBatchPod(ctx, f, pods)
	defer deletePodsSync(ctx, f, pods)

	ginkgo.By("Waiting for all Pods to be observed by the watch...")

	gomega.Eventually(ctx, func() bool {
		return len(watchTimes) == testArg.podsNr
	}, 10*time.Minute, 10*time.Second).Should(gomega.BeTrue())

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

	return batchLag, e2eLags
}

// newTestPods creates a list of pods (specification) for test.
func newTestPods(numPods int, imageName, podType string) []*v1.Pod {
	var pods []*v1.Pod
	for i := 0; i < numPods; i++ {
		podName := "test-" + string(uuid.NewUUID())
		labels := map[string]string{
			"type": podType,
			"name": podName,
		}

		pods = append(pods,
			&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   podName,
					Labels: labels,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Image: imageName,
							Name:  podName,
						},
					},
				},
			},
		)
	}
	return pods
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
	lw := &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			options.LabelSelector = labels.SelectorFromSet(labels.Set{"type": podType}).String()
			obj, err := f.ClientSet.CoreV1().Pods(ns).List(ctx, options)
			return runtime.Object(obj), err
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			options.LabelSelector = labels.SelectorFromSet(labels.Set{"type": podType}).String()
			return f.ClientSet.CoreV1().Pods(ns).Watch(ctx, options)
		},
	}
	handler := cache.ResourceEventHandlerFuncs{
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
	}

	options := cache.InformerOptions{
		ListerWatcher: lw,
		ObjectType:    &v1.Pod{},
		Handler:       handler,
	}
	_, controller := cache.NewInformerWithOptions(options)

	return controller
}

// createBatchPod creates a batch of pods concurrently
func createBatchPod(ctx context.Context, f *framework.Framework, pods []*v1.Pod) map[string]metav1.Time {
	createTimes := make(map[string]metav1.Time)
	for i := range pods {
		pod := pods[i]
		createTimes[pod.ObjectMeta.Name] = metav1.Now()
		go e2epod.NewPodClient(f).Create(ctx, pod)
	}
	return createTimes
}

// deletePodsSync deletes a list of pods and block until pods disappear.
func deletePodsSync(ctx context.Context, f *framework.Framework, pods []*v1.Pod) {
	var wg sync.WaitGroup
	for i := range pods {
		pod := pods[i]
		wg.Add(1)
		go func() {
			defer ginkgo.GinkgoRecover()
			defer wg.Done()

			err := e2epod.NewPodClient(f).Delete(ctx, pod.ObjectMeta.Name, *metav1.NewDeleteOptions(30))
			if apierrors.IsNotFound(err) {
				framework.Failf("Unexpected error trying to delete pod %s: %v", pod.Name, err)
			}

			framework.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, pod.ObjectMeta.Name, f.Namespace.Name, 10*time.Minute))
		}()
	}
	wg.Wait()
	return
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
