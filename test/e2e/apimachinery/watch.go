/*
Copyright 2018 The Kubernetes Authors.

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

package apimachinery

import (
	"context"
	"fmt"
	"math/rand"
	"strconv"
	"time"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	cachetools "k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

const (
	watchConfigMapLabelKey = "watch-this-configmap"

	multipleWatchersLabelValueA   = "multiple-watchers-A"
	multipleWatchersLabelValueB   = "multiple-watchers-B"
	fromResourceVersionLabelValue = "from-resource-version"
	watchRestartedLabelValue      = "watch-closed-and-restarted"
	toBeChangedLabelValue         = "label-changed-and-restored"
)

var _ = SIGDescribe("Watchers", func() {
	f := framework.NewDefaultFramework("watch")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	/*
		    Release: v1.11
		    Testname: watch-configmaps-with-multiple-watchers
		    Description: Ensure that multiple watchers are able to receive all add,
			update, and delete notifications on configmaps that match a label selector and do
			not receive notifications for configmaps which do not match that label selector.
	*/
	framework.ConformanceIt("should observe add, update, and delete watch notifications on configmaps", func(ctx context.Context) {
		c := f.ClientSet
		ns := f.Namespace.Name

		ginkgo.By("creating a watch on configmaps with label A")
		watchA, err := watchConfigMaps(ctx, f, "", multipleWatchersLabelValueA)
		framework.ExpectNoError(err, "failed to create a watch on configmaps with label: %s", multipleWatchersLabelValueA)

		ginkgo.By("creating a watch on configmaps with label B")
		watchB, err := watchConfigMaps(ctx, f, "", multipleWatchersLabelValueB)
		framework.ExpectNoError(err, "failed to create a watch on configmaps with label: %s", multipleWatchersLabelValueB)

		ginkgo.By("creating a watch on configmaps with label A or B")
		watchAB, err := watchConfigMaps(ctx, f, "", multipleWatchersLabelValueA, multipleWatchersLabelValueB)
		framework.ExpectNoError(err, "failed to create a watch on configmaps with label %s or %s", multipleWatchersLabelValueA, multipleWatchersLabelValueB)

		testConfigMapA := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name: "e2e-watch-test-configmap-a",
				Labels: map[string]string{
					watchConfigMapLabelKey: multipleWatchersLabelValueA,
				},
			},
		}
		testConfigMapB := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name: "e2e-watch-test-configmap-b",
				Labels: map[string]string{
					watchConfigMapLabelKey: multipleWatchersLabelValueB,
				},
			},
		}

		ginkgo.By("creating a configmap with label A and ensuring the correct watchers observe the notification")
		testConfigMapA, err = c.CoreV1().ConfigMaps(ns).Create(ctx, testConfigMapA, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create a configmap with label %s in namespace: %s", multipleWatchersLabelValueA, ns)
		expectEvent(watchA, watch.Added, testConfigMapA)
		expectEvent(watchAB, watch.Added, testConfigMapA)

		ginkgo.By("modifying configmap A and ensuring the correct watchers observe the notification")
		testConfigMapA, err = updateConfigMap(ctx, c, ns, testConfigMapA.GetName(), func(cm *v1.ConfigMap) {
			setConfigMapData(cm, "mutation", "1")
		})
		framework.ExpectNoError(err, "failed to update configmap %s in namespace: %s", testConfigMapA.GetName(), ns)
		expectEvent(watchA, watch.Modified, testConfigMapA)
		expectEvent(watchAB, watch.Modified, testConfigMapA)

		ginkgo.By("modifying configmap A again and ensuring the correct watchers observe the notification")
		testConfigMapA, err = updateConfigMap(ctx, c, ns, testConfigMapA.GetName(), func(cm *v1.ConfigMap) {
			setConfigMapData(cm, "mutation", "2")
		})
		framework.ExpectNoError(err, "failed to update configmap %s in namespace: %s", testConfigMapA.GetName(), ns)
		expectEvent(watchA, watch.Modified, testConfigMapA)
		expectEvent(watchAB, watch.Modified, testConfigMapA)

		ginkgo.By("deleting configmap A and ensuring the correct watchers observe the notification")
		err = c.CoreV1().ConfigMaps(ns).Delete(ctx, testConfigMapA.GetName(), metav1.DeleteOptions{})
		framework.ExpectNoError(err, "failed to delete configmap %s in namespace: %s", testConfigMapA.GetName(), ns)
		expectEvent(watchA, watch.Deleted, nil)
		expectEvent(watchAB, watch.Deleted, nil)

		ginkgo.By("creating a configmap with label B and ensuring the correct watchers observe the notification")
		testConfigMapB, err = c.CoreV1().ConfigMaps(ns).Create(ctx, testConfigMapB, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create configmap %s in namespace: %s", testConfigMapB, ns)
		expectEvent(watchB, watch.Added, testConfigMapB)
		expectEvent(watchAB, watch.Added, testConfigMapB)
		expectNoEvent(watchA, watch.Added, testConfigMapB)

		ginkgo.By("deleting configmap B and ensuring the correct watchers observe the notification")
		err = c.CoreV1().ConfigMaps(ns).Delete(ctx, testConfigMapB.GetName(), metav1.DeleteOptions{})
		framework.ExpectNoError(err, "failed to delete configmap %s in namespace: %s", testConfigMapB.GetName(), ns)
		expectEvent(watchB, watch.Deleted, nil)
		expectEvent(watchAB, watch.Deleted, nil)
		expectNoEvent(watchA, watch.Deleted, nil)
	})

	/*
		    Release: v1.11
		    Testname: watch-configmaps-from-resource-version
		    Description: Ensure that a watch can be opened from a particular resource version
			in the past and only notifications happening after that resource version are observed.
	*/
	framework.ConformanceIt("should be able to start watching from a specific resource version", func(ctx context.Context) {
		c := f.ClientSet
		ns := f.Namespace.Name

		testConfigMap := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name: "e2e-watch-test-resource-version",
				Labels: map[string]string{
					watchConfigMapLabelKey: fromResourceVersionLabelValue,
				},
			},
		}

		ginkgo.By("creating a new configmap")
		testConfigMap, err := c.CoreV1().ConfigMaps(ns).Create(ctx, testConfigMap, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create configmap %s in namespace: %s", testConfigMap.GetName(), ns)

		ginkgo.By("modifying the configmap once")
		testConfigMapFirstUpdate, err := updateConfigMap(ctx, c, ns, testConfigMap.GetName(), func(cm *v1.ConfigMap) {
			setConfigMapData(cm, "mutation", "1")
		})
		framework.ExpectNoError(err, "failed to update configmap %s in namespace: %s", testConfigMap.GetName(), ns)

		ginkgo.By("modifying the configmap a second time")
		testConfigMapSecondUpdate, err := updateConfigMap(ctx, c, ns, testConfigMap.GetName(), func(cm *v1.ConfigMap) {
			setConfigMapData(cm, "mutation", "2")
		})
		framework.ExpectNoError(err, "failed to update configmap %s in namespace %s a second time", testConfigMap.GetName(), ns)

		ginkgo.By("deleting the configmap")
		err = c.CoreV1().ConfigMaps(ns).Delete(ctx, testConfigMap.GetName(), metav1.DeleteOptions{})
		framework.ExpectNoError(err, "failed to delete configmap %s in namespace: %s", testConfigMap.GetName(), ns)

		ginkgo.By("creating a watch on configmaps from the resource version returned by the first update")
		testWatch, err := watchConfigMaps(ctx, f, testConfigMapFirstUpdate.ObjectMeta.ResourceVersion, fromResourceVersionLabelValue)
		framework.ExpectNoError(err, "failed to create a watch on configmaps from the resource version %s returned by the first update", testConfigMapFirstUpdate.ObjectMeta.ResourceVersion)

		ginkgo.By("Expecting to observe notifications for all changes to the configmap after the first update")
		expectEvent(testWatch, watch.Modified, testConfigMapSecondUpdate)
		expectEvent(testWatch, watch.Deleted, nil)
	})

	/*
		    Release: v1.11
		    Testname: watch-configmaps-closed-and-restarted
		    Description: Ensure that a watch can be reopened from the last resource version
			observed by the previous watch, and it will continue delivering notifications from
			that point in time.
	*/
	framework.ConformanceIt("should be able to restart watching from the last resource version observed by the previous watch", func(ctx context.Context) {
		c := f.ClientSet
		ns := f.Namespace.Name

		configMapName := "e2e-watch-test-watch-closed"
		testConfigMap := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name: configMapName,
				Labels: map[string]string{
					watchConfigMapLabelKey: watchRestartedLabelValue,
				},
			},
		}

		ginkgo.By("creating a watch on configmaps")
		testWatchBroken, err := watchConfigMaps(ctx, f, "", watchRestartedLabelValue)
		framework.ExpectNoError(err, "failed to create a watch on configmap with label: %s", watchRestartedLabelValue)

		ginkgo.By("creating a new configmap")
		testConfigMap, err = c.CoreV1().ConfigMaps(ns).Create(ctx, testConfigMap, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create configmap %s in namespace: %s", configMapName, ns)

		ginkgo.By("modifying the configmap once")
		_, err = updateConfigMap(ctx, c, ns, testConfigMap.GetName(), func(cm *v1.ConfigMap) {
			setConfigMapData(cm, "mutation", "1")
		})
		framework.ExpectNoError(err, "failed to update configmap %s in namespace: %s", configMapName, ns)

		ginkgo.By("closing the watch once it receives two notifications")
		expectEvent(testWatchBroken, watch.Added, testConfigMap)
		lastEvent, ok := waitForEvent(testWatchBroken, watch.Modified, nil, 1*time.Minute)
		if !ok {
			framework.Failf("Timed out waiting for second watch notification")
		}
		testWatchBroken.Stop()

		ginkgo.By("modifying the configmap a second time, while the watch is closed")
		testConfigMapSecondUpdate, err := updateConfigMap(ctx, c, ns, testConfigMap.GetName(), func(cm *v1.ConfigMap) {
			setConfigMapData(cm, "mutation", "2")
		})
		framework.ExpectNoError(err, "failed to update configmap %s in namespace %s a second time", configMapName, ns)

		ginkgo.By("creating a new watch on configmaps from the last resource version observed by the first watch")
		lastEventConfigMap, ok := lastEvent.Object.(*v1.ConfigMap)
		if !ok {
			framework.Failf("Expected last notification to refer to a configmap but got: %v", lastEvent)
		}
		testWatchRestarted, err := watchConfigMaps(ctx, f, lastEventConfigMap.ObjectMeta.ResourceVersion, watchRestartedLabelValue)
		framework.ExpectNoError(err, "failed to create a new watch on configmaps from the last resource version %s observed by the first watch", lastEventConfigMap.ObjectMeta.ResourceVersion)

		ginkgo.By("deleting the configmap")
		err = c.CoreV1().ConfigMaps(ns).Delete(ctx, testConfigMap.GetName(), metav1.DeleteOptions{})
		framework.ExpectNoError(err, "failed to delete configmap %s in namespace: %s", configMapName, ns)

		ginkgo.By("Expecting to observe notifications for all changes to the configmap since the first watch closed")
		expectEvent(testWatchRestarted, watch.Modified, testConfigMapSecondUpdate)
		expectEvent(testWatchRestarted, watch.Deleted, nil)
	})

	/*
		    Release: v1.11
		    Testname: watch-configmaps-label-changed
		    Description: Ensure that a watched object stops meeting the requirements of
			a watch's selector, the watch will observe a delete, and will not observe
			notifications for that object until it meets the selector's requirements again.
	*/
	framework.ConformanceIt("should observe an object deletion if it stops meeting the requirements of the selector", func(ctx context.Context) {
		c := f.ClientSet
		ns := f.Namespace.Name

		configMapName := "e2e-watch-test-label-changed"
		testConfigMap := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name: configMapName,
				Labels: map[string]string{
					watchConfigMapLabelKey: toBeChangedLabelValue,
				},
			},
		}

		ginkgo.By("creating a watch on configmaps with a certain label")
		testWatch, err := watchConfigMaps(ctx, f, "", toBeChangedLabelValue)
		framework.ExpectNoError(err, "failed to create a watch on configmap with label: %s", toBeChangedLabelValue)

		ginkgo.By("creating a new configmap")
		testConfigMap, err = c.CoreV1().ConfigMaps(ns).Create(ctx, testConfigMap, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create configmap %s in namespace: %s", configMapName, ns)

		ginkgo.By("modifying the configmap once")
		testConfigMapFirstUpdate, err := updateConfigMap(ctx, c, ns, testConfigMap.GetName(), func(cm *v1.ConfigMap) {
			setConfigMapData(cm, "mutation", "1")
		})
		framework.ExpectNoError(err, "failed to update configmap %s in namespace: %s", configMapName, ns)

		ginkgo.By("changing the label value of the configmap")
		_, err = updateConfigMap(ctx, c, ns, testConfigMap.GetName(), func(cm *v1.ConfigMap) {
			cm.ObjectMeta.Labels[watchConfigMapLabelKey] = "wrong-value"
		})
		framework.ExpectNoError(err, "failed to update configmap %s in namespace %s by changing label value", configMapName, ns)

		ginkgo.By("Expecting to observe a delete notification for the watched object")
		expectEvent(testWatch, watch.Added, testConfigMap)
		expectEvent(testWatch, watch.Modified, testConfigMapFirstUpdate)
		expectEvent(testWatch, watch.Deleted, nil)

		ginkgo.By("modifying the configmap a second time")
		testConfigMapSecondUpdate, err := updateConfigMap(ctx, c, ns, testConfigMap.GetName(), func(cm *v1.ConfigMap) {
			setConfigMapData(cm, "mutation", "2")
		})
		framework.ExpectNoError(err, "failed to update configmap %s in namespace %s a second time", configMapName, ns)

		ginkgo.By("Expecting not to observe a notification because the object no longer meets the selector's requirements")
		expectNoEvent(testWatch, watch.Modified, testConfigMapSecondUpdate)

		ginkgo.By("changing the label value of the configmap back")
		testConfigMapLabelRestored, err := updateConfigMap(ctx, c, ns, testConfigMap.GetName(), func(cm *v1.ConfigMap) {
			cm.ObjectMeta.Labels[watchConfigMapLabelKey] = toBeChangedLabelValue
		})
		framework.ExpectNoError(err, "failed to update configmap %s in namespace %s by changing label value back", configMapName, ns)

		ginkgo.By("modifying the configmap a third time")
		testConfigMapThirdUpdate, err := updateConfigMap(ctx, c, ns, testConfigMap.GetName(), func(cm *v1.ConfigMap) {
			setConfigMapData(cm, "mutation", "3")
		})
		framework.ExpectNoError(err, "failed to update configmap %s in namespace %s a third time", configMapName, ns)

		ginkgo.By("deleting the configmap")
		err = c.CoreV1().ConfigMaps(ns).Delete(ctx, testConfigMap.GetName(), metav1.DeleteOptions{})
		framework.ExpectNoError(err, "failed to delete configmap %s in namespace: %s", configMapName, ns)

		ginkgo.By("Expecting to observe an add notification for the watched object when the label value was restored")
		expectEvent(testWatch, watch.Added, testConfigMapLabelRestored)
		expectEvent(testWatch, watch.Modified, testConfigMapThirdUpdate)
		expectEvent(testWatch, watch.Deleted, nil)
	})

	/*
	   Release: v1.15
	   Testname: watch-consistency
	   Description: Ensure that concurrent watches are consistent with each other by initiating an additional watch
	   for events received from the first watch, initiated at the resource version of the event, and checking that all
	   resource versions of all events match. Events are produced from writes on a background goroutine.
	*/
	framework.ConformanceIt("should receive events on concurrent watches in same order", func(ctx context.Context) {
		c := f.ClientSet
		ns := f.Namespace.Name

		iterations := 100

		ginkgo.By("getting a starting resourceVersion")
		configmaps, err := c.CoreV1().ConfigMaps(ns).List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err, "Failed to list configmaps in the namespace %s", ns)
		resourceVersion := configmaps.ResourceVersion

		ginkgo.By("starting a background goroutine to produce watch events")
		donec := make(chan struct{})
		stopc := make(chan struct{})
		go func() {
			defer ginkgo.GinkgoRecover()
			defer close(donec)
			produceConfigMapEvents(ctx, f, stopc, 5*time.Millisecond)
		}()

		listWatcher := &cachetools.ListWatch{
			WatchFunc: func(listOptions metav1.ListOptions) (watch.Interface, error) {
				return c.CoreV1().ConfigMaps(ns).Watch(ctx, listOptions)
			},
		}

		ginkgo.By("creating watches starting from each resource version of the events produced and verifying they all receive resource versions in the same order")
		wcs := []watch.Interface{}
		for i := 0; i < iterations; i++ {
			wc, err := watchtools.NewRetryWatcher(resourceVersion, listWatcher)
			framework.ExpectNoError(err, "Failed to watch configmaps in the namespace %s", ns)
			wcs = append(wcs, wc)
			resourceVersion = waitForNextConfigMapEvent(wcs[0]).ResourceVersion
			for _, wc := range wcs[1:] {
				e := waitForNextConfigMapEvent(wc)
				if resourceVersion != e.ResourceVersion {
					framework.Failf("resource version mismatch, expected %s but got %s", resourceVersion, e.ResourceVersion)
				}
			}
		}
		close(stopc)
		for _, wc := range wcs {
			wc.Stop()
		}
		<-donec
	})
})

func watchConfigMaps(ctx context.Context, f *framework.Framework, resourceVersion string, labels ...string) (watch.Interface, error) {
	c := f.ClientSet
	ns := f.Namespace.Name
	opts := metav1.ListOptions{
		ResourceVersion: resourceVersion,
		LabelSelector: metav1.FormatLabelSelector(&metav1.LabelSelector{
			MatchExpressions: []metav1.LabelSelectorRequirement{
				{
					Key:      watchConfigMapLabelKey,
					Operator: metav1.LabelSelectorOpIn,
					Values:   labels,
				},
			},
		}),
	}
	return c.CoreV1().ConfigMaps(ns).Watch(ctx, opts)
}

func setConfigMapData(cm *v1.ConfigMap, key, value string) {
	if cm.Data == nil {
		cm.Data = make(map[string]string)
	}
	cm.Data[key] = value
}

func expectEvent(w watch.Interface, eventType watch.EventType, object runtime.Object) {
	if event, ok := waitForEvent(w, eventType, object, 1*time.Minute); !ok {
		framework.Failf("Timed out waiting for expected watch notification: %v", event)
	}
}

func expectNoEvent(w watch.Interface, eventType watch.EventType, object runtime.Object) {
	if event, ok := waitForEvent(w, eventType, object, 10*time.Second); ok {
		framework.Failf("Unexpected watch notification observed: %v", event)
	}
}

func waitForEvent(w watch.Interface, expectType watch.EventType, expectObject runtime.Object, duration time.Duration) (watch.Event, bool) {
	stopTimer := time.NewTimer(duration)
	defer stopTimer.Stop()
	for {
		select {
		case actual, ok := <-w.ResultChan():
			if ok {
				framework.Logf("Got : %v %v", actual.Type, actual.Object)
			} else {
				framework.Failf("Watch closed unexpectedly")
			}
			if expectType == actual.Type && (expectObject == nil || apiequality.Semantic.DeepEqual(expectObject, actual.Object)) {
				return actual, true
			}
		case <-stopTimer.C:
			expected := watch.Event{
				Type:   expectType,
				Object: expectObject,
			}
			return expected, false
		}
	}
}

func waitForNextConfigMapEvent(watch watch.Interface) *v1.ConfigMap {
	select {
	case event, ok := <-watch.ResultChan():
		if !ok {
			framework.Failf("Watch closed unexpectedly")
		}
		if configMap, ok := event.Object.(*v1.ConfigMap); ok {
			return configMap
		}
		framework.Failf("expected config map, got %T", event.Object)
	case <-time.After(10 * time.Second):
		framework.Failf("timed out waiting for watch event")
	}
	return nil // should never happen
}

const (
	createEvent = iota
	updateEvent
	deleteEvent
)

func produceConfigMapEvents(ctx context.Context, f *framework.Framework, stopc <-chan struct{}, minWaitBetweenEvents time.Duration) {
	c := f.ClientSet
	ns := f.Namespace.Name

	name := func(i int) string {
		return fmt.Sprintf("cm-%d", i)
	}

	existing := []int{}
	tc := time.NewTicker(minWaitBetweenEvents)
	defer tc.Stop()
	i := 0
	updates := 0
	for range tc.C {
		op := rand.Intn(3)
		if len(existing) == 0 {
			op = createEvent
		}

		switch op {
		case createEvent:
			cm := &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: name(i),
				},
			}
			_, err := c.CoreV1().ConfigMaps(ns).Create(ctx, cm, metav1.CreateOptions{})
			framework.ExpectNoError(err, "Failed to create configmap %s in namespace %s", cm.Name, ns)
			existing = append(existing, i)
			i++
		case updateEvent:
			idx := rand.Intn(len(existing))
			cm := &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: name(existing[idx]),
					Labels: map[string]string{
						"mutated": strconv.Itoa(updates),
					},
				},
			}
			_, err := c.CoreV1().ConfigMaps(ns).Update(ctx, cm, metav1.UpdateOptions{})
			framework.ExpectNoError(err, "Failed to update configmap %s in namespace %s", cm.Name, ns)
			updates++
		case deleteEvent:
			idx := rand.Intn(len(existing))
			err := c.CoreV1().ConfigMaps(ns).Delete(ctx, name(existing[idx]), metav1.DeleteOptions{})
			framework.ExpectNoError(err, "Failed to delete configmap %s in namespace %s", name(existing[idx]), ns)
			existing = append(existing[:idx], existing[idx+1:]...)
		default:
			framework.Failf("Unsupported event operation: %d", op)
		}
		select {
		case <-stopc:
			return
		default:
		}
	}
}
