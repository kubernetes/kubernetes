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
	"fmt"
	"math/rand"
	"time"

	"k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
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

	/*
		    Testname: watch-configmaps-with-multiple-watchers
		    Description: Ensure that multiple watchers are able to receive all add,
			update, and delete notifications on configmaps that match a label selector and do
			not receive notifications for configmaps which do not match that label selector.
	*/
	framework.ConformanceIt("should observe add, update, and delete watch notifications on configmaps", func() {
		c := f.ClientSet
		ns := f.Namespace.Name

		By("creating a watch on configmaps with label A")
		watchA, err := watchConfigMaps(f, "", multipleWatchersLabelValueA)
		Expect(err).NotTo(HaveOccurred(), "failed to create a watch on configmaps with label: %s", multipleWatchersLabelValueA)

		By("creating a watch on configmaps with label B")
		watchB, err := watchConfigMaps(f, "", multipleWatchersLabelValueB)
		Expect(err).NotTo(HaveOccurred(), "failed to create a watch on configmaps with label: %s", multipleWatchersLabelValueB)

		By("creating a watch on configmaps with label A or B")
		watchAB, err := watchConfigMaps(f, "", multipleWatchersLabelValueA, multipleWatchersLabelValueB)
		Expect(err).NotTo(HaveOccurred(), "failed to create a watch on configmaps with label %s or %s", multipleWatchersLabelValueA, multipleWatchersLabelValueB)

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

		By("creating a configmap with label A and ensuring the correct watchers observe the notification")
		testConfigMapA, err = c.CoreV1().ConfigMaps(ns).Create(testConfigMapA)
		Expect(err).NotTo(HaveOccurred(), "failed to create a configmap with label %s in namespace: %s", multipleWatchersLabelValueA, ns)
		expectEvent(watchA, watch.Added, testConfigMapA)
		expectEvent(watchAB, watch.Added, testConfigMapA)
		expectNoEvent(watchB, watch.Added, testConfigMapA)

		By("modifying configmap A and ensuring the correct watchers observe the notification")
		testConfigMapA, err = updateConfigMap(c, ns, testConfigMapA.GetName(), func(cm *v1.ConfigMap) {
			setConfigMapData(cm, "mutation", "1")
		})
		Expect(err).NotTo(HaveOccurred(), "failed to update configmap %s in namespace: %s", testConfigMapA.GetName(), ns)
		expectEvent(watchA, watch.Modified, testConfigMapA)
		expectEvent(watchAB, watch.Modified, testConfigMapA)
		expectNoEvent(watchB, watch.Modified, testConfigMapA)

		By("modifying configmap A again and ensuring the correct watchers observe the notification")
		testConfigMapA, err = updateConfigMap(c, ns, testConfigMapA.GetName(), func(cm *v1.ConfigMap) {
			setConfigMapData(cm, "mutation", "2")
		})
		Expect(err).NotTo(HaveOccurred(), "failed to update configmap %s in namespace: %s", testConfigMapA.GetName(), ns)
		expectEvent(watchA, watch.Modified, testConfigMapA)
		expectEvent(watchAB, watch.Modified, testConfigMapA)
		expectNoEvent(watchB, watch.Modified, testConfigMapA)

		By("deleting configmap A and ensuring the correct watchers observe the notification")
		err = c.CoreV1().ConfigMaps(ns).Delete(testConfigMapA.GetName(), nil)
		Expect(err).NotTo(HaveOccurred(), "failed to delete configmap %s in namespace: %s", testConfigMapA.GetName(), ns)
		expectEvent(watchA, watch.Deleted, nil)
		expectEvent(watchAB, watch.Deleted, nil)
		expectNoEvent(watchB, watch.Deleted, nil)

		By("creating a configmap with label B and ensuring the correct watchers observe the notification")
		testConfigMapB, err = c.CoreV1().ConfigMaps(ns).Create(testConfigMapB)
		Expect(err).NotTo(HaveOccurred(), "failed to create configmap %s in namespace: %s", testConfigMapB, ns)
		expectEvent(watchB, watch.Added, testConfigMapB)
		expectEvent(watchAB, watch.Added, testConfigMapB)
		expectNoEvent(watchA, watch.Added, testConfigMapB)

		By("deleting configmap B and ensuring the correct watchers observe the notification")
		err = c.CoreV1().ConfigMaps(ns).Delete(testConfigMapB.GetName(), nil)
		Expect(err).NotTo(HaveOccurred(), "failed to delete configmap %s in namespace: %s", testConfigMapB.GetName(), ns)
		expectEvent(watchB, watch.Deleted, nil)
		expectEvent(watchAB, watch.Deleted, nil)
		expectNoEvent(watchA, watch.Deleted, nil)
	})

	/*
		    Testname: watch-configmaps-from-resource-version
		    Description: Ensure that a watch can be opened from a particular resource version
			in the past and only notifications happening after that resource version are observed.
	*/
	framework.ConformanceIt("should be able to start watching from a specific resource version", func() {
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

		By("creating a new configmap")
		testConfigMap, err := c.CoreV1().ConfigMaps(ns).Create(testConfigMap)
		Expect(err).NotTo(HaveOccurred(), "failed to create configmap %s in namespace: %s", testConfigMap.GetName(), ns)

		By("modifying the configmap once")
		testConfigMapFirstUpdate, err := updateConfigMap(c, ns, testConfigMap.GetName(), func(cm *v1.ConfigMap) {
			setConfigMapData(cm, "mutation", "1")
		})
		Expect(err).NotTo(HaveOccurred(), "failed to update configmap %s in namespace: %s", testConfigMap.GetName(), ns)

		By("modifying the configmap a second time")
		testConfigMapSecondUpdate, err := updateConfigMap(c, ns, testConfigMap.GetName(), func(cm *v1.ConfigMap) {
			setConfigMapData(cm, "mutation", "2")
		})
		Expect(err).NotTo(HaveOccurred(), "failed to update configmap %s in namespace %s a second time", testConfigMap.GetName(), ns)

		By("deleting the configmap")
		err = c.CoreV1().ConfigMaps(ns).Delete(testConfigMap.GetName(), nil)
		Expect(err).NotTo(HaveOccurred(), "failed to delete configmap %s in namespace: %s", testConfigMap.GetName(), ns)

		By("creating a watch on configmaps from the resource version returned by the first update")
		testWatch, err := watchConfigMaps(f, testConfigMapFirstUpdate.ObjectMeta.ResourceVersion, fromResourceVersionLabelValue)
		Expect(err).NotTo(HaveOccurred(), "failed to create a watch on configmaps from the resource version %s returned by the first update", testConfigMapFirstUpdate.ObjectMeta.ResourceVersion)

		By("Expecting to observe notifications for all changes to the configmap after the first update")
		expectEvent(testWatch, watch.Modified, testConfigMapSecondUpdate)
		expectEvent(testWatch, watch.Deleted, nil)
	})

	/*
		    Testname: watch-configmaps-closed-and-restarted
		    Description: Ensure that a watch can be reopened from the last resource version
			observed by the previous watch, and it will continue delivering notifications from
			that point in time.
	*/
	framework.ConformanceIt("should be able to restart watching from the last resource version observed by the previous watch", func() {
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

		By("creating a watch on configmaps")
		testWatchBroken, err := watchConfigMaps(f, "", watchRestartedLabelValue)
		Expect(err).NotTo(HaveOccurred(), "failed to create a watch on configmap with label: %s", watchRestartedLabelValue)

		By("creating a new configmap")
		testConfigMap, err = c.CoreV1().ConfigMaps(ns).Create(testConfigMap)
		Expect(err).NotTo(HaveOccurred(), "failed to create configmap %s in namespace: %s", configMapName, ns)

		By("modifying the configmap once")
		_, err = updateConfigMap(c, ns, testConfigMap.GetName(), func(cm *v1.ConfigMap) {
			setConfigMapData(cm, "mutation", "1")
		})
		Expect(err).NotTo(HaveOccurred(), "failed to update configmap %s in namespace: %s", configMapName, ns)

		By("closing the watch once it receives two notifications")
		expectEvent(testWatchBroken, watch.Added, testConfigMap)
		lastEvent, ok := waitForEvent(testWatchBroken, watch.Modified, nil, 1*time.Minute)
		if !ok {
			framework.Failf("Timed out waiting for second watch notification")
		}
		testWatchBroken.Stop()

		By("modifying the configmap a second time, while the watch is closed")
		testConfigMapSecondUpdate, err := updateConfigMap(c, ns, testConfigMap.GetName(), func(cm *v1.ConfigMap) {
			setConfigMapData(cm, "mutation", "2")
		})
		Expect(err).NotTo(HaveOccurred(), "failed to update configmap %s in namespace %s a second time", configMapName, ns)

		By("creating a new watch on configmaps from the last resource version observed by the first watch")
		lastEventConfigMap, ok := lastEvent.Object.(*v1.ConfigMap)
		if !ok {
			framework.Failf("Expected last notfication to refer to a configmap but got: %v", lastEvent)
		}
		testWatchRestarted, err := watchConfigMaps(f, lastEventConfigMap.ObjectMeta.ResourceVersion, watchRestartedLabelValue)
		Expect(err).NotTo(HaveOccurred(), "failed to create a new watch on configmaps from the last resource version %s observed by the first watch", lastEventConfigMap.ObjectMeta.ResourceVersion)

		By("deleting the configmap")
		err = c.CoreV1().ConfigMaps(ns).Delete(testConfigMap.GetName(), nil)
		Expect(err).NotTo(HaveOccurred(), "failed to delete configmap %s in namespace: %s", configMapName, ns)

		By("Expecting to observe notifications for all changes to the configmap since the first watch closed")
		expectEvent(testWatchRestarted, watch.Modified, testConfigMapSecondUpdate)
		expectEvent(testWatchRestarted, watch.Deleted, nil)
	})

	/*
		    Testname: watch-configmaps-label-changed
		    Description: Ensure that a watched object stops meeting the requirements of
			a watch's selector, the watch will observe a delete, and will not observe
			notifications for that object until it meets the selector's requirements again.
	*/
	framework.ConformanceIt("should observe an object deletion if it stops meeting the requirements of the selector", func() {
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

		By("creating a watch on configmaps with a certain label")
		testWatch, err := watchConfigMaps(f, "", toBeChangedLabelValue)
		Expect(err).NotTo(HaveOccurred(), "failed to create a watch on configmap with label: %s", toBeChangedLabelValue)

		By("creating a new configmap")
		testConfigMap, err = c.CoreV1().ConfigMaps(ns).Create(testConfigMap)
		Expect(err).NotTo(HaveOccurred(), "failed to create configmap %s in namespace: %s", configMapName, ns)

		By("modifying the configmap once")
		testConfigMapFirstUpdate, err := updateConfigMap(c, ns, testConfigMap.GetName(), func(cm *v1.ConfigMap) {
			setConfigMapData(cm, "mutation", "1")
		})
		Expect(err).NotTo(HaveOccurred(), "failed to update configmap %s in namespace: %s", configMapName, ns)

		By("changing the label value of the configmap")
		_, err = updateConfigMap(c, ns, testConfigMap.GetName(), func(cm *v1.ConfigMap) {
			cm.ObjectMeta.Labels[watchConfigMapLabelKey] = "wrong-value"
		})
		Expect(err).NotTo(HaveOccurred(), "failed to update configmap %s in namespace %s by changing label value", configMapName, ns)

		By("Expecting to observe a delete notification for the watched object")
		expectEvent(testWatch, watch.Added, testConfigMap)
		expectEvent(testWatch, watch.Modified, testConfigMapFirstUpdate)
		expectEvent(testWatch, watch.Deleted, nil)

		By("modifying the configmap a second time")
		testConfigMapSecondUpdate, err := updateConfigMap(c, ns, testConfigMap.GetName(), func(cm *v1.ConfigMap) {
			setConfigMapData(cm, "mutation", "2")
		})
		Expect(err).NotTo(HaveOccurred(), "failed to update configmap %s in namespace %s a second time", configMapName, ns)

		By("Expecting not to observe a notification because the object no longer meets the selector's requirements")
		expectNoEvent(testWatch, watch.Modified, testConfigMapSecondUpdate)

		By("changing the label value of the configmap back")
		testConfigMapLabelRestored, err := updateConfigMap(c, ns, testConfigMap.GetName(), func(cm *v1.ConfigMap) {
			cm.ObjectMeta.Labels[watchConfigMapLabelKey] = toBeChangedLabelValue
		})
		Expect(err).NotTo(HaveOccurred(), "failed to update configmap %s in namespace %s by changing label value back", configMapName, ns)

		By("modifying the configmap a third time")
		testConfigMapThirdUpdate, err := updateConfigMap(c, ns, testConfigMap.GetName(), func(cm *v1.ConfigMap) {
			setConfigMapData(cm, "mutation", "3")
		})
		Expect(err).NotTo(HaveOccurred(), "failed to update configmap %s in namespace %s a third time", configMapName, ns)

		By("deleting the configmap")
		err = c.CoreV1().ConfigMaps(ns).Delete(testConfigMap.GetName(), nil)
		Expect(err).NotTo(HaveOccurred(), "failed to delete configmap %s in namespace: %s", configMapName, ns)

		By("Expecting to observe an add notification for the watched object when the label value was restored")
		expectEvent(testWatch, watch.Added, testConfigMapLabelRestored)
		expectEvent(testWatch, watch.Modified, testConfigMapThirdUpdate)
		expectEvent(testWatch, watch.Deleted, nil)
	})

	/*
	   Testname: watch-consistency
	   Description: Ensure that concurrent watches are consistent with each other by initiating an additional watch
	   for events received from the first watch, initiated at the resource version of the event, and checking that all
	   resource versions of all events match. Events are produced from writes on a background goroutine.
	*/
	It("should receive events on concurrent watches in same order", func() {
		c := f.ClientSet
		ns := f.Namespace.Name

		iterations := 100

		By("starting a background goroutine to produce watch events")
		donec := make(chan struct{})
		stopc := make(chan struct{})
		go func() {
			defer GinkgoRecover()
			defer close(donec)
			produceConfigMapEvents(f, stopc, 5*time.Millisecond)
		}()

		By("creating watches starting from each resource version of the events produced and verifying they all receive resource versions in the same order")
		wcs := []watch.Interface{}
		resourceVersion := "0"
		for i := 0; i < iterations; i++ {
			wc, err := c.CoreV1().ConfigMaps(ns).Watch(metav1.ListOptions{ResourceVersion: resourceVersion})
			Expect(err).NotTo(HaveOccurred())
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

func watchConfigMaps(f *framework.Framework, resourceVersion string, labels ...string) (watch.Interface, error) {
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
	return c.CoreV1().ConfigMaps(ns).Watch(opts)
}

func int64ptr(i int) *int64 {
	i64 := int64(i)
	return &i64
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
	case event := <-watch.ResultChan():
		if configMap, ok := event.Object.(*v1.ConfigMap); ok {
			return configMap
		} else {
			framework.Failf("expected config map")
		}
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

func produceConfigMapEvents(f *framework.Framework, stopc <-chan struct{}, minWaitBetweenEvents time.Duration) {
	c := f.ClientSet
	ns := f.Namespace.Name

	name := func(i int) string {
		return fmt.Sprintf("cm-%d", i)
	}

	existing := []int{}
	tc := time.NewTicker(minWaitBetweenEvents)
	defer tc.Stop()
	i := 0
	for range tc.C {
		op := rand.Intn(3)
		if len(existing) == 0 {
			op = createEvent
		}

		cm := &v1.ConfigMap{}
		switch op {
		case createEvent:
			cm.Name = name(i)
			_, err := c.CoreV1().ConfigMaps(ns).Create(cm)
			Expect(err).NotTo(HaveOccurred())
			existing = append(existing, i)
			i += 1
		case updateEvent:
			idx := rand.Intn(len(existing))
			cm.Name = name(existing[idx])
			_, err := c.CoreV1().ConfigMaps(ns).Update(cm)
			Expect(err).NotTo(HaveOccurred())
		case deleteEvent:
			idx := rand.Intn(len(existing))
			err := c.CoreV1().ConfigMaps(ns).Delete(name(existing[idx]), &metav1.DeleteOptions{})
			Expect(err).NotTo(HaveOccurred())
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
