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
		Expect(err).NotTo(HaveOccurred())

		By("creating a watch on configmaps with label B")
		watchB, err := watchConfigMaps(f, "", multipleWatchersLabelValueB)
		Expect(err).NotTo(HaveOccurred())

		By("creating a watch on configmaps with label A or B")
		watchAB, err := watchConfigMaps(f, "", multipleWatchersLabelValueA, multipleWatchersLabelValueB)
		Expect(err).NotTo(HaveOccurred())

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
		Expect(err).NotTo(HaveOccurred())
		expectEvent(watchA, watch.Added, testConfigMapA)
		expectEvent(watchAB, watch.Added, testConfigMapA)
		expectNoEvent(watchB, watch.Added, testConfigMapA)

		By("modifying configmap A and ensuring the correct watchers observe the notification")
		testConfigMapA, err = updateConfigMap(c, ns, testConfigMapA.GetName(), func(cm *v1.ConfigMap) {
			setConfigMapData(cm, "mutation", "1")
		})
		Expect(err).NotTo(HaveOccurred())
		expectEvent(watchA, watch.Modified, testConfigMapA)
		expectEvent(watchAB, watch.Modified, testConfigMapA)
		expectNoEvent(watchB, watch.Modified, testConfigMapA)

		By("modifying configmap A again and ensuring the correct watchers observe the notification")
		testConfigMapA, err = updateConfigMap(c, ns, testConfigMapA.GetName(), func(cm *v1.ConfigMap) {
			setConfigMapData(cm, "mutation", "2")
		})
		Expect(err).NotTo(HaveOccurred())
		expectEvent(watchA, watch.Modified, testConfigMapA)
		expectEvent(watchAB, watch.Modified, testConfigMapA)
		expectNoEvent(watchB, watch.Modified, testConfigMapA)

		By("deleting configmap A and ensuring the correct watchers observe the notification")
		err = c.CoreV1().ConfigMaps(ns).Delete(testConfigMapA.GetName(), nil)
		Expect(err).NotTo(HaveOccurred())
		expectEvent(watchA, watch.Deleted, nil)
		expectEvent(watchAB, watch.Deleted, nil)
		expectNoEvent(watchB, watch.Deleted, nil)

		By("creating a configmap with label B and ensuring the correct watchers observe the notification")
		testConfigMapB, err = c.CoreV1().ConfigMaps(ns).Create(testConfigMapB)
		Expect(err).NotTo(HaveOccurred())
		expectEvent(watchB, watch.Added, testConfigMapB)
		expectEvent(watchAB, watch.Added, testConfigMapB)
		expectNoEvent(watchA, watch.Added, testConfigMapB)

		By("deleting configmap B and ensuring the correct watchers observe the notification")
		err = c.CoreV1().ConfigMaps(ns).Delete(testConfigMapB.GetName(), nil)
		Expect(err).NotTo(HaveOccurred())
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
		Expect(err).NotTo(HaveOccurred())

		By("modifying the configmap once")
		testConfigMapFirstUpdate, err := updateConfigMap(c, ns, testConfigMap.GetName(), func(cm *v1.ConfigMap) {
			setConfigMapData(cm, "mutation", "1")
		})
		Expect(err).NotTo(HaveOccurred())

		By("modifying the configmap a second time")
		testConfigMapSecondUpdate, err := updateConfigMap(c, ns, testConfigMap.GetName(), func(cm *v1.ConfigMap) {
			setConfigMapData(cm, "mutation", "2")
		})
		Expect(err).NotTo(HaveOccurred())

		By("deleting the configmap")
		err = c.CoreV1().ConfigMaps(ns).Delete(testConfigMap.GetName(), nil)
		Expect(err).NotTo(HaveOccurred())

		By("creating a watch on configmaps from the resource version returned by the first update")
		testWatch, err := watchConfigMaps(f, testConfigMapFirstUpdate.ObjectMeta.ResourceVersion, fromResourceVersionLabelValue)
		Expect(err).NotTo(HaveOccurred())

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

		testConfigMap := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name: "e2e-watch-test-watch-closed",
				Labels: map[string]string{
					watchConfigMapLabelKey: watchRestartedLabelValue,
				},
			},
		}

		By("creating a watch on configmaps")
		testWatchBroken, err := watchConfigMaps(f, "", watchRestartedLabelValue)
		Expect(err).NotTo(HaveOccurred())

		By("creating a new configmap")
		testConfigMap, err = c.CoreV1().ConfigMaps(ns).Create(testConfigMap)
		Expect(err).NotTo(HaveOccurred())

		By("modifying the configmap once")
		_, err = updateConfigMap(c, ns, testConfigMap.GetName(), func(cm *v1.ConfigMap) {
			setConfigMapData(cm, "mutation", "1")
		})
		Expect(err).NotTo(HaveOccurred())

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
		Expect(err).NotTo(HaveOccurred())

		By("creating a new watch on configmaps from the last resource version observed by the first watch")
		lastEventConfigMap, ok := lastEvent.Object.(*v1.ConfigMap)
		if !ok {
			framework.Failf("Expected last notfication to refer to a configmap but got: %v", lastEvent)
		}
		testWatchRestarted, err := watchConfigMaps(f, lastEventConfigMap.ObjectMeta.ResourceVersion, watchRestartedLabelValue)
		Expect(err).NotTo(HaveOccurred())

		By("deleting the configmap")
		err = c.CoreV1().ConfigMaps(ns).Delete(testConfigMap.GetName(), nil)
		Expect(err).NotTo(HaveOccurred())

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

		testConfigMap := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name: "e2e-watch-test-label-changed",
				Labels: map[string]string{
					watchConfigMapLabelKey: toBeChangedLabelValue,
				},
			},
		}

		By("creating a watch on configmaps with a certain label")
		testWatch, err := watchConfigMaps(f, "", toBeChangedLabelValue)
		Expect(err).NotTo(HaveOccurred())

		By("creating a new configmap")
		testConfigMap, err = c.CoreV1().ConfigMaps(ns).Create(testConfigMap)
		Expect(err).NotTo(HaveOccurred())

		By("modifying the configmap once")
		testConfigMapFirstUpdate, err := updateConfigMap(c, ns, testConfigMap.GetName(), func(cm *v1.ConfigMap) {
			setConfigMapData(cm, "mutation", "1")
		})
		Expect(err).NotTo(HaveOccurred())

		By("changing the label value of the configmap")
		_, err = updateConfigMap(c, ns, testConfigMap.GetName(), func(cm *v1.ConfigMap) {
			cm.ObjectMeta.Labels[watchConfigMapLabelKey] = "wrong-value"
		})
		Expect(err).NotTo(HaveOccurred())

		By("Expecting to observe a delete notification for the watched object")
		expectEvent(testWatch, watch.Added, testConfigMap)
		expectEvent(testWatch, watch.Modified, testConfigMapFirstUpdate)
		expectEvent(testWatch, watch.Deleted, nil)

		By("modifying the configmap a second time")
		testConfigMapSecondUpdate, err := updateConfigMap(c, ns, testConfigMap.GetName(), func(cm *v1.ConfigMap) {
			setConfigMapData(cm, "mutation", "2")
		})
		Expect(err).NotTo(HaveOccurred())

		By("Expecting not to observe a notification because the object no longer meets the selector's requirements")
		expectNoEvent(testWatch, watch.Modified, testConfigMapSecondUpdate)

		By("changing the label value of the configmap back")
		testConfigMapLabelRestored, err := updateConfigMap(c, ns, testConfigMap.GetName(), func(cm *v1.ConfigMap) {
			cm.ObjectMeta.Labels[watchConfigMapLabelKey] = toBeChangedLabelValue
		})
		Expect(err).NotTo(HaveOccurred())

		By("modifying the configmap a third time")
		testConfigMapThirdUpdate, err := updateConfigMap(c, ns, testConfigMap.GetName(), func(cm *v1.ConfigMap) {
			setConfigMapData(cm, "mutation", "3")
		})
		Expect(err).NotTo(HaveOccurred())

		By("deleting the configmap")
		err = c.CoreV1().ConfigMaps(ns).Delete(testConfigMap.GetName(), nil)
		Expect(err).NotTo(HaveOccurred())

		By("Expecting to observe an add notification for the watched object when the label value was restored")
		expectEvent(testWatch, watch.Added, testConfigMapLabelRestored)
		expectEvent(testWatch, watch.Modified, testConfigMapThirdUpdate)
		expectEvent(testWatch, watch.Deleted, nil)
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
