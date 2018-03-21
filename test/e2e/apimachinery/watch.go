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
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	watchPodLabelKey    = "watch-this-pod"
	watchPodLabelValueA = "AAA"
	watchPodLabelValueB = "BBB"
)

var _ = SIGDescribe("Watchers", func() {
	f := framework.NewDefaultFramework("watch")

	It("should observe add, update, and delete events on pods", func() {
		c := f.ClientSet
		ns := f.Namespace.Name

		By("creating multiple similar watches on pods")
		watchA, err := watchPodsWithLabels(f, watchPodLabelValueA)
		Expect(err).NotTo(HaveOccurred())

		watchB, err := watchPodsWithLabels(f, watchPodLabelValueB)
		Expect(err).NotTo(HaveOccurred())

		watchAB, err := watchPodsWithLabels(f, watchPodLabelValueA, watchPodLabelValueB)
		Expect(err).NotTo(HaveOccurred())

		By("creating, modifying, and deleting pods")
		testPodA := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "e2e-watch-test-pod-a",
				Labels: map[string]string{
					watchPodLabelKey: watchPodLabelValueA,
				},
			},
			Spec: v1.PodSpec{
				ActiveDeadlineSeconds: int64ptr(20),
				Containers: []v1.Container{
					{
						Name:  "example",
						Image: framework.GetPauseImageName(c),
					},
				},
			},
		}
		testPodB := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "e2e-watch-test-pod-b",
				Labels: map[string]string{
					watchPodLabelKey: watchPodLabelValueB,
				},
			},
			Spec: v1.PodSpec{
				ActiveDeadlineSeconds: int64ptr(20),
				Containers: []v1.Container{
					{
						Name:  "example",
						Image: framework.GetPauseImageName(c),
					},
				},
			},
		}
		testPodA, err = c.CoreV1().Pods(ns).Create(testPodA)
		Expect(err).NotTo(HaveOccurred())
		expectEvent(watchA, watch.Added, testPodA)
		expectEvent(watchAB, watch.Added, testPodA)
		expectNoEvent(watchB, watch.Added, testPodA)

		testPodA, err = updatePod(f, testPodA.GetName(), func(p *v1.Pod) {
			p.Spec.ActiveDeadlineSeconds = int64ptr(10)
		})
		Expect(err).NotTo(HaveOccurred())
		expectEvent(watchA, watch.Modified, testPodA)
		expectEvent(watchAB, watch.Modified, testPodA)
		expectNoEvent(watchB, watch.Modified, testPodA)

		testPodA, err = updatePod(f, testPodA.GetName(), func(p *v1.Pod) {
			p.Spec.ActiveDeadlineSeconds = int64ptr(5)
		})
		Expect(err).NotTo(HaveOccurred())
		expectEvent(watchA, watch.Modified, testPodA)
		expectEvent(watchAB, watch.Modified, testPodA)
		expectNoEvent(watchB, watch.Modified, testPodA)

		err = c.CoreV1().Pods(ns).Delete(testPodA.GetName(), nil)
		Expect(err).NotTo(HaveOccurred())
		expectEvent(watchA, watch.Deleted, nil)
		expectEvent(watchAB, watch.Deleted, nil)
		expectNoEvent(watchB, watch.Deleted, nil)

		testPodB, err = c.CoreV1().Pods(ns).Create(testPodB)
		Expect(err).NotTo(HaveOccurred())
		expectEvent(watchB, watch.Added, testPodB)
		expectEvent(watchAB, watch.Added, testPodB)
		expectNoEvent(watchA, watch.Added, testPodB)

		err = c.CoreV1().Pods(ns).Delete(testPodB.GetName(), nil)
		Expect(err).NotTo(HaveOccurred())
		expectEvent(watchB, watch.Deleted, nil)
		expectEvent(watchAB, watch.Deleted, nil)
		expectNoEvent(watchA, watch.Deleted, nil)
	})
})

func watchPodsWithLabels(f *framework.Framework, labels ...string) (watch.Interface, error) {
	c := f.ClientSet
	ns := f.Namespace.Name
	opts := metav1.ListOptions{
		LabelSelector: metav1.FormatLabelSelector(&metav1.LabelSelector{
			MatchExpressions: []metav1.LabelSelectorRequirement{
				{
					Key:      watchPodLabelKey,
					Operator: metav1.LabelSelectorOpIn,
					Values:   labels,
				},
			},
		}),
	}
	return c.CoreV1().Pods(ns).Watch(opts)
}

func int64ptr(i int) *int64 {
	i64 := int64(i)
	return &i64
}

type updatePodFunc func(p *v1.Pod)

func updatePod(f *framework.Framework, name string, update updatePodFunc) (*v1.Pod, error) {
	c := f.ClientSet
	ns := f.Namespace.Name
	var p *v1.Pod
	pollErr := wait.PollImmediate(2*time.Second, 1*time.Minute, func() (bool, error) {
		var err error
		if p, err = c.CoreV1().Pods(ns).Get(name, metav1.GetOptions{}); err != nil {
			return false, err
		}
		update(p)
		if p, err = c.CoreV1().Pods(ns).Update(p); err == nil {
			return true, nil
		}
		// Only retry update on conflict
		if !errors.IsConflict(err) {
			return false, err
		}
		return false, nil
	})
	return p, pollErr
}

func expectEvent(w watch.Interface, eventType watch.EventType, object runtime.Object) {
	if event, ok := waitForEvent(w, eventType, object); !ok {
		framework.Failf("Timed out waiting for expected event: %v", event)
	}
}

func expectNoEvent(w watch.Interface, eventType watch.EventType, object runtime.Object) {
	if event, ok := waitForEvent(w, eventType, object); ok {
		framework.Failf("Unexpected event occurred: %v", event)
	}
}

func waitForEvent(w watch.Interface, expectType watch.EventType, expectObject runtime.Object) (watch.Event, bool) {
	stopTimer := time.NewTimer(1 * time.Minute)
	for {
		select {
		case actual := <-w.ResultChan():
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
