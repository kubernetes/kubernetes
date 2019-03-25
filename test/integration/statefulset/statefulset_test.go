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

package statefulset

import (
	"fmt"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestSpecReplicasChange(t *testing.T) {
	s, closeFn, rm, informers, c := scSetup(t)
	defer closeFn()
	ns := framework.CreateTestingNamespace("test-spec-replicas-change", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)
	stopCh := runControllerAndInformers(rm, informers)
	defer close(stopCh)

	createHeadlessService(t, c, newHeadlessService(ns.Name))
	sts := newSTS("sts", ns.Name, 2)
	stss, _ := createSTSsPods(t, c, []*appsv1.StatefulSet{sts}, []*v1.Pod{})
	sts = stss[0]
	waitSTSStable(t, c, sts)

	// Update .Spec.Replicas and verify .Status.Replicas is changed accordingly
	scaleSTS(t, c, sts, 3)
	scaleSTS(t, c, sts, 0)
	scaleSTS(t, c, sts, 2)

	// Add a template annotation change to test STS's status does update
	// without .Spec.Replicas change
	stsClient := c.AppsV1().StatefulSets(ns.Name)
	var oldGeneration int64
	newSTS := updateSTS(t, stsClient, sts.Name, func(sts *appsv1.StatefulSet) {
		oldGeneration = sts.Generation
		sts.Spec.Template.Annotations = map[string]string{"test": "annotation"}
	})
	savedGeneration := newSTS.Generation
	if savedGeneration == oldGeneration {
		t.Fatalf("failed to verify .Generation has incremented for sts %s", sts.Name)
	}

	if err := wait.PollImmediate(pollInterval, pollTimeout, func() (bool, error) {
		newSTS, err := stsClient.Get(sts.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return newSTS.Status.ObservedGeneration >= savedGeneration, nil
	}); err != nil {
		t.Fatalf("failed to verify .Status.ObservedGeneration has incremented for sts %s: %v", sts.Name, err)
	}
}

func TestDeletingAndFailedPods(t *testing.T) {
	s, closeFn, rm, informers, c := scSetup(t)
	defer closeFn()
	ns := framework.CreateTestingNamespace("test-deleting-and-failed-pods", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)
	stopCh := runControllerAndInformers(rm, informers)
	defer close(stopCh)

	labelMap := labelMap()
	sts := newSTS("sts", ns.Name, 2)
	stss, _ := createSTSsPods(t, c, []*appsv1.StatefulSet{sts}, []*v1.Pod{})
	sts = stss[0]
	waitSTSStable(t, c, sts)

	// Verify STS creates 2 pods
	podClient := c.CoreV1().Pods(ns.Name)
	pods := getPods(t, podClient, labelMap)
	if len(pods.Items) != 2 {
		t.Fatalf("len(pods) = %d, want 2", len(pods.Items))
	}

	// Set first pod as deleting pod
	// Set finalizers for the pod to simulate pending deletion status
	deletingPod := &pods.Items[0]
	updatePod(t, podClient, deletingPod.Name, func(pod *v1.Pod) {
		pod.Finalizers = []string{"fake.example.com/blockDeletion"}
	})
	if err := c.CoreV1().Pods(ns.Name).Delete(deletingPod.Name, &metav1.DeleteOptions{}); err != nil {
		t.Fatalf("error deleting pod %s: %v", deletingPod.Name, err)
	}

	// Set second pod as failed pod
	failedPod := &pods.Items[1]
	updatePodStatus(t, podClient, failedPod.Name, func(pod *v1.Pod) {
		pod.Status.Phase = v1.PodFailed
	})

	if err := wait.PollImmediate(pollInterval, pollTimeout, func() (bool, error) {
		// Verify only 2 pods exist: deleting pod and new pod replacing failed pod
		pods = getPods(t, podClient, labelMap)
		if len(pods.Items) != 2 {
			return false, nil
		}
		// Verify deleting pod still exists
		// Immediately return false with an error if it does not exist
		if pods.Items[0].UID != deletingPod.UID && pods.Items[1].UID != deletingPod.UID {
			return false, fmt.Errorf("expected deleting pod %s still exists, but it is not found", deletingPod.Name)
		}
		// Verify failed pod does not exist anymore
		if pods.Items[0].UID == failedPod.UID || pods.Items[1].UID == failedPod.UID {
			return false, nil
		}
		// Verify both pods have non-failed status
		return pods.Items[0].Status.Phase != v1.PodFailed && pods.Items[1].Status.Phase != v1.PodFailed, nil
	}); err != nil {
		t.Fatalf("failed to verify failed pod %s has been replaced with a new non-failed pod, and deleting pod %s survives: %v", failedPod.Name, deletingPod.Name, err)
	}

	// Remove finalizers of deleting pod to simulate successful deletion
	updatePod(t, podClient, deletingPod.Name, func(pod *v1.Pod) {
		pod.Finalizers = []string{}
	})

	if err := wait.PollImmediate(pollInterval, pollTimeout, func() (bool, error) {
		// Verify only 2 pods exist: new non-deleting pod replacing deleting pod and the non-failed pod
		pods = getPods(t, podClient, labelMap)
		if len(pods.Items) != 2 {
			return false, nil
		}
		// Verify deleting pod does not exist anymore
		return pods.Items[0].UID != deletingPod.UID && pods.Items[1].UID != deletingPod.UID, nil
	}); err != nil {
		t.Fatalf("failed to verify deleting pod %s has been replaced with a new non-deleting pod: %v", deletingPod.Name, err)
	}
}
