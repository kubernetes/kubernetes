/*
Copyright 2019 The Kubernetes Authors.

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

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
)

// WaitForPartitionedRollingUpdate waits for all Pods in set to exist and have the correct revision. set must have
// a RollingUpdateStatefulSetStrategyType with a non-nil RollingUpdate and Partition. All Pods with ordinals less
// than or equal to the Partition are expected to be at set's current revision. All other Pods are expected to be
// at its update revision.
func WaitForPartitionedRollingUpdate(c clientset.Interface, set *appsv1.StatefulSet) (*appsv1.StatefulSet, *v1.PodList) {
	var pods *v1.PodList
	if set.Spec.UpdateStrategy.Type != appsv1.RollingUpdateStatefulSetStrategyType {
		e2elog.Failf("StatefulSet %s/%s attempt to wait for partitioned update with updateStrategy %s",
			set.Namespace,
			set.Name,
			set.Spec.UpdateStrategy.Type)
	}
	if set.Spec.UpdateStrategy.RollingUpdate == nil || set.Spec.UpdateStrategy.RollingUpdate.Partition == nil {
		e2elog.Failf("StatefulSet %s/%s attempt to wait for partitioned update with nil RollingUpdate or nil Partition",
			set.Namespace,
			set.Name)
	}
	WaitForState(c, set, func(set2 *appsv1.StatefulSet, pods2 *v1.PodList) (bool, error) {
		set = set2
		pods = pods2
		partition := int(*set.Spec.UpdateStrategy.RollingUpdate.Partition)
		if len(pods.Items) < int(*set.Spec.Replicas) {
			return false, nil
		}
		if partition <= 0 && set.Status.UpdateRevision != set.Status.CurrentRevision {
			e2elog.Logf("Waiting for StatefulSet %s/%s to complete update",
				set.Namespace,
				set.Name,
			)
			SortStatefulPods(pods)
			for i := range pods.Items {
				if pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel] != set.Status.UpdateRevision {
					e2elog.Logf("Waiting for Pod %s/%s to have revision %s update revision %s",
						pods.Items[i].Namespace,
						pods.Items[i].Name,
						set.Status.UpdateRevision,
						pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel])
				}
			}
			return false, nil
		}
		for i := int(*set.Spec.Replicas) - 1; i >= partition; i-- {
			if pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel] != set.Status.UpdateRevision {
				e2elog.Logf("Waiting for Pod %s/%s to have revision %s update revision %s",
					pods.Items[i].Namespace,
					pods.Items[i].Name,
					set.Status.UpdateRevision,
					pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel])
				return false, nil
			}
		}
		return true, nil
	})
	return set, pods
}

// WaitForRunning waits for numPodsRunning in ss to be Running and for the first
// numPodsReady ordinals to be Ready.
func WaitForRunning(c clientset.Interface, numPodsRunning, numPodsReady int32, ss *appsv1.StatefulSet) {
	pollErr := wait.PollImmediate(StatefulSetPoll, StatefulSetTimeout,
		func() (bool, error) {
			podList := GetPodList(c, ss)
			SortStatefulPods(podList)
			if int32(len(podList.Items)) < numPodsRunning {
				e2elog.Logf("Found %d stateful pods, waiting for %d", len(podList.Items), numPodsRunning)
				return false, nil
			}
			if int32(len(podList.Items)) > numPodsRunning {
				return false, fmt.Errorf("too many pods scheduled, expected %d got %d", numPodsRunning, len(podList.Items))
			}
			for _, p := range podList.Items {
				shouldBeReady := getStatefulPodOrdinal(&p) < int(numPodsReady)
				isReady := podutil.IsPodReady(&p)
				desiredReadiness := shouldBeReady == isReady
				e2elog.Logf("Waiting for pod %v to enter %v - Ready=%v, currently %v - Ready=%v", p.Name, v1.PodRunning, shouldBeReady, p.Status.Phase, isReady)
				if p.Status.Phase != v1.PodRunning || !desiredReadiness {
					return false, nil
				}
			}
			return true, nil
		})
	if pollErr != nil {
		e2elog.Failf("Failed waiting for pods to enter running: %v", pollErr)
	}
}

// WaitForState periodically polls for the ss and its pods until the until function returns either true or an error
func WaitForState(c clientset.Interface, ss *appsv1.StatefulSet, until func(*appsv1.StatefulSet, *v1.PodList) (bool, error)) {
	pollErr := wait.PollImmediate(StatefulSetPoll, StatefulSetTimeout,
		func() (bool, error) {
			ssGet, err := c.AppsV1().StatefulSets(ss.Namespace).Get(ss.Name, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			podList := GetPodList(c, ssGet)
			return until(ssGet, podList)
		})
	if pollErr != nil {
		e2elog.Failf("Failed waiting for state update: %v", pollErr)
	}
}

// WaitForStatus waits for the StatefulSetStatus's ObservedGeneration to be greater than or equal to set's Generation.
// The returned StatefulSet contains such a StatefulSetStatus
func WaitForStatus(c clientset.Interface, set *appsv1.StatefulSet) *appsv1.StatefulSet {
	WaitForState(c, set, func(set2 *appsv1.StatefulSet, pods *v1.PodList) (bool, error) {
		if set2.Status.ObservedGeneration >= set.Generation {
			set = set2
			return true, nil
		}
		return false, nil
	})
	return set
}

// WaitForRunningAndReady waits for numStatefulPods in ss to be Running and Ready.
func WaitForRunningAndReady(c clientset.Interface, numStatefulPods int32, ss *appsv1.StatefulSet) {
	WaitForRunning(c, numStatefulPods, numStatefulPods, ss)
}

// WaitForPodReady waits for the Pod named podName in set to exist and have a Ready condition.
func WaitForPodReady(c clientset.Interface, set *appsv1.StatefulSet, podName string) (*appsv1.StatefulSet, *v1.PodList) {
	var pods *v1.PodList
	WaitForState(c, set, func(set2 *appsv1.StatefulSet, pods2 *v1.PodList) (bool, error) {
		set = set2
		pods = pods2
		for i := range pods.Items {
			if pods.Items[i].Name == podName {
				return podutil.IsPodReady(&pods.Items[i]), nil
			}
		}
		return false, nil
	})
	return set, pods
}

// WaitForPodNotReady waits for the Pod named podName in set to exist and to not have a Ready condition.
func WaitForPodNotReady(c clientset.Interface, set *appsv1.StatefulSet, podName string) (*appsv1.StatefulSet, *v1.PodList) {
	var pods *v1.PodList
	WaitForState(c, set, func(set2 *appsv1.StatefulSet, pods2 *v1.PodList) (bool, error) {
		set = set2
		pods = pods2
		for i := range pods.Items {
			if pods.Items[i].Name == podName {
				return !podutil.IsPodReady(&pods.Items[i]), nil
			}
		}
		return false, nil
	})
	return set, pods
}

// WaitForRollingUpdate waits for all Pods in set to exist and have the correct revision and for the RollingUpdate to
// complete. set must have a RollingUpdateStatefulSetStrategyType.
func WaitForRollingUpdate(c clientset.Interface, set *appsv1.StatefulSet) (*appsv1.StatefulSet, *v1.PodList) {
	var pods *v1.PodList
	if set.Spec.UpdateStrategy.Type != appsv1.RollingUpdateStatefulSetStrategyType {
		e2elog.Failf("StatefulSet %s/%s attempt to wait for rolling update with updateStrategy %s",
			set.Namespace,
			set.Name,
			set.Spec.UpdateStrategy.Type)
	}
	WaitForState(c, set, func(set2 *appsv1.StatefulSet, pods2 *v1.PodList) (bool, error) {
		set = set2
		pods = pods2
		if len(pods.Items) < int(*set.Spec.Replicas) {
			return false, nil
		}
		if set.Status.UpdateRevision != set.Status.CurrentRevision {
			e2elog.Logf("Waiting for StatefulSet %s/%s to complete update",
				set.Namespace,
				set.Name,
			)
			SortStatefulPods(pods)
			for i := range pods.Items {
				if pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel] != set.Status.UpdateRevision {
					e2elog.Logf("Waiting for Pod %s/%s to have revision %s update revision %s",
						pods.Items[i].Namespace,
						pods.Items[i].Name,
						set.Status.UpdateRevision,
						pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel])
				}
			}
			return false, nil
		}
		return true, nil
	})
	return set, pods
}

// WaitForRunningAndNotReady waits for numStatefulPods in ss to be Running and not Ready.
func WaitForRunningAndNotReady(c clientset.Interface, numStatefulPods int32, ss *appsv1.StatefulSet) {
	WaitForRunning(c, numStatefulPods, 0, ss)
}

// WaitForStatusReadyReplicas waits for the ss.Status.ReadyReplicas to be equal to expectedReplicas
func WaitForStatusReadyReplicas(c clientset.Interface, ss *appsv1.StatefulSet, expectedReplicas int32) {
	e2elog.Logf("Waiting for statefulset status.replicas updated to %d", expectedReplicas)

	ns, name := ss.Namespace, ss.Name
	pollErr := wait.PollImmediate(StatefulSetPoll, StatefulSetTimeout,
		func() (bool, error) {
			ssGet, err := c.AppsV1().StatefulSets(ns).Get(name, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			if ssGet.Status.ObservedGeneration < ss.Generation {
				return false, nil
			}
			if ssGet.Status.ReadyReplicas != expectedReplicas {
				e2elog.Logf("Waiting for stateful set status.readyReplicas to become %d, currently %d", expectedReplicas, ssGet.Status.ReadyReplicas)
				return false, nil
			}
			return true, nil
		})
	if pollErr != nil {
		e2elog.Failf("Failed waiting for stateful set status.readyReplicas updated to %d: %v", expectedReplicas, pollErr)
	}
}

// WaitForStatusReplicas waits for the ss.Status.Replicas to be equal to expectedReplicas
func WaitForStatusReplicas(c clientset.Interface, ss *appsv1.StatefulSet, expectedReplicas int32) {
	e2elog.Logf("Waiting for statefulset status.replicas updated to %d", expectedReplicas)

	ns, name := ss.Namespace, ss.Name
	pollErr := wait.PollImmediate(StatefulSetPoll, StatefulSetTimeout,
		func() (bool, error) {
			ssGet, err := c.AppsV1().StatefulSets(ns).Get(name, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			if ssGet.Status.ObservedGeneration < ss.Generation {
				return false, nil
			}
			if ssGet.Status.Replicas != expectedReplicas {
				e2elog.Logf("Waiting for stateful set status.replicas to become %d, currently %d", expectedReplicas, ssGet.Status.Replicas)
				return false, nil
			}
			return true, nil
		})
	if pollErr != nil {
		e2elog.Failf("Failed waiting for stateful set status.replicas updated to %d: %v", expectedReplicas, pollErr)
	}
}

// Saturate waits for all Pods in ss to become Running and Ready.
func Saturate(c clientset.Interface, ss *appsv1.StatefulSet) {
	var i int32
	for i = 0; i < *(ss.Spec.Replicas); i++ {
		e2elog.Logf("Waiting for stateful pod at index %v to enter Running", i)
		WaitForRunning(c, i+1, i, ss)
		e2elog.Logf("Resuming stateful pod at index %v", i)
		ResumeNextPod(c, ss)
	}
}
