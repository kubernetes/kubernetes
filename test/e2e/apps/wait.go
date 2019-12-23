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

package apps

import (
	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"

	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/test/e2e/framework"
	e2esset "k8s.io/kubernetes/test/e2e/framework/statefulset"
)

// waitForPartitionedRollingUpdate waits for all Pods in set to exist and have the correct revision. set must have
// a RollingUpdateStatefulSetStrategyType with a non-nil RollingUpdate and Partition. All Pods with ordinals less
// than or equal to the Partition are expected to be at set's current revision. All other Pods are expected to be
// at its update revision.
func waitForPartitionedRollingUpdate(c clientset.Interface, set *appsv1.StatefulSet) (*appsv1.StatefulSet, *v1.PodList) {
	var pods *v1.PodList
	if set.Spec.UpdateStrategy.Type != appsv1.RollingUpdateStatefulSetStrategyType {
		framework.Failf("StatefulSet %s/%s attempt to wait for partitioned update with updateStrategy %s",
			set.Namespace,
			set.Name,
			set.Spec.UpdateStrategy.Type)
	}
	if set.Spec.UpdateStrategy.RollingUpdate == nil || set.Spec.UpdateStrategy.RollingUpdate.Partition == nil {
		framework.Failf("StatefulSet %s/%s attempt to wait for partitioned update with nil RollingUpdate or nil Partition",
			set.Namespace,
			set.Name)
	}
	e2esset.WaitForState(c, set, func(set2 *appsv1.StatefulSet, pods2 *v1.PodList) (bool, error) {
		set = set2
		pods = pods2
		partition := int(*set.Spec.UpdateStrategy.RollingUpdate.Partition)
		if len(pods.Items) < int(*set.Spec.Replicas) {
			return false, nil
		}
		if partition <= 0 && set.Status.UpdateRevision != set.Status.CurrentRevision {
			framework.Logf("Waiting for StatefulSet %s/%s to complete update",
				set.Namespace,
				set.Name,
			)
			e2esset.SortStatefulPods(pods)
			for i := range pods.Items {
				if pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel] != set.Status.UpdateRevision {
					framework.Logf("Waiting for Pod %s/%s to have revision %s update revision %s",
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
				framework.Logf("Waiting for Pod %s/%s to have revision %s update revision %s",
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

// waitForStatus waits for the StatefulSetStatus's ObservedGeneration to be greater than or equal to set's Generation.
// The returned StatefulSet contains such a StatefulSetStatus
func waitForStatus(c clientset.Interface, set *appsv1.StatefulSet) *appsv1.StatefulSet {
	e2esset.WaitForState(c, set, func(set2 *appsv1.StatefulSet, pods *v1.PodList) (bool, error) {
		if set2.Status.ObservedGeneration >= set.Generation {
			set = set2
			return true, nil
		}
		return false, nil
	})
	return set
}

// waitForPodNotReady waits for the Pod named podName in set to exist and to not have a Ready condition.
func waitForPodNotReady(c clientset.Interface, set *appsv1.StatefulSet, podName string) (*appsv1.StatefulSet, *v1.PodList) {
	var pods *v1.PodList
	e2esset.WaitForState(c, set, func(set2 *appsv1.StatefulSet, pods2 *v1.PodList) (bool, error) {
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

// waitForRollingUpdate waits for all Pods in set to exist and have the correct revision and for the RollingUpdate to
// complete. set must have a RollingUpdateStatefulSetStrategyType.
func waitForRollingUpdate(c clientset.Interface, set *appsv1.StatefulSet) (*appsv1.StatefulSet, *v1.PodList) {
	var pods *v1.PodList
	if set.Spec.UpdateStrategy.Type != appsv1.RollingUpdateStatefulSetStrategyType {
		framework.Failf("StatefulSet %s/%s attempt to wait for rolling update with updateStrategy %s",
			set.Namespace,
			set.Name,
			set.Spec.UpdateStrategy.Type)
	}
	e2esset.WaitForState(c, set, func(set2 *appsv1.StatefulSet, pods2 *v1.PodList) (bool, error) {
		set = set2
		pods = pods2
		if len(pods.Items) < int(*set.Spec.Replicas) {
			return false, nil
		}
		if set.Status.UpdateRevision != set.Status.CurrentRevision {
			framework.Logf("Waiting for StatefulSet %s/%s to complete update",
				set.Namespace,
				set.Name,
			)
			e2esset.SortStatefulPods(pods)
			for i := range pods.Items {
				if pods.Items[i].Labels[appsv1.StatefulSetRevisionLabel] != set.Status.UpdateRevision {
					framework.Logf("Waiting for Pod %s/%s to have revision %s update revision %s",
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

// waitForRunningAndNotReady waits for numStatefulPods in ss to be Running and not Ready.
func waitForRunningAndNotReady(c clientset.Interface, numStatefulPods int32, ss *appsv1.StatefulSet) {
	e2esset.WaitForRunning(c, numStatefulPods, 0, ss)
}
