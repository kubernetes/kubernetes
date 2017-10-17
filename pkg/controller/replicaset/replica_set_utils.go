/*
Copyright 2016 The Kubernetes Authors.

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

// If you make changes to this file, you should also make the corresponding change in ReplicationController.

package replicaset

import (
	"fmt"
	"reflect"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	unversionedextensions "k8s.io/client-go/kubernetes/typed/extensions/v1beta1"
	"k8s.io/client-go/tools/cache"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller"
)

// updateReplicaSetStatus attempts to update the Status.Replicas of the given ReplicaSet, with a single GET/PUT retry.
func updateReplicaSetStatus(c unversionedextensions.ReplicaSetInterface, rs *extensions.ReplicaSet, newStatus extensions.ReplicaSetStatus) (*extensions.ReplicaSet, error) {
	// This is the steady state. It happens when the ReplicaSet doesn't have any expectations, since
	// we do a periodic relist every 30s. If the generations differ but the replicas are
	// the same, a caller might've resized to the same replica count.
	if rs.Status.Replicas == newStatus.Replicas &&
		rs.Status.FullyLabeledReplicas == newStatus.FullyLabeledReplicas &&
		rs.Status.ReadyReplicas == newStatus.ReadyReplicas &&
		rs.Status.AvailableReplicas == newStatus.AvailableReplicas &&
		rs.Generation == rs.Status.ObservedGeneration &&
		reflect.DeepEqual(rs.Status.Conditions, newStatus.Conditions) {
		return rs, nil
	}

	// Save the generation number we acted on, otherwise we might wrongfully indicate
	// that we've seen a spec update when we retry.
	// TODO: This can clobber an update if we allow multiple agents to write to the
	// same status.
	newStatus.ObservedGeneration = rs.Generation

	var getErr, updateErr error
	var updatedRS *extensions.ReplicaSet
	for i, rs := 0, rs; ; i++ {
		glog.V(4).Infof(fmt.Sprintf("Updating status for ReplicaSet: %s/%s, ", rs.Namespace, rs.Name) +
			fmt.Sprintf("replicas %d->%d (need %d), ", rs.Status.Replicas, newStatus.Replicas, *(rs.Spec.Replicas)) +
			fmt.Sprintf("fullyLabeledReplicas %d->%d, ", rs.Status.FullyLabeledReplicas, newStatus.FullyLabeledReplicas) +
			fmt.Sprintf("readyReplicas %d->%d, ", rs.Status.ReadyReplicas, newStatus.ReadyReplicas) +
			fmt.Sprintf("availableReplicas %d->%d, ", rs.Status.AvailableReplicas, newStatus.AvailableReplicas) +
			fmt.Sprintf("sequence No: %v->%v", rs.Status.ObservedGeneration, newStatus.ObservedGeneration))

		rs.Status = newStatus
		updatedRS, updateErr = c.UpdateStatus(rs)
		if updateErr == nil {
			return updatedRS, nil
		}
		// Stop retrying if we exceed statusUpdateRetries - the replicaSet will be requeued with a rate limit.
		if i >= statusUpdateRetries {
			break
		}
		// Update the ReplicaSet with the latest resource version for the next poll
		if rs, getErr = c.Get(rs.Name, metav1.GetOptions{}); getErr != nil {
			// If the GET fails we can't trust status.Replicas anymore. This error
			// is bound to be more interesting than the update failure.
			return nil, getErr
		}
	}

	return nil, updateErr
}

func calculateStatus(rs *extensions.ReplicaSet, filteredPods []*v1.Pod, manageReplicasErr error) extensions.ReplicaSetStatus {
	newStatus := rs.Status
	// Count the number of pods that have labels matching the labels of the pod
	// template of the replica set, the matching pods may have more
	// labels than are in the template. Because the label of podTemplateSpec is
	// a superset of the selector of the replica set, so the possible
	// matching pods must be part of the filteredPods.
	fullyLabeledReplicasCount := 0
	readyReplicasCount := 0
	availableReplicasCount := 0
	templateLabel := labels.Set(rs.Spec.Template.Labels).AsSelectorPreValidated()
	for _, pod := range filteredPods {
		if templateLabel.Matches(labels.Set(pod.Labels)) {
			fullyLabeledReplicasCount++
		}
		if podutil.IsPodReady(pod) {
			readyReplicasCount++
			if podutil.IsPodAvailable(pod, rs.Spec.MinReadySeconds, metav1.Now()) {
				availableReplicasCount++
			}
		}
	}

	failureCond := GetCondition(rs.Status, extensions.ReplicaSetReplicaFailure)
	if manageReplicasErr != nil && failureCond == nil {
		var reason string
		if diff := len(filteredPods) - int(*(rs.Spec.Replicas)); diff < 0 {
			reason = "FailedCreate"
		} else if diff > 0 {
			reason = "FailedDelete"
		}
		cond := NewReplicaSetCondition(extensions.ReplicaSetReplicaFailure, v1.ConditionTrue, reason, manageReplicasErr.Error())
		SetCondition(&newStatus, cond)
	} else if manageReplicasErr == nil && failureCond != nil {
		RemoveCondition(&newStatus, extensions.ReplicaSetReplicaFailure)
	}

	newStatus.Replicas = int32(len(filteredPods))
	newStatus.FullyLabeledReplicas = int32(fullyLabeledReplicasCount)
	newStatus.ReadyReplicas = int32(readyReplicasCount)
	newStatus.AvailableReplicas = int32(availableReplicasCount)
	return newStatus
}

// NewReplicaSetCondition creates a new replicaset condition.
func NewReplicaSetCondition(condType extensions.ReplicaSetConditionType, status v1.ConditionStatus, reason, msg string) extensions.ReplicaSetCondition {
	return extensions.ReplicaSetCondition{
		Type:               condType,
		Status:             status,
		LastTransitionTime: metav1.Now(),
		Reason:             reason,
		Message:            msg,
	}
}

// GetCondition returns a replicaset condition with the provided type if it exists.
func GetCondition(status extensions.ReplicaSetStatus, condType extensions.ReplicaSetConditionType) *extensions.ReplicaSetCondition {
	for _, c := range status.Conditions {
		if c.Type == condType {
			return &c
		}
	}
	return nil
}

// SetCondition adds/replaces the given condition in the replicaset status. If the condition that we
// are about to add already exists and has the same status and reason then we are not going to update.
func SetCondition(status *extensions.ReplicaSetStatus, condition extensions.ReplicaSetCondition) {
	currentCond := GetCondition(*status, condition.Type)
	if currentCond != nil && currentCond.Status == condition.Status && currentCond.Reason == condition.Reason {
		return
	}
	newConditions := filterOutCondition(status.Conditions, condition.Type)
	status.Conditions = append(newConditions, condition)
}

// RemoveCondition removes the condition with the provided type from the replicaset status.
func RemoveCondition(status *extensions.ReplicaSetStatus, condType extensions.ReplicaSetConditionType) {
	status.Conditions = filterOutCondition(status.Conditions, condType)
}

// filterOutCondition returns a new slice of replicaset conditions without conditions with the provided type.
func filterOutCondition(conditions []extensions.ReplicaSetCondition, condType extensions.ReplicaSetConditionType) []extensions.ReplicaSetCondition {
	var newConditions []extensions.ReplicaSetCondition
	for _, c := range conditions {
		if c.Type == condType {
			continue
		}
		newConditions = append(newConditions, c)
	}
	return newConditions
}

// Obtain all RSs affected by pod creation event. The codes should be similar with addPod(),
// but return RSs instead of enqueuing them to controller's workqueue.
func getAffectedRSsByPodCreation(pod *v1.Pod, rsc *ReplicaSetController) []*extensions.ReplicaSet {
	if pod.DeletionTimestamp != nil {
		return getAffectedRSsByPodDeletion(pod, rsc)
	}

	if controllerRef := metav1.GetControllerOf(pod); controllerRef != nil {
		rs := rsc.resolveControllerRef(pod.Namespace, controllerRef)
		if rs == nil {
			return nil
		}
		rsKey, err := controller.KeyFunc(rs)
		if err != nil {
			return nil
		}
		glog.V(4).Infof("Pod %s created: %#v.", pod.Name, pod)
		rsc.expectations.CreationObserved(rsKey)
		return []*extensions.ReplicaSet{rs}
	}

	glog.V(4).Infof("Orphan Pod %s created: %#v.", pod.Name, pod)
	return rsc.getPodReplicaSets(pod)
}

// Obtain all RSs affected by pod update event. The codes should be similar with updatePod(),
// but return RSs instead of enqueuing them to controller's workqueue.
func getAffectedRSsByPodUpdate(oldPod, curPod *v1.Pod, rsc *ReplicaSetController) []*extensions.ReplicaSet {
	var rss []*extensions.ReplicaSet
	if curPod.ResourceVersion == oldPod.ResourceVersion {
		return rss
	}

	labelChanged := !reflect.DeepEqual(curPod.Labels, oldPod.Labels)
	if curPod.DeletionTimestamp != nil {
		rss = getAffectedRSsByPodDeletion(curPod, rsc)
		if labelChanged {
			affectedOldRSs := getAffectedRSsByPodDeletion(oldPod, rsc)
			for _, rs := range affectedOldRSs {
				rss = append(rss, rs)
			}
		}
		return rss
	}

	curControllerRef := metav1.GetControllerOf(curPod)
	oldControllerRef := metav1.GetControllerOf(oldPod)
	controllerRefChanged := !reflect.DeepEqual(curControllerRef, oldControllerRef)
	if controllerRefChanged && oldControllerRef != nil {
		if rs := rsc.resolveControllerRef(oldPod.Namespace, oldControllerRef); rs != nil {
			rss = append(rss, rs)
		}
	}
	if curControllerRef != nil {
		rs := rsc.resolveControllerRef(curPod.Namespace, curControllerRef)
		if rs == nil {
			return rss
		}
		glog.V(4).Infof("Pod %s updated, objectMeta %+v -> %+v.", curPod.Name, oldPod.ObjectMeta, curPod.ObjectMeta)
		rss = append(rss, rs)
		if !podutil.IsPodReady(oldPod) && podutil.IsPodReady(curPod) && rs.Spec.MinReadySeconds > 0 {
			glog.V(2).Infof("ReplicaSet %q will be enqueued after %ds for availability check", rs.Name, rs.Spec.MinReadySeconds)
			// explicitly enqueue rs for second time if its MinReadySeconds > 0
			return append(rss, rs)
		}
		return rss
	}
	if labelChanged || controllerRefChanged {
		matchingRSs := rsc.getPodReplicaSets(curPod)
		if len(matchingRSs) == 0 {
			return rss
		}
		glog.V(4).Infof("Orphan Pod %s updated, objectMeta %+v -> %+v.", curPod.Name, oldPod.ObjectMeta, curPod.ObjectMeta)
		for _, rs := range matchingRSs {
			rss = append(rss, rs)
		}
	}
	return rss
}

// Obtain all RSs affected by pod deletion event. The codes should be similar with deletePod(),
// but return RSs instead of enqueuing them to controller's workqueue.
func getAffectedRSsByPodDeletion(obj interface{}, rsc *ReplicaSetController) []*extensions.ReplicaSet {
	var rss []*extensions.ReplicaSet
	pod, ok := obj.(*v1.Pod)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("couldn't get object from tombstone %+v", obj))
			return rss
		}
		pod, ok = tombstone.Obj.(*v1.Pod)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("tombstone contained object that is not a pod %#v", obj))
			return rss
		}
	}

	controllerRef := metav1.GetControllerOf(pod)
	if controllerRef == nil {
		return rss
	}
	rs := rsc.resolveControllerRef(pod.Namespace, controllerRef)
	if rs == nil {
		return rss
	}
	rsKey, err := controller.KeyFunc(rs)
	if err != nil {
		return rss
	}
	glog.V(4).Infof("Pod %s/%s deleted through %v, timestamp %+v: %#v.", pod.Namespace, pod.Name, utilruntime.GetCaller(), pod.DeletionTimestamp, pod)
	rsc.expectations.DeletionObserved(rsKey, controller.PodKey(pod))
	return append(rss, rs)
}
