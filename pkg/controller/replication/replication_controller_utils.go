/*
Copyright 2015 The Kubernetes Authors.

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

// If you make changes to this file, you should also make the corresponding change in ReplicaSet.

package replication

import (
	"fmt"
	"reflect"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	unversionedcore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	"k8s.io/kubernetes/pkg/labels"
)

// updateReplicationControllerStatus attempts to update the Status.Replicas of the given controller, with a single GET/PUT retry.
func updateReplicationControllerStatus(c unversionedcore.ReplicationControllerInterface, rc api.ReplicationController, newStatus api.ReplicationControllerStatus) (updateErr error) {
	// This is the steady state. It happens when the rc doesn't have any expectations, since
	// we do a periodic relist every 30s. If the generations differ but the replicas are
	// the same, a caller might've resized to the same replica count.
	if rc.Status.Replicas == newStatus.Replicas &&
		rc.Status.FullyLabeledReplicas == newStatus.FullyLabeledReplicas &&
		rc.Status.ReadyReplicas == newStatus.ReadyReplicas &&
		rc.Status.AvailableReplicas == newStatus.AvailableReplicas &&
		rc.Generation == rc.Status.ObservedGeneration &&
		reflect.DeepEqual(rc.Status.Conditions, newStatus.Conditions) {
		return nil
	}
	// Save the generation number we acted on, otherwise we might wrongfully indicate
	// that we've seen a spec update when we retry.
	// TODO: This can clobber an update if we allow multiple agents to write to the
	// same status.
	newStatus.ObservedGeneration = rc.Generation

	var getErr error
	for i, rc := 0, &rc; ; i++ {
		glog.V(4).Infof(fmt.Sprintf("Updating replica count for rc: %s/%s, ", rc.Namespace, rc.Name) +
			fmt.Sprintf("replicas %d->%d (need %d), ", rc.Status.Replicas, newStatus.Replicas, rc.Spec.Replicas) +
			fmt.Sprintf("fullyLabeledReplicas %d->%d, ", rc.Status.FullyLabeledReplicas, newStatus.FullyLabeledReplicas) +
			fmt.Sprintf("readyReplicas %d->%d, ", rc.Status.ReadyReplicas, newStatus.ReadyReplicas) +
			fmt.Sprintf("availableReplicas %d->%d, ", rc.Status.AvailableReplicas, newStatus.AvailableReplicas) +
			fmt.Sprintf("sequence No: %v->%v", rc.Status.ObservedGeneration, newStatus.ObservedGeneration))

		rc.Status = newStatus
		_, updateErr = c.UpdateStatus(rc)
		if updateErr == nil || i >= statusUpdateRetries {
			return updateErr
		}
		// Update the controller with the latest resource version for the next poll
		if rc, getErr = c.Get(rc.Name); getErr != nil {
			// If the GET fails we can't trust status.Replicas anymore. This error
			// is bound to be more interesting than the update failure.
			return getErr
		}
	}
}

// OverlappingControllers sorts a list of controllers by creation timestamp, using their names as a tie breaker.
type OverlappingControllers []*api.ReplicationController

func (o OverlappingControllers) Len() int      { return len(o) }
func (o OverlappingControllers) Swap(i, j int) { o[i], o[j] = o[j], o[i] }

func (o OverlappingControllers) Less(i, j int) bool {
	if o[i].CreationTimestamp.Equal(o[j].CreationTimestamp) {
		return o[i].Name < o[j].Name
	}
	return o[i].CreationTimestamp.Before(o[j].CreationTimestamp)
}

func calculateStatus(rc api.ReplicationController, filteredPods []*api.Pod, manageReplicasErr error) api.ReplicationControllerStatus {
	newStatus := rc.Status
	// Count the number of pods that have labels matching the labels of the pod
	// template of the replication controller, the matching pods may have more
	// labels than are in the template. Because the label of podTemplateSpec is
	// a superset of the selector of the replication controller, so the possible
	// matching pods must be part of the filteredPods.
	fullyLabeledReplicasCount := 0
	readyReplicasCount := 0
	availableReplicasCount := 0
	templateLabel := labels.Set(rc.Spec.Template.Labels).AsSelectorPreValidated()
	for _, pod := range filteredPods {
		if templateLabel.Matches(labels.Set(pod.Labels)) {
			fullyLabeledReplicasCount++
		}
		if api.IsPodReady(pod) {
			readyReplicasCount++
			if api.IsPodAvailable(pod, rc.Spec.MinReadySeconds, unversioned.Now()) {
				availableReplicasCount++
			}
		}
	}

	failureCond := GetCondition(rc.Status, api.ReplicationControllerReplicaFailure)
	if manageReplicasErr != nil && failureCond == nil {
		var reason string
		if diff := len(filteredPods) - int(rc.Spec.Replicas); diff < 0 {
			reason = "FailedCreate"
		} else if diff > 0 {
			reason = "FailedDelete"
		}
		cond := NewReplicationControllerCondition(api.ReplicationControllerReplicaFailure, api.ConditionTrue, reason, manageReplicasErr.Error())
		SetCondition(&newStatus, cond)
	} else if manageReplicasErr == nil && failureCond != nil {
		RemoveCondition(&newStatus, api.ReplicationControllerReplicaFailure)
	}

	newStatus.Replicas = int32(len(filteredPods))
	newStatus.FullyLabeledReplicas = int32(fullyLabeledReplicasCount)
	newStatus.ReadyReplicas = int32(readyReplicasCount)
	newStatus.AvailableReplicas = int32(availableReplicasCount)
	return newStatus
}

// NewReplicationControllerCondition creates a new replication controller condition.
func NewReplicationControllerCondition(condType api.ReplicationControllerConditionType, status api.ConditionStatus, reason, msg string) api.ReplicationControllerCondition {
	return api.ReplicationControllerCondition{
		Type:               condType,
		Status:             status,
		LastTransitionTime: unversioned.Now(),
		Reason:             reason,
		Message:            msg,
	}
}

// GetCondition returns a replication controller condition with the provided type if it exists.
func GetCondition(status api.ReplicationControllerStatus, condType api.ReplicationControllerConditionType) *api.ReplicationControllerCondition {
	for i := range status.Conditions {
		c := status.Conditions[i]
		if c.Type == condType {
			return &c
		}
	}
	return nil
}

// SetCondition adds/replaces the given condition in the replication controller status.
func SetCondition(status *api.ReplicationControllerStatus, condition api.ReplicationControllerCondition) {
	currentCond := GetCondition(*status, condition.Type)
	if currentCond != nil && currentCond.Status == condition.Status && currentCond.Reason == condition.Reason {
		return
	}
	newConditions := filterOutCondition(status.Conditions, condition.Type)
	status.Conditions = append(newConditions, condition)
}

// RemoveCondition removes the condition with the provided type from the replication controller status.
func RemoveCondition(status *api.ReplicationControllerStatus, condType api.ReplicationControllerConditionType) {
	status.Conditions = filterOutCondition(status.Conditions, condType)
}

// filterOutCondition returns a new slice of replication controller conditions without conditions with the provided type.
func filterOutCondition(conditions []api.ReplicationControllerCondition, condType api.ReplicationControllerConditionType) []api.ReplicationControllerCondition {
	var newConditions []api.ReplicationControllerCondition
	for _, c := range conditions {
		if c.Type == condType {
			continue
		}
		newConditions = append(newConditions, c)
	}
	return newConditions
}
