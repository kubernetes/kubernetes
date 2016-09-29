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
	unversionedcore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/unversioned"
	"k8s.io/kubernetes/pkg/labels"
)

// updateReplicationControllerStatus attempts to update the Status.Replicas of the given controller, with a single GET/PUT retry.
func updateReplicationControllerStatus(
	rcClient unversionedcore.ReplicationControllerInterface,
	controller api.ReplicationController,
	newStatus api.ReplicationControllerStatus,
) (updateErr error) {
	// This is the steady state. It happens when the rc doesn't have any expectations, since
	// we do a periodic relist every 30s. If the generations differ but the replicas are
	// the same, a caller might've resized to the same replica count.
	if controller.Status.Replicas == newStatus.Replicas &&
		controller.Status.FullyLabeledReplicas == newStatus.FullyLabeledReplicas &&
		controller.Status.ReadyReplicas == newStatus.ReadyReplicas &&
		controller.Status.AvailableReplicas == newStatus.AvailableReplicas &&
		controller.Generation == controller.Status.ObservedGeneration &&
		reflect.DeepEqual(controller.Status.Conditions, newStatus.Conditions) {
		return nil
	}
	// Save the generation number we acted on, otherwise we might wrongfully indicate
	// that we've seen a spec update when we retry.
	// TODO: This can clobber an update if we allow multiple agents to write to the
	// same status.
	newStatus.ObservedGeneration = controller.Generation

	var getErr error
	for i, rc := 0, &controller; ; i++ {
		glog.V(4).Infof(fmt.Sprintf("Updating replica count for rc: %s/%s, ", controller.Namespace, controller.Name) +
			fmt.Sprintf("replicas %d->%d (need %d), ", controller.Status.Replicas, newStatus.Replicas, controller.Spec.Replicas) +
			fmt.Sprintf("fullyLabeledReplicas %d->%d, ", controller.Status.FullyLabeledReplicas, newStatus.FullyLabeledReplicas) +
			fmt.Sprintf("readyReplicas %d->%d, ", controller.Status.ReadyReplicas, newStatus.ReadyReplicas) +
			fmt.Sprintf("availableReplicas %d->%d, ", controller.Status.AvailableReplicas, newStatus.AvailableReplicas) +
			fmt.Sprintf("sequence No: %v->%v", controller.Status.ObservedGeneration, newStatus.ObservedGeneration))

		rc.Status = newStatus
		_, updateErr = rcClient.UpdateStatus(rc)
		if updateErr == nil || i >= statusUpdateRetries {
			return updateErr
		}
		// Update the controller with the latest resource version for the next poll
		if rc, getErr = rcClient.Get(controller.Name); getErr != nil {
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

	if manageReplicasErr != nil {
		var reason string
		if diff := len(filteredPods) - int(rc.Spec.Replicas); diff < 0 {
			reason = "FailedCreate"
		} else if diff > 0 {
			reason = "FailedDelete"
		}
		cond := NewReplicationControllerCondition(api.ReplicationControllerReplicaFailure, api.ConditionTrue, reason, manageReplicasErr.Error())
		SetCondition(&newStatus, cond)
	} else {
		RemoveCondition(&newStatus, api.ReplicationControllerReplicaFailure)
	}

	newStatus.Replicas = int32(len(filteredPods))
	newStatus.FullyLabeledReplicas = int32(fullyLabeledReplicasCount)
	newStatus.ReadyReplicas = int32(readyReplicasCount)
	newStatus.AvailableReplicas = int32(availableReplicasCount)
	return newStatus
}

func NewReplicationControllerCondition(condType api.ReplicationControllerConditionType, status api.ConditionStatus, reason, msg string) api.ReplicationControllerCondition {
	return api.ReplicationControllerCondition{
		Type:               condType,
		Status:             status,
		LastTransitionTime: unversioned.Now(),
		Reason:             reason,
		Message:            msg,
	}
}

func SetCondition(status *api.ReplicationControllerStatus, cond api.ReplicationControllerCondition) {
	newConditions := filterOutCondition(status.Conditions, cond.Type)
	status.Conditions = append(newConditions, cond)
}

func RemoveCondition(status *api.ReplicationControllerStatus, condType api.ReplicationControllerConditionType) {
	status.Conditions = filterOutCondition(status.Conditions, condType)
}

// filterOutCondition returns a new slice of deployment conditions without conditions with the provided type.
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
