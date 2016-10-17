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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	unversionedextensions "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/extensions/internalversion"
	"k8s.io/kubernetes/pkg/labels"
)

// updateReplicaSetStatus attempts to update the Status.Replicas of the given ReplicaSet, with a single GET/PUT retry.
func updateReplicaSetStatus(c unversionedextensions.ReplicaSetInterface, rs extensions.ReplicaSet, newStatus extensions.ReplicaSetStatus) (updateErr error) {
	// This is the steady state. It happens when the ReplicaSet doesn't have any expectations, since
	// we do a periodic relist every 30s. If the generations differ but the replicas are
	// the same, a caller might've resized to the same replica count.
	if rs.Status.Replicas == newStatus.Replicas &&
		rs.Status.FullyLabeledReplicas == newStatus.FullyLabeledReplicas &&
		rs.Status.ReadyReplicas == newStatus.ReadyReplicas &&
		rs.Status.AvailableReplicas == newStatus.AvailableReplicas &&
		rs.Generation == rs.Status.ObservedGeneration &&
		reflect.DeepEqual(rs.Status.Conditions, newStatus.Conditions) {
		return nil
	}

	// deep copy to avoid mutation now.
	// TODO this method need some work.  Retry on conflict probably, though I suspect this is stomping status to something it probably shouldn't
	copyObj, err := api.Scheme.DeepCopy(rs)
	if err != nil {
		return err
	}
	rs = copyObj.(extensions.ReplicaSet)

	// Save the generation number we acted on, otherwise we might wrongfully indicate
	// that we've seen a spec update when we retry.
	// TODO: This can clobber an update if we allow multiple agents to write to the
	// same status.
	newStatus.ObservedGeneration = rs.Generation

	var getErr error
	for i, rs := 0, &rs; ; i++ {
		glog.V(4).Infof(fmt.Sprintf("Updating replica count for ReplicaSet: %s/%s, ", rs.Namespace, rs.Name) +
			fmt.Sprintf("replicas %d->%d (need %d), ", rs.Status.Replicas, newStatus.Replicas, rs.Spec.Replicas) +
			fmt.Sprintf("fullyLabeledReplicas %d->%d, ", rs.Status.FullyLabeledReplicas, newStatus.FullyLabeledReplicas) +
			fmt.Sprintf("readyReplicas %d->%d, ", rs.Status.ReadyReplicas, newStatus.ReadyReplicas) +
			fmt.Sprintf("availableReplicas %d->%d, ", rs.Status.AvailableReplicas, newStatus.AvailableReplicas) +
			fmt.Sprintf("sequence No: %v->%v", rs.Status.ObservedGeneration, newStatus.ObservedGeneration))

		rs.Status = newStatus
		_, updateErr = c.UpdateStatus(rs)
		if updateErr == nil || i >= statusUpdateRetries {
			return updateErr
		}
		// Update the ReplicaSet with the latest resource version for the next poll
		if rs, getErr = c.Get(rs.Name); getErr != nil {
			// If the GET fails we can't trust status.Replicas anymore. This error
			// is bound to be more interesting than the update failure.
			return getErr
		}
	}
}

// overlappingReplicaSets sorts a list of ReplicaSets by creation timestamp, using their names as a tie breaker.
type overlappingReplicaSets []*extensions.ReplicaSet

func (o overlappingReplicaSets) Len() int      { return len(o) }
func (o overlappingReplicaSets) Swap(i, j int) { o[i], o[j] = o[j], o[i] }

func (o overlappingReplicaSets) Less(i, j int) bool {
	if o[i].CreationTimestamp.Equal(o[j].CreationTimestamp) {
		return o[i].Name < o[j].Name
	}
	return o[i].CreationTimestamp.Before(o[j].CreationTimestamp)
}

func calculateStatus(rs extensions.ReplicaSet, filteredPods []*api.Pod, manageReplicasErr error) extensions.ReplicaSetStatus {
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
		if api.IsPodReady(pod) {
			readyReplicasCount++
			if api.IsPodAvailable(pod, rs.Spec.MinReadySeconds, unversioned.Now()) {
				availableReplicasCount++
			}
		}
	}

	failureCond := GetCondition(rs.Status, extensions.ReplicaSetReplicaFailure)
	if manageReplicasErr != nil && failureCond == nil {
		var reason string
		if diff := len(filteredPods) - int(rs.Spec.Replicas); diff < 0 {
			reason = "FailedCreate"
		} else if diff > 0 {
			reason = "FailedDelete"
		}
		cond := NewReplicaSetCondition(extensions.ReplicaSetReplicaFailure, api.ConditionTrue, reason, manageReplicasErr.Error())
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

// NewReplicaSetCondition creates a new replica set condition.
func NewReplicaSetCondition(condType extensions.ReplicaSetConditionType, status api.ConditionStatus, reason, msg string) extensions.ReplicaSetCondition {
	return extensions.ReplicaSetCondition{
		Type:               condType,
		Status:             status,
		LastTransitionTime: unversioned.Now(),
		Reason:             reason,
		Message:            msg,
	}
}

// GetCondition returns a replica set condition with the provided type if it exists.
func GetCondition(status extensions.ReplicaSetStatus, condType extensions.ReplicaSetConditionType) *extensions.ReplicaSetCondition {
	for i := range status.Conditions {
		c := status.Conditions[i]
		if c.Type == condType {
			return &c
		}
	}
	return nil
}

// SetCondition adds/replaces the given condition in the replica set status. If the condition that we
// are about to add already exists and has the same status and reason then we are not going to update.
func SetCondition(status *extensions.ReplicaSetStatus, condition extensions.ReplicaSetCondition) {
	currentCond := GetCondition(*status, condition.Type)
	if currentCond != nil && currentCond.Status == condition.Status && currentCond.Reason == condition.Reason {
		return
	}
	newConditions := filterOutCondition(status.Conditions, condition.Type)
	status.Conditions = append(newConditions, condition)
}

// RemoveCondition removes the condition with the provided type from the replica set status.
func RemoveCondition(status *extensions.ReplicaSetStatus, condType extensions.ReplicaSetConditionType) {
	status.Conditions = filterOutCondition(status.Conditions, condType)
}

// filterOutCondition returns a new slice of replica set conditions without conditions with the provided type.
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
