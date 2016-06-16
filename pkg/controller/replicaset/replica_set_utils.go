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
	"strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/apis/extensions"
	client "k8s.io/kubernetes/pkg/client/unversioned"
)

// updateReplicaCount attempts to update the Status.Replicas of the given ReplicaSet, with a single GET/PUT retry.
func updateReplicaCount(rsClient client.ReplicaSetInterface, rs extensions.ReplicaSet, numReplicas, numFullyLabeledReplicas int) (updateErr error) {
	// This is the steady state. It happens when the ReplicaSet doesn't have any expectations, since
	// we do a periodic relist every 30s. If the generations differ but the replicas are
	// the same, a caller might've resized to the same replica count.
	if int(rs.Status.Replicas) == numReplicas &&
		int(rs.Status.FullyLabeledReplicas) == numFullyLabeledReplicas &&
		rs.Generation == rs.Status.ObservedGeneration {
		return nil
	}
	// Save the generation number we acted on, otherwise we might wrongfully indicate
	// that we've seen a spec update when we retry.
	// TODO: This can clobber an update if we allow multiple agents to write to the
	// same status.
	generation := rs.Generation

	var getErr error
	for i, rs := 0, &rs; ; i++ {
		glog.V(4).Infof(fmt.Sprintf("Updating replica count for ReplicaSet: %s/%s, ", rs.Namespace, rs.Name) +
			fmt.Sprintf("replicas %d->%d (need %d), ", rs.Status.Replicas, numReplicas, rs.Spec.Replicas) +
			fmt.Sprintf("fullyLabeledReplicas %d->%d, ", rs.Status.FullyLabeledReplicas, numFullyLabeledReplicas) +
			fmt.Sprintf("sequence No: %v->%v", rs.Status.ObservedGeneration, generation))

		rs.Status = extensions.ReplicaSetStatus{Replicas: int32(numReplicas), FullyLabeledReplicas: int32(numFullyLabeledReplicas), ObservedGeneration: generation}
		_, updateErr = rsClient.UpdateStatus(rs)
		if updateErr == nil || i >= statusUpdateRetries {
			return updateErr
		}
		// Update the ReplicaSet with the latest resource version for the next poll
		if rs, getErr = rsClient.Get(rs.Name); getErr != nil {
			// If the GET fails we can't trust status.Replicas anymore. This error
			// is bound to be more interesting than the update failure.
			return getErr
		}
	}
}

// overlappingReplicaSets sorts a list of ReplicaSets by creation timestamp, using their names as a tie breaker.
type overlappingReplicaSets []extensions.ReplicaSet

func (o overlappingReplicaSets) Len() int      { return len(o) }
func (o overlappingReplicaSets) Swap(i, j int) { o[i], o[j] = o[j], o[i] }

func (o overlappingReplicaSets) Less(i, j int) bool {
	if o[i].CreationTimestamp.Equal(o[j].CreationTimestamp) {
		return o[i].Name < o[j].Name
	}
	return o[i].CreationTimestamp.Before(o[j].CreationTimestamp)
}

// ReplicaSetsByCreationTimestamp sorts a list of ReplicationSets by creation timestamp, using their names as a tie breaker.
type ReplicaSetsByCreationTimestamp []*extensions.ReplicaSet

func (o ReplicaSetsByCreationTimestamp) Len() int      { return len(o) }
func (o ReplicaSetsByCreationTimestamp) Swap(i, j int) { o[i], o[j] = o[j], o[i] }

func (o ReplicaSetsByCreationTimestamp) Less(i, j int) bool {
	if o[i].CreationTimestamp.Equal(o[j].CreationTimestamp) {
		return o[i].Name < o[j].Name
	}
	return o[i].CreationTimestamp.Before(o[j].CreationTimestamp)
}

// ReplicaSetsBySizeOlder sorts a list of ReplicaSet by size in descending order, using their creation timestamp or name as a tie breaker.
// By using the creation timestamp, this sorts from old to new replica sets.
type ReplicaSetsBySizeOlder []*extensions.ReplicaSet

func (o ReplicaSetsBySizeOlder) Len() int      { return len(o) }
func (o ReplicaSetsBySizeOlder) Swap(i, j int) { o[i], o[j] = o[j], o[i] }
func (o ReplicaSetsBySizeOlder) Less(i, j int) bool {
	if o[i].Spec.Replicas == o[j].Spec.Replicas {
		return ReplicaSetsByCreationTimestamp(o).Less(i, j)
	}
	return o[i].Spec.Replicas > o[j].Spec.Replicas
}

// ReplicaSetsBySizeNewer sorts a list of ReplicaSet by size in descending order, using their creation timestamp or name as a tie breaker.
// By using the creation timestamp, this sorts from new to old replica sets.
type ReplicaSetsBySizeNewer []*extensions.ReplicaSet

func (o ReplicaSetsBySizeNewer) Len() int      { return len(o) }
func (o ReplicaSetsBySizeNewer) Swap(i, j int) { o[i], o[j] = o[j], o[i] }
func (o ReplicaSetsBySizeNewer) Less(i, j int) bool {
	if o[i].Spec.Replicas == o[j].Spec.Replicas {
		return ReplicaSetsByCreationTimestamp(o).Less(j, i)
	}
	return o[i].Spec.Replicas > o[j].Spec.Replicas
}

// ReplicaSetsByActiveness implements sort.Interface for []extensions.ReplicaSet
// The primary key is whether the spec has active (>0) replicas, while age is
// the secondary.
type ReplicaSetsByActiveness []extensions.ReplicaSet

func (list ReplicaSetsByActiveness) Len() int {
	fmt.Println(len(list))
	return len(list)
}

func (list ReplicaSetsByActiveness) Swap(i, j int) {
	list[i], list[j] = list[j], list[i]
}

func (list ReplicaSetsByActiveness) Less(i, j int) bool {
	iElem := list[i]
	jElem := list[j]
	iIsActive := iElem.Spec.Replicas > 0
	jIsActive := jElem.Spec.Replicas > 0
	if iIsActive == jIsActive {
		// break the tie
		iName := strings.TrimSuffix(iElem.Namespace+"/"+iElem.Name,
			iElem.Labels["pod-template-hash"])
		jName := strings.TrimSuffix(jElem.Namespace+"/"+jElem.Name,
			jElem.Labels["pod-template-hash"])
		if iName == jName {
			return jElem.CreationTimestamp.Time.Before(iElem.CreationTimestamp.Time)
		}
		return iName < jName
	}
	return iIsActive
}
