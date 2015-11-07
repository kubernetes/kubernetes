/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package replication

import (
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
)

// updateReplicaCount attempts to update the Status.Replicas of the given controller, with a single GET/PUT retry.
func updateReplicaCount(rcClient client.ReplicationControllerInterface, controller api.ReplicationController, numReplicas int) (updateErr error) {
	// This is the steady state. It happens when the rc doesn't have any expectations, since
	// we do a periodic relist every 30s. If the generations differ but the replicas are
	// the same, a caller might've resized to the same replica count.
	if controller.Status.Replicas == numReplicas &&
		controller.Generation == controller.Status.ObservedGeneration {
		return nil
	}
	// Save the generation number we acted on, otherwise we might wrongfully indicate
	// that we've seen a spec update when we retry.
	// TODO: This can clobber an update if we allow multiple agents to write to the
	// same status.
	generation := controller.Generation

	var getErr error
	for i, rc := 0, &controller; ; i++ {
		glog.V(4).Infof("Updating replica count for rc: %v, %d->%d (need %d), sequence No: %v->%v",
			controller.Name, controller.Status.Replicas, numReplicas, controller.Spec.Replicas, controller.Status.ObservedGeneration, generation)

		rc.Status = api.ReplicationControllerStatus{Replicas: numReplicas, ObservedGeneration: generation}
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
type overlappingControllers []api.ReplicationController

func (o overlappingControllers) Len() int      { return len(o) }
func (o overlappingControllers) Swap(i, j int) { o[i], o[j] = o[j], o[i] }

func (o overlappingControllers) Less(i, j int) bool {
	if o[i].CreationTimestamp.Equal(o[j].CreationTimestamp) {
		return o[i].Name < o[j].Name
	}
	return o[i].CreationTimestamp.Before(o[j].CreationTimestamp)
}
