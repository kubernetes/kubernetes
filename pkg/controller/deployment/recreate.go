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

package deployment

import (
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/retry"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/watch"
)

// rolloutRecreate implements the logic for recreating a replica set.
func (dc *DeploymentController) rolloutRecreate(deployment *extensions.Deployment) error {
	// Don't create a new RS if not already existed, so that we avoid scaling up before scaling down
	newRS, oldRSs, err := dc.getAllReplicaSetsAndSyncRevision(deployment, false)
	if err != nil {
		return err
	}
	allRSs := append(oldRSs, newRS)
	activeOldRSs := controller.FilterActiveReplicaSets(oldRSs)

	// scale down old replica sets
	scaledDown, err := dc.scaleDownOldReplicaSetsForRecreate(activeOldRSs, deployment)
	if err != nil {
		return err
	}
	if scaledDown {
		// Update DeploymentStatus
		return dc.syncRolloutStatus(allRSs, newRS, deployment)
	}

	// If we need to create a new RS, create it now
	// TODO: Create a new RS without re-listing all RSs.
	if newRS == nil {
		// Wait for all old replica set to scale down to zero.
		if err := dc.waitForInactiveReplicaSets(activeOldRSs); err != nil {
			return err
		}
		// Wait for all pods to be deleted.
		if err := dc.waitForNoPods(deployment); err != nil {
			return err
		}

		newRS, oldRSs, err = dc.getAllReplicaSetsAndSyncRevision(deployment, true)
		if err != nil {
			return err
		}
		allRSs = append(oldRSs, newRS)
	}

	// scale up new replica set
	scaledUp, err := dc.scaleUpNewReplicaSetForRecreate(newRS, deployment)
	if err != nil {
		return err
	}
	if scaledUp {
		// Update DeploymentStatus
		return dc.syncRolloutStatus(allRSs, newRS, deployment)
	}

	dc.cleanupDeployment(oldRSs, deployment)

	// Sync deployment status
	return dc.syncRolloutStatus(allRSs, newRS, deployment)
}

// scaleDownOldReplicaSetsForRecreate scales down old replica sets when deployment strategy is "Recreate"
func (dc *DeploymentController) scaleDownOldReplicaSetsForRecreate(oldRSs []*extensions.ReplicaSet, deployment *extensions.Deployment) (bool, error) {
	scaled := false
	for i := range oldRSs {
		rs := oldRSs[i]
		// Scaling not required.
		if rs.Spec.Replicas == 0 {
			continue
		}
		scaledRS, updatedRS, err := dc.scaleReplicaSetAndRecordEvent(rs, 0, deployment)
		if err != nil {
			return false, err
		}
		if scaledRS {
			oldRSs[i] = updatedRS
			scaled = true
		}
	}
	return scaled, nil
}

// waitForInactiveReplicaSets will wait until all passed replica sets are inactive and have been noticed
// by the replica set controller.
func (dc *DeploymentController) waitForInactiveReplicaSets(oldRSs []*extensions.ReplicaSet) error {
	for i := range oldRSs {
		rs := oldRSs[i]
		desiredGeneration := rs.Generation
		observedGeneration := rs.Status.ObservedGeneration
		specReplicas := rs.Spec.Replicas
		statusReplicas := rs.Status.Replicas

		if err := wait.ExponentialBackoff(retry.DefaultRetry, func() (bool, error) {
			replicaSet, err := dc.rsLister.ReplicaSets(rs.Namespace).Get(rs.Name)
			if err != nil {
				return false, err
			}

			specReplicas = replicaSet.Spec.Replicas
			statusReplicas = replicaSet.Status.Replicas
			observedGeneration = replicaSet.Status.ObservedGeneration

			// TODO: We also need to wait for terminating replicas to actually terminate.
			// See https://github.com/kubernetes/kubernetes/issues/32567
			return observedGeneration >= desiredGeneration && replicaSet.Spec.Replicas == 0 && replicaSet.Status.Replicas == 0, nil
		}); err != nil {
			if err == wait.ErrWaitTimeout {
				err = fmt.Errorf("replica set %q never became inactive: synced=%t, spec.replicas=%d, status.replicas=%d",
					rs.Name, observedGeneration >= desiredGeneration, specReplicas, statusReplicas)
			}
			return err
		}
	}
	return nil
}

// waitForNoPods will wait until all pods for the provided deployment are deleted.
func (dc *DeploymentController) waitForNoPods(deployment *extensions.Deployment) error {
	selector, err := unversioned.LabelSelectorAsSelector(deployment.Spec.Selector)
	if err != nil {
		return err
	}
	pods, err := dc.podLister.Pods(deployment.Namespace).List(selector)
	if err != nil {
		return err
	}
	if len(pods) == 0 {
		return nil
	}
	options := api.ListOptions{LabelSelector: selector, ResourceVersion: pods[0].ResourceVersion}
	w, err := dc.client.Core().Pods(deployment.Namespace).Watch(options)
	if err != nil {
		return err
	}
	defer w.Stop()

	deletionsNeeded := len(pods)
	condition := func(event watch.Event) (bool, error) {
		if event.Type == watch.Deleted {
			deletionsNeeded--
		}
		return deletionsNeeded == 0, nil
	}
	// TODO: Wait some time proportionate to the size of deletionsNeeded.
	_, err = watch.Until(2*time.Minute, w, condition)
	return err
}

// scaleUpNewReplicaSetForRecreate scales up new replica set when deployment strategy is "Recreate"
func (dc *DeploymentController) scaleUpNewReplicaSetForRecreate(newRS *extensions.ReplicaSet, deployment *extensions.Deployment) (bool, error) {
	scaled, _, err := dc.scaleReplicaSetAndRecordEvent(newRS, deployment.Spec.Replicas, deployment)
	return scaled, err
}
