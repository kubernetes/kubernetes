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
	"context"
	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/deployment/util"
)

// rolloutRecreate implements the logic for recreating a replica set.
func (dc *DeploymentController) rolloutRecreate(ctx context.Context, d *apps.Deployment, rsList []*apps.ReplicaSet, podMap map[types.UID][]*v1.Pod) error {
	// Don't create a new RS if not already existed, so that we avoid scaling up before scaling down.
	newRS, oldRSs, err := dc.getAllReplicaSetsAndSyncRevision(ctx, d, rsList, false)
	if err != nil {
		return err
	}
	allRSs := append(oldRSs, newRS)
	activeOldRSs := controller.FilterActiveReplicaSets(oldRSs)

	// scale down old replica sets.
	scaledDown, err := dc.scaleDownOldReplicaSetsForRecreate(ctx, activeOldRSs, d)
	if err != nil {
		return err
	}
	if scaledDown {
		// Update DeploymentStatus.
		return dc.syncRolloutStatus(ctx, allRSs, newRS, d)
	}

	// Do not process a deployment when it has old pods running.
	if oldPodsRunning(newRS, oldRSs, podMap) {
		return dc.syncRolloutStatus(ctx, allRSs, newRS, d)
	}

	// If we need to create a new RS, create it now.
	if newRS == nil {
		newRS, oldRSs, err = dc.getAllReplicaSetsAndSyncRevision(ctx, d, rsList, true)
		if err != nil {
			return err
		}
		allRSs = append(oldRSs, newRS)
	}

	// scale up new replica set.
	if _, err := dc.scaleUpNewReplicaSetForRecreate(ctx, newRS, d); err != nil {
		return err
	}

	if util.DeploymentComplete(d, &d.Status) {
		if err := dc.cleanupDeployment(ctx, oldRSs, d); err != nil {
			return err
		}
	}

	// Sync deployment status.
	return dc.syncRolloutStatus(ctx, allRSs, newRS, d)
}

// scaleDownOldReplicaSetsForRecreate scales down old replica sets when deployment strategy is "Recreate".
func (dc *DeploymentController) scaleDownOldReplicaSetsForRecreate(ctx context.Context, oldRSs []*apps.ReplicaSet, deployment *apps.Deployment) (bool, error) {
	scaled := false
	for i := range oldRSs {
		rs := oldRSs[i]
		// Scaling not required.
		if *(rs.Spec.Replicas) == 0 {
			continue
		}
		scaledRS, updatedRS, err := dc.scaleReplicaSetAndRecordEvent(ctx, rs, 0, deployment)
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

// oldPodsRunning returns whether there are old pods running or any of the old ReplicaSets thinks that it runs pods.
func oldPodsRunning(newRS *apps.ReplicaSet, oldRSs []*apps.ReplicaSet, podMap map[types.UID][]*v1.Pod) bool {
	if oldPods := util.GetActualReplicaCountForReplicaSets(oldRSs); oldPods > 0 {
		return true
	}
	for rsUID, podList := range podMap {
		// If the pods belong to the new ReplicaSet, ignore.
		if newRS != nil && newRS.UID == rsUID {
			continue
		}
		for _, pod := range podList {
			switch pod.Status.Phase {
			case v1.PodFailed, v1.PodSucceeded:
				// Don't count pods in terminal state.
				continue
			case v1.PodUnknown:
				// v1.PodUnknown is a deprecated status.
				// This logic is kept for backward compatibility.
				// This used to happen in situation like when the node is temporarily disconnected from the cluster.
				// If we can't be sure that the pod is not running, we have to count it.
				return true
			default:
				// Pod is not in terminal phase.
				return true
			}
		}
	}
	return false
}

// scaleUpNewReplicaSetForRecreate scales up new replica set when deployment strategy is "Recreate".
func (dc *DeploymentController) scaleUpNewReplicaSetForRecreate(ctx context.Context, newRS *apps.ReplicaSet, deployment *apps.Deployment) (bool, error) {
	scaled, _, err := dc.scaleReplicaSetAndRecordEvent(ctx, newRS, *(deployment.Spec.Replicas), deployment)
	return scaled, err
}
