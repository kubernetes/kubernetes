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
	"reflect"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/controller/deployment/util"
)

// hasFailed determines if a deployment has timed out or not by estimating its progress.
// Progress for a deployment is considered when a new replica set is created or adopted,
// and when new pods scale up or old pods scale down. Progress is not estimated for paused
// deployments or when users don't really care about it ie. progressDeadlineSeconds is not
// specified.
func (dc *DeploymentController) hasFailed(d *extensions.Deployment) (bool, error) {
	if d.Spec.ProgressDeadlineSeconds == nil || d.Spec.RollbackTo != nil || d.Spec.Paused {
		return false, nil
	}

	newRS, oldRSs, err := dc.getAllReplicaSetsAndSyncRevision(d, false)
	if err != nil {
		return false, err
	}

	// There is a template change so we don't need to check for any progress right now.
	if newRS == nil {
		return false, nil
	}

	// Look at the status of the deployment; if there is already a NewRSAvailableReason
	// then we don't need to estimate any progress. This is needed in order to avoid
	// estimating progress for scaling events after a rollout has finished.
	cond := util.GetDeploymentCondition(d.Status, extensions.DeploymentProgressing)
	if cond != nil && cond.Reason == util.NewRSAvailableReason {
		return false, nil
	}

	allRSs := append(oldRSs, newRS)
	newStatus := dc.calculateStatus(allRSs, newRS, d)

	// If the deployment is complete or is progressing, there is no need to check if it
	// has timed out.
	if util.IsDeploymentComplete(d, &newStatus) || util.IsDeploymentProgressing(d, &newStatus) {
		return false, nil
	}

	return util.IsDeploymentFailed(d, &newStatus), nil
}

// syncRolloutStatus updates the status of a deployment during a rollout or
// when it has failed progressing.
func (dc *DeploymentController) syncRolloutStatus(allRSs []*extensions.ReplicaSet, newRS *extensions.ReplicaSet, d *extensions.Deployment) error {
	newStatus := dc.calculateStatus(allRSs, newRS, d)

	if d.Spec.ProgressDeadlineSeconds != nil {
		switch {
		case util.IsDeploymentComplete(d, &newStatus):
			// Update the deployment conditions with a message for the new replica set that
			// was successfully deployed. If the condition already exists, ignore this update.
			// Cleanup any condition that reports lack of progress.
			msg := fmt.Sprintf("Replica set %q has successfully progressed.", newRS.Name)
			condition := util.NewDeploymentCondition(extensions.DeploymentProgressing, api.ConditionTrue, util.NewRSAvailableReason, msg)
			util.SetDeploymentCondition(&newStatus, *condition)

			// Cleanup conditions that denote any failures since it may be confusing for users.
			util.RemoveDeploymentCondition(&newStatus, extensions.DeploymentReplicaFailure)

		case util.IsDeploymentProgressing(d, &newStatus):
			// If there is any progress made, continue by not checking if the deployment failed. This
			// behavior emulates the rolling updater progressDeadline check.

			msg := fmt.Sprintf("Replica set %q is progressing.", newRS.Name)
			condition := util.NewDeploymentCondition(extensions.DeploymentProgressing, api.ConditionTrue, util.ReplicaSetUpdatedReason, msg)

			// Update the previous Progressing condition or add a new one if it doesn't exist.
			// If a Progressing condition with status=true already exists, we should update
			// everything but lastTransitionTime. SetDeploymentCondition already does that but
			// it also is not updating conditions when the reason of the new condition is the
			// same as the old. The Progressing condition is a special case because we want to
			// update with the same reason and change just lastUpdateTime iff we notice any
			// progress since then. That's why we handle it here.
			oldCond := util.GetDeploymentCondition(newStatus, extensions.DeploymentProgressing)
			if oldCond != nil {
				if oldCond.Status == api.ConditionTrue {
					condition.LastTransitionTime = oldCond.LastTransitionTime
				}
				util.RemoveDeploymentCondition(&newStatus, extensions.DeploymentProgressing)
			}
			util.SetDeploymentCondition(&newStatus, *condition)

		case util.IsDeploymentFailed(d, &newStatus):
			// Update the deployment with a timeout condition. If the condition already exists,
			// ignore this update.
			msg := fmt.Sprintf("Replica set %q has timed out progressing.", newRS.Name)
			condition := util.NewDeploymentCondition(extensions.DeploymentProgressing, api.ConditionFalse, util.TimedOutReason, msg)
			util.SetDeploymentCondition(&newStatus, *condition)
		}
	}

	// Move warning events of the replica set in deployment conditions. Let's not display
	// these warnings once a deployment completes, otherwise it may be confusing for users.
	if !util.IsDeploymentComplete(d, &newStatus) {
		replicaFailureCond := dc.getReplicaFailures(newRS)
		if len(replicaFailureCond) > 0 {
			// There will be only one ReplicaFailure condition on the replica set.
			util.SetDeploymentCondition(&newStatus, replicaFailureCond[0])
		}
	}

	// Do not update if there is nothing new to add.
	if reflect.DeepEqual(d.Status, newStatus) {
		return nil
	}

	newDeployment := d
	newDeployment.Status = newStatus
	_, err := dc.client.Extensions().Deployments(newDeployment.Namespace).UpdateStatus(newDeployment)
	return err
}

// getReplicaFailures will convert replica failure conditions from replica set
// to deployment conditions.
func (dc *DeploymentController) getReplicaFailures(rs *extensions.ReplicaSet) []extensions.DeploymentCondition {
	var conditions []extensions.DeploymentCondition
	for _, c := range rs.Status.Conditions {
		if c.Type != extensions.ReplicaSetReplicaFailure {
			continue
		}
		conditions = append(conditions, util.ReplicaSetToDeploymentCondition(c))
	}
	return conditions
}
