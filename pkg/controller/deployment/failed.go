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
	"sort"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/events"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/controller/deployment/util"
)

// hasTimedOut determines if a deployment has timed out or not by estimating its progress.
// Progress for a deployment is considered when a new replica set is created or adopted,
// and when new pods scale up or old pods scale down. Progress is not estimated for paused
// deployments, or when the deployment needs to rollback, or when users don't really care
// about it ie. progressDeadlineSeconds is not specified.
func (dc *DeploymentController) hasTimedOut(d *extensions.Deployment) (bool, error) {
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
	newStatus, err := dc.calculateStatus(allRSs, newRS, d)
	if err != nil {
		return false, err
	}

	if util.IsDeploymentComplete(d, &newStatus) || util.IsDeploymentProgressing(d, &newStatus) {
		return false, nil
	}

	return util.IsDeploymentFailed(d, &newStatus), nil
}

func (dc *DeploymentController) syncFailed(deployment *extensions.Deployment) error {
	newRS, oldRSs, err := dc.getAllReplicaSetsAndSyncRevision(deployment, false)
	if err != nil {
		return err
	}

	allRSs := append(oldRSs, newRS)
	return dc.syncRolloutStatus(allRSs, newRS, deployment)
}

// syncRolloutStatus updates the status of a deployment during a rollout or
// when it has failed progressing.
func (dc *DeploymentController) syncRolloutStatus(allRSs []*extensions.ReplicaSet, newRS *extensions.ReplicaSet, d *extensions.Deployment) error {
	newStatus, err := dc.calculateStatus(allRSs, newRS, d)
	if err != nil {
		return err
	}

	if d.Spec.ProgressDeadlineSeconds != nil {
		switch {
		case util.IsDeploymentComplete(d, &newStatus):
			// Update the deployment conditions with a message for the new replica set that
			// was successfully deployed. If the condition already exists, ignore this update.
			// Cleanup any condition that reports lack of progress.
			cond := util.GetDeploymentCondition(d.Status, extensions.DeploymentProgressing)
			if cond == nil || cond.Reason != util.NewRSAvailableReason {
				msg := fmt.Sprintf("Replica set %q has successfully progressed.", newRS.Name)
				condition := util.NewDeploymentCondition(extensions.DeploymentProgressing, api.ConditionTrue, util.NewRSAvailableReason, msg)
				util.SetDeploymentCondition(&newStatus, *condition)
			}
			// Cleanup conditions that denote any failures since it may be confusing for users.
			util.RemoveDeploymentCondition(&newStatus, extensions.DeploymentReplicaFailure)

		case util.IsDeploymentProgressing(d, &newStatus):
			// If there is any progress made, continue by not checking if the deployment failed. This
			// behavior emulates the rolling updater progressDeadline check.
			msg := fmt.Sprintf("Replica set %q is progressing.", newRS.Name)
			condition := util.NewDeploymentCondition(extensions.DeploymentProgressing, api.ConditionTrue, util.ReplicaSetUpdatedReason, msg)
			util.SetDeploymentCondition(&newStatus, *condition)

		case util.IsDeploymentFailed(d, &newStatus):
			// Update the deployment with a timeout condition. If the condition already exists,
			// ignore this update.
			cond := util.GetDeploymentCondition(d.Status, extensions.DeploymentProgressing)
			if cond == nil || cond.Reason != util.TimedOutReason {
				msg := fmt.Sprintf("Replica set %q has timed out progressing.", newRS.Name)
				condition := util.NewDeploymentCondition(extensions.DeploymentProgressing, api.ConditionFalse, util.TimedOutReason, msg)
				util.SetDeploymentCondition(&newStatus, *condition)
			}
		}
	}

	// Move warning events of the replica set in deployment conditions. Let's not display
	// these warnings once a deployment completes, otherwise it may be confusing for users.
	if !util.IsDeploymentComplete(d, &newStatus) {
		if warnings := dc.getReplicaFailures(newRS); len(warnings) > 0 {
			// Add the oldest found warning - we can only accomodate one instanse
			// of a Condition type in the deployment status.
			util.SetDeploymentCondition(&newStatus, warnings[0])
		}
	}

	if reflect.DeepEqual(d.Status, newStatus) {
		return nil
	}

	newDeployment := d
	newDeployment.Status = newStatus
	_, err = dc.client.Extensions().Deployments(newDeployment.Namespace).UpdateStatus(newDeployment)
	return err
}

// getReplicaFailures transitions all the warning events found for a replica set into
// deployment conditions. It's up to the caller to dedupe between the conditions and
// add just a single condition, ideally the oldest one.
// TODO: Once #32863 is fixed we should stop listing events and instead look into replica
// set Conditions for failures.
func (dc *DeploymentController) getReplicaFailures(rs *extensions.ReplicaSet) []extensions.DeploymentCondition {
	eventList, err := dc.client.Core().Events(rs.Namespace).Search(rs)
	if err != nil {
		glog.V(2).Infof("Cannot list events for replica set %q: %v", err)
		return nil
	}

	sort.Sort(events.SortableEvents(eventList.Items))

	var conditions []extensions.DeploymentCondition
	for i := range eventList.Items {
		event := eventList.Items[i]

		if cond := util.ConditionFromWarningEvent(event); cond != nil {
			conditions = append(conditions, *cond)
		}
	}
	return conditions
}
