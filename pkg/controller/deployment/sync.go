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
	"strconv"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/kubernetes/pkg/controller"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	labelsutil "k8s.io/kubernetes/pkg/util/labels"
)

// syncStatusOnly only updates Deployments Status and doesn't take any mutating actions.
func (dc *DeploymentController) syncStatusOnly(d *extensions.Deployment, rsList []*extensions.ReplicaSet, podMap map[types.UID]*v1.PodList) error {
	newRS, oldRSs, err := dc.getAllReplicaSetsAndSyncRevision(d, rsList, podMap, false)
	if err != nil {
		return err
	}

	allRSs := append(oldRSs, newRS)
	return dc.syncDeploymentStatus(allRSs, newRS, d)
}

// sync is responsible for reconciling deployments on scaling events or when they
// are paused.
func (dc *DeploymentController) sync(d *extensions.Deployment, rsList []*extensions.ReplicaSet, podMap map[types.UID]*v1.PodList) error {
	newRS, oldRSs, err := dc.getAllReplicaSetsAndSyncRevision(d, rsList, podMap, false)
	if err != nil {
		return err
	}
	if err := dc.scale(d, newRS, oldRSs); err != nil {
		// If we get an error while trying to scale, the deployment will be requeued
		// so we can abort this resync
		return err
	}

	// Clean up the deployment when it's paused and no rollback is in flight.
	if d.Spec.Paused && d.Spec.RollbackTo == nil {
		if err := dc.cleanupDeployment(oldRSs, d); err != nil {
			return err
		}
	}

	allRSs := append(oldRSs, newRS)
	return dc.syncDeploymentStatus(allRSs, newRS, d)
}

// checkPausedConditions checks if the given deployment is paused or not and adds an appropriate condition.
// These conditions are needed so that we won't accidentally report lack of progress for resumed deployments
// that were paused for longer than progressDeadlineSeconds.
func (dc *DeploymentController) checkPausedConditions(d *extensions.Deployment) error {
	if d.Spec.ProgressDeadlineSeconds == nil {
		return nil
	}
	cond := deploymentutil.GetDeploymentCondition(d.Status, extensions.DeploymentProgressing)
	if cond != nil && cond.Reason == deploymentutil.TimedOutReason {
		// If we have reported lack of progress, do not overwrite it with a paused condition.
		return nil
	}
	pausedCondExists := cond != nil && cond.Reason == deploymentutil.PausedDeployReason

	needsUpdate := false
	if d.Spec.Paused && !pausedCondExists {
		condition := deploymentutil.NewDeploymentCondition(extensions.DeploymentProgressing, v1.ConditionUnknown, deploymentutil.PausedDeployReason, "Deployment is paused")
		deploymentutil.SetDeploymentCondition(&d.Status, *condition)
		needsUpdate = true
	} else if !d.Spec.Paused && pausedCondExists {
		condition := deploymentutil.NewDeploymentCondition(extensions.DeploymentProgressing, v1.ConditionUnknown, deploymentutil.ResumedDeployReason, "Deployment is resumed")
		deploymentutil.SetDeploymentCondition(&d.Status, *condition)
		needsUpdate = true
	}

	if !needsUpdate {
		return nil
	}

	var err error
	d, err = dc.client.Extensions().Deployments(d.Namespace).UpdateStatus(d)
	return err
}

// getAllReplicaSetsAndSyncRevision returns all the replica sets for the provided deployment (new and all old), with new RS's and deployment's revision updated.
//
// rsList should come from getReplicaSetsForDeployment(d).
// podMap should come from getPodMapForDeployment(d, rsList).
//
// 1. Get all old RSes this deployment targets, and calculate the max revision number among them (maxOldV).
// 2. Get new RS this deployment targets (whose pod template matches deployment's), and update new RS's revision number to (maxOldV + 1),
//    only if its revision number is smaller than (maxOldV + 1). If this step failed, we'll update it in the next deployment sync loop.
// 3. Copy new RS's revision number to deployment (update deployment's revision). If this step failed, we'll update it in the next deployment sync loop.
//
// Note that currently the deployment controller is using caches to avoid querying the server for reads.
// This may lead to stale reads of replica sets, thus incorrect deployment status.
func (dc *DeploymentController) getAllReplicaSetsAndSyncRevision(d *extensions.Deployment, rsList []*extensions.ReplicaSet, podMap map[types.UID]*v1.PodList, createIfNotExisted bool) (*extensions.ReplicaSet, []*extensions.ReplicaSet, error) {
	// List the deployment's RSes & Pods and apply pod-template-hash info to deployment's adopted RSes/Pods
	rsList, err := dc.rsAndPodsWithHashKeySynced(d, rsList, podMap)
	if err != nil {
		return nil, nil, fmt.Errorf("error labeling replica sets and pods with pod-template-hash: %v", err)
	}
	_, allOldRSs, err := deploymentutil.FindOldReplicaSets(d, rsList)
	if err != nil {
		return nil, nil, err
	}

	// Get new replica set with the updated revision number
	newRS, err := dc.getNewReplicaSet(d, rsList, allOldRSs, createIfNotExisted)
	if err != nil {
		return nil, nil, err
	}

	return newRS, allOldRSs, nil
}

// rsAndPodsWithHashKeySynced returns the RSes and pods the given deployment
// targets, with pod-template-hash information synced.
//
// rsList should come from getReplicaSetsForDeployment(d).
// podMap should come from getPodMapForDeployment(d, rsList).
func (dc *DeploymentController) rsAndPodsWithHashKeySynced(d *extensions.Deployment, rsList []*extensions.ReplicaSet, podMap map[types.UID]*v1.PodList) ([]*extensions.ReplicaSet, error) {
	var syncedRSList []*extensions.ReplicaSet
	for _, rs := range rsList {
		// Add pod-template-hash information if it's not in the RS.
		// Otherwise, new RS produced by Deployment will overlap with pre-existing ones
		// that aren't constrained by the pod-template-hash.
		syncedRS, err := dc.addHashKeyToRSAndPods(rs, podMap[rs.UID], d.Status.CollisionCount)
		if err != nil {
			return nil, err
		}
		syncedRSList = append(syncedRSList, syncedRS)
	}
	return syncedRSList, nil
}

// addHashKeyToRSAndPods adds pod-template-hash information to the given rs, if it's not already there, with the following steps:
// 1. Add hash label to the rs's pod template, and make sure the controller sees this update so that no orphaned pods will be created
// 2. Add hash label to all pods this rs owns, wait until replicaset controller reports rs.Status.FullyLabeledReplicas equal to the desired number of replicas
// 3. Add hash label to the rs's label and selector
func (dc *DeploymentController) addHashKeyToRSAndPods(rs *extensions.ReplicaSet, podList *v1.PodList, collisionCount *int64) (*extensions.ReplicaSet, error) {
	// If the rs already has the new hash label in its selector, it's done syncing
	if labelsutil.SelectorHasLabel(rs.Spec.Selector, extensions.DefaultDeploymentUniqueLabelKey) {
		return rs, nil
	}
	hash, err := deploymentutil.GetReplicaSetHash(rs, collisionCount)
	if err != nil {
		return nil, err
	}
	// 1. Add hash template label to the rs. This ensures that any newly created pods will have the new label.
	updatedRS, err := deploymentutil.UpdateRSWithRetries(dc.client.Extensions().ReplicaSets(rs.Namespace), dc.rsLister, rs.Namespace, rs.Name,
		func(updated *extensions.ReplicaSet) error {
			// Precondition: the RS doesn't contain the new hash in its pod template label.
			if updated.Spec.Template.Labels[extensions.DefaultDeploymentUniqueLabelKey] == hash {
				return utilerrors.ErrPreconditionViolated
			}
			updated.Spec.Template.Labels = labelsutil.AddLabel(updated.Spec.Template.Labels, extensions.DefaultDeploymentUniqueLabelKey, hash)
			return nil
		})
	if err != nil {
		return nil, fmt.Errorf("error updating replica set %s/%s pod template label with template hash: %v", rs.Namespace, rs.Name, err)
	}
	// Make sure rs pod template is updated so that it won't create pods without the new label (orphaned pods).
	if updatedRS.Generation > updatedRS.Status.ObservedGeneration {
		// TODO: Revisit if we really need to wait here as opposed to returning and
		// potentially unblocking this worker (can wait up to 1min before timing out).
		if err = deploymentutil.WaitForReplicaSetUpdated(dc.rsLister, updatedRS.Generation, updatedRS.Namespace, updatedRS.Name); err != nil {
			return nil, fmt.Errorf("error waiting for replica set %s/%s to be observed by controller: %v", updatedRS.Namespace, updatedRS.Name, err)
		}
		glog.V(4).Infof("Observed the update of replica set %s/%s's pod template with hash %s.", rs.Namespace, rs.Name, hash)
	}

	// 2. Update all pods managed by the rs to have the new hash label, so they will be correctly adopted.
	if err := deploymentutil.LabelPodsWithHash(podList, dc.client, dc.podLister, rs.Namespace, rs.Name, hash); err != nil {
		return nil, fmt.Errorf("error in adding template hash label %s to pods %+v: %s", hash, podList, err)
	}

	// We need to wait for the replicaset controller to observe the pods being
	// labeled with pod template hash. Because previously we've called
	// WaitForReplicaSetUpdated, the replicaset controller should have dropped
	// FullyLabeledReplicas to 0 already, we only need to wait it to increase
	// back to the number of replicas in the spec.
	// TODO: Revisit if we really need to wait here as opposed to returning and
	// potentially unblocking this worker (can wait up to 1min before timing out).
	if err := deploymentutil.WaitForPodsHashPopulated(dc.rsLister, updatedRS.Generation, updatedRS.Namespace, updatedRS.Name); err != nil {
		return nil, fmt.Errorf("Replica set %s/%s: error waiting for replicaset controller to observe pods being labeled with template hash: %v", updatedRS.Namespace, updatedRS.Name, err)
	}

	// 3. Update rs label and selector to include the new hash label
	// Copy the old selector, so that we can scrub out any orphaned pods
	updatedRS, err = deploymentutil.UpdateRSWithRetries(dc.client.Extensions().ReplicaSets(rs.Namespace), dc.rsLister, rs.Namespace, rs.Name, func(updated *extensions.ReplicaSet) error {
		// Precondition: the RS doesn't contain the new hash in its label and selector.
		if updated.Labels[extensions.DefaultDeploymentUniqueLabelKey] == hash && updated.Spec.Selector.MatchLabels[extensions.DefaultDeploymentUniqueLabelKey] == hash {
			return utilerrors.ErrPreconditionViolated
		}
		updated.Labels = labelsutil.AddLabel(updated.Labels, extensions.DefaultDeploymentUniqueLabelKey, hash)
		updated.Spec.Selector = labelsutil.AddLabelToSelector(updated.Spec.Selector, extensions.DefaultDeploymentUniqueLabelKey, hash)
		return nil
	})
	// If the RS isn't actually updated, that's okay, we'll retry in the
	// next sync loop since its selector isn't updated yet.
	if err != nil {
		return nil, fmt.Errorf("error updating ReplicaSet %s/%s label and selector with template hash: %v", updatedRS.Namespace, updatedRS.Name, err)
	}

	// TODO: look for orphaned pods and label them in the background somewhere else periodically
	return updatedRS, nil
}

// Returns a replica set that matches the intent of the given deployment. Returns nil if the new replica set doesn't exist yet.
// 1. Get existing new RS (the RS that the given deployment targets, whose pod template is the same as deployment's).
// 2. If there's existing new RS, update its revision number if it's smaller than (maxOldRevision + 1), where maxOldRevision is the max revision number among all old RSes.
// 3. If there's no existing new RS and createIfNotExisted is true, create one with appropriate revision number (maxOldRevision + 1) and replicas.
// Note that the pod-template-hash will be added to adopted RSes and pods.
func (dc *DeploymentController) getNewReplicaSet(d *extensions.Deployment, rsList, oldRSs []*extensions.ReplicaSet, createIfNotExisted bool) (*extensions.ReplicaSet, error) {
	existingNewRS, err := deploymentutil.FindNewReplicaSet(d, rsList)
	if err != nil {
		return nil, err
	}

	// Calculate the max revision number among all old RSes
	maxOldRevision := deploymentutil.MaxRevision(oldRSs)
	// Calculate revision number for this new replica set
	newRevision := strconv.FormatInt(maxOldRevision+1, 10)

	// Latest replica set exists. We need to sync its annotations (includes copying all but
	// annotationsToSkip from the parent deployment, and update revision, desiredReplicas,
	// and maxReplicas) and also update the revision annotation in the deployment with the
	// latest revision.
	if existingNewRS != nil {
		objCopy, err := scheme.Scheme.Copy(existingNewRS)
		if err != nil {
			return nil, err
		}
		rsCopy := objCopy.(*extensions.ReplicaSet)

		// Set existing new replica set's annotation
		annotationsUpdated := deploymentutil.SetNewReplicaSetAnnotations(d, rsCopy, newRevision, true)
		minReadySecondsNeedsUpdate := rsCopy.Spec.MinReadySeconds != d.Spec.MinReadySeconds
		if annotationsUpdated || minReadySecondsNeedsUpdate {
			rsCopy.Spec.MinReadySeconds = d.Spec.MinReadySeconds
			return dc.client.Extensions().ReplicaSets(rsCopy.ObjectMeta.Namespace).Update(rsCopy)
		}

		// Should use the revision in existingNewRS's annotation, since it set by before
		needsUpdate := deploymentutil.SetDeploymentRevision(d, rsCopy.Annotations[deploymentutil.RevisionAnnotation])
		// If no other Progressing condition has been recorded and we need to estimate the progress
		// of this deployment then it is likely that old users started caring about progress. In that
		// case we need to take into account the first time we noticed their new replica set.
		cond := deploymentutil.GetDeploymentCondition(d.Status, extensions.DeploymentProgressing)
		if d.Spec.ProgressDeadlineSeconds != nil && cond == nil {
			msg := fmt.Sprintf("Found new replica set %q", rsCopy.Name)
			condition := deploymentutil.NewDeploymentCondition(extensions.DeploymentProgressing, v1.ConditionTrue, deploymentutil.FoundNewRSReason, msg)
			deploymentutil.SetDeploymentCondition(&d.Status, *condition)
			needsUpdate = true
		}

		if needsUpdate {
			if d, err = dc.client.Extensions().Deployments(d.Namespace).UpdateStatus(d); err != nil {
				return nil, err
			}
		}
		return rsCopy, nil
	}

	if !createIfNotExisted {
		return nil, nil
	}

	// new ReplicaSet does not exist, create one.
	templateCopy, err := scheme.Scheme.DeepCopy(d.Spec.Template)
	if err != nil {
		return nil, err
	}
	newRSTemplate := templateCopy.(v1.PodTemplateSpec)
	podTemplateSpecHash := fmt.Sprintf("%d", controller.ComputeHash(&newRSTemplate, d.Status.CollisionCount))
	newRSTemplate.Labels = labelsutil.CloneAndAddLabel(d.Spec.Template.Labels, extensions.DefaultDeploymentUniqueLabelKey, podTemplateSpecHash)
	// Add podTemplateHash label to selector.
	newRSSelector := labelsutil.CloneSelectorAndAddLabel(d.Spec.Selector, extensions.DefaultDeploymentUniqueLabelKey, podTemplateSpecHash)

	// Create new ReplicaSet
	newRS := extensions.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			// Make the name deterministic, to ensure idempotence
			Name:            d.Name + "-" + podTemplateSpecHash,
			Namespace:       d.Namespace,
			OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(d, controllerKind)},
		},
		Spec: extensions.ReplicaSetSpec{
			Replicas:        new(int32),
			MinReadySeconds: d.Spec.MinReadySeconds,
			Selector:        newRSSelector,
			Template:        newRSTemplate,
		},
	}
	allRSs := append(oldRSs, &newRS)
	newReplicasCount, err := deploymentutil.NewRSNewReplicas(d, allRSs, &newRS)
	if err != nil {
		return nil, err
	}

	*(newRS.Spec.Replicas) = newReplicasCount
	// Set new replica set's annotation
	deploymentutil.SetNewReplicaSetAnnotations(d, &newRS, newRevision, false)
	// Create the new ReplicaSet. If it already exists, then we need to check for possible
	// hash collisions. If there is any other error, we need to report it in the status of
	// the Deployment.
	alreadyExists := false
	createdRS, err := dc.client.Extensions().ReplicaSets(d.Namespace).Create(&newRS)
	switch {
	// We may end up hitting this due to a slow cache or a fast resync of the Deployment.
	// Fetch a copy of the ReplicaSet. If its PodTemplateSpec is semantically deep equal
	// with the PodTemplateSpec of the Deployment, then that is our new ReplicaSet. Otherwise,
	// this is a hash collision and we need to increment the collisionCount field in the
	// status of the Deployment and try the creation again.
	case errors.IsAlreadyExists(err):
		alreadyExists = true
		rs, rsErr := dc.rsLister.ReplicaSets(newRS.Namespace).Get(newRS.Name)
		if rsErr != nil {
			return nil, rsErr
		}
		isEqual, equalErr := deploymentutil.EqualIgnoreHash(&d.Spec.Template, &rs.Spec.Template)
		if equalErr != nil {
			return nil, equalErr
		}
		// Matching ReplicaSet is not equal - increment the collisionCount in the DeploymentStatus
		// and requeue the Deployment.
		if !isEqual {
			if d.Status.CollisionCount == nil {
				d.Status.CollisionCount = new(int64)
			}
			preCollisionCount := *d.Status.CollisionCount
			*d.Status.CollisionCount++
			// Update the collisionCount for the Deployment and let it requeue by returning the original
			// error.
			_, dErr := dc.client.Extensions().Deployments(d.Namespace).UpdateStatus(d)
			if dErr == nil {
				glog.V(2).Infof("Found a hash collision for deployment %q - bumping collisionCount (%d->%d) to resolve it", d.Name, preCollisionCount, *d.Status.CollisionCount)
			}
			return nil, err
		}
		// Pass through the matching ReplicaSet as the new ReplicaSet.
		createdRS = rs
		err = nil
	case err != nil:
		msg := fmt.Sprintf("Failed to create new replica set %q: %v", newRS.Name, err)
		if d.Spec.ProgressDeadlineSeconds != nil {
			cond := deploymentutil.NewDeploymentCondition(extensions.DeploymentProgressing, v1.ConditionFalse, deploymentutil.FailedRSCreateReason, msg)
			deploymentutil.SetDeploymentCondition(&d.Status, *cond)
			// We don't really care about this error at this point, since we have a bigger issue to report.
			// TODO: Identify which errors are permanent and switch DeploymentIsFailed to take into account
			// these reasons as well. Related issue: https://github.com/kubernetes/kubernetes/issues/18568
			_, _ = dc.client.Extensions().Deployments(d.Namespace).UpdateStatus(d)
		}
		dc.eventRecorder.Eventf(d, v1.EventTypeWarning, deploymentutil.FailedRSCreateReason, msg)
		return nil, err
	}
	if !alreadyExists && newReplicasCount > 0 {
		dc.eventRecorder.Eventf(d, v1.EventTypeNormal, "ScalingReplicaSet", "Scaled up replica set %s to %d", createdRS.Name, newReplicasCount)
	}

	needsUpdate := deploymentutil.SetDeploymentRevision(d, newRevision)
	if !alreadyExists && d.Spec.ProgressDeadlineSeconds != nil {
		msg := fmt.Sprintf("Created new replica set %q", createdRS.Name)
		condition := deploymentutil.NewDeploymentCondition(extensions.DeploymentProgressing, v1.ConditionTrue, deploymentutil.NewReplicaSetReason, msg)
		deploymentutil.SetDeploymentCondition(&d.Status, *condition)
		needsUpdate = true
	}
	if needsUpdate {
		_, err = dc.client.Extensions().Deployments(d.Namespace).UpdateStatus(d)
	}
	return createdRS, err
}

// scale scales proportionally in order to mitigate risk. Otherwise, scaling up can increase the size
// of the new replica set and scaling down can decrease the sizes of the old ones, both of which would
// have the effect of hastening the rollout progress, which could produce a higher proportion of unavailable
// replicas in the event of a problem with the rolled out template. Should run only on scaling events or
// when a deployment is paused and not during the normal rollout process.
func (dc *DeploymentController) scale(deployment *extensions.Deployment, newRS *extensions.ReplicaSet, oldRSs []*extensions.ReplicaSet) error {
	// If there is only one active replica set then we should scale that up to the full count of the
	// deployment. If there is no active replica set, then we should scale up the newest replica set.
	if activeOrLatest := deploymentutil.FindActiveOrLatest(newRS, oldRSs); activeOrLatest != nil {
		if *(activeOrLatest.Spec.Replicas) == *(deployment.Spec.Replicas) {
			return nil
		}
		_, _, err := dc.scaleReplicaSetAndRecordEvent(activeOrLatest, *(deployment.Spec.Replicas), deployment)
		return err
	}

	// If the new replica set is saturated, old replica sets should be fully scaled down.
	// This case handles replica set adoption during a saturated new replica set.
	if deploymentutil.IsSaturated(deployment, newRS) {
		for _, old := range controller.FilterActiveReplicaSets(oldRSs) {
			if _, _, err := dc.scaleReplicaSetAndRecordEvent(old, 0, deployment); err != nil {
				return err
			}
		}
		return nil
	}

	// There are old replica sets with pods and the new replica set is not saturated.
	// We need to proportionally scale all replica sets (new and old) in case of a
	// rolling deployment.
	if deploymentutil.IsRollingUpdate(deployment) {
		allRSs := controller.FilterActiveReplicaSets(append(oldRSs, newRS))
		allRSsReplicas := deploymentutil.GetReplicaCountForReplicaSets(allRSs)

		allowedSize := int32(0)
		if *(deployment.Spec.Replicas) > 0 {
			allowedSize = *(deployment.Spec.Replicas) + deploymentutil.MaxSurge(*deployment)
		}

		// Number of additional replicas that can be either added or removed from the total
		// replicas count. These replicas should be distributed proportionally to the active
		// replica sets.
		deploymentReplicasToAdd := allowedSize - allRSsReplicas

		// The additional replicas should be distributed proportionally amongst the active
		// replica sets from the larger to the smaller in size replica set. Scaling direction
		// drives what happens in case we are trying to scale replica sets of the same size.
		// In such a case when scaling up, we should scale up newer replica sets first, and
		// when scaling down, we should scale down older replica sets first.
		var scalingOperation string
		switch {
		case deploymentReplicasToAdd > 0:
			sort.Sort(controller.ReplicaSetsBySizeNewer(allRSs))
			scalingOperation = "up"

		case deploymentReplicasToAdd < 0:
			sort.Sort(controller.ReplicaSetsBySizeOlder(allRSs))
			scalingOperation = "down"
		}

		// Iterate over all active replica sets and estimate proportions for each of them.
		// The absolute value of deploymentReplicasAdded should never exceed the absolute
		// value of deploymentReplicasToAdd.
		deploymentReplicasAdded := int32(0)
		nameToSize := make(map[string]int32)
		for i := range allRSs {
			rs := allRSs[i]

			// Estimate proportions if we have replicas to add, otherwise simply populate
			// nameToSize with the current sizes for each replica set.
			if deploymentReplicasToAdd != 0 {
				proportion := deploymentutil.GetProportion(rs, *deployment, deploymentReplicasToAdd, deploymentReplicasAdded)

				nameToSize[rs.Name] = *(rs.Spec.Replicas) + proportion
				deploymentReplicasAdded += proportion
			} else {
				nameToSize[rs.Name] = *(rs.Spec.Replicas)
			}
		}

		// Update all replica sets
		for i := range allRSs {
			rs := allRSs[i]

			// Add/remove any leftovers to the largest replica set.
			if i == 0 && deploymentReplicasToAdd != 0 {
				leftover := deploymentReplicasToAdd - deploymentReplicasAdded
				nameToSize[rs.Name] = nameToSize[rs.Name] + leftover
				if nameToSize[rs.Name] < 0 {
					nameToSize[rs.Name] = 0
				}
			}

			// TODO: Use transactions when we have them.
			if _, _, err := dc.scaleReplicaSet(rs, nameToSize[rs.Name], deployment, scalingOperation); err != nil {
				// Return as soon as we fail, the deployment is requeued
				return err
			}
		}
	}
	return nil
}

func (dc *DeploymentController) scaleReplicaSetAndRecordEvent(rs *extensions.ReplicaSet, newScale int32, deployment *extensions.Deployment) (bool, *extensions.ReplicaSet, error) {
	// No need to scale
	if *(rs.Spec.Replicas) == newScale {
		return false, rs, nil
	}
	var scalingOperation string
	if *(rs.Spec.Replicas) < newScale {
		scalingOperation = "up"
	} else {
		scalingOperation = "down"
	}
	scaled, newRS, err := dc.scaleReplicaSet(rs, newScale, deployment, scalingOperation)
	return scaled, newRS, err
}

func (dc *DeploymentController) scaleReplicaSet(rs *extensions.ReplicaSet, newScale int32, deployment *extensions.Deployment, scalingOperation string) (bool, *extensions.ReplicaSet, error) {
	objCopy, err := scheme.Scheme.Copy(rs)
	if err != nil {
		return false, nil, err
	}
	rsCopy := objCopy.(*extensions.ReplicaSet)

	sizeNeedsUpdate := *(rsCopy.Spec.Replicas) != newScale
	// TODO: Do not mutate the replica set here, instead simply compare the annotation and if they mismatch
	// call SetReplicasAnnotations inside the following if clause. Then we can also move the deep-copy from
	// above inside the if too.
	annotationsNeedUpdate := deploymentutil.SetReplicasAnnotations(rsCopy, *(deployment.Spec.Replicas), *(deployment.Spec.Replicas)+deploymentutil.MaxSurge(*deployment))

	scaled := false
	if sizeNeedsUpdate || annotationsNeedUpdate {
		*(rsCopy.Spec.Replicas) = newScale
		rs, err = dc.client.Extensions().ReplicaSets(rsCopy.Namespace).Update(rsCopy)
		if err == nil && sizeNeedsUpdate {
			scaled = true
			dc.eventRecorder.Eventf(deployment, v1.EventTypeNormal, "ScalingReplicaSet", "Scaled %s replica set %s to %d", scalingOperation, rs.Name, newScale)
		}
	}
	return scaled, rs, err
}

// cleanupDeployment is responsible for cleaning up a deployment ie. retains all but the latest N old replica sets
// where N=d.Spec.RevisionHistoryLimit. Old replica sets are older versions of the podtemplate of a deployment kept
// around by default 1) for historical reasons and 2) for the ability to rollback a deployment.
func (dc *DeploymentController) cleanupDeployment(oldRSs []*extensions.ReplicaSet, deployment *extensions.Deployment) error {
	if deployment.Spec.RevisionHistoryLimit == nil {
		return nil
	}

	// Avoid deleting replica set with deletion timestamp set
	aliveFilter := func(rs *extensions.ReplicaSet) bool {
		return rs != nil && rs.ObjectMeta.DeletionTimestamp == nil
	}
	cleanableRSes := controller.FilterReplicaSets(oldRSs, aliveFilter)

	diff := int32(len(cleanableRSes)) - *deployment.Spec.RevisionHistoryLimit
	if diff <= 0 {
		return nil
	}

	sort.Sort(controller.ReplicaSetsByCreationTimestamp(cleanableRSes))
	glog.V(4).Infof("Looking to cleanup old replica sets for deployment %q", deployment.Name)

	for i := int32(0); i < diff; i++ {
		rs := cleanableRSes[i]
		// Avoid delete replica set with non-zero replica counts
		if rs.Status.Replicas != 0 || *(rs.Spec.Replicas) != 0 || rs.Generation > rs.Status.ObservedGeneration || rs.DeletionTimestamp != nil {
			continue
		}
		glog.V(4).Infof("Trying to cleanup replica set %q for deployment %q", rs.Name, deployment.Name)
		if err := dc.client.Extensions().ReplicaSets(rs.Namespace).Delete(rs.Name, nil); err != nil && !errors.IsNotFound(err) {
			// Return error instead of aggregating and continuing DELETEs on the theory
			// that we may be overloading the api server.
			return err
		}
	}

	return nil
}

// syncDeploymentStatus checks if the status is up-to-date and sync it if necessary
func (dc *DeploymentController) syncDeploymentStatus(allRSs []*extensions.ReplicaSet, newRS *extensions.ReplicaSet, d *extensions.Deployment) error {
	newStatus := calculateStatus(allRSs, newRS, d)

	if reflect.DeepEqual(d.Status, newStatus) {
		return nil
	}

	newDeployment := d
	newDeployment.Status = newStatus
	_, err := dc.client.Extensions().Deployments(newDeployment.Namespace).UpdateStatus(newDeployment)
	return err
}

// calculateStatus calculates the latest status for the provided deployment by looking into the provided replica sets.
func calculateStatus(allRSs []*extensions.ReplicaSet, newRS *extensions.ReplicaSet, deployment *extensions.Deployment) extensions.DeploymentStatus {
	availableReplicas := deploymentutil.GetAvailableReplicaCountForReplicaSets(allRSs)
	totalReplicas := deploymentutil.GetReplicaCountForReplicaSets(allRSs)
	unavailableReplicas := totalReplicas - availableReplicas
	// If unavailableReplicas is negative, then that means the Deployment has more available replicas running than
	// desired, e.g. whenever it scales down. In such a case we should simply default unavailableReplicas to zero.
	if unavailableReplicas < 0 {
		unavailableReplicas = 0
	}

	status := extensions.DeploymentStatus{
		// TODO: Ensure that if we start retrying status updates, we won't pick up a new Generation value.
		ObservedGeneration:  deployment.Generation,
		Replicas:            deploymentutil.GetActualReplicaCountForReplicaSets(allRSs),
		UpdatedReplicas:     deploymentutil.GetActualReplicaCountForReplicaSets([]*extensions.ReplicaSet{newRS}),
		ReadyReplicas:       deploymentutil.GetReadyReplicaCountForReplicaSets(allRSs),
		AvailableReplicas:   availableReplicas,
		UnavailableReplicas: unavailableReplicas,
		CollisionCount:      deployment.Status.CollisionCount,
	}

	// Copy conditions one by one so we won't mutate the original object.
	conditions := deployment.Status.Conditions
	for i := range conditions {
		status.Conditions = append(status.Conditions, conditions[i])
	}

	if availableReplicas >= *(deployment.Spec.Replicas)-deploymentutil.MaxUnavailable(*deployment) {
		minAvailability := deploymentutil.NewDeploymentCondition(extensions.DeploymentAvailable, v1.ConditionTrue, deploymentutil.MinimumReplicasAvailable, "Deployment has minimum availability.")
		deploymentutil.SetDeploymentCondition(&status, *minAvailability)
	} else {
		noMinAvailability := deploymentutil.NewDeploymentCondition(extensions.DeploymentAvailable, v1.ConditionFalse, deploymentutil.MinimumReplicasUnavailable, "Deployment does not have minimum availability.")
		deploymentutil.SetDeploymentCondition(&status, *noMinAvailability)
	}

	return status
}

// isScalingEvent checks whether the provided deployment has been updated with a scaling event
// by looking at the desired-replicas annotation in the active replica sets of the deployment.
//
// rsList should come from getReplicaSetsForDeployment(d).
// podMap should come from getPodMapForDeployment(d, rsList).
func (dc *DeploymentController) isScalingEvent(d *extensions.Deployment, rsList []*extensions.ReplicaSet, podMap map[types.UID]*v1.PodList) (bool, error) {
	newRS, oldRSs, err := dc.getAllReplicaSetsAndSyncRevision(d, rsList, podMap, false)
	if err != nil {
		return false, err
	}
	allRSs := append(oldRSs, newRS)
	for _, rs := range controller.FilterActiveReplicaSets(allRSs) {
		desired, ok := deploymentutil.GetDesiredReplicasAnnotation(rs)
		if !ok {
			continue
		}
		if desired != *(d.Spec.Replicas) {
			return true, nil
		}
	}
	return false, nil
}
