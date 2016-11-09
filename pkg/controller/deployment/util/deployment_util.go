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

package util

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/annotations"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/integer"
	intstrutil "k8s.io/kubernetes/pkg/util/intstr"
	labelsutil "k8s.io/kubernetes/pkg/util/labels"
	podutil "k8s.io/kubernetes/pkg/util/pod"
	"k8s.io/kubernetes/pkg/util/wait"
)

const (
	// RevisionAnnotation is the revision annotation of a deployment's replica sets which records its rollout sequence
	RevisionAnnotation = "deployment.kubernetes.io/revision"
	// RevisionHistoryAnnotation maintains the history of all old revisions that a replica set has served for a deployment.
	RevisionHistoryAnnotation = "deployment.kubernetes.io/revision-history"
	// DesiredReplicasAnnotation is the desired replicas for a deployment recorded as an annotation
	// in its replica sets. Helps in separating scaling events from the rollout process and for
	// determining if the new replica set for a deployment is really saturated.
	DesiredReplicasAnnotation = "deployment.kubernetes.io/desired-replicas"
	// MaxReplicasAnnotation is the maximum replicas a deployment can have at a given point, which
	// is deployment.spec.replicas + maxSurge. Used by the underlying replica sets to estimate their
	// proportions in case the deployment has surge replicas.
	MaxReplicasAnnotation = "deployment.kubernetes.io/max-replicas"

	// RollbackRevisionNotFound is not found rollback event reason
	RollbackRevisionNotFound = "DeploymentRollbackRevisionNotFound"
	// RollbackTemplateUnchanged is the template unchanged rollback event reason
	RollbackTemplateUnchanged = "DeploymentRollbackTemplateUnchanged"
	// RollbackDone is the done rollback event reason
	RollbackDone = "DeploymentRollback"
	// OverlapAnnotation marks deployments with overlapping selector with other deployments
	// TODO: Delete this annotation when we gracefully handle overlapping selectors.
	// See https://github.com/kubernetes/kubernetes/issues/2210
	OverlapAnnotation = "deployment.kubernetes.io/error-selector-overlapping-with"
	// SelectorUpdateAnnotation marks the last time deployment selector update
	// TODO: Delete this annotation when we gracefully handle overlapping selectors.
	// See https://github.com/kubernetes/kubernetes/issues/2210
	SelectorUpdateAnnotation = "deployment.kubernetes.io/selector-updated-at"

	// Reasons for deployment conditions
	//
	// Progressing:
	//
	// ReplicaSetUpdatedReason is added in a deployment when one of its replica sets is updated as part
	// of the rollout process.
	ReplicaSetUpdatedReason = "ReplicaSetUpdated"
	// FailedRSCreateReason is added in a deployment when it cannot create a new replica set.
	FailedRSCreateReason = "ReplicaSetCreateError"
	// NewReplicaSetReason is added in a deployment when it creates a new replica set.
	NewReplicaSetReason = "NewReplicaSetCreated"
	// FoundNewRSReason is added in a deployment when it adopts an existing replica set.
	FoundNewRSReason = "FoundNewReplicaSet"
	// NewRSAvailableReason is added in a deployment when its newest replica set is made available
	// ie. the number of new pods that have passed readiness checks and run for at least minReadySeconds
	// is at least the minimum available pods that need to run for the deployment.
	NewRSAvailableReason = "NewReplicaSetAvailable"
	// TimedOutReason is added in a deployment when its newest replica set fails to show any progress
	// within the given deadline (progressDeadlineSeconds).
	TimedOutReason = "ProgressDeadlineExceeded"
	// PausedDeployReason is added in a deployment when it is paused. Lack of progress shouldn't be
	// estimated once a deployment is paused.
	PausedDeployReason = "DeploymentPaused"
	// ResumedDeployReason is added in a deployment when it is resumed. Useful for not failing accidentally
	// deployments that paused amidst a rollout and are bounded by a deadline.
	ResumedDeployReason = "DeploymentResumed"
	//
	// Available:
	//
	// MinimumReplicasAvailable is added in a deployment when it has its minimum replicas required available.
	MinimumReplicasAvailable = "MinimumReplicasAvailable"
	// MinimumReplicasUnavailable is added in a deployment when it doesn't have the minimum required replicas
	// available.
	MinimumReplicasUnavailable = "MinimumReplicasUnavailable"
)

// NewDeploymentCondition creates a new deployment condition.
func NewDeploymentCondition(condType extensions.DeploymentConditionType, status api.ConditionStatus, reason, message string) *extensions.DeploymentCondition {
	return &extensions.DeploymentCondition{
		Type:               condType,
		Status:             status,
		LastUpdateTime:     unversioned.Now(),
		LastTransitionTime: unversioned.Now(),
		Reason:             reason,
		Message:            message,
	}
}

// GetDeploymentCondition returns the condition with the provided type.
func GetDeploymentCondition(status extensions.DeploymentStatus, condType extensions.DeploymentConditionType) *extensions.DeploymentCondition {
	for i := range status.Conditions {
		c := status.Conditions[i]
		if c.Type == condType {
			return &c
		}
	}
	return nil
}

// SetDeploymentCondition updates the deployment to include the provided condition. If the condition that
// we are about to add already exists and has the same status and reason then we are not going to update.
func SetDeploymentCondition(status *extensions.DeploymentStatus, condition extensions.DeploymentCondition) {
	currentCond := GetDeploymentCondition(*status, condition.Type)
	if currentCond != nil && currentCond.Status == condition.Status && currentCond.Reason == condition.Reason {
		return
	}
	// Do not update lastTransitionTime if the status of the condition doesn't change.
	if currentCond != nil && currentCond.Status == condition.Status {
		condition.LastTransitionTime = currentCond.LastTransitionTime
	}
	newConditions := filterOutCondition(status.Conditions, condition.Type)
	status.Conditions = append(newConditions, condition)
}

// RemoveDeploymentCondition removes the deployment condition with the provided type.
func RemoveDeploymentCondition(status *extensions.DeploymentStatus, condType extensions.DeploymentConditionType) {
	status.Conditions = filterOutCondition(status.Conditions, condType)
}

// filterOutCondition returns a new slice of deployment conditions without conditions with the provided type.
func filterOutCondition(conditions []extensions.DeploymentCondition, condType extensions.DeploymentConditionType) []extensions.DeploymentCondition {
	var newConditions []extensions.DeploymentCondition
	for _, c := range conditions {
		if c.Type == condType {
			continue
		}
		newConditions = append(newConditions, c)
	}
	return newConditions
}

// ReplicaSetToDeploymentCondition converts a replica set condition into a deployment condition.
// Useful for promoting replica set failure conditions into deployments.
func ReplicaSetToDeploymentCondition(cond extensions.ReplicaSetCondition) extensions.DeploymentCondition {
	return extensions.DeploymentCondition{
		Type:               extensions.DeploymentConditionType(cond.Type),
		Status:             cond.Status,
		LastTransitionTime: cond.LastTransitionTime,
		LastUpdateTime:     cond.LastTransitionTime,
		Reason:             cond.Reason,
		Message:            cond.Message,
	}
}

// SetDeploymentRevision updates the revision for a deployment.
func SetDeploymentRevision(deployment *extensions.Deployment, revision string) bool {
	updated := false

	if deployment.Annotations == nil {
		deployment.Annotations = make(map[string]string)
	}
	if deployment.Annotations[RevisionAnnotation] != revision {
		deployment.Annotations[RevisionAnnotation] = revision
		updated = true
	}

	return updated
}

// MaxRevision finds the highest revision in the replica sets
func MaxRevision(allRSs []*extensions.ReplicaSet) int64 {
	max := int64(0)
	for _, rs := range allRSs {
		if v, err := Revision(rs); err != nil {
			// Skip the replica sets when it failed to parse their revision information
			glog.V(4).Infof("Error: %v. Couldn't parse revision for replica set %#v, deployment controller will skip it when reconciling revisions.", err, rs)
		} else if v > max {
			max = v
		}
	}
	return max
}

// LastRevision finds the second max revision number in all replica sets (the last revision)
func LastRevision(allRSs []*extensions.ReplicaSet) int64 {
	max, secMax := int64(0), int64(0)
	for _, rs := range allRSs {
		if v, err := Revision(rs); err != nil {
			// Skip the replica sets when it failed to parse their revision information
			glog.V(4).Infof("Error: %v. Couldn't parse revision for replica set %#v, deployment controller will skip it when reconciling revisions.", err, rs)
		} else if v >= max {
			secMax = max
			max = v
		} else if v > secMax {
			secMax = v
		}
	}
	return secMax
}

// Revision returns the revision number of the input object.
func Revision(obj runtime.Object) (int64, error) {
	acc, err := meta.Accessor(obj)
	if err != nil {
		return 0, err
	}
	v, ok := acc.GetAnnotations()[RevisionAnnotation]
	if !ok {
		return 0, nil
	}
	return strconv.ParseInt(v, 10, 64)
}

// SetNewReplicaSetAnnotations sets new replica set's annotations appropriately by updating its revision and
// copying required deployment annotations to it; it returns true if replica set's annotation is changed.
func SetNewReplicaSetAnnotations(deployment *extensions.Deployment, newRS *extensions.ReplicaSet, newRevision string, exists bool) bool {
	// First, copy deployment's annotations (except for apply and revision annotations)
	annotationChanged := copyDeploymentAnnotationsToReplicaSet(deployment, newRS)
	// Then, update replica set's revision annotation
	if newRS.Annotations == nil {
		newRS.Annotations = make(map[string]string)
	}
	oldRevision, ok := newRS.Annotations[RevisionAnnotation]
	// The newRS's revision should be the greatest among all RSes. Usually, its revision number is newRevision (the max revision number
	// of all old RSes + 1). However, it's possible that some of the old RSes are deleted after the newRS revision being updated, and
	// newRevision becomes smaller than newRS's revision. We should only update newRS revision when it's smaller than newRevision.
	if oldRevision < newRevision {
		newRS.Annotations[RevisionAnnotation] = newRevision
		annotationChanged = true
		glog.V(4).Infof("Updating replica set %q revision to %s", newRS.Name, newRevision)
	}
	// If a revision annotation already existed and this replica set was updated with a new revision
	// then that means we are rolling back to this replica set. We need to preserve the old revisions
	// for historical information.
	if ok && annotationChanged {
		revisionHistoryAnnotation := newRS.Annotations[RevisionHistoryAnnotation]
		oldRevisions := strings.Split(revisionHistoryAnnotation, ",")
		if len(oldRevisions[0]) == 0 {
			newRS.Annotations[RevisionHistoryAnnotation] = oldRevision
		} else {
			oldRevisions = append(oldRevisions, oldRevision)
			newRS.Annotations[RevisionHistoryAnnotation] = strings.Join(oldRevisions, ",")
		}
	}
	// If the new replica set is about to be created, we need to add replica annotations to it.
	if !exists && SetReplicasAnnotations(newRS, deployment.Spec.Replicas, deployment.Spec.Replicas+MaxSurge(*deployment)) {
		annotationChanged = true
	}
	return annotationChanged
}

var annotationsToSkip = map[string]bool{
	annotations.LastAppliedConfigAnnotation: true,
	RevisionAnnotation:                      true,
	RevisionHistoryAnnotation:               true,
	DesiredReplicasAnnotation:               true,
	MaxReplicasAnnotation:                   true,
	OverlapAnnotation:                       true,
	SelectorUpdateAnnotation:                true,
}

// skipCopyAnnotation returns true if we should skip copying the annotation with the given annotation key
// TODO: How to decide which annotations should / should not be copied?
//       See https://github.com/kubernetes/kubernetes/pull/20035#issuecomment-179558615
func skipCopyAnnotation(key string) bool {
	return annotationsToSkip[key]
}

// copyDeploymentAnnotationsToReplicaSet copies deployment's annotations to replica set's annotations,
// and returns true if replica set's annotation is changed.
// Note that apply and revision annotations are not copied.
func copyDeploymentAnnotationsToReplicaSet(deployment *extensions.Deployment, rs *extensions.ReplicaSet) bool {
	rsAnnotationsChanged := false
	if rs.Annotations == nil {
		rs.Annotations = make(map[string]string)
	}
	for k, v := range deployment.Annotations {
		// newRS revision is updated automatically in getNewReplicaSet, and the deployment's revision number is then updated
		// by copying its newRS revision number. We should not copy deployment's revision to its newRS, since the update of
		// deployment revision number may fail (revision becomes stale) and the revision number in newRS is more reliable.
		if skipCopyAnnotation(k) || rs.Annotations[k] == v {
			continue
		}
		rs.Annotations[k] = v
		rsAnnotationsChanged = true
	}
	return rsAnnotationsChanged
}

// SetDeploymentAnnotationsTo sets deployment's annotations as given RS's annotations.
// This action should be done if and only if the deployment is rolling back to this rs.
// Note that apply and revision annotations are not changed.
func SetDeploymentAnnotationsTo(deployment *extensions.Deployment, rollbackToRS *extensions.ReplicaSet) {
	deployment.Annotations = getSkippedAnnotations(deployment.Annotations)
	for k, v := range rollbackToRS.Annotations {
		if !skipCopyAnnotation(k) {
			deployment.Annotations[k] = v
		}
	}
}

func getSkippedAnnotations(annotations map[string]string) map[string]string {
	skippedAnnotations := make(map[string]string)
	for k, v := range annotations {
		if skipCopyAnnotation(k) {
			skippedAnnotations[k] = v
		}
	}
	return skippedAnnotations
}

// FindActiveOrLatest returns the only active or the latest replica set in case there is at most one active
// replica set. If there are more active replica sets, then we should proportionally scale them.
func FindActiveOrLatest(newRS *extensions.ReplicaSet, oldRSs []*extensions.ReplicaSet) *extensions.ReplicaSet {
	if newRS == nil && len(oldRSs) == 0 {
		return nil
	}

	sort.Sort(sort.Reverse(controller.ReplicaSetsByCreationTimestamp(oldRSs)))
	allRSs := controller.FilterActiveReplicaSets(append(oldRSs, newRS))

	switch len(allRSs) {
	case 0:
		// If there is no active replica set then we should return the newest.
		if newRS != nil {
			return newRS
		}
		return oldRSs[0]
	case 1:
		return allRSs[0]
	default:
		return nil
	}
}

// GetDesiredReplicasAnnotation returns the number of desired replicas
func GetDesiredReplicasAnnotation(rs *extensions.ReplicaSet) (int32, bool) {
	return getIntFromAnnotation(rs, DesiredReplicasAnnotation)
}

func getMaxReplicasAnnotation(rs *extensions.ReplicaSet) (int32, bool) {
	return getIntFromAnnotation(rs, MaxReplicasAnnotation)
}

func getIntFromAnnotation(rs *extensions.ReplicaSet, annotationKey string) (int32, bool) {
	annotationValue, ok := rs.Annotations[annotationKey]
	if !ok {
		return int32(0), false
	}
	intValue, err := strconv.Atoi(annotationValue)
	if err != nil {
		glog.Warningf("Cannot convert the value %q with annotation key %q for the replica set %q",
			annotationValue, annotationKey, rs.Name)
		return int32(0), false
	}
	return int32(intValue), true
}

// SetReplicasAnnotations sets the desiredReplicas and maxReplicas into the annotations
func SetReplicasAnnotations(rs *extensions.ReplicaSet, desiredReplicas, maxReplicas int32) bool {
	updated := false
	if rs.Annotations == nil {
		rs.Annotations = make(map[string]string)
	}
	desiredString := fmt.Sprintf("%d", desiredReplicas)
	if hasString := rs.Annotations[DesiredReplicasAnnotation]; hasString != desiredString {
		rs.Annotations[DesiredReplicasAnnotation] = desiredString
		updated = true
	}
	maxString := fmt.Sprintf("%d", maxReplicas)
	if hasString := rs.Annotations[MaxReplicasAnnotation]; hasString != maxString {
		rs.Annotations[MaxReplicasAnnotation] = maxString
		updated = true
	}
	return updated
}

// MaxUnavailable returns the maximum unavailable pods a rolling deployment can take.
func MaxUnavailable(deployment extensions.Deployment) int32 {
	if !IsRollingUpdate(&deployment) {
		return int32(0)
	}
	// Error caught by validation
	_, maxUnavailable, _ := ResolveFenceposts(&deployment.Spec.Strategy.RollingUpdate.MaxSurge, &deployment.Spec.Strategy.RollingUpdate.MaxUnavailable, deployment.Spec.Replicas)
	return maxUnavailable
}

// MinAvailable returns the minimum vailable pods of a given deployment
func MinAvailable(deployment *extensions.Deployment) int32 {
	if !IsRollingUpdate(deployment) {
		return int32(0)
	}
	return deployment.Spec.Replicas - MaxUnavailable(*deployment)
}

// MaxSurge returns the maximum surge pods a rolling deployment can take.
func MaxSurge(deployment extensions.Deployment) int32 {
	if !IsRollingUpdate(&deployment) {
		return int32(0)
	}
	// Error caught by validation
	maxSurge, _, _ := ResolveFenceposts(&deployment.Spec.Strategy.RollingUpdate.MaxSurge, &deployment.Spec.Strategy.RollingUpdate.MaxUnavailable, deployment.Spec.Replicas)
	return maxSurge
}

// GetProportion will estimate the proportion for the provided replica set using 1. the current size
// of the parent deployment, 2. the replica count that needs be added on the replica sets of the
// deployment, and 3. the total replicas added in the replica sets of the deployment so far.
func GetProportion(rs *extensions.ReplicaSet, d extensions.Deployment, deploymentReplicasToAdd, deploymentReplicasAdded int32) int32 {
	if rs == nil || rs.Spec.Replicas == 0 || deploymentReplicasToAdd == 0 || deploymentReplicasToAdd == deploymentReplicasAdded {
		return int32(0)
	}

	rsFraction := getReplicaSetFraction(*rs, d)
	allowed := deploymentReplicasToAdd - deploymentReplicasAdded

	if deploymentReplicasToAdd > 0 {
		// Use the minimum between the replica set fraction and the maximum allowed replicas
		// when scaling up. This way we ensure we will not scale up more than the allowed
		// replicas we can add.
		return integer.Int32Min(rsFraction, allowed)
	}
	// Use the maximum between the replica set fraction and the maximum allowed replicas
	// when scaling down. This way we ensure we will not scale down more than the allowed
	// replicas we can remove.
	return integer.Int32Max(rsFraction, allowed)
}

// getReplicaSetFraction estimates the fraction of replicas a replica set can have in
// 1. a scaling event during a rollout or 2. when scaling a paused deployment.
func getReplicaSetFraction(rs extensions.ReplicaSet, d extensions.Deployment) int32 {
	// If we are scaling down to zero then the fraction of this replica set is its whole size (negative)
	if d.Spec.Replicas == int32(0) {
		return -rs.Spec.Replicas
	}

	deploymentReplicas := d.Spec.Replicas + MaxSurge(d)
	annotatedReplicas, ok := getMaxReplicasAnnotation(&rs)
	if !ok {
		// If we cannot find the annotation then fallback to the current deployment size. Note that this
		// will not be an accurate proportion estimation in case other replica sets have different values
		// which means that the deployment was scaled at some point but we at least will stay in limits
		// due to the min-max comparisons in getProportion.
		annotatedReplicas = d.Status.Replicas
	}

	// We should never proportionally scale up from zero which means rs.spec.replicas and annotatedReplicas
	// will never be zero here.
	newRSsize := (float64(rs.Spec.Replicas * deploymentReplicas)) / float64(annotatedReplicas)
	return integer.RoundToInt32(newRSsize) - rs.Spec.Replicas
}

// GetAllReplicaSets returns the old and new replica sets targeted by the given Deployment. It gets PodList and ReplicaSetList from client interface.
// Note that the first set of old replica sets doesn't include the ones with no pods, and the second set of old replica sets include all old replica sets.
// The third returned value is the new replica set, and it may be nil if it doesn't exist yet.
func GetAllReplicaSets(deployment *extensions.Deployment, c clientset.Interface) ([]*extensions.ReplicaSet, []*extensions.ReplicaSet, *extensions.ReplicaSet, error) {
	rsList, err := listReplicaSets(deployment, c)
	if err != nil {
		return nil, nil, nil, err
	}
	podList, err := listPods(deployment, c)
	if err != nil {
		return nil, nil, nil, err
	}
	oldRSes, allOldRSes, err := FindOldReplicaSets(deployment, rsList, podList)
	if err != nil {
		return nil, nil, nil, err
	}
	newRS, err := FindNewReplicaSet(deployment, rsList)
	if err != nil {
		return nil, nil, nil, err
	}
	return oldRSes, allOldRSes, newRS, nil
}

// GetOldReplicaSets returns the old replica sets targeted by the given Deployment; get PodList and ReplicaSetList from client interface.
// Note that the first set of old replica sets doesn't include the ones with no pods, and the second set of old replica sets include all old replica sets.
func GetOldReplicaSets(deployment *extensions.Deployment, c clientset.Interface) ([]*extensions.ReplicaSet, []*extensions.ReplicaSet, error) {
	rsList, err := listReplicaSets(deployment, c)
	if err != nil {
		return nil, nil, err
	}
	podList, err := listPods(deployment, c)
	if err != nil {
		return nil, nil, err
	}
	return FindOldReplicaSets(deployment, rsList, podList)
}

// GetNewReplicaSet returns a replica set that matches the intent of the given deployment; get ReplicaSetList from client interface.
// Returns nil if the new replica set doesn't exist yet.
func GetNewReplicaSet(deployment *extensions.Deployment, c clientset.Interface) (*extensions.ReplicaSet, error) {
	rsList, err := listReplicaSets(deployment, c)
	if err != nil {
		return nil, err
	}
	return FindNewReplicaSet(deployment, rsList)
}

// listReplicaSets lists all RSes the given deployment targets with the given client interface.
func listReplicaSets(deployment *extensions.Deployment, c clientset.Interface) ([]*extensions.ReplicaSet, error) {
	return ListReplicaSets(deployment,
		func(namespace string, options api.ListOptions) ([]*extensions.ReplicaSet, error) {
			rsList, err := c.Extensions().ReplicaSets(namespace).List(options)
			if err != nil {
				return nil, err
			}
			ret := []*extensions.ReplicaSet{}
			for i := range rsList.Items {
				ret = append(ret, &rsList.Items[i])
			}
			return ret, err
		})
}

// listReplicaSets lists all Pods the given deployment targets with the given client interface.
func listPods(deployment *extensions.Deployment, c clientset.Interface) (*api.PodList, error) {
	return ListPods(deployment,
		func(namespace string, options api.ListOptions) (*api.PodList, error) {
			return c.Core().Pods(namespace).List(options)
		})
}

// TODO: switch this to full namespacers
type rsListFunc func(string, api.ListOptions) ([]*extensions.ReplicaSet, error)
type podListFunc func(string, api.ListOptions) (*api.PodList, error)

// ListReplicaSets returns a slice of RSes the given deployment targets.
func ListReplicaSets(deployment *extensions.Deployment, getRSList rsListFunc) ([]*extensions.ReplicaSet, error) {
	// TODO: Right now we list replica sets by their labels. We should list them by selector, i.e. the replica set's selector
	//       should be a superset of the deployment's selector, see https://github.com/kubernetes/kubernetes/issues/19830;
	//       or use controllerRef, see https://github.com/kubernetes/kubernetes/issues/2210
	namespace := deployment.Namespace
	selector, err := unversioned.LabelSelectorAsSelector(deployment.Spec.Selector)
	if err != nil {
		return nil, err
	}
	options := api.ListOptions{LabelSelector: selector}
	return getRSList(namespace, options)
}

// ListPods returns a list of pods the given deployment targets.
func ListPods(deployment *extensions.Deployment, getPodList podListFunc) (*api.PodList, error) {
	namespace := deployment.Namespace
	selector, err := unversioned.LabelSelectorAsSelector(deployment.Spec.Selector)
	if err != nil {
		return nil, err
	}
	options := api.ListOptions{LabelSelector: selector}
	return getPodList(namespace, options)
}

// equalIgnoreHash returns true if two given podTemplateSpec are equal, ignoring the diff in value of Labels[pod-template-hash]
// We ignore pod-template-hash because the hash result would be different upon podTemplateSpec API changes
// (e.g. the addition of a new field will cause the hash code to change)
// Note that we assume input podTemplateSpecs contain non-empty labels
func equalIgnoreHash(template1, template2 api.PodTemplateSpec) (bool, error) {
	// First, compare template.Labels (ignoring hash)
	labels1, labels2 := template1.Labels, template2.Labels
	// The podTemplateSpec must have a non-empty label so that label selectors can find them.
	// This is checked by validation (of resources contain a podTemplateSpec).
	if len(labels1) == 0 || len(labels2) == 0 {
		return false, fmt.Errorf("Unexpected empty labels found in given template")
	}
	if len(labels1) > len(labels2) {
		labels1, labels2 = labels2, labels1
	}
	// We make sure len(labels2) >= len(labels1)
	for k, v := range labels2 {
		if labels1[k] != v && k != extensions.DefaultDeploymentUniqueLabelKey {
			return false, nil
		}
	}

	// Then, compare the templates without comparing their labels
	template1.Labels, template2.Labels = nil, nil
	result := api.Semantic.DeepEqual(template1, template2)
	return result, nil
}

// FindNewReplicaSet returns the new RS this given deployment targets (the one with the same pod template).
func FindNewReplicaSet(deployment *extensions.Deployment, rsList []*extensions.ReplicaSet) (*extensions.ReplicaSet, error) {
	newRSTemplate := GetNewReplicaSetTemplate(deployment)
	for i := range rsList {
		equal, err := equalIgnoreHash(rsList[i].Spec.Template, newRSTemplate)
		if err != nil {
			return nil, err
		}
		if equal {
			// This is the new ReplicaSet.
			return rsList[i], nil
		}
	}
	// new ReplicaSet does not exist.
	return nil, nil
}

// FindOldReplicaSets returns the old replica sets targeted by the given Deployment, with the given PodList and slice of RSes.
// Note that the first set of old replica sets doesn't include the ones with no pods, and the second set of old replica sets include all old replica sets.
func FindOldReplicaSets(deployment *extensions.Deployment, rsList []*extensions.ReplicaSet, podList *api.PodList) ([]*extensions.ReplicaSet, []*extensions.ReplicaSet, error) {
	// Find all pods whose labels match deployment.Spec.Selector, and corresponding replica sets for pods in podList.
	// All pods and replica sets are labeled with pod-template-hash to prevent overlapping
	oldRSs := map[string]*extensions.ReplicaSet{}
	allOldRSs := map[string]*extensions.ReplicaSet{}
	newRSTemplate := GetNewReplicaSetTemplate(deployment)
	for _, pod := range podList.Items {
		podLabelsSelector := labels.Set(pod.ObjectMeta.Labels)
		for _, rs := range rsList {
			rsLabelsSelector, err := unversioned.LabelSelectorAsSelector(rs.Spec.Selector)
			if err != nil {
				return nil, nil, fmt.Errorf("invalid label selector: %v", err)
			}
			// Filter out replica set that has the same pod template spec as the deployment - that is the new replica set.
			equal, err := equalIgnoreHash(rs.Spec.Template, newRSTemplate)
			if err != nil {
				return nil, nil, err
			}
			if equal {
				continue
			}
			allOldRSs[rs.ObjectMeta.Name] = rs
			if rsLabelsSelector.Matches(podLabelsSelector) {
				oldRSs[rs.ObjectMeta.Name] = rs
			}
		}
	}
	requiredRSs := []*extensions.ReplicaSet{}
	for key := range oldRSs {
		value := oldRSs[key]
		requiredRSs = append(requiredRSs, value)
	}
	allRSs := []*extensions.ReplicaSet{}
	for key := range allOldRSs {
		value := allOldRSs[key]
		allRSs = append(allRSs, value)
	}
	return requiredRSs, allRSs, nil
}

// WaitForReplicaSetUpdated polls the replica set until it is updated.
func WaitForReplicaSetUpdated(c clientset.Interface, desiredGeneration int64, namespace, name string) error {
	return wait.Poll(10*time.Millisecond, 1*time.Minute, func() (bool, error) {
		rs, err := c.Extensions().ReplicaSets(namespace).Get(name)
		if err != nil {
			return false, err
		}
		return rs.Status.ObservedGeneration >= desiredGeneration, nil
	})
}

// WaitForPodsHashPopulated polls the replica set until updated and fully labeled.
func WaitForPodsHashPopulated(c clientset.Interface, desiredGeneration int64, namespace, name string) error {
	return wait.Poll(1*time.Second, 1*time.Minute, func() (bool, error) {
		rs, err := c.Extensions().ReplicaSets(namespace).Get(name)
		if err != nil {
			return false, err
		}
		return rs.Status.ObservedGeneration >= desiredGeneration &&
			rs.Status.FullyLabeledReplicas == rs.Spec.Replicas, nil
	})
}

// LabelPodsWithHash labels all pods in the given podList with the new hash label.
// The returned bool value can be used to tell if all pods are actually labeled.
func LabelPodsWithHash(podList *api.PodList, rs *extensions.ReplicaSet, c clientset.Interface, namespace, hash string) (bool, error) {
	allPodsLabeled := true
	for _, pod := range podList.Items {
		// Only label the pod that doesn't already have the new hash
		if pod.Labels[extensions.DefaultDeploymentUniqueLabelKey] != hash {
			if _, podUpdated, err := podutil.UpdatePodWithRetries(c.Core().Pods(namespace), &pod,
				func(podToUpdate *api.Pod) error {
					// Precondition: the pod doesn't contain the new hash in its label.
					if podToUpdate.Labels[extensions.DefaultDeploymentUniqueLabelKey] == hash {
						return errors.ErrPreconditionViolated
					}
					podToUpdate.Labels = labelsutil.AddLabel(podToUpdate.Labels, extensions.DefaultDeploymentUniqueLabelKey, hash)
					return nil
				}); err != nil {
				return false, fmt.Errorf("error in adding template hash label %s to pod %+v: %s", hash, pod, err)
			} else if podUpdated {
				glog.V(4).Infof("Labeled %s %s/%s of %s %s/%s with hash %s.", pod.Kind, pod.Namespace, pod.Name, rs.Kind, rs.Namespace, rs.Name, hash)
			} else {
				// If the pod wasn't updated but didn't return error when we try to update it, we've hit "pod not found" or "precondition violated" error.
				// Then we can't say all pods are labeled
				allPodsLabeled = false
			}
		}
	}
	return allPodsLabeled, nil
}

// GetNewReplicaSetTemplate returns the desired PodTemplateSpec for the new ReplicaSet corresponding to the given ReplicaSet.
func GetNewReplicaSetTemplate(deployment *extensions.Deployment) api.PodTemplateSpec {
	// newRS will have the same template as in deployment spec, plus a unique label in some cases.
	newRSTemplate := api.PodTemplateSpec{
		ObjectMeta: deployment.Spec.Template.ObjectMeta,
		Spec:       deployment.Spec.Template.Spec,
	}
	newRSTemplate.ObjectMeta.Labels = labelsutil.CloneAndAddLabel(
		deployment.Spec.Template.ObjectMeta.Labels,
		extensions.DefaultDeploymentUniqueLabelKey,
		podutil.GetPodTemplateSpecHash(newRSTemplate))
	return newRSTemplate
}

// SetFromReplicaSetTemplate sets the desired PodTemplateSpec from a replica set template to the given deployment.
func SetFromReplicaSetTemplate(deployment *extensions.Deployment, template api.PodTemplateSpec) *extensions.Deployment {
	deployment.Spec.Template.ObjectMeta = template.ObjectMeta
	deployment.Spec.Template.Spec = template.Spec
	deployment.Spec.Template.ObjectMeta.Labels = labelsutil.CloneAndRemoveLabel(
		deployment.Spec.Template.ObjectMeta.Labels,
		extensions.DefaultDeploymentUniqueLabelKey)
	return deployment
}

// GetReplicaCountForReplicaSets returns the sum of Replicas of the given replica sets.
func GetReplicaCountForReplicaSets(replicaSets []*extensions.ReplicaSet) int32 {
	totalReplicas := int32(0)
	for _, rs := range replicaSets {
		if rs != nil {
			totalReplicas += rs.Spec.Replicas
		}
	}
	return totalReplicas
}

// GetActualReplicaCountForReplicaSets returns the sum of actual replicas of the given replica sets.
func GetActualReplicaCountForReplicaSets(replicaSets []*extensions.ReplicaSet) int32 {
	totalActualReplicas := int32(0)
	for _, rs := range replicaSets {
		if rs != nil {
			totalActualReplicas += rs.Status.Replicas
		}
	}
	return totalActualReplicas
}

// GetAvailableReplicaCountForReplicaSets returns the number of available pods corresponding to the given replica sets.
func GetAvailableReplicaCountForReplicaSets(replicaSets []*extensions.ReplicaSet) int32 {
	totalAvailableReplicas := int32(0)
	for _, rs := range replicaSets {
		if rs != nil {
			totalAvailableReplicas += rs.Status.AvailableReplicas
		}
	}
	return totalAvailableReplicas
}

// IsPodAvailable return true if the pod is available.
// TODO: Remove this once we start using replica set status for calculating available pods
// for a deployment.
func IsPodAvailable(pod *api.Pod, minReadySeconds int32, now time.Time) bool {
	if !controller.IsPodActive(pod) {
		return false
	}
	// Check if we've passed minReadySeconds since LastTransitionTime
	// If so, this pod is ready
	for _, c := range pod.Status.Conditions {
		// we only care about pod ready conditions
		if c.Type == api.PodReady && c.Status == api.ConditionTrue {
			glog.V(4).Infof("Comparing pod %s/%s ready condition last transition time %s + minReadySeconds %d with now %s.", pod.Namespace, pod.Name, c.LastTransitionTime.String(), minReadySeconds, now.String())
			// 2 cases that this ready condition is valid (passed minReadySeconds, i.e. the pod is available):
			// 1. minReadySeconds == 0, or
			// 2. LastTransitionTime (is set) + minReadySeconds (>0) < current time
			minReadySecondsDuration := time.Duration(minReadySeconds) * time.Second
			if minReadySeconds == 0 || !c.LastTransitionTime.IsZero() && c.LastTransitionTime.Add(minReadySecondsDuration).Before(now) {
				return true
			}
		}
	}
	return false
}

// IsRollingUpdate returns true if the strategy type is a rolling update.
func IsRollingUpdate(deployment *extensions.Deployment) bool {
	return deployment.Spec.Strategy.Type == extensions.RollingUpdateDeploymentStrategyType
}

// DeploymentComplete considers a deployment to be complete once its desired replicas equals its
// updatedReplicas and it doesn't violate minimum availability.
func DeploymentComplete(deployment *extensions.Deployment, newStatus *extensions.DeploymentStatus) bool {
	return newStatus.UpdatedReplicas == deployment.Spec.Replicas &&
		newStatus.AvailableReplicas >= deployment.Spec.Replicas-MaxUnavailable(*deployment)
}

// DeploymentProgressing reports progress for a deployment. Progress is estimated by comparing the
// current with the new status of the deployment that the controller is observing. The following
// algorithm is already used in the kubectl rolling updater to report lack of progress.
func DeploymentProgressing(deployment *extensions.Deployment, newStatus *extensions.DeploymentStatus) bool {
	oldStatus := deployment.Status

	// Old replicas that need to be scaled down
	oldStatusOldReplicas := oldStatus.Replicas - oldStatus.UpdatedReplicas
	newStatusOldReplicas := newStatus.Replicas - newStatus.UpdatedReplicas

	return (newStatus.UpdatedReplicas > oldStatus.UpdatedReplicas) || (newStatusOldReplicas < oldStatusOldReplicas)
}

// used for unit testing
var nowFn = func() time.Time { return time.Now() }

// DeploymentTimedOut considers a deployment to have timed out once its condition that reports progress
// is older than progressDeadlineSeconds or a Progressing condition with a TimedOutReason reason already
// exists.
func DeploymentTimedOut(deployment *extensions.Deployment, newStatus *extensions.DeploymentStatus) bool {
	if deployment.Spec.ProgressDeadlineSeconds == nil {
		return false
	}

	// Look for the Progressing condition. If it doesn't exist, we have no base to estimate progress.
	// If it's already set with a TimedOutReason reason, we have already timed out, no need to check
	// again.
	condition := GetDeploymentCondition(*newStatus, extensions.DeploymentProgressing)
	if condition == nil {
		return false
	}
	if condition.Reason == TimedOutReason {
		return true
	}

	// Look at the difference in seconds between now and the last time we reported any
	// progress or tried to create a replica set, or resumed a paused deployment and
	// compare against progressDeadlineSeconds.
	from := condition.LastUpdateTime
	delta := time.Duration(*deployment.Spec.ProgressDeadlineSeconds) * time.Second
	return from.Add(delta).Before(nowFn())
}

// NewRSNewReplicas calculates the number of replicas a deployment's new RS should have.
// When one of the followings is true, we're rolling out the deployment; otherwise, we're scaling it.
// 1) The new RS is saturated: newRS's replicas == deployment's replicas
// 2) Max number of pods allowed is reached: deployment's replicas + maxSurge == all RSs' replicas
func NewRSNewReplicas(deployment *extensions.Deployment, allRSs []*extensions.ReplicaSet, newRS *extensions.ReplicaSet) (int32, error) {
	switch deployment.Spec.Strategy.Type {
	case extensions.RollingUpdateDeploymentStrategyType:
		// Check if we can scale up.
		maxSurge, err := intstrutil.GetValueFromIntOrPercent(&deployment.Spec.Strategy.RollingUpdate.MaxSurge, int(deployment.Spec.Replicas), true)
		if err != nil {
			return 0, err
		}
		// Find the total number of pods
		currentPodCount := GetReplicaCountForReplicaSets(allRSs)
		maxTotalPods := deployment.Spec.Replicas + int32(maxSurge)
		if currentPodCount >= maxTotalPods {
			// Cannot scale up.
			return newRS.Spec.Replicas, nil
		}
		// Scale up.
		scaleUpCount := maxTotalPods - currentPodCount
		// Do not exceed the number of desired replicas.
		scaleUpCount = int32(integer.IntMin(int(scaleUpCount), int(deployment.Spec.Replicas-newRS.Spec.Replicas)))
		return newRS.Spec.Replicas + scaleUpCount, nil
	case extensions.RecreateDeploymentStrategyType:
		return deployment.Spec.Replicas, nil
	default:
		return 0, fmt.Errorf("deployment type %v isn't supported", deployment.Spec.Strategy.Type)
	}
}

// IsSaturated checks if the new replica set is saturated by comparing its size with its deployment size.
// Both the deployment and the replica set have to believe this replica set can own all of the desired
// replicas in the deployment and the annotation helps in achieving that.
func IsSaturated(deployment *extensions.Deployment, rs *extensions.ReplicaSet) bool {
	if rs == nil {
		return false
	}
	desiredString := rs.Annotations[DesiredReplicasAnnotation]
	desired, err := strconv.Atoi(desiredString)
	if err != nil {
		return false
	}
	return rs.Spec.Replicas == deployment.Spec.Replicas && int32(desired) == deployment.Spec.Replicas
}

// WaitForObservedDeployment polls for deployment to be updated so that deployment.Status.ObservedGeneration >= desiredGeneration.
// Returns error if polling timesout.
func WaitForObservedDeployment(getDeploymentFunc func() (*extensions.Deployment, error), desiredGeneration int64, interval, timeout time.Duration) error {
	// TODO: This should take clientset.Interface when all code is updated to use clientset. Keeping it this way allows the function to be used by callers who have client.Interface.
	return wait.Poll(interval, timeout, func() (bool, error) {
		deployment, err := getDeploymentFunc()
		if err != nil {
			return false, err
		}
		return deployment.Status.ObservedGeneration >= desiredGeneration, nil
	})
}

// ResolveFenceposts resolves both maxSurge and maxUnavailable. This needs to happen in one
// step. For example:
//
// 2 desired, max unavailable 1%, surge 0% - should scale old(-1), then new(+1), then old(-1), then new(+1)
// 1 desired, max unavailable 1%, surge 0% - should scale old(-1), then new(+1)
// 2 desired, max unavailable 25%, surge 1% - should scale new(+1), then old(-1), then new(+1), then old(-1)
// 1 desired, max unavailable 25%, surge 1% - should scale new(+1), then old(-1)
// 2 desired, max unavailable 0%, surge 1% - should scale new(+1), then old(-1), then new(+1), then old(-1)
// 1 desired, max unavailable 0%, surge 1% - should scale new(+1), then old(-1)
func ResolveFenceposts(maxSurge, maxUnavailable *intstrutil.IntOrString, desired int32) (int32, int32, error) {
	surge, err := intstrutil.GetValueFromIntOrPercent(maxSurge, int(desired), true)
	if err != nil {
		return 0, 0, err
	}
	unavailable, err := intstrutil.GetValueFromIntOrPercent(maxUnavailable, int(desired), false)
	if err != nil {
		return 0, 0, err
	}

	if surge == 0 && unavailable == 0 {
		// Validation should never allow the user to explicitly use zero values for both maxSurge
		// maxUnavailable. Due to rounding down maxUnavailable though, it may resolve to zero.
		// If both fenceposts resolve to zero, then we should set maxUnavailable to 1 on the
		// theory that surge might not work due to quota.
		unavailable = 1
	}

	return int32(surge), int32(unavailable), nil
}

func DeploymentDeepCopy(deployment *extensions.Deployment) (*extensions.Deployment, error) {
	objCopy, err := api.Scheme.DeepCopy(deployment)
	if err != nil {
		return nil, err
	}
	copied, ok := objCopy.(*extensions.Deployment)
	if !ok {
		return nil, fmt.Errorf("expected Deployment, got %#v", objCopy)
	}
	return copied, nil
}

// SelectorUpdatedBefore returns true if the former deployment's selector
// is updated before the latter, false otherwise
func SelectorUpdatedBefore(d1, d2 *extensions.Deployment) bool {
	t1, t2 := LastSelectorUpdate(d1), LastSelectorUpdate(d2)
	return t1.Before(t2)
}

// LastSelectorUpdate returns the last time given deployment's selector is updated
func LastSelectorUpdate(d *extensions.Deployment) unversioned.Time {
	t := d.Annotations[SelectorUpdateAnnotation]
	if len(t) > 0 {
		parsedTime, err := time.Parse(t, time.RFC3339)
		// If failed to parse the time, use creation timestamp instead
		if err != nil {
			return d.CreationTimestamp
		}
		return unversioned.Time{Time: parsedTime}
	}
	// If it's never updated, use creation timestamp instead
	return d.CreationTimestamp
}

// BySelectorLastUpdateTime sorts a list of deployments by the last update time of their selector,
// first using their creation timestamp and then their names as a tie breaker.
type BySelectorLastUpdateTime []*extensions.Deployment

func (o BySelectorLastUpdateTime) Len() int      { return len(o) }
func (o BySelectorLastUpdateTime) Swap(i, j int) { o[i], o[j] = o[j], o[i] }
func (o BySelectorLastUpdateTime) Less(i, j int) bool {
	ti, tj := LastSelectorUpdate(o[i]), LastSelectorUpdate(o[j])
	if ti.Equal(tj) {
		if o[i].CreationTimestamp.Equal(o[j].CreationTimestamp) {
			return o[i].Name < o[j].Name
		}
		return o[i].CreationTimestamp.Before(o[j].CreationTimestamp)
	}
	return ti.Before(tj)
}

// OverlapsWith returns true when two given deployments are different and overlap with each other
func OverlapsWith(current, other *extensions.Deployment) (bool, error) {
	if current.UID == other.UID {
		return false, nil
	}
	currentSelector, err := unversioned.LabelSelectorAsSelector(current.Spec.Selector)
	if err != nil {
		return false, fmt.Errorf("deployment %s/%s has invalid label selector: %v", current.Namespace, current.Name, err)
	}
	otherSelector, err := unversioned.LabelSelectorAsSelector(other.Spec.Selector)
	if err != nil {
		// Broken selectors from other deployments shouldn't block current deployment. Just log the error and continue.
		glog.V(2).Infof("Skip overlapping check: deployment %s/%s has invalid label selector: %v", other.Namespace, other.Name, err)
		return false, nil
	}
	return (!currentSelector.Empty() && currentSelector.Matches(labels.Set(other.Spec.Template.Labels))) ||
		(!otherSelector.Empty() && otherSelector.Matches(labels.Set(current.Spec.Template.Labels))), nil
}
