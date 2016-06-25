/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"sort"
	"strconv"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/controller"
	deploymentutil "k8s.io/kubernetes/pkg/util/deployment"
	"k8s.io/kubernetes/pkg/util/integer"
)

// findActiveOrLatest returns the only active or the latest replica set in case there is at most one active
// replica set. If there are more active replica sets, then we should proportionally scale them.
func findActiveOrLatest(newRS *extensions.ReplicaSet, oldRSs []*extensions.ReplicaSet) *extensions.ReplicaSet {
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

func getDesiredReplicasAnnotation(rs *extensions.ReplicaSet) (int32, bool) {
	return getIntFromAnnotation(rs, deploymentutil.DesiredReplicasAnnotation)
}

func getMaxReplicasAnnotation(rs *extensions.ReplicaSet) (int32, bool) {
	return getIntFromAnnotation(rs, deploymentutil.MaxReplicasAnnotation)
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

func setReplicasAnnotations(rs *extensions.ReplicaSet, desiredReplicas, maxReplicas int32) bool {
	updated := false
	if rs.Annotations == nil {
		rs.Annotations = make(map[string]string)
	}
	desiredString := fmt.Sprintf("%d", desiredReplicas)
	if hasString := rs.Annotations[deploymentutil.DesiredReplicasAnnotation]; hasString != desiredString {
		rs.Annotations[deploymentutil.DesiredReplicasAnnotation] = desiredString
		updated = true
	}
	maxString := fmt.Sprintf("%d", maxReplicas)
	if hasString := rs.Annotations[deploymentutil.MaxReplicasAnnotation]; hasString != maxString {
		rs.Annotations[deploymentutil.MaxReplicasAnnotation] = maxString
		updated = true
	}
	return updated
}

// maxUnavailable returns the maximum unavailable pods a rolling deployment can take.
func maxUnavailable(deployment extensions.Deployment) int32 {
	if !deploymentutil.IsRollingUpdate(&deployment) {
		return int32(0)
	}
	// Error caught by validation
	_, maxUnavailable, _ := deploymentutil.ResolveFenceposts(&deployment.Spec.Strategy.RollingUpdate.MaxSurge, &deployment.Spec.Strategy.RollingUpdate.MaxUnavailable, deployment.Spec.Replicas)
	return maxUnavailable
}

// maxSurge returns the maximum surge pods a rolling deployment can take.
func maxSurge(deployment extensions.Deployment) int32 {
	if !deploymentutil.IsRollingUpdate(&deployment) {
		return int32(0)
	}
	// Error caught by validation
	maxSurge, _, _ := deploymentutil.ResolveFenceposts(&deployment.Spec.Strategy.RollingUpdate.MaxSurge, &deployment.Spec.Strategy.RollingUpdate.MaxUnavailable, deployment.Spec.Replicas)
	return maxSurge
}

// getProportion will estimate the proportion for the provided replica set using 1. the current size
// of the parent deployment, 2. the replica count that needs be added on the replica sets of the
// deployment, and 3. the total replicas added in the replica sets of the deployment so far.
func getProportion(rs *extensions.ReplicaSet, d extensions.Deployment, deploymentReplicasToAdd, deploymentReplicasAdded int32) int32 {
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

	deploymentReplicas := d.Spec.Replicas + maxSurge(d)
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
