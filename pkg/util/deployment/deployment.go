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

package deployment

import (
	"fmt"
	"strconv"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/integer"
	intstrutil "k8s.io/kubernetes/pkg/util/intstr"
	labelsutil "k8s.io/kubernetes/pkg/util/labels"
	podutil "k8s.io/kubernetes/pkg/util/pod"
)

const (
	// The revision annotation of a deployment's replica sets which records its rollout sequence
	RevisionAnnotation = "deployment.kubernetes.io/revision"

	// Here are the possible rollback event reasons
	RollbackRevisionNotFound  = "DeploymentRollbackRevisionNotFound"
	RollbackTemplateUnchanged = "DeploymentRollbackTemplateUnchanged"
	RollbackDone              = "DeploymentRollback"
)

// GetOldReplicaSets returns the old replica sets targeted by the given Deployment; get PodList and ReplicaSetList from client interface.
// Note that the first set of old replica sets doesn't include the ones with no pods, and the second set of old replica sets include all old replica sets.
func GetOldReplicaSets(deployment extensions.Deployment, c clientset.Interface) ([]*extensions.ReplicaSet, []*extensions.ReplicaSet, error) {
	return GetOldReplicaSetsFromLists(deployment, c,
		func(namespace string, options api.ListOptions) (*api.PodList, error) {
			return c.Core().Pods(namespace).List(options)
		},
		func(namespace string, options api.ListOptions) ([]extensions.ReplicaSet, error) {
			rsList, err := c.Extensions().ReplicaSets(namespace).List(options)
			return rsList.Items, err
		})
}

// GetOldReplicaSetsFromLists returns two sets of old replica sets targeted by the given Deployment; get PodList and ReplicaSetList with input functions.
// Note that the first set of old replica sets doesn't include the ones with no pods, and the second set of old replica sets include all old replica sets.
func GetOldReplicaSetsFromLists(deployment extensions.Deployment, c clientset.Interface, getPodList func(string, api.ListOptions) (*api.PodList, error), getRSList func(string, api.ListOptions) ([]extensions.ReplicaSet, error)) ([]*extensions.ReplicaSet, []*extensions.ReplicaSet, error) {
	namespace := deployment.ObjectMeta.Namespace
	selector, err := unversioned.LabelSelectorAsSelector(deployment.Spec.Selector)
	if err != nil {
		return nil, nil, fmt.Errorf("invalid label selector: %v", err)
	}

	// 1. Find all pods whose labels match deployment.Spec.Selector
	options := api.ListOptions{LabelSelector: selector}
	podList, err := getPodList(namespace, options)
	if err != nil {
		return nil, nil, fmt.Errorf("error listing pods: %v", err)
	}
	// 2. Find the corresponding replica sets for pods in podList.
	// TODO: Right now we list all replica sets and then filter. We should add an API for this.
	oldRSs := map[string]extensions.ReplicaSet{}
	allOldRSs := map[string]extensions.ReplicaSet{}
	rsList, err := getRSList(namespace, options)
	if err != nil {
		return nil, nil, fmt.Errorf("error listing replica sets: %v", err)
	}
	newRSTemplate := GetNewReplicaSetTemplate(deployment)
	for _, pod := range podList.Items {
		podLabelsSelector := labels.Set(pod.ObjectMeta.Labels)
		for _, rs := range rsList {
			rsLabelsSelector, err := unversioned.LabelSelectorAsSelector(rs.Spec.Selector)
			if err != nil {
				return nil, nil, fmt.Errorf("invalid label selector: %v", err)
			}
			// Filter out replica set that has the same pod template spec as the deployment - that is the new replica set.
			if api.Semantic.DeepEqual(rs.Spec.Template, &newRSTemplate) {
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
		requiredRSs = append(requiredRSs, &value)
	}
	allRSs := []*extensions.ReplicaSet{}
	for key := range allOldRSs {
		value := allOldRSs[key]
		allRSs = append(allRSs, &value)
	}
	return requiredRSs, allRSs, nil
}

// GetNewReplicaSet returns a replica set that matches the intent of the given deployment; get ReplicaSetList from client interface.
// Returns nil if the new replica set doesn't exist yet.
func GetNewReplicaSet(deployment extensions.Deployment, c clientset.Interface) (*extensions.ReplicaSet, error) {
	return GetNewReplicaSetFromList(deployment, c,
		func(namespace string, options api.ListOptions) ([]extensions.ReplicaSet, error) {
			rsList, err := c.Extensions().ReplicaSets(namespace).List(options)
			return rsList.Items, err
		})
}

// GetNewReplicaSetFromList returns a replica set that matches the intent of the given deployment; get ReplicaSetList with the input function.
// Returns nil if the new replica set doesn't exist yet.
func GetNewReplicaSetFromList(deployment extensions.Deployment, c clientset.Interface, getRSList func(string, api.ListOptions) ([]extensions.ReplicaSet, error)) (*extensions.ReplicaSet, error) {
	namespace := deployment.ObjectMeta.Namespace
	selector, err := unversioned.LabelSelectorAsSelector(deployment.Spec.Selector)
	if err != nil {
		return nil, fmt.Errorf("invalid label selector: %v", err)
	}

	rsList, err := getRSList(namespace, api.ListOptions{LabelSelector: selector})
	if err != nil {
		return nil, fmt.Errorf("error listing ReplicaSets: %v", err)
	}
	newRSTemplate := GetNewReplicaSetTemplate(deployment)

	for i := range rsList {
		if api.Semantic.DeepEqual(rsList[i].Spec.Template, &newRSTemplate) {
			// This is the new ReplicaSet.
			return &rsList[i], nil
		}
	}
	// new ReplicaSet does not exist.
	return nil, nil
}

// Returns the desired PodTemplateSpec for the new ReplicaSet corresponding to the given ReplicaSet.
func GetNewReplicaSetTemplate(deployment extensions.Deployment) api.PodTemplateSpec {
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

// Returns the sum of Replicas of the given replica sets.
func GetReplicaCountForReplicaSets(replicaSets []*extensions.ReplicaSet) int {
	totalReplicaCount := 0
	for _, rs := range replicaSets {
		totalReplicaCount += rs.Spec.Replicas
	}
	return totalReplicaCount
}

// Returns the number of available pods corresponding to the given replica sets.
func GetAvailablePodsForReplicaSets(c clientset.Interface, rss []*extensions.ReplicaSet, minReadySeconds int) (int, error) {
	allPods, err := GetPodsForReplicaSets(c, rss)
	if err != nil {
		return 0, err
	}
	return getReadyPodsCount(allPods, minReadySeconds), nil
}

func getReadyPodsCount(pods []api.Pod, minReadySeconds int) int {
	readyPodCount := 0
	for _, pod := range pods {
		if IsPodAvailable(&pod, minReadySeconds) {
			readyPodCount++
		}
	}
	return readyPodCount
}

func IsPodAvailable(pod *api.Pod, minReadySeconds int) bool {
	// Check if we've passed minReadySeconds since LastTransitionTime
	// If so, this pod is ready
	for _, c := range pod.Status.Conditions {
		// we only care about pod ready conditions
		if c.Type == api.PodReady && c.Status == api.ConditionTrue {
			// 2 cases that this ready condition is valid (passed minReadySeconds, i.e. the pod is ready):
			// 1. minReadySeconds <= 0
			// 2. LastTransitionTime (is set) + minReadySeconds (>0) < current time
			minReadySecondsDuration := time.Duration(minReadySeconds) * time.Second
			if minReadySeconds <= 0 || !c.LastTransitionTime.IsZero() && c.LastTransitionTime.Add(minReadySecondsDuration).Before(time.Now()) {
				return true
			}
		}
	}
	return false
}

func GetPodsForReplicaSets(c clientset.Interface, replicaSets []*extensions.ReplicaSet) ([]api.Pod, error) {
	allPods := []api.Pod{}
	for _, rs := range replicaSets {
		selector, err := unversioned.LabelSelectorAsSelector(rs.Spec.Selector)
		if err != nil {
			return nil, fmt.Errorf("invalid label selector: %v", err)
		}
		options := api.ListOptions{LabelSelector: selector}
		podList, err := c.Core().Pods(rs.ObjectMeta.Namespace).List(options)
		if err != nil {
			return allPods, fmt.Errorf("error listing pods: %v", err)
		}
		allPods = append(allPods, podList.Items...)
	}
	return allPods, nil
}

// Revision returns the revision number of the input replica set
func Revision(rs *extensions.ReplicaSet) (int64, error) {
	v, ok := rs.Annotations[RevisionAnnotation]
	if !ok {
		return 0, nil
	}
	return strconv.ParseInt(v, 10, 64)
}

func IsRollingUpdate(deployment *extensions.Deployment) bool {
	return deployment.Spec.Strategy.Type == extensions.RollingUpdateDeploymentStrategyType
}

// NewRSNewReplicas calculates the number of replicas a deployment's new RS should have.
// When one of the followings is true, we're rolling out the deployment; otherwise, we're scaling it.
// 1) The new RS is saturated: newRS's replicas == deployment's replicas
// 2) Max number of pods allowed is reached: deployment's replicas + maxSurge == all RSs' replicas
func NewRSNewReplicas(deployment *extensions.Deployment, allRSs []*extensions.ReplicaSet, newRS *extensions.ReplicaSet) (int, error) {
	switch deployment.Spec.Strategy.Type {
	case extensions.RollingUpdateDeploymentStrategyType:
		// Check if we can scale up.
		maxSurge, err := intstrutil.GetValueFromIntOrPercent(&deployment.Spec.Strategy.RollingUpdate.MaxSurge, deployment.Spec.Replicas)
		if err != nil {
			return 0, err
		}
		// Find the total number of pods
		currentPodCount := GetReplicaCountForReplicaSets(allRSs)
		maxTotalPods := deployment.Spec.Replicas + maxSurge
		if currentPodCount >= maxTotalPods {
			// Cannot scale up.
			return newRS.Spec.Replicas, nil
		}
		// Scale up.
		scaleUpCount := maxTotalPods - currentPodCount
		// Do not exceed the number of desired replicas.
		scaleUpCount = integer.IntMin(scaleUpCount, deployment.Spec.Replicas-newRS.Spec.Replicas)
		return newRS.Spec.Replicas + scaleUpCount, nil
	case extensions.RecreateDeploymentStrategyType:
		return deployment.Spec.Replicas, nil
	default:
		return 0, fmt.Errorf("deployment type %v isn't supported", deployment.Spec.Strategy.Type)
	}
}
