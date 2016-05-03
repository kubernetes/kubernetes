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

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/integer"
	intstrutil "k8s.io/kubernetes/pkg/util/intstr"
	labelsutil "k8s.io/kubernetes/pkg/util/labels"
	podutil "k8s.io/kubernetes/pkg/util/pod"
	"k8s.io/kubernetes/pkg/util/wait"
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
func GetOldReplicaSets(deployment *extensions.Deployment, c clientset.Interface) ([]*extensions.ReplicaSet, []*extensions.ReplicaSet, error) {
	rsList, err := ListReplicaSets(deployment,
		func(namespace string, options api.ListOptions) ([]extensions.ReplicaSet, error) {
			rsList, err := c.Extensions().ReplicaSets(namespace).List(options)
			return rsList.Items, err
		})
	if err != nil {
		return nil, nil, fmt.Errorf("error listing ReplicaSets: %v", err)
	}
	podList, err := ListPods(deployment,
		func(namespace string, options api.ListOptions) (*api.PodList, error) {
			return c.Core().Pods(namespace).List(options)
		})
	if err != nil {
		return nil, nil, fmt.Errorf("error listing Pods: %v", err)
	}
	return FindOldReplicaSets(deployment, rsList, podList)
}

// GetNewReplicaSet returns a replica set that matches the intent of the given deployment; get ReplicaSetList from client interface.
// Returns nil if the new replica set doesn't exist yet.
func GetNewReplicaSet(deployment *extensions.Deployment, c clientset.Interface) (*extensions.ReplicaSet, error) {
	rsList, err := ListReplicaSets(deployment,
		func(namespace string, options api.ListOptions) ([]extensions.ReplicaSet, error) {
			rsList, err := c.Extensions().ReplicaSets(namespace).List(options)
			return rsList.Items, err
		})
	if err != nil {
		return nil, fmt.Errorf("error listing ReplicaSets: %v", err)
	}
	return FindNewReplicaSet(deployment, rsList)
}

// TODO: switch this to full namespacers
type rsListFunc func(string, api.ListOptions) ([]extensions.ReplicaSet, error)
type podListFunc func(string, api.ListOptions) (*api.PodList, error)

// ListReplicaSets returns a slice of RSes the given deployment targets.
func ListReplicaSets(deployment *extensions.Deployment, getRSList rsListFunc) ([]extensions.ReplicaSet, error) {
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

// FindNewReplicaSet returns the new RS this given deployment targets (the one with the same pod template).
func FindNewReplicaSet(deployment *extensions.Deployment, rsList []extensions.ReplicaSet) (*extensions.ReplicaSet, error) {
	newRSTemplate := GetNewReplicaSetTemplate(deployment)
	for i := range rsList {
		if api.Semantic.DeepEqual(rsList[i].Spec.Template, newRSTemplate) {
			// This is the new ReplicaSet.
			return &rsList[i], nil
		}
	}
	// new ReplicaSet does not exist.
	return nil, nil
}

// FindOldReplicaSets returns the old replica sets targeted by the given Deployment, with the given PodList and slice of RSes.
// Note that the first set of old replica sets doesn't include the ones with no pods, and the second set of old replica sets include all old replica sets.
func FindOldReplicaSets(deployment *extensions.Deployment, rsList []extensions.ReplicaSet, podList *api.PodList) ([]*extensions.ReplicaSet, []*extensions.ReplicaSet, error) {
	// Find all pods whose labels match deployment.Spec.Selector, and corresponding replica sets for pods in podList.
	// All pods and replica sets are labeled with pod-template-hash to prevent overlapping
	oldRSs := map[string]extensions.ReplicaSet{}
	allOldRSs := map[string]extensions.ReplicaSet{}
	newRSTemplate := GetNewReplicaSetTemplate(deployment)
	for _, pod := range podList.Items {
		podLabelsSelector := labels.Set(pod.ObjectMeta.Labels)
		for _, rs := range rsList {
			rsLabelsSelector, err := unversioned.LabelSelectorAsSelector(rs.Spec.Selector)
			if err != nil {
				return nil, nil, fmt.Errorf("invalid label selector: %v", err)
			}
			// Filter out replica set that has the same pod template spec as the deployment - that is the new replica set.
			if api.Semantic.DeepEqual(rs.Spec.Template, newRSTemplate) {
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

func WaitForReplicaSetUpdated(c clientset.Interface, desiredGeneration int64, namespace, name string) error {
	return wait.Poll(10*time.Millisecond, 1*time.Minute, func() (bool, error) {
		rs, err := c.Extensions().ReplicaSets(namespace).Get(name)
		if err != nil {
			return false, err
		}
		return rs.Status.ObservedGeneration >= desiredGeneration, nil
	})
}

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

// Returns the desired PodTemplateSpec for the new ReplicaSet corresponding to the given ReplicaSet.
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

// Returns the sum of Replicas of the given replica sets.
func GetReplicaCountForReplicaSets(replicaSets []*extensions.ReplicaSet) int32 {
	totalReplicaCount := int32(0)
	for _, rs := range replicaSets {
		if rs != nil {
			totalReplicaCount += rs.Spec.Replicas
		}
	}
	return totalReplicaCount
}

// GetActualReplicaCountForReplicaSets returns the sum of actual replicas of the given replica sets.
func GetActualReplicaCountForReplicaSets(replicaSets []*extensions.ReplicaSet) int32 {
	totalReplicaCount := int32(0)
	for _, rs := range replicaSets {
		if rs != nil {
			totalReplicaCount += rs.Status.Replicas
		}
	}
	return totalReplicaCount
}

// Returns the number of available pods corresponding to the given replica sets.
func GetAvailablePodsForReplicaSets(c clientset.Interface, rss []*extensions.ReplicaSet, minReadySeconds int32) (int32, error) {
	allPods, err := GetPodsForReplicaSets(c, rss)
	if err != nil {
		return 0, err
	}
	return getReadyPodsCount(allPods, minReadySeconds), nil
}

func getReadyPodsCount(pods []api.Pod, minReadySeconds int32) int32 {
	readyPodCount := int32(0)
	for _, pod := range pods {
		if IsPodAvailable(&pod, minReadySeconds) {
			readyPodCount++
		}
	}
	return readyPodCount
}

func IsPodAvailable(pod *api.Pod, minReadySeconds int32) bool {
	if !controller.IsPodActive(*pod) {
		return false
	}
	// Check if we've passed minReadySeconds since LastTransitionTime
	// If so, this pod is ready
	for _, c := range pod.Status.Conditions {
		// we only care about pod ready conditions
		if c.Type == api.PodReady && c.Status == api.ConditionTrue {
			// 2 cases that this ready condition is valid (passed minReadySeconds, i.e. the pod is available):
			// 1. minReadySeconds == 0, or
			// 2. LastTransitionTime (is set) + minReadySeconds (>0) < current time
			minReadySecondsDuration := time.Duration(minReadySeconds) * time.Second
			if minReadySeconds == 0 || !c.LastTransitionTime.IsZero() && c.LastTransitionTime.Add(minReadySecondsDuration).Before(time.Now()) {
				return true
			}
		}
	}
	return false
}

func GetPodsForReplicaSets(c clientset.Interface, replicaSets []*extensions.ReplicaSet) ([]api.Pod, error) {
	allPods := map[string]api.Pod{}
	for _, rs := range replicaSets {
		if rs != nil {
			selector, err := unversioned.LabelSelectorAsSelector(rs.Spec.Selector)
			if err != nil {
				return nil, fmt.Errorf("invalid label selector: %v", err)
			}
			options := api.ListOptions{LabelSelector: selector}
			podList, err := c.Core().Pods(rs.ObjectMeta.Namespace).List(options)
			if err != nil {
				return nil, fmt.Errorf("error listing pods: %v", err)
			}
			for _, pod := range podList.Items {
				allPods[pod.Name] = pod
			}
		}
	}
	requiredPods := []api.Pod{}
	for _, pod := range allPods {
		requiredPods = append(requiredPods, pod)
	}
	return requiredPods, nil
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

// Polls for deployment to be updated so that deployment.Status.ObservedGeneration >= desiredGeneration.
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
