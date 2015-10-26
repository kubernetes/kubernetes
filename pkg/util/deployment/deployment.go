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
	"hash/adler32"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"
)

// Returns the old RCs targetted by the given Deployment.
func GetOldRCs(deployment extensions.Deployment, c client.Interface) ([]*api.ReplicationController, error) {
	namespace := deployment.ObjectMeta.Namespace
	// 1. Find all pods whose labels match deployment.Spec.Selector
	podList, err := c.Pods(namespace).List(labels.SelectorFromSet(deployment.Spec.Selector), fields.Everything())
	if err != nil {
		return nil, fmt.Errorf("error listing pods: %v", err)
	}
	// 2. Find the corresponding RCs for pods in podList.
	// TODO: Right now we list all RCs and then filter. We should add an API for this.
	oldRCs := map[string]api.ReplicationController{}
	rcList, err := c.ReplicationControllers(namespace).List(labels.Everything(), fields.Everything())
	if err != nil {
		return nil, fmt.Errorf("error listing replication controllers: %v", err)
	}
	for _, pod := range podList.Items {
		podLabelsSelector := labels.Set(pod.ObjectMeta.Labels)
		for _, rc := range rcList.Items {
			rcLabelsSelector := labels.SelectorFromSet(rc.Spec.Selector)
			if rcLabelsSelector.Matches(podLabelsSelector) {
				// Filter out RC that has the same pod template spec as the deployment - that is the new RC.
				if api.Semantic.DeepEqual(rc.Spec.Template, GetNewRCTemplate(deployment)) {
					continue
				}
				oldRCs[rc.ObjectMeta.Name] = rc
			}
		}
	}
	requiredRCs := []*api.ReplicationController{}
	for _, value := range oldRCs {
		requiredRCs = append(requiredRCs, &value)
	}
	return requiredRCs, nil
}

// Returns an RC that matches the intent of the given deployment.
// Returns nil if the new RC doesnt exist yet.
func GetNewRC(deployment extensions.Deployment, c client.Interface) (*api.ReplicationController, error) {
	namespace := deployment.ObjectMeta.Namespace
	rcList, err := c.ReplicationControllers(namespace).List(labels.Everything(), fields.Everything())
	if err != nil {
		return nil, fmt.Errorf("error listing replication controllers: %v", err)
	}
	newRCTemplate := GetNewRCTemplate(deployment)

	for _, rc := range rcList.Items {
		if api.Semantic.DeepEqual(rc.Spec.Template, newRCTemplate) {
			// This is the new RC.
			return &rc, nil
		}
	}
	// new RC does not exist.
	return nil, nil
}

// Returns the desired PodTemplateSpec for the new RC corresponding to the given RC.
func GetNewRCTemplate(deployment extensions.Deployment) *api.PodTemplateSpec {
	// newRC will have the same template as in deployment spec, plus a unique label in some cases.
	newRCTemplate := &api.PodTemplateSpec{
		ObjectMeta: deployment.Spec.Template.ObjectMeta,
		Spec:       deployment.Spec.Template.Spec,
	}
	newRCTemplate.ObjectMeta.Labels = CloneAndAddLabel(
		deployment.Spec.Template.ObjectMeta.Labels,
		deployment.Spec.UniqueLabelKey,
		GetPodTemplateSpecHash(newRCTemplate))
	return newRCTemplate
}

// Clones the given map and returns a new map with the given key and value added.
// Returns the given map, if labelKey is empty.
func CloneAndAddLabel(labels map[string]string, labelKey string, labelValue uint32) map[string]string {
	if labelKey == "" {
		// Dont need to add a label.
		return labels
	}
	// Clone.
	newLabels := map[string]string{}
	for key, value := range labels {
		newLabels[key] = value
	}
	newLabels[labelKey] = fmt.Sprintf("%d", labelValue)
	return newLabels
}

func GetPodTemplateSpecHash(template *api.PodTemplateSpec) uint32 {
	podTemplateSpecHasher := adler32.New()
	util.DeepHashObject(podTemplateSpecHasher, template)
	return podTemplateSpecHasher.Sum32()
}

// Returns the sum of Replicas of the given replication controllers.
func GetReplicaCountForRCs(replicationControllers []*api.ReplicationController) int {
	totalReplicaCount := 0
	for _, rc := range replicationControllers {
		totalReplicaCount += rc.Spec.Replicas
	}
	return totalReplicaCount
}

// Returns the number of available pods corresponding to the given RCs.
func GetAvailablePodsForRCs(c client.Interface, rcs []*api.ReplicationController) (int, error) {
	// TODO: Use MinReadySeconds once https://github.com/kubernetes/kubernetes/pull/12894 is merged.
	allPods, err := getPodsForRCs(c, rcs)
	if err != nil {
		return 0, err
	}
	readyPodCount := 0
	for _, pod := range allPods {
		if api.IsPodReady(&pod) {
			readyPodCount++
		}
	}
	return readyPodCount, nil
}

func getPodsForRCs(c client.Interface, replicationControllers []*api.ReplicationController) ([]api.Pod, error) {
	allPods := []api.Pod{}
	for _, rc := range replicationControllers {
		podList, err := c.Pods(rc.ObjectMeta.Namespace).List(labels.SelectorFromSet(rc.Spec.Selector), fields.Everything())
		if err != nil {
			return allPods, fmt.Errorf("error listing pods: %v", err)
		}
		allPods = append(allPods, podList.Items...)
	}
	return allPods, nil
}
