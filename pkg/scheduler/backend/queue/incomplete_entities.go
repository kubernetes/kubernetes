/*
Copyright The Kubernetes Authors.

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

package queue

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// incompleteEntities stores pod infos that wait for their corresponding PodGroup object to be observed.
type incompleteEntities struct {
	// Key is the pod group key: namespace/pod-group-name
	// TODO(CompositePodGroup): For multi-level hierarchies, pods will be keyed by the highest-level (C)PG key?
	podGroupToPodInfos map[string][]*framework.QueuedPodInfo
}

func newIncompleteEntities() *incompleteEntities {
	return &incompleteEntities{
		podGroupToPodInfos: make(map[string][]*framework.QueuedPodInfo),
	}
}

// add adds a pod info waiting for the pod group.
// TODO(CompositePodGroup): Pass the highest-level (C)PG key as well?
func (p *incompleteEntities) add(pInfo *framework.QueuedPodInfo) {
	key := podGroupKeyForPod(pInfo.Pod)
	if key == "" {
		return
	}
	p.podGroupToPodInfos[key] = append(p.podGroupToPodInfos[key], pInfo)
}

// get returns the pod infos waiting for the pod group.
func (p *incompleteEntities) getPod(pod *v1.Pod) *framework.QueuedPodInfo {
	key := podGroupKeyForPod(pod)
	for _, pInfo := range p.podGroupToPodInfos[key] {
		if pInfo.Pod.Name == pod.Name && pInfo.Pod.Namespace == pod.Namespace {
			return pInfo
		}
	}
	return nil
}

// update updates the pod inside the incomplete entities.
// It returns the updated pod info if the pod was found, nil otherwise.
func (p *incompleteEntities) update(pod *v1.Pod) *framework.QueuedPodInfo {
	key := podGroupKeyForPod(pod)
	pInfos, ok := p.podGroupToPodInfos[key]
	if !ok {
		return nil
	}
	for _, pInfo := range pInfos {
		if pInfo.Pod.Name == pod.Name && pInfo.Pod.Namespace == pod.Namespace {
			pInfo.Pod = pod
			return pInfo
		}
	}
	return nil
}

// delete removes a specific pod and returns its QueuedPodInfo.
// If the pod list for the group becomes empty, it cleans up the key.
func (p *incompleteEntities) delete(pod *v1.Pod) *framework.QueuedPodInfo {
	key := podGroupKeyForPod(pod)
	pInfos, ok := p.podGroupToPodInfos[key]
	if !ok {
		return nil
	}
	for i, pInfo := range pInfos {
		if pInfo.Pod.UID == pod.UID {
			p.podGroupToPodInfos[key] = append(pInfos[:i], pInfos[i+1:]...)
			if len(p.podGroupToPodInfos[key]) == 0 {
				delete(p.podGroupToPodInfos, key)
			}
			return pInfo
		}
	}
	return nil
}

// clear removes and returns all pod infos waiting for the pod group.
func (p *incompleteEntities) clear(podGroup *schedulingv1alpha3.PodGroup) []*framework.QueuedPodInfo {
	pgKey := podGroupKey(podGroup)
	if pods, ok := p.podGroupToPodInfos[pgKey]; ok {
		delete(p.podGroupToPodInfos, pgKey)
		return pods
	}
	return nil
}

func podGroupKeyForPod(pod *v1.Pod) string {
	return fmt.Sprintf("%s/%s", pod.Namespace, *pod.Spec.SchedulingGroup.PodGroupName)
}

func podGroupKey(podGroup *schedulingv1alpha3.PodGroup) string {
	return fmt.Sprintf("%s/%s", podGroup.Namespace, podGroup.Name)
}
