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
	// Key is the pod group key: type/namespace/pod-group-name
	podGroupToPodInfos map[string][]*framework.QueuedPodInfo
}

func newIncompleteEntities() *incompleteEntities {
	return &incompleteEntities{
		podGroupToPodInfos: make(map[string][]*framework.QueuedPodInfo),
	}
}

// add adds a pod info waiting for the pod group.
func (p *incompleteEntities) add(pInfo *framework.QueuedPodInfo, highestLevelAncestorKey string) {
	if highestLevelAncestorKey == "" {
		return
	}
	p.podGroupToPodInfos[highestLevelAncestorKey] = append(p.podGroupToPodInfos[highestLevelAncestorKey], pInfo)
}

// get returns the pod infos waiting for the pod group.
func (p *incompleteEntities) getPod(pod *v1.Pod, highestLevelAncestorKey string) *framework.QueuedPodInfo {
	if highestLevelAncestorKey == "" {
		return nil
	}
	for _, pInfo := range p.podGroupToPodInfos[highestLevelAncestorKey] {
		if pInfo.Pod.Name == pod.Name && pInfo.Pod.Namespace == pod.Namespace {
			return pInfo
		}
	}
	return nil
}

// update updates the pod inside the incomplete entities.
// It returns the updated pod info if the pod was found, nil otherwise.
func (p *incompleteEntities) update(pod *v1.Pod, highestLevelAncestorKey string) *framework.QueuedPodInfo {
	if highestLevelAncestorKey == "" {
		return nil
	}
	pInfos, ok := p.podGroupToPodInfos[highestLevelAncestorKey]
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
func (p *incompleteEntities) delete(pod *v1.Pod, highestLevelAncestorKey string) *framework.QueuedPodInfo {
	if highestLevelAncestorKey == "" {
		return nil
	}
	pInfos, ok := p.podGroupToPodInfos[highestLevelAncestorKey]
	if !ok {
		return nil
	}
	for i, pInfo := range pInfos {
		if pInfo.Pod.UID == pod.UID {
			p.podGroupToPodInfos[highestLevelAncestorKey] = append(pInfos[:i], pInfos[i+1:]...)
			if len(p.podGroupToPodInfos[highestLevelAncestorKey]) == 0 {
				delete(p.podGroupToPodInfos, highestLevelAncestorKey)
			}
			return pInfo
		}
	}
	return nil
}

// clear removes and returns all pod infos waiting for the (composite) pod group.
func (p *incompleteEntities) clear(pgKey string) []*framework.QueuedPodInfo {
	if pods, ok := p.podGroupToPodInfos[pgKey]; ok {
		delete(p.podGroupToPodInfos, pgKey)
		return pods
	}
	return nil
}

func podGroupKeyForPod(pod *v1.Pod) string {
	return fmt.Sprintf("%s/%s/%s", framework.PodGroupKeyType, pod.Namespace, *pod.Spec.SchedulingGroup.PodGroupName)
}

func podGroupKey(podGroup *schedulingv1alpha3.PodGroup) string {
	return fmt.Sprintf("%s/%s/%s", framework.PodGroupKeyType, podGroup.Namespace, podGroup.Name)
}

func compositePodGroupKey(compositePodGroup *schedulingv1alpha3.CompositePodGroup) string {
	return fmt.Sprintf("%s/%s/%s", framework.CompositePodGroupKeyType, compositePodGroup.Namespace, compositePodGroup.Name)
}

func compositePodGroupKeyFromName(name, namespace string) string {
	return fmt.Sprintf("%s/%s/%s", framework.CompositePodGroupKeyType, namespace, name)
}
