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
	"slices"

	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// pendingPodGroupMemberPods stores all pending pods that wait for their corresponding pod group to be requeued.
type pendingPodGroupMemberPods struct {
	podGroupToPods map[string][]*framework.QueuedPodInfo
}

func newPendingPodGroupMemberPods() *pendingPodGroupMemberPods {
	return &pendingPodGroupMemberPods{
		podGroupToPods: make(map[string][]*framework.QueuedPodInfo),
	}
}

// add adds a pod info to the specified pod group.
func (p *pendingPodGroupMemberPods) add(pgInfoLookup framework.QueuedEntityInfo, pInfo *framework.QueuedPodInfo) {
	pgKey := queuedEntityKeyFunc(pgInfoLookup)
	p.podGroupToPods[pgKey] = append(p.podGroupToPods[pgKey], pInfo)
}

// get returns all pod infos associated with the specified pod group.
func (p *pendingPodGroupMemberPods) get(pgInfoLookup framework.QueuedEntityInfo) []*framework.QueuedPodInfo {
	pgKey := queuedEntityKeyFunc(pgInfoLookup)
	return p.podGroupToPods[pgKey]
}

// has checks if the specified pod group has any pending pods.
func (p *pendingPodGroupMemberPods) has(pgInfoLookup framework.QueuedEntityInfo) bool {
	pgKey := queuedEntityKeyFunc(pgInfoLookup)
	_, ok := p.podGroupToPods[pgKey]
	return ok
}

// len returns the number of pending pods.
func (p *pendingPodGroupMemberPods) len() int {
	count := 0
	for _, pods := range p.podGroupToPods {
		count += len(pods)
	}
	return count
}

// getPod searches for a specific pod within the provided pod group.
func (p *pendingPodGroupMemberPods) getPod(pgInfoLookup framework.QueuedEntityInfo, pod *v1.Pod) *framework.QueuedPodInfo {
	pgKey := queuedEntityKeyFunc(pgInfoLookup)
	for _, pInfo := range p.podGroupToPods[pgKey] {
		if pInfo.Pod.Name == pod.Name && pInfo.Pod.Namespace == pod.Namespace {
			return pInfo
		}
	}
	return nil
}

// update refreshes the pod object for a member of the specified pod group.
// It returns the updated pod info if the pod was found, nil otherwise.
func (p *pendingPodGroupMemberPods) update(pgInfoLookup framework.QueuedEntityInfo, newPod *v1.Pod) *framework.QueuedPodInfo {
	pgKey := queuedEntityKeyFunc(pgInfoLookup)
	for _, pInfo := range p.podGroupToPods[pgKey] {
		if pInfo.Pod.Name == newPod.Name && pInfo.Pod.Namespace == newPod.Namespace {
			pInfo.Pod = newPod
			return pInfo
		}
	}
	return nil
}

// delete removes a specific pod from the tracked members of a pod group.
// It returns the removed pod info if found, nil otherwise.
func (p *pendingPodGroupMemberPods) delete(pgInfoLookup framework.QueuedEntityInfo, pod *v1.Pod) *framework.QueuedPodInfo {
	pgKey := queuedEntityKeyFunc(pgInfoLookup)
	for i, pInfo := range p.podGroupToPods[pgKey] {
		if pInfo.Pod.Name == pod.Name && pInfo.Pod.Namespace == pod.Namespace {
			p.podGroupToPods[pgKey] = slices.Delete(p.podGroupToPods[pgKey], i, i+1)
			if len(p.podGroupToPods[pgKey]) == 0 {
				delete(p.podGroupToPods, pgKey)
			}
			return pInfo
		}
	}
	return nil
}

// clear removes all pods associated with the specified pod group.
func (p *pendingPodGroupMemberPods) clear(pgInfoLookup framework.QueuedEntityInfo) {
	pgKey := queuedEntityKeyFunc(pgInfoLookup)
	delete(p.podGroupToPods, pgKey)
}
