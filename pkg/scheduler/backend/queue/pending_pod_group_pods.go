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
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// pendingPodGroupMemberPods stores all pending member pods of the currently scheduling pod group.
type pendingPodGroupMemberPods struct {
	keyToPod map[string]*framework.QueuedPodInfo
}

func newPendingPodGroupMemberPods() *pendingPodGroupMemberPods {
	return &pendingPodGroupMemberPods{
		keyToPod: make(map[string]*framework.QueuedPodInfo),
	}
}

// add adds a pod info.
func (p *pendingPodGroupMemberPods) add(pInfo *framework.QueuedPodInfo) {
	p.keyToPod[pendingPodKey(pInfo.Pod)] = pInfo
}

// has checks if the pod is tracked.
func (p *pendingPodGroupMemberPods) has(podLookup *v1.Pod) bool {
	_, ok := p.keyToPod[pendingPodKey(podLookup)]
	return ok
}

// len returns the number of pending pods.
func (p *pendingPodGroupMemberPods) len() int {
	return len(p.keyToPod)
}

// get returns the queued pod info for the given pod.
func (p *pendingPodGroupMemberPods) get(podLookup *v1.Pod) *framework.QueuedPodInfo {
	return p.keyToPod[pendingPodKey(podLookup)]
}

// update refreshes the pod object and returns the updated pod info if found.
func (p *pendingPodGroupMemberPods) update(newPod *v1.Pod) *framework.QueuedPodInfo {
	pInfo, ok := p.keyToPod[pendingPodKey(newPod)]
	if !ok {
		return nil
	}
	pInfo.Pod = newPod
	return pInfo
}

// delete removes a specific pod and returns its pod info if found.
func (p *pendingPodGroupMemberPods) delete(podLookup *v1.Pod) *framework.QueuedPodInfo {
	pInfo, ok := p.keyToPod[pendingPodKey(podLookup)]
	if !ok {
		return nil
	}
	delete(p.keyToPod, pendingPodKey(podLookup))
	return pInfo
}

// clear removes all tracked pods and returns them.
func (p *pendingPodGroupMemberPods) clear() []*framework.QueuedPodInfo {
	pods := make([]*framework.QueuedPodInfo, 0, len(p.keyToPod))
	for _, pInfo := range p.keyToPod {
		pods = append(pods, pInfo)
	}
	p.keyToPod = make(map[string]*framework.QueuedPodInfo)
	return pods
}

func pendingPodKey(pod *v1.Pod) string {
	return fmt.Sprintf("%s/%s", pod.Namespace, pod.Name)
}
