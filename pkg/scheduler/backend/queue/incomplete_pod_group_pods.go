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
	v1 "k8s.io/api/core/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// incompletePodGroupPods stores pod infos that wait for their corresponding PodGroup object to be observed.
// This structure is not thread-safe and should be accessed only under the lock of the PriorityQueue.
type incompletePodGroupPods struct {
	// podGroupToPodInfos stores QueuedPodInfos keyed by their pod group key (pg-type/namespace/pg-name),
	// and then by pod key (namespace/pg-name).
	podGroupToPodInfos map[string]map[string]*framework.QueuedPodInfo
}

func newIncompletePodGroupPods() *incompletePodGroupPods {
	return &incompletePodGroupPods{
		podGroupToPodInfos: make(map[string]map[string]*framework.QueuedPodInfo),
	}
}

// add adds a pod info waiting for the pod group.
func (ip *incompletePodGroupPods) add(pInfo *framework.QueuedPodInfo) {
	pgKey, pKey := podGroupKeyForPod(pInfo.Pod), podKey(pInfo.Pod)
	if ip.podGroupToPodInfos[pgKey] == nil {
		ip.podGroupToPodInfos[pgKey] = make(map[string]*framework.QueuedPodInfo)
	}
	ip.podGroupToPodInfos[pgKey][pKey] = pInfo
}

// get returns the pod info for the given pod.
func (ip *incompletePodGroupPods) get(pod *v1.Pod) *framework.QueuedPodInfo {
	pgKey, pKey := podGroupKeyForPod(pod), podKey(pod)
	return ip.podGroupToPodInfos[pgKey][pKey]
}

// has returns true if the pod is present in the incompletePodGroupPods.
func (ip *incompletePodGroupPods) has(pod *v1.Pod) bool {
	pgKey, pKey := podGroupKeyForPod(pod), podKey(pod)
	return ip.podGroupToPodInfos[pgKey][pKey] != nil
}

// len returns the number of incomplete pods.
func (ip *incompletePodGroupPods) len() int {
	l := 0
	for _, pods := range ip.podGroupToPodInfos {
		l += len(pods)
	}
	return l
}

// update updates the pod inside the incompletePodGroupPods.
// It returns the updated pod info if the pod was found, nil otherwise.
func (ip *incompletePodGroupPods) update(pod *v1.Pod) *framework.QueuedPodInfo {
	pgKey, pKey := podGroupKeyForPod(pod), podKey(pod)
	if pInfo, ok := ip.podGroupToPodInfos[pgKey][pKey]; ok {
		pInfo.Pod = pod
		return pInfo
	}
	return nil
}

// delete removes a specific pod and returns its QueuedPodInfo.
// If the pod list for the group becomes empty, it cleans up the key.
func (ip *incompletePodGroupPods) delete(pod *v1.Pod) *framework.QueuedPodInfo {
	pgKey, pKey := podGroupKeyForPod(pod), podKey(pod)
	if pInfo, ok := ip.podGroupToPodInfos[pgKey][pKey]; ok {
		delete(ip.podGroupToPodInfos[pgKey], pKey)
		if len(ip.podGroupToPodInfos[pgKey]) == 0 {
			delete(ip.podGroupToPodInfos, pgKey)
		}
		return pInfo
	}
	return nil
}

// clear removes and returns all pod infos waiting for the pod group.
func (ip *incompletePodGroupPods) clear(podGroup *schedulingv1alpha3.PodGroup) []*framework.QueuedPodInfo {
	pgKey := podGroupKey(podGroup)
	if pods, ok := ip.podGroupToPodInfos[pgKey]; ok {
		delete(ip.podGroupToPodInfos, pgKey)
		var pInfoList []*framework.QueuedPodInfo
		for _, pInfo := range pods {
			pInfoList = append(pInfoList, pInfo)
		}
		return pInfoList
	}
	return nil
}
