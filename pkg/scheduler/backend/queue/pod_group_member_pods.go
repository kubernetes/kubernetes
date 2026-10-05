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
	"k8s.io/component-base/metrics"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// podGroupMemberPods stores member pods of pod groups.
// This structure is not thread-safe and should be accessed only under the lock of the PriorityQueue.
type podGroupMemberPods struct {
	// podGroupToPodInfos stores QueuedPodInfos keyed by their pod group key (pg-type/namespace/pg-name),
	// and then by pod key (namespace/pg-name).
	podGroupToPodInfos map[fwk.EntityKey]map[fwk.EntityKey]*framework.QueuedPodInfo
	podsRecorder       metrics.GaugeMetric
}

func newPodGroupMemberPods(podsRecorder metrics.GaugeMetric) *podGroupMemberPods {
	return &podGroupMemberPods{
		podGroupToPodInfos: make(map[fwk.EntityKey]map[fwk.EntityKey]*framework.QueuedPodInfo),
		podsRecorder:       podsRecorder,
	}
}

// add adds a pod info.
// It also increments the metric counter.
func (p *podGroupMemberPods) add(pInfo *framework.QueuedPodInfo) {
	pgKey, pKey := podGroupKeyForPod(pInfo.Pod), podKey(pInfo.Pod)
	if p.podGroupToPodInfos[pgKey] == nil {
		p.podGroupToPodInfos[pgKey] = make(map[fwk.EntityKey]*framework.QueuedPodInfo)
	}
	p.podGroupToPodInfos[pgKey][pKey] = pInfo
	p.podsRecorder.Inc()
}

// has checks if the pod is tracked.
func (p *podGroupMemberPods) has(podLookup *v1.Pod) bool {
	pgKey, pKey := podGroupKeyForPod(podLookup), podKey(podLookup)
	return p.podGroupToPodInfos[pgKey][pKey] != nil
}

// len returns the total number of pods.
func (p *podGroupMemberPods) len() int {
	l := 0
	for _, pods := range p.podGroupToPodInfos {
		l += len(pods)
	}
	return l
}

// get returns the queued pod info for the given pod.
func (p *podGroupMemberPods) get(podLookup *v1.Pod) *framework.QueuedPodInfo {
	pgKey, pKey := podGroupKeyForPod(podLookup), podKey(podLookup)
	return p.podGroupToPodInfos[pgKey][pKey]
}

// update refreshes the pod object and returns the updated pod info if found.
func (p *podGroupMemberPods) update(newPod *v1.Pod) *framework.QueuedPodInfo {
	pgKey, pKey := podGroupKeyForPod(newPod), podKey(newPod)
	if pInfo, ok := p.podGroupToPodInfos[pgKey][pKey]; ok {
		pInfo.Pod = newPod
		return pInfo
	}
	return nil
}

// delete removes a specific pod and returns its pod info if found.
// It also decrements the metric counter.
func (p *podGroupMemberPods) delete(podLookup *v1.Pod) *framework.QueuedPodInfo {
	pgKey, pKey := podGroupKeyForPod(podLookup), podKey(podLookup)
	if pInfo, ok := p.podGroupToPodInfos[pgKey][pKey]; ok {
		delete(p.podGroupToPodInfos[pgKey], pKey)
		if len(p.podGroupToPodInfos[pgKey]) == 0 {
			delete(p.podGroupToPodInfos, pgKey)
		}
		p.podsRecorder.Dec()
		return pInfo
	}
	return nil
}

// list returns all tracked pods.
func (p *podGroupMemberPods) list() []*v1.Pod {
	var pods []*v1.Pod
	for _, pgPods := range p.podGroupToPodInfos {
		for _, pInfo := range pgPods {
			pods = append(pods, pInfo.Pod)
		}
	}
	return pods
}

// clear removes and returns all pod infos for a specific pod group namespace and name.
// It also decrements the metric counter for all cleared pods.
func (p *podGroupMemberPods) clear(namespace, name string) []*framework.QueuedPodInfo {
	pgKey := fwk.PodGroupKey(namespace, name)
	if pInfos, ok := p.podGroupToPodInfos[pgKey]; ok {
		delete(p.podGroupToPodInfos, pgKey)
		var pInfoList []*framework.QueuedPodInfo
		for _, pInfo := range pInfos {
			pInfoList = append(pInfoList, pInfo)
			p.podsRecorder.Dec()
		}
		return pInfoList
	}
	return nil
}
