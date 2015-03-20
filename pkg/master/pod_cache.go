/*
Copyright 2014 Google Inc. All rights reserved.

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

package master

import (
	"sync"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/pod"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/golang/glog"
)

// PodCache contains both a cache of container information, as well as the mechanism for keeping
// that cache up to date.
type PodCache struct {
	containerInfo client.PodInfoGetter
	pods          pod.Registry
	// For confirming existance of a node
	nodes client.NodeInterface

	// lock protects access to all fields below
	lock sync.Mutex
	// cached pod statuses.
	podStatus map[objKey]api.PodStatus
	// nodes that we know exist. Cleared at the beginning of each
	// UpdateAllPods call.
	currentNodes map[objKey]api.NodeStatus
}

type objKey struct {
	namespace, name string
}

// NewPodCache returns a new PodCache which watches container information
// registered in the given PodRegistry.
// TODO(lavalamp): pods should be a client.PodInterface.
func NewPodCache(info client.PodInfoGetter, nodes client.NodeInterface, pods pod.Registry) *PodCache {
	return &PodCache{
		containerInfo: info,
		pods:          pods,
		nodes:         nodes,
		currentNodes:  map[objKey]api.NodeStatus{},
		podStatus:     map[objKey]api.PodStatus{},
	}
}

// GetPodStatus gets the stored pod status.
func (p *PodCache) GetPodStatus(namespace, name string) (*api.PodStatus, error) {
	status := p.getPodStatusInternal(namespace, name)
	if status != nil {
		return status, nil
	}
	return p.updateCacheAndReturn(namespace, name)
}

func (p *PodCache) updateCacheAndReturn(namespace, name string) (*api.PodStatus, error) {
	pod, err := p.pods.GetPod(api.WithNamespace(api.NewContext(), namespace), name)
	if err != nil {
		return nil, err
	}
	if err := p.updatePodStatus(pod); err != nil {
		return nil, err
	}
	status := p.getPodStatusInternal(namespace, name)
	if status == nil {
		glog.Warningf("nil status after successful update.  that's odd... (%s %s)", namespace, name)
		return nil, client.ErrPodInfoNotAvailable
	}
	return status, nil
}

func (p *PodCache) getPodStatusInternal(namespace, name string) *api.PodStatus {
	p.lock.Lock()
	defer p.lock.Unlock()
	value, ok := p.podStatus[objKey{namespace, name}]
	if !ok {
		return nil
	}
	// Make a copy
	return &value
}

func (p *PodCache) ClearPodStatus(namespace, name string) {
	p.lock.Lock()
	defer p.lock.Unlock()

	delete(p.podStatus, objKey{namespace, name})
}

func (p *PodCache) getNodeStatusInCache(name string) (*api.NodeStatus, bool) {
	p.lock.Lock()
	defer p.lock.Unlock()
	nodeStatus, cacheHit := p.currentNodes[objKey{"", name}]
	return &nodeStatus, cacheHit
}

// lock must *not* be held
func (p *PodCache) getNodeStatus(name string) (*api.NodeStatus, error) {
	nodeStatus, cacheHit := p.getNodeStatusInCache(name)
	if cacheHit {
		return nodeStatus, nil
	}
	// TODO: suppose there's N concurrent requests for node "foo"; in that case
	// it might be useful to block all of them and only look up "foo" once.
	// (This code will make up to N lookups.) One way of doing that would be to
	// have a pool of M mutexes and require that before looking up "foo" you must
	// lock mutex hash("foo") % M.
	node, err := p.nodes.Get(name)
	if err != nil {
		glog.Errorf("Unexpected error verifying node existence: %+v", err)
		return nil, err
	}

	p.lock.Lock()
	defer p.lock.Unlock()
	p.currentNodes[objKey{"", name}] = node.Status
	return &node.Status, nil
}

func (p *PodCache) clearNodeStatus() {
	p.lock.Lock()
	defer p.lock.Unlock()
	p.currentNodes = map[objKey]api.NodeStatus{}
}

func (p *PodCache) getHostAddress(addresses []api.NodeAddress) string {
	addressMap := make(map[api.NodeAddressType][]api.NodeAddress)
	for i := range addresses {
		addressMap[addresses[i].Type] = append(addressMap[addresses[i].Type], addresses[i])
	}
	if addresses, ok := addressMap[api.NodeLegacyHostIP]; ok {
		return addresses[0].Address
	}
	if addresses, ok := addressMap[api.NodeInternalIP]; ok {
		return addresses[0].Address
	}
	if addresses, ok := addressMap[api.NodeExternalIP]; ok {
		return addresses[0].Address
	}
	return ""
}

// TODO: once Host gets moved to spec, this can take a podSpec + metadata instead of an
// entire pod?
func (p *PodCache) updatePodStatus(pod *api.Pod) error {
	newStatus, err := p.computePodStatus(pod)

	p.lock.Lock()
	defer p.lock.Unlock()
	// Map accesses must be locked.
	if err == nil {
		p.podStatus[objKey{pod.Namespace, pod.Name}] = newStatus
	}

	return err
}

// computePodStatus always returns a new status, even if it also returns a non-nil error.
// TODO: once Host gets moved to spec, this can take a podSpec + metadata instead of an
// entire pod?
func (p *PodCache) computePodStatus(pod *api.Pod) (api.PodStatus, error) {
	newStatus := pod.Status

	if pod.Status.Host == "" {
		// Not assigned.
		newStatus.Phase = api.PodPending
		newStatus.Conditions = append(newStatus.Conditions, pod.Status.Conditions...)
		return newStatus, nil
	}

	nodeStatus, err := p.getNodeStatus(pod.Status.Host)

	// Assigned to non-existing node.
	if err != nil || len(nodeStatus.Conditions) == 0 {
		glog.V(5).Infof("node doesn't exist: %v %v, setting pod %q status to unknown", err, nodeStatus, pod.Name)
		newStatus.Phase = api.PodUnknown
		newStatus.Conditions = append(newStatus.Conditions, pod.Status.Conditions...)
		return newStatus, nil
	}

	// Assigned to an unhealthy node.
	for _, condition := range nodeStatus.Conditions {
		if (condition.Type == api.NodeReady || condition.Type == api.NodeReachable) && condition.Status == api.ConditionNone {
			glog.V(5).Infof("node status: %v, setting pod %q status to unknown", condition, pod.Name)
			newStatus.Phase = api.PodUnknown
			newStatus.Conditions = append(newStatus.Conditions, pod.Status.Conditions...)
			return newStatus, nil
		}
	}

	result, err := p.containerInfo.GetPodStatus(pod.Status.Host, pod.Namespace, pod.Name)

	if err != nil {
		glog.V(5).Infof("error getting pod %s status: %v, retry later", pod.Name, err)
	} else {
		newStatus.HostIP = p.getHostAddress(nodeStatus.Addresses)
		newStatus.Info = result.Status.Info
		newStatus.PodIP = result.Status.PodIP
		newStatus.Phase = result.Status.Phase
		newStatus.Conditions = result.Status.Conditions
	}
	return newStatus, err
}

func (p *PodCache) GarbageCollectPodStatus() {
	pods, err := p.pods.ListPods(api.NewContext(), labels.Everything())
	if err != nil {
		glog.Errorf("Error getting pod list: %v", err)
		return
	}
	keys := map[objKey]bool{}
	for _, pod := range pods.Items {
		keys[objKey{pod.Namespace, pod.Name}] = true
	}
	p.lock.Lock()
	defer p.lock.Unlock()
	for key := range p.podStatus {
		if _, found := keys[key]; !found {
			glog.Infof("Deleting orphaned cache entry: %v", key)
			delete(p.podStatus, key)
		}
	}
}

// UpdateAllContainers updates information about all containers.
// Callers should let one call to UpdateAllContainers finish before
// calling again, or risk having new info getting clobbered by delayed
// old info.
func (p *PodCache) UpdateAllContainers() {
	// TODO: this is silly, we should pro-actively update the pod status when
	// the API server makes changes.
	p.clearNodeStatus()

	ctx := api.NewContext()
	pods, err := p.pods.ListPods(ctx, labels.Everything())
	if err != nil {
		glog.Errorf("Error getting pod list: %v", err)
		return
	}

	// TODO: this algorithm is 1 goroutine & RPC per pod. With a little work,
	// it should be possible to make it 1 per *node*, which will be important
	// at very large scales. (To be clear, the goroutines shouldn't matter--
	// it's the RPCs that need to be minimized.)
	var wg sync.WaitGroup
	for i := range pods.Items {
		pod := &pods.Items[i]
		wg.Add(1)
		go func() {
			defer util.HandleCrash()
			defer wg.Done()
			err := p.updatePodStatus(pod)
			if err != nil && err != client.ErrPodInfoNotAvailable {
				glog.Errorf("Error getting info for pod %v/%v: %v", pod.Namespace, pod.Name, err)
			}
		}()
	}
	wg.Wait()
}
