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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/pod"

	"github.com/golang/glog"
)

type IPGetter interface {
	GetInstanceIP(host string) (ip string)
}

// PodCache contains both a cache of container information, as well as the mechanism for keeping
// that cache up to date.
type PodCache struct {
	ipCache       IPGetter
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
	currentNodes map[objKey]bool
}

type objKey struct {
	namespace, name string
}

// NewPodCache returns a new PodCache which watches container information
// registered in the given PodRegistry.
// TODO(lavalamp): pods should be a client.PodInterface.
func NewPodCache(ipCache IPGetter, info client.PodInfoGetter, nodes client.NodeInterface, pods pod.Registry) *PodCache {
	return &PodCache{
		ipCache:       ipCache,
		containerInfo: info,
		pods:          pods,
		nodes:         nodes,
		currentNodes:  map[objKey]bool{},
		podStatus:     map[objKey]api.PodStatus{},
	}
}

// GetPodStatus gets the stored pod status.
func (p *PodCache) GetPodStatus(namespace, name string) (*api.PodStatus, error) {
	p.lock.Lock()
	defer p.lock.Unlock()
	value, ok := p.podStatus[objKey{namespace, name}]
	if !ok {
		return nil, client.ErrPodInfoNotAvailable
	}
	// Make a copy
	return &value, nil
}

// lock must *not* be held
func (p *PodCache) nodeExists(name string) bool {
	p.lock.Lock()
	defer p.lock.Unlock()
	exists, cacheHit := p.currentNodes[objKey{"", name}]
	if cacheHit {
		return exists
	}
	// Don't block everyone while looking up this minion.
	// Because this may require an RPC to our storage (e.g. etcd).
	func() {
		p.lock.Unlock()
		defer p.lock.Lock()
		_, err := p.nodes.Get(name)
		exists = true
		if err != nil {
			exists = false
			if !errors.IsNotFound(err) {
				glog.Errorf("Unexpected error type verifying minion existence: %+v", err)
			}
		}
	}()
	p.currentNodes[objKey{"", name}] = exists
	return exists
}

// TODO: once Host gets moved to spec, this can take a podSpec + metadata instead of an
// entire pod?
func (p *PodCache) updatePodStatus(pod *api.Pod) error {
	newStatus := pod.Status
	if pod.Status.Host == "" {
		p.lock.Lock()
		defer p.lock.Unlock()
		// Not assigned.
		newStatus.Phase = api.PodPending
		p.podStatus[objKey{pod.Namespace, pod.Name}] = newStatus
		return nil
	}

	if !p.nodeExists(pod.Status.Host) {
		p.lock.Lock()
		defer p.lock.Unlock()
		// Assigned to non-existing node.
		newStatus.Phase = api.PodFailed
		p.podStatus[objKey{pod.Namespace, pod.Name}] = newStatus
		return nil
	}

	info, err := p.containerInfo.GetPodInfo(pod.Status.Host, pod.Namespace, pod.Name)
	newStatus.HostIP = p.ipCache.GetInstanceIP(pod.Status.Host)

	if err != nil {
		newStatus.Phase = api.PodUnknown
	} else {
		newStatus.Info = info.ContainerInfo
		newStatus.Phase = getPhase(&pod.Spec, newStatus.Info)
		if netContainerInfo, ok := newStatus.Info["net"]; ok {
			if netContainerInfo.PodIP != "" {
				newStatus.PodIP = netContainerInfo.PodIP
			}
		}
	}
	p.lock.Lock()
	defer p.lock.Unlock()
	p.podStatus[objKey{pod.Namespace, pod.Name}] = newStatus
	return err
}

// UpdateAllContainers updates information about all containers.
func (p *PodCache) UpdateAllContainers() {
	func() {
		// Reset which nodes we think exist
		p.lock.Lock()
		defer p.lock.Unlock()
		p.currentNodes = map[objKey]bool{}
	}()

	ctx := api.NewContext()
	pods, err := p.pods.ListPods(ctx, labels.Everything())
	if err != nil {
		glog.Errorf("Error synchronizing container list: %v", err)
		return
	}
	var wg sync.WaitGroup
	for i := range pods.Items {
		pod := &pods.Items[i]
		wg.Add(1)
		go func() {
			defer wg.Done()
			err := p.updatePodStatus(pod)
			if err != nil && err != client.ErrPodInfoNotAvailable {
				glog.Errorf("Error synchronizing container: %v", err)
			}
		}()
	}
	wg.Wait()
}

// getPhase returns the phase of a pod given its container info.
// TODO(dchen1107): push this all the way down into kubelet.
func getPhase(spec *api.PodSpec, info api.PodInfo) api.PodPhase {
	if info == nil {
		return api.PodPending
	}
	running := 0
	waiting := 0
	stopped := 0
	failed := 0
	succeeded := 0
	unknown := 0
	for _, container := range spec.Containers {
		if containerStatus, ok := info[container.Name]; ok {
			if containerStatus.State.Running != nil {
				running++
			} else if containerStatus.State.Termination != nil {
				stopped++
				if containerStatus.State.Termination.ExitCode == 0 {
					succeeded++
				} else {
					failed++
				}
			} else if containerStatus.State.Waiting != nil {
				waiting++
			} else {
				unknown++
			}
		} else {
			unknown++
		}
	}
	switch {
	case waiting > 0:
		// One or more containers has not been started
		return api.PodPending
	case running > 0 && unknown == 0:
		// All containers have been started, and at least
		// one container is running
		return api.PodRunning
	case running == 0 && stopped > 0 && unknown == 0:
		// All containers are terminated
		if spec.RestartPolicy.Always != nil {
			// All containers are in the process of restarting
			return api.PodRunning
		}
		if stopped == succeeded {
			// RestartPolicy is not Always, and all
			// containers are terminated in success
			return api.PodSucceeded
		}
		if spec.RestartPolicy.Never != nil {
			// RestartPolicy is Never, and all containers are
			// terminated with at least one in failure
			return api.PodFailed
		}
		// RestartPolicy is OnFailure, and at least one in failure
		// and in the process of restarting
		return api.PodRunning
	default:
		return api.PodPending
	}
}
