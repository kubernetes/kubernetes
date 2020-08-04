/*
Copyright 2018 The Kubernetes Authors.

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

package podresources

import (
	"context"
	"sync"

	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/apis/podresources/v1alpha1"
)

// DevicesProvider knows how to provide the devices used by the given container
type DevicesProvider interface {
	GetDevices(podUID, containerName string) []*v1alpha1.ContainerDevices
	UpdateAllocatedDevices()
}

// TODO fix the name
// must be compatible with pod.Manager
type PodNotifier interface {
	AddPod(pod *v1.Pod)
	UpdatePod(pod *v1.Pod)
	DeletePod(pod *v1.Pod)
}

// PodsProvider knows how to provide the pods admitted by the node
type PodsProvider interface {
	GetPods() []*v1.Pod
}

type podInfo struct {
	Action v1alpha1.WatchPodAction
	Pod    *v1.Pod
}

// podResourcesServer implements PodResourcesListerServer
type podResourcesServer struct {
	podsProvider    PodsProvider
	devicesProvider DevicesProvider
	podSource       chan podInfo
	lock            sync.RWMutex
	sinkId          int
	podSinks        map[int]chan podInfo
}

// NewPodResourcesServer returns a PodResourcesListerServer which lists pods provided by the PodsProvider
// with device information provided by the DevicesProvider
func NewPodResourcesServer(podsProvider PodsProvider, devicesProvider DevicesProvider) (v1alpha1.PodResourcesListerServer, PodNotifier) {
	p := &podResourcesServer{
		podsProvider:    podsProvider,
		devicesProvider: devicesProvider,
		podSource:       make(chan podInfo),
		podSinks:        make(map[int]chan podInfo),
	}
	go p.dispatchPods()
	return p, p
}

func (p *podResourcesServer) makePodResources(pod *v1.Pod) *v1alpha1.PodResources {
	pRes := v1alpha1.PodResources{
		Name:       pod.Name,
		Namespace:  pod.Namespace,
		Containers: make([]*v1alpha1.ContainerResources, len(pod.Spec.Containers)),
	}

	for j, container := range pod.Spec.Containers {
		pRes.Containers[j] = &v1alpha1.ContainerResources{
			Name:    container.Name,
			Devices: p.devicesProvider.GetDevices(string(pod.UID), container.Name),
		}
	}
	return &pRes
}

// List returns information about the resources assigned to pods on the node
func (p *podResourcesServer) List(ctx context.Context, req *v1alpha1.ListPodResourcesRequest) (*v1alpha1.ListPodResourcesResponse, error) {
	pods := p.podsProvider.GetPods()
	podResources := make([]*v1alpha1.PodResources, len(pods))
	p.devicesProvider.UpdateAllocatedDevices()

	for i, pod := range pods {
		podResources[i] = p.makePodResources(pod)
	}

	return &v1alpha1.ListPodResourcesResponse{
		PodResources: podResources,
	}, nil
}

func (p *podResourcesServer) AddPod(pod *v1.Pod) {
	p.podSource <- podInfo{
		Action: v1alpha1.WatchPodAction_ADDED,
		Pod:    pod,
	}
}

func (p *podResourcesServer) UpdatePod(pod *v1.Pod) {
	p.podSource <- podInfo{
		Action: v1alpha1.WatchPodAction_MODIFIED,
		Pod:    pod,
	}
}

func (p *podResourcesServer) DeletePod(pod *v1.Pod) {
	p.podSource <- podInfo{
		Action: v1alpha1.WatchPodAction_DELETED,
		Pod:    pod,
	}
}

func (p *podResourcesServer) dispatchPods() {
	for {
		info := <-p.podSource

		p.lock.RLock()
		for _, ch := range p.podSinks {
			ch <- info
		}
		p.lock.RUnlock()
	}
}

func (p *podResourcesServer) registerWatcher() (int, chan podInfo) {
	p.lock.Lock()
	defer p.lock.Unlock()
	sinkChan := make(chan podInfo)
	sinkId := p.sinkId
	p.sinkId++
	p.podSinks[sinkId] = sinkChan
	return sinkId, sinkChan
}

func (p *podResourcesServer) unregisterWatcher(sinkId int) {
	p.lock.Lock()
	defer p.lock.Unlock()
	// TODO: sink close?
	delete(p.podSinks, sinkId)
}

func (p *podResourcesServer) makeWatchPodResponse(info podInfo) *v1alpha1.WatchPodResourcesResponse {
	resp := v1alpha1.WatchPodResourcesResponse{
		Action: info.Action,
		PodResources: []*v1alpha1.PodResources{
			p.makePodResources(info.Pod),
		},
	}
	return &resp
}

func (p *podResourcesServer) Watch(req *v1alpha1.WatchPodResourcesRequest, srv v1alpha1.PodResourcesLister_WatchServer) error {
	sinkId, sinkChan := p.registerWatcher()
	defer p.unregisterWatcher(sinkId)
	for {
		pod := <-sinkChan
		resp := p.makeWatchPodResponse(pod)
		err := srv.Send(resp)
		if err != nil {
			return err
		}
	}
	return nil
}
