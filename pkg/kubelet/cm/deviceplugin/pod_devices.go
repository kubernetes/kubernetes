/*
Copyright 2017 The Kubernetes Authors.

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

package deviceplugin

import (
	"fmt"
	"sync"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/util/sets"
	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1alpha"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

type devAllocation interface {
	// Check if the specified pod uid is known to have allocations
	hasPod(pod string) bool

	// Enumerate all devices known to be already allocated by returning a map
	// keyed by resource name.
	devices() map[string]sets.String

	// Update the allocation registry based on the provided list of active
	// pods. Records for inactive pods are pruned.
	purge(activePods sets.String)

	// Process the device requests from a Pod and return the set of device IDs
	// to allocate.
	reserve(allDevices map[string]sets.String, req *allocRequest) (containerMap, error)

	// Confirm the device allocations by changing the device status
	confirm(pod string, reservation containerMap) error

	// Abandon all allocations, reservations for a Pod
	rollback(pod string)

	// Marshal the registry into data for checkpointing
	toCheckpointData() []podDevicesCheckpointEntry

	// Unmarshal checkpointed data into registry
	fromCheckpointData(data []podDevicesCheckpointEntry)

	// Collect device information into container runtime settings.
	collectRunOptions(pod, container string) *DeviceRunContainerOptions

	// The following methods are for testing only
	// insert a new allocation record
	insert(pod, container, resource string, devices sets.String, resp *pluginapi.AllocateResponse)
	// get the set of pod IDs known to the device manager
	pods() sets.String
	// return all pod devices
	allocations() map[string]containerMap
	// Enumerate all known device allocations for the given container on
	// specified resource.
	devicesByContainer(pod, container, resource string) sets.String
}

type deviceAllocationStatus string

const (
	devReserved  deviceAllocationStatus = "Reserved"
	devAllocated deviceAllocationStatus = "Allocated"
)

type deviceInfo struct {
	// status represents the allocation status of the devices
	status deviceAllocationStatus
	// ids contains device Ids allocated to this container for the given resourceName.
	ids sets.String
	// allocResp contains cached rpc AllocateResponse.
	allocResp *pluginapi.AllocateResponse
}

type resourceMap map[string]deviceInfo   // Keyed by resourceName.
type containerMap map[string]resourceMap // Keyed by containerName.

// devAllocationImpl is a struct that implements the allocation interface.
type devAllocationImpl struct {
	sync.RWMutex
	// registry is a map from Pod UID to per-container device allocations
	registry map[string]containerMap
}

// allocRequest is a struct containing all device allocation requests by a Pod.
type allocRequest struct {
	// pod is the UID of a Pod
	pod string
	// inits is a map from initcontainer to resource to count
	inits map[string]map[string]int
	// containers is a map from container to resource to count
	containers map[string]map[string]int
}

// podDevicesCheckpointEntry is used to record <pod, container> to device allocation information.
type podDevicesCheckpointEntry struct {
	PodUID        string
	ContainerName string
	ResourceName  string
	DeviceIDs     []string
	AllocResp     []byte
}

// newAllocation instantiates a devAllocationImpl object to be used as an
// implementation of devAllocation interface.
func newAllocation() *devAllocationImpl {
	return &devAllocationImpl{
		registry: make(map[string]containerMap),
	}
}

// hasPod checks if there are records associated with the given pod.
func (a *devAllocationImpl) hasPod(pod string) bool {
	a.RLock()
	defer a.RUnlock()
	_, found := a.registry[pod]
	return found
}

// devices returns all devices already allocated or reserved, keyed by resource name
func (a *devAllocationImpl) devices() map[string]sets.String {
	a.RLock()
	defer a.RUnlock()
	return a.getDevices()
}

// getDevices is an internal function returning all devices allocated or
// reserved
func (a *devAllocationImpl) getDevices() map[string]sets.String {
	ret := make(map[string]sets.String)
	for _, cmap := range a.registry {
		for _, rmap := range cmap {
			for r, dinfo := range rmap {
				if _, ok := ret[r]; !ok {
					ret[r] = sets.NewString()
				}
				ret[r] = ret[r].Union(dinfo.ids)
			}
		}
	}
	return ret
}

// purge updates the pod allocation registry by list of active pods.
// The function returns a map from resource name to set of device IDs.
func (a *devAllocationImpl) purge(activePods sets.String) {
	a.Lock()
	defer a.Unlock()

	for p := range a.registry {
		if !activePods.Has(p) {
			delete(a.registry, p)
		}
	}
}

// reserve processes the device requests from a Pod and return the set of
// device IDs to allocate.
func (a *devAllocationImpl) reserve(allDevices map[string]sets.String, req *allocRequest) (containerMap, error) {
	a.Lock()
	defer a.Unlock()

	allocated := a.getDevices()
	reservation := make(containerMap)
	for container, rescount := range req.inits {
		for resource, requested := range rescount {
			// 1. filter out devices already allocated, if any.
			// TODO: revise next line by using devices() directly
			existing := a.devicesByContainer(req.pod, container, resource)
			if existing != nil {
				glog.V(6).Infof("existing allocation found for container %q/%q: %s=%v",
					req.pod, container, resource, existing.List())
				// TODO: allow reallocation when there is a need
				if existing.Len() != requested {
					return nil, fmt.Errorf("container %v/%v resource request changed: %v = %v to %v",
						req.pod, container, resource, existing.Len(), requested)
				}
				// skip allocation because there is no need
				continue
			}

			// 2. try find candidate devices
			// Note that 'allocated' is not updated afterwards so the same
			// device may get reused across initContainers.
			candidates := allDevices[resource].Difference(allocated[resource])
			if int(candidates.Len()) < requested {
				return nil, fmt.Errorf("insufficient resource %s: requested=%d, available=%d",
					resource, requested, candidates.Len())
			}
			devList := candidates.List()[:requested]

			// 3. record reservation
			if _, found := reservation[container]; !found {
				reservation[container] = make(resourceMap)
			}
			reservation[container][resource] = deviceInfo{
				status: devReserved,
				ids:    sets.NewString(devList...),
			}
		}
	}

	for container, rescount := range req.containers {
		for resource, requested := range rescount {
			// 1. filter out devices already allocated, if any.
			existing := a.devicesByContainer(req.pod, container, resource)
			if existing != nil {
				glog.V(6).Infof("existing allocation found for container %q/%q: %s=%v",
					req.pod, container, resource, existing.List())
				// TODO: allow reallocation when there is a need
				if existing.Len() != requested {
					return nil, fmt.Errorf("container %v/%v resource request changed: %v = %v to %v",
						req.pod, container, resource, existing.Len(), requested)
				}
				continue
			}

			// 2. try find candidate devices
			// Note that 'allocated' is updated during each iteration so
			// devices are not reused across these containers
			candidates := allDevices[resource].Difference(allocated[resource])
			if int(candidates.Len()) < requested {
				return nil, fmt.Errorf("insufficient resource %s: requested=%d, available=%d",
					resource, requested, candidates.Len())
			}
			devList := candidates.List()[:requested]
			selected := sets.NewString(devList...)
			allocated[resource] = allocated[resource].Union(selected)

			// 3. record reservation
			if _, found := reservation[container]; !found {
				reservation[container] = make(resourceMap)
			}
			reservation[container][resource] = deviceInfo{
				status:    devReserved,
				ids:       selected,
				allocResp: nil,
			}
		}
	}

	a.recordReservation(req.pod, reservation)

	return reservation, nil
}

// devicesByContainer returns a set of device IDs allocated to the given container
// for the given resource. nil is returned no record is found.
// This function should be called with read lock held on registry
func (a *devAllocationImpl) devicesByContainer(pod, container, resource string) sets.String {
	if _, found := a.registry[pod]; !found {
		return nil
	}
	if _, found := a.registry[pod][container]; !found {
		return nil
	}
	dInfo, found := a.registry[pod][container][resource]
	if !found || dInfo.status != devAllocated {
		return nil
	}
	return dInfo.ids
}

// recordReservation writes device revservation with reserved status
// This function is called with write lock held on registry
func (a *devAllocationImpl) recordReservation(pod string, reservation containerMap) {
	if _, ok := a.registry[pod]; !ok {
		a.registry[pod] = make(containerMap)
	}
	for c, rmap := range reservation {
		if _, ok := a.registry[pod][c]; !ok {
			a.registry[pod][c] = make(resourceMap)
		}
		for r, dinfo := range rmap {
			a.registry[pod][c][r] = deviceInfo{
				status:    devReserved,
				ids:       dinfo.ids,
				allocResp: dinfo.allocResp,
			}
		}
	}
}

// confirm changes all device reservations into allocated status
func (a *devAllocationImpl) confirm(pod string, reservation containerMap) error {
	a.Lock()
	defer a.Unlock()

	for c, rmap := range reservation {
		for r, dinfo := range rmap {
			a.registry[pod][c][r] = deviceInfo{
				status:    devAllocated,
				ids:       dinfo.ids,
				allocResp: dinfo.allocResp,
			}
		}
	}

	return nil
}

// rollback abandons all device reservations and allocations for a pod
func (a *devAllocationImpl) rollback(pod string) {
	a.Lock()
	defer a.Unlock()

	if _, ok := a.registry[pod]; ok {
		delete(a.registry, pod)
	}
}

// toCheckpointData translates the devAllocation to data to be checkpointed.
func (a *devAllocationImpl) toCheckpointData() []podDevicesCheckpointEntry {
	a.RLock()
	defer a.RUnlock()

	var data []podDevicesCheckpointEntry
	for pod, cmap := range a.registry {
		for c, rmap := range cmap {
			for r, devinfo := range rmap {
				devIds := devinfo.ids.UnsortedList()
				if devinfo.allocResp == nil {
					glog.Warningf("allocation response is missing for %v %v %v", pod, c, r)
					continue
				}

				allocResp, err := devinfo.allocResp.Marshal()
				if err != nil {
					glog.Errorf("failed to marshal allocation response for %v %v %v: %v", pod, c, r, err)
					continue
				}
				data = append(data, podDevicesCheckpointEntry{pod, c, r, devIds, allocResp})
			}
		}
	}
	return data
}

// fromCheckpointData populates the registry from the provided checkpointData.
func (a *devAllocationImpl) fromCheckpointData(data []podDevicesCheckpointEntry) {
	a.Lock()
	defer a.Unlock()

	for _, d := range data {
		glog.V(4).Infof("parsing checkpoint entry: %v %v %v %v %v\n",
			d.PodUID, d.ContainerName, d.ResourceName, d.DeviceIDs, d.AllocResp)
		devIDs := sets.NewString()
		for _, devID := range d.DeviceIDs {
			devIDs.Insert(devID)
		}
		allocResp := &pluginapi.AllocateResponse{}
		err := allocResp.Unmarshal(d.AllocResp)
		if err != nil {
			glog.Errorf("failed to unmarshal allocation response for %v %v %v: %v", d.PodUID, d.ContainerName, d.ResourceName, err)
			continue
		}
		a.insert(d.PodUID, d.ContainerName, d.ResourceName, devIDs, allocResp)
	}
}

// collectRunOptions collects container runtime settings from the device
// allocation response for a container to consume the allocated devices.
// The collected data include environment variables, mount points and device
// files, all of which are generated on a per-resource per-container basis.
func (a *devAllocationImpl) collectRunOptions(pod, container string) *DeviceRunContainerOptions {
	a.RLock()
	defer a.RUnlock()

	cmap, found := a.registry[pod]
	if !found {
		return nil
	}
	dmap, found := cmap[container]
	if !found {
		return nil
	}

	opts := &DeviceRunContainerOptions{}
	// Maps to detect duplicate settings.
	devsMap := make(map[string]string)
	mntsMap := make(map[string]string)
	envsMap := make(map[string]string)

	// loops through AllocationResponses of all cached device resources.
	for _, d := range dmap {
		resp := d.allocResp
		// Environment variables
		for k, v := range resp.Envs {
			if e, ok := envsMap[k]; ok {
				if e != v {
					glog.Warningf("environment variable conflict detected on %s: %s and %s", k, e, v)
				}
				continue
			}
			glog.V(4).Infof("environment variable added: %s=%s", k, v)
			envsMap[k] = v
			opts.Envs = append(opts.Envs, kubecontainer.EnvVar{Name: k, Value: v})
		}

		// device files
		for _, dev := range resp.Devices {
			if d, ok := devsMap[dev.ContainerPath]; ok {
				if d != dev.HostPath {
					glog.Warningf("device map conflict detected on %s: %s and %s",
						dev.ContainerPath, d, dev.HostPath)
				}
				continue
			}
			glog.V(4).Infof("device file added: %s=%s", dev.ContainerPath, dev.HostPath)
			devsMap[dev.ContainerPath] = dev.HostPath
			opts.Devices = append(opts.Devices, kubecontainer.DeviceInfo{
				PathOnHost:      dev.HostPath,
				PathInContainer: dev.ContainerPath,
				Permissions:     dev.Permissions,
			})
		}

		// Mount points
		for _, mnt := range resp.Mounts {
			if m, ok := mntsMap[mnt.ContainerPath]; ok {
				if m != mnt.HostPath {
					glog.Warningf("mount point conflict detected on %s: %s and %s",
						mnt.ContainerPath, m, mnt.HostPath)
				}
				continue
			}
			glog.V(4).Infof("add mount %s %s", mnt.ContainerPath, mnt.HostPath)
			mntsMap[mnt.ContainerPath] = mnt.HostPath
			opts.Mounts = append(opts.Mounts, kubecontainer.Mount{
				Name:          mnt.ContainerPath,
				ContainerPath: mnt.ContainerPath,
				HostPath:      mnt.HostPath,
				ReadOnly:      mnt.ReadOnly,
				// TODO: This may need to be part of Device plugin API.
				SELinuxRelabel: false,
			})
		}
	}
	return opts
}

// allocations returns the set of Pods currently known.
func (a *devAllocationImpl) allocations() map[string]containerMap {
	a.RLock()
	defer a.RUnlock()

	ret := make(map[string]containerMap)
	for p, cmap := range a.registry {
		ret[p] = make(map[string]resourceMap)
		for c, dmap := range cmap {
			ret[p][c] = make(map[string]deviceInfo)
			for r, dinfo := range dmap {
				ret[p][c][r] = deviceInfo{
					status:    dinfo.status,
					ids:       dinfo.ids,
					allocResp: dinfo.allocResp,
				}
			}
		}
	}
	return ret
}

// pods returns the set of Pods currently known.
func (a *devAllocationImpl) pods() (r sets.String) {
	a.RLock()
	defer a.RUnlock()

	r = sets.NewString()
	for k := range a.registry {
		r.Insert(k)
	}
	return r
}

// insert adds a new record to the allocation registry.
// Note this method is supposed to be called with the write lock held.
// It is used for reservation when resp is nil.
func (a *devAllocationImpl) insert(pod, container, resource string, devices sets.String, resp *pluginapi.AllocateResponse) {
	if _, found := a.registry[pod]; !found {
		a.registry[pod] = make(containerMap)
	}
	if _, found := a.registry[pod][container]; !found {
		a.registry[pod][container] = make(resourceMap)
	}
	if resp != nil {
		a.registry[pod][container][resource] = deviceInfo{
			status:    devAllocated,
			ids:       devices,
			allocResp: resp,
		}
	} else {
		a.registry[pod][container][resource] = deviceInfo{
			status:    devReserved,
			ids:       devices,
			allocResp: nil,
		}
	}
}
