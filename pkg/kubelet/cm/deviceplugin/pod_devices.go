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
	"sync"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1alpha"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

type devAllocation interface {
	// get a deep copy of all maps managed.
	allocations() map[string]containerMap

	// get the set of pod IDs known to the device manager
	pods() sets.String

	// Check if the specified pod uid is known to have allocations
	hasPod(pod string) bool

	// Enumerate all devices known to be already allocated by returning a map
	// keyed by resource name.
	devices() map[string]sets.String

	// Enumerate all known device allocations for the given container on
	// specified resource.
	devicesByContainer(pod, container, resource string) sets.String

	// Insert new allocation record to the registry.
	insert(pod, container, resource string, devices sets.String, resp *pluginapi.AllocateResponse)

	// Update the allocation registry based on the provided list of active
	// pods. Records for inactive pods are pruned.
	update(activePods []*v1.Pod) map[string]sets.String

	// Marshal the registry into data for checkpointing
	toCheckpointData() []podDevicesCheckpointEntry

	// Unmarshal checkpointed data into registry
	fromCheckpointData(data []podDevicesCheckpointEntry)

	// Collect device information into container runtime settings.
	collectRunOptions(pod, container string) *DeviceRunContainerOptions
}

type deviceInfo struct {
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

// podDevicesCheckpointEntry is used to record <pod, container> to device allocation information.
type podDevicesCheckpointEntry struct {
	PodUID        string
	ContainerName string
	ResourceName  string
	DeviceIDs     []string
	AllocResp     []byte
}

// newAllocation() instantiates a devAllocationImpl object to be used as an
// implementation of devAllocation interface.
func newAllocation() *devAllocationImpl {
	return &devAllocationImpl{
		registry: make(map[string]containerMap),
	}
}

// allocations() returns the set of Pods currently known.
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
					ids:       dinfo.ids,
					allocResp: dinfo.allocResp,
				}
			}
		}
	}
	return ret
}

// pods() returns the set of Pods currently known.
func (a *devAllocationImpl) pods() (r sets.String) {
	a.RLock()
	defer a.RUnlock()

	r = sets.NewString()
	for k := range a.registry {
		r.Insert(k)
	}
	return r
}

// hasPod() checks if there are records associated with the given pod.
func (a *devAllocationImpl) hasPod(pod string) bool {
	a.RLock()
	defer a.RUnlock()
	_, found := a.registry[pod]
	return found
}

// insert() adds a new record to the allocation map.
func (a *devAllocationImpl) insert(pod, container, resource string, devices sets.String, resp *pluginapi.AllocateResponse) {
	a.Lock()
	defer a.Unlock()
	a.doInsert(pod, container, resource, devices, resp)
}

// doInsert() adds a new record to the allocation map.
// Note this method is supposed to be called with the write lock held.
func (a *devAllocationImpl) doInsert(pod, container, resource string, devices sets.String, resp *pluginapi.AllocateResponse) {
	if _, found := a.registry[pod]; !found {
		a.registry[pod] = make(containerMap)
	}
	if _, found := a.registry[pod][container]; !found {
		a.registry[pod][container] = make(resourceMap)
	}
	a.registry[pod][container][resource] = deviceInfo{
		ids:       devices,
		allocResp: resp,
	}
}

// update() updates the pod allocation registry by list of active pods.
// The function returns a map from resource name to set of device IDs.
func (a *devAllocationImpl) update(activePods []*v1.Pod) map[string]sets.String {
	a.Lock()
	defer a.Unlock()

	active := sets.NewString()
	for _, p := range activePods {
		active.Insert(string(p.UID))
	}

	ret := make(map[string]sets.String)
	for p, cmap := range a.registry {
		if active.Has(p) {
			for _, rmap := range cmap {
				for r, dinfo := range rmap {
					if _, ok := ret[r]; !ok {
						ret[r] = sets.NewString()
					}
					ret[r] = ret[r].Union(dinfo.ids)
				}
			}
			continue
		} else {
			delete(a.registry, p)
			glog.V(4).Infof("removed inactive pod: %v", p)
		}
	}
	return ret
}

// devicesByContainer() returns a set of device IDs allocated to the given container
// for the given resource. nil is returned no record is found.
func (a *devAllocationImpl) devicesByContainer(pod, container, resource string) sets.String {
	a.RLock()
	defer a.RUnlock()

	if _, found := a.registry[pod]; !found {
		return nil
	}
	if _, found := a.registry[pod][container]; !found {
		return nil
	}
	dInfo, found := a.registry[pod][container][resource]
	if !found {
		return nil
	}
	return dInfo.ids
}

// devices() returns all devices already allocated
func (a *devAllocationImpl) devices() map[string]sets.String {
	a.RLock()
	defer a.RUnlock()

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

// toCheckpointData() translates the devAllocation to data to be checkpointed.
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

// Populates registry from the passed in checkpointData.
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
		a.doInsert(d.PodUID, d.ContainerName, d.ResourceName, devIDs, allocResp)
	}
}

// deviceRunContainerOptions() collects container runtime settings from the
// device allocation response for a container to consume the allocated devices.
// The collected information include environment variables, mount points and
// device files, all of which are generated on a per resource per container
// basis.
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
				} else {
					glog.V(4).Infof("skipping duplicated environment variable: %s=%s", k, v)
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
				} else {
					glog.V(4).Infof("skipping duplicated device file %s=%s", dev.ContainerPath, dev.HostPath)
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
				} else {
					glog.V(4).Infof("skipping duplicated mount point %s=%s", mnt.ContainerPath, mnt.HostPath)
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
