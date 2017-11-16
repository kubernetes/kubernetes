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
	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/util/sets"
	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1alpha"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

type deviceAllocateInfo struct {
	// deviceIds contains device Ids allocated to this container for the given resourceName.
	deviceIds sets.String
	// allocResp contains cached rpc AllocateResponse.
	allocResp *pluginapi.AllocateResponse
}

type resourceAllocateInfo map[string]deviceAllocateInfo // Keyed by resourceName.
type containerDevices map[string]resourceAllocateInfo   // Keyed by containerName.
type podDevices map[string]containerDevices             // Keyed by podUID.

func (pdev podDevices) pods() sets.String {
	ret := sets.NewString()
	for k := range pdev {
		ret.Insert(k)
	}
	return ret
}

func (pdev podDevices) insert(podUID, contName, resource string, devices sets.String, resp *pluginapi.AllocateResponse) {
	if _, podExists := pdev[podUID]; !podExists {
		pdev[podUID] = make(containerDevices)
	}
	if _, contExists := pdev[podUID][contName]; !contExists {
		pdev[podUID][contName] = make(resourceAllocateInfo)
	}
	pdev[podUID][contName][resource] = deviceAllocateInfo{
		deviceIds: devices,
		allocResp: resp,
	}
}

func (pdev podDevices) delete(pods []string) {
	for _, uid := range pods {
		delete(pdev, uid)
	}
}

// Returns list of device Ids allocated to the given container for the given resource.
// Returns nil if we don't have cached state for the given <podUID, contName, resource>.
func (pdev podDevices) containerDevices(podUID, contName, resource string) sets.String {
	if _, podExists := pdev[podUID]; !podExists {
		return nil
	}
	if _, contExists := pdev[podUID][contName]; !contExists {
		return nil
	}
	devs, resourceExists := pdev[podUID][contName][resource]
	if !resourceExists {
		return nil
	}
	return devs.deviceIds
}

// Returns all of devices allocated to the pods being tracked, keyed by resourceName.
func (pdev podDevices) devices() map[string]sets.String {
	ret := make(map[string]sets.String)
	for _, containerDevices := range pdev {
		for _, resources := range containerDevices {
			for resource, devices := range resources {
				if _, exists := ret[resource]; !exists {
					ret[resource] = sets.NewString()
				}
				ret[resource] = ret[resource].Union(devices.deviceIds)
			}
		}
	}
	return ret
}

type checkpointEntry struct {
	PodUID        string
	ContainerName string
	ResourceName  string
	DeviceIDs     []string
	AllocResp     []byte
}

// checkpointData struct is used to store pod to device allocation information
// in a checkpoint file.
// TODO: add version control when we need to change checkpoint format.
type checkpointData struct {
	Entries []checkpointEntry
}

// Turns podDevices to checkpointData.
func (pdev podDevices) toCheckpointData() checkpointData {
	var data checkpointData
	for podUID, containerDevices := range pdev {
		for conName, resources := range containerDevices {
			for resource, devices := range resources {
				devIds := devices.deviceIds.UnsortedList()
				allocResp, err := devices.allocResp.Marshal()
				if err != nil {
					glog.Errorf("Can't marshal allocResp for %v %v %v: %v", podUID, conName, resource, err)
					continue
				}
				data.Entries = append(data.Entries, checkpointEntry{podUID, conName, resource, devIds, allocResp})
			}
		}
	}
	return data
}

// Populates podDevices from the passed in checkpointData.
func (pdev podDevices) fromCheckpointData(data checkpointData) {
	for _, entry := range data.Entries {
		glog.V(2).Infof("Get checkpoint entry: %v %v %v %v %v\n",
			entry.PodUID, entry.ContainerName, entry.ResourceName, entry.DeviceIDs, entry.AllocResp)
		devIDs := sets.NewString()
		for _, devID := range entry.DeviceIDs {
			devIDs.Insert(devID)
		}
		allocResp := &pluginapi.AllocateResponse{}
		err := allocResp.Unmarshal(entry.AllocResp)
		if err != nil {
			glog.Errorf("Can't unmarshal allocResp for %v %v %v: %v", entry.PodUID, entry.ContainerName, entry.ResourceName, err)
			continue
		}
		pdev.insert(entry.PodUID, entry.ContainerName, entry.ResourceName, devIDs, allocResp)
	}
}

// Returns combined container runtime settings to consume the container's allocated devices.
func (pdev podDevices) deviceRunContainerOptions(podUID, contName string) *DeviceRunContainerOptions {
	containers, exists := pdev[podUID]
	if !exists {
		return nil
	}
	resources, exists := containers[contName]
	if !exists {
		return nil
	}
	opts := &DeviceRunContainerOptions{}
	// Maps to detect duplicate settings.
	devsMap := make(map[string]string)
	mountsMap := make(map[string]string)
	envsMap := make(map[string]string)
	// Loops through AllocationResponses of all cached device resources.
	for _, devices := range resources {
		resp := devices.allocResp
		// Each Allocate response has the following artifacts.
		// Environment variables
		// Mount points
		// Device files
		// These artifacts are per resource per container.
		// Updates RunContainerOptions.Envs.
		for k, v := range resp.Envs {
			if e, ok := envsMap[k]; ok {
				glog.V(3).Infof("skip existing env %s %s", k, v)
				if e != v {
					glog.Errorf("Environment variable %s has conflicting setting: %s and %s", k, e, v)
				}
				continue
			}
			glog.V(4).Infof("add env %s %s", k, v)
			envsMap[k] = v
			opts.Envs = append(opts.Envs, kubecontainer.EnvVar{Name: k, Value: v})
		}

		// Updates RunContainerOptions.Devices.
		for _, dev := range resp.Devices {
			if d, ok := devsMap[dev.ContainerPath]; ok {
				glog.V(3).Infof("skip existing device %s %s", dev.ContainerPath, dev.HostPath)
				if d != dev.HostPath {
					glog.Errorf("Container device %s has conflicting mapping host devices: %s and %s",
						dev.ContainerPath, d, dev.HostPath)
				}
				continue
			}
			glog.V(4).Infof("add device %s %s", dev.ContainerPath, dev.HostPath)
			devsMap[dev.ContainerPath] = dev.HostPath
			opts.Devices = append(opts.Devices, kubecontainer.DeviceInfo{
				PathOnHost:      dev.HostPath,
				PathInContainer: dev.ContainerPath,
				Permissions:     dev.Permissions,
			})
		}
		// Updates RunContainerOptions.Mounts.
		for _, mount := range resp.Mounts {
			if m, ok := mountsMap[mount.ContainerPath]; ok {
				glog.V(3).Infof("skip existing mount %s %s", mount.ContainerPath, mount.HostPath)
				if m != mount.HostPath {
					glog.Errorf("Container mount %s has conflicting mapping host mounts: %s and %s",
						mount.ContainerPath, m, mount.HostPath)
				}
				continue
			}
			glog.V(4).Infof("add mount %s %s", mount.ContainerPath, mount.HostPath)
			mountsMap[mount.ContainerPath] = mount.HostPath
			opts.Mounts = append(opts.Mounts, kubecontainer.Mount{
				Name:          mount.ContainerPath,
				ContainerPath: mount.ContainerPath,
				HostPath:      mount.HostPath,
				ReadOnly:      mount.ReadOnly,
				// TODO: This may need to be part of Device plugin API.
				SELinuxRelabel: false,
			})
		}
	}
	return opts
}
