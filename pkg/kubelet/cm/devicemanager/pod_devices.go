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

package devicemanager

import (
	"maps"
	"sync"

	"google.golang.org/protobuf/proto"
	"k8s.io/klog/v2"

	"k8s.io/apimachinery/pkg/util/sets"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
	"k8s.io/kubernetes/pkg/kubelet/cm/devicemanager/checkpoint"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

type deviceAllocateInfo struct {
	// deviceIds contains device Ids allocated or reserved for this container for the given resourceName.
	deviceIds checkpoint.DevicesPerNUMA
	// allocResp contains the cached AllocateResponse for a committed allocation.
	allocResp *pluginapi.ContainerAllocateResponse
	// state distinguishes devices reserved for an in-flight Allocate call from
	// devices whose Allocate response has been committed.
	state allocationState
}

type allocationState uint8

const (
	// allocationCommitted is intentionally the zero value so checkpoint-restored
	// and test-created deviceAllocateInfo values retain their existing semantics.
	allocationCommitted allocationState = iota
	allocationReserved
)

type resourceAllocateInfo map[string]deviceAllocateInfo // Keyed by resourceName.
type containerDevices map[string]resourceAllocateInfo   // Keyed by containerName.
type podDevices struct {
	sync.RWMutex
	devs map[string]containerDevices // Keyed by podUID.
}

// NewPodDevices is a function that returns object of podDevices type with its own guard
// RWMutex and a map where key is a pod UID and value contains
// container devices information of type containerDevices.
func newPodDevices() *podDevices {
	return &podDevices{devs: make(map[string]containerDevices)}
}

// pods returns pods with at least one committed device allocation. Pods that
// contain only reservations are excluded from garbage collection while their
// RPC is in flight.
func (pdev *podDevices) pods() sets.Set[string] {
	pdev.RLock()
	defer pdev.RUnlock()
	ret := sets.New[string]()
	for podUID, containers := range pdev.devs {
		for _, resources := range containers {
			for _, devices := range resources {
				if devices.state == allocationCommitted {
					ret.Insert(podUID)
					break
				}
			}
			if ret.Has(podUID) {
				break
			}
		}
	}
	return ret
}

func (pdev *podDevices) size() int {
	pdev.RLock()
	defer pdev.RUnlock()
	return len(pdev.devs)
}

func (pdev *podDevices) hasPod(podUID string) bool {
	pdev.RLock()
	defer pdev.RUnlock()
	for _, resources := range pdev.devs[podUID] {
		for _, devices := range resources {
			if devices.state == allocationCommitted {
				return true
			}
		}
	}
	return false
}

func (pdev *podDevices) insert(podUID, contName, resource string, devices checkpoint.DevicesPerNUMA, resp *pluginapi.ContainerAllocateResponse) {
	pdev.Lock()
	defer pdev.Unlock()
	if _, podExists := pdev.devs[podUID]; !podExists {
		pdev.devs[podUID] = make(containerDevices)
	}
	if _, contExists := pdev.devs[podUID][contName]; !contExists {
		pdev.devs[podUID][contName] = make(resourceAllocateInfo)
	}
	pdev.devs[podUID][contName][resource] = deviceAllocateInfo{
		deviceIds: devices,
		allocResp: resp,
		state:     allocationCommitted,
	}
}

// reserve creates an empty reservation for a pod, container and resource. The
// empty entry prevents another allocation for the same owner from starting
// while this one is in flight.
func (pdev *podDevices) reserve(podUID, contName, resource string) bool {
	pdev.Lock()
	defer pdev.Unlock()
	if resources, podExists := pdev.devs[podUID]; podExists {
		if allocations, contExists := resources[contName]; contExists {
			if _, resourceExists := allocations[resource]; resourceExists {
				return false
			}
		}
	}
	if _, podExists := pdev.devs[podUID]; !podExists {
		pdev.devs[podUID] = make(containerDevices)
	}
	if _, contExists := pdev.devs[podUID][contName]; !contExists {
		pdev.devs[podUID][contName] = make(resourceAllocateInfo)
	}
	pdev.devs[podUID][contName][resource] = deviceAllocateInfo{
		deviceIds: checkpoint.NewDevicesPerNUMA(),
		state:     allocationReserved,
	}
	return true
}

// addDevicesToReservation records devices selected for an in-flight Allocate
// call. It may be called more than once as device selection progresses around
// GetPreferredAllocation RPCs.
func (pdev *podDevices) addDevicesToReservation(podUID, contName, resource string, devices sets.Set[string]) bool {
	pdev.Lock()
	defer pdev.Unlock()
	allocation, exists := pdev.devs[podUID][contName][resource]
	if !exists || allocation.state != allocationReserved {
		return false
	}
	reserved := allocation.deviceIds.Devices().Union(devices)
	allocation.deviceIds = checkpoint.DevicesPerNUMA{nodeWithoutTopology: reserved.UnsortedList()}
	pdev.devs[podUID][contName][resource] = allocation
	return true
}

// commitReservation replaces a reservation with the successful Allocate
// result. The caller is responsible for serializing this transition with
// allocatedDevices rebuilds.
func (pdev *podDevices) commitReservation(podUID, contName, resource string, devices checkpoint.DevicesPerNUMA, resp *pluginapi.ContainerAllocateResponse) bool {
	pdev.Lock()
	defer pdev.Unlock()
	allocation, exists := pdev.devs[podUID][contName][resource]
	if !exists || allocation.state != allocationReserved {
		return false
	}
	pdev.devs[podUID][contName][resource] = deviceAllocateInfo{
		deviceIds: devices,
		allocResp: resp,
		state:     allocationCommitted,
	}
	return true
}

// rollbackReservation removes an in-flight reservation without disturbing
// committed allocations for the same device, such as reusable init-container
// devices.
func (pdev *podDevices) rollbackReservation(podUID, contName, resource string) bool {
	pdev.Lock()
	defer pdev.Unlock()
	allocation, exists := pdev.devs[podUID][contName][resource]
	if !exists || allocation.state != allocationReserved {
		return false
	}
	pdev.deleteResourceLocked(podUID, contName, resource)
	return true
}

func (pdev *podDevices) deleteResourceLocked(podUID, contName, resource string) {
	delete(pdev.devs[podUID][contName], resource)
	if len(pdev.devs[podUID][contName]) == 0 {
		delete(pdev.devs[podUID], contName)
	}
	if len(pdev.devs[podUID]) == 0 {
		delete(pdev.devs, podUID)
	}
}

// delete removes committed allocations for the specified pods while preserving
// any reservations whose Allocate RPCs are still in flight.
func (pdev *podDevices) delete(pods []string) {
	pdev.Lock()
	defer pdev.Unlock()
	for _, podUID := range pods {
		for contName, resources := range pdev.devs[podUID] {
			for resource, devices := range resources {
				if devices.state == allocationCommitted {
					pdev.deleteResourceLocked(podUID, contName, resource)
				}
			}
		}
	}
}

// Returns list of device Ids allocated to the given pod for the given resource.
// Returns nil if we don't have cached state for the given <podUID, resource>.
func (pdev *podDevices) podDevices(podUID, resource string) sets.Set[string] {
	pdev.RLock()
	defer pdev.RUnlock()

	ret := sets.New[string]()
	for _, resources := range pdev.devs[podUID] {
		devices, exists := resources[resource]
		if exists && devices.state == allocationCommitted {
			ret = ret.Union(devices.deviceIds.Devices())
		}
	}
	return ret
}

// Returns list of device Ids allocated to the given container for the given resource.
// Returns nil if we don't have cached state for the given <podUID, contName, resource>.
func (pdev *podDevices) containerDevices(podUID, contName, resource string) sets.Set[string] {
	pdev.RLock()
	defer pdev.RUnlock()
	if _, podExists := pdev.devs[podUID]; !podExists {
		return nil
	}
	if _, contExists := pdev.devs[podUID][contName]; !contExists {
		return nil
	}
	devs, resourceExists := pdev.devs[podUID][contName][resource]
	if !resourceExists || devs.state != allocationCommitted {
		return nil
	}
	return devs.deviceIds.Devices()
}

// Populates allocatedResources with the device resources allocated to the specified <podUID, contName>.
func (pdev *podDevices) addContainerAllocatedResources(podUID, contName string, allocatedResources map[string]sets.Set[string]) {
	pdev.RLock()
	defer pdev.RUnlock()
	containers, exists := pdev.devs[podUID]
	if !exists {
		return
	}
	resources, exists := containers[contName]
	if !exists {
		return
	}
	for resource, devices := range resources {
		if devices.state != allocationCommitted {
			continue
		}
		allocatedResources[resource] = allocatedResources[resource].Union(devices.deviceIds.Devices())
	}
}

// Removes the device resources allocated to the specified <podUID, contName> from allocatedResources.
func (pdev *podDevices) removeContainerAllocatedResources(podUID, contName string, allocatedResources map[string]sets.Set[string]) {
	pdev.RLock()
	defer pdev.RUnlock()
	containers, exists := pdev.devs[podUID]
	if !exists {
		return
	}
	resources, exists := containers[contName]
	if !exists {
		return
	}
	for resource, devices := range resources {
		if devices.state != allocationCommitted {
			continue
		}
		allocatedResources[resource] = allocatedResources[resource].Difference(devices.deviceIds.Devices())
	}
}

// Returns all committed and reserved devices being tracked, keyed by resourceName.
func (pdev *podDevices) devices() map[string]sets.Set[string] {
	ret := make(map[string]sets.Set[string])
	pdev.RLock()
	defer pdev.RUnlock()
	for _, containerDevices := range pdev.devs {
		for _, resources := range containerDevices {
			for resource, devices := range resources {
				if devices.state == allocationReserved && devices.deviceIds.Devices().Len() == 0 {
					continue
				}
				if _, exists := ret[resource]; !exists {
					ret[resource] = sets.New[string]()
				}
				ret[resource] = ret[resource].Union(devices.deviceIds.Devices())
			}
		}
	}
	return ret
}

// Returns podUID and containerName for a device allocated under the given resource.
// The lookup must be scoped by resourceName: device IDs are only unique within a
// resource, so different plugins may expose devices with identical IDs.
func (pdev *podDevices) getPodAndContainerForDevice(resourceName, deviceID string) (string, string) {
	pdev.RLock()
	defer pdev.RUnlock()
	for podUID, containerDevices := range pdev.devs {
		for containerName, resources := range containerDevices {
			if devices, ok := resources[resourceName]; ok && devices.state == allocationCommitted {
				if devices.deviceIds.Devices().Has(deviceID) {
					return podUID, containerName
				}
			}
		}
	}
	return "", ""
}

// Turns podDevices to checkpointData.
func (pdev *podDevices) toCheckpointData(logger klog.Logger) []checkpoint.PodDevicesEntry {
	var data []checkpoint.PodDevicesEntry
	pdev.RLock()
	defer pdev.RUnlock()
	for podUID, containerDevices := range pdev.devs {
		for conName, resources := range containerDevices {
			for resource, devices := range resources {
				if devices.state != allocationCommitted {
					continue
				}
				if devices.allocResp == nil {
					logger.Error(nil, "Can't marshal allocResp, allocation response is missing", "podUID", podUID, "containerName", conName, "resourceName", resource)
					continue
				}

				allocResp, err := proto.Marshal(devices.allocResp)
				if err != nil {
					logger.Error(err, "Can't marshal allocResp", "podUID", podUID, "containerName", conName, "resourceName", resource)
					continue
				}
				data = append(data, checkpoint.PodDevicesEntry{
					PodUID:        podUID,
					ContainerName: conName,
					ResourceName:  resource,
					DeviceIDs:     devices.deviceIds,
					AllocResp:     allocResp})
			}
		}
	}
	return data
}

// Populates podDevices from the passed in checkpointData.
func (pdev *podDevices) fromCheckpointData(logger klog.Logger, data []checkpoint.PodDevicesEntry) {
	for _, entry := range data {
		logger.V(2).Info("Get checkpoint entry",
			"podUID", entry.PodUID, "containerName", entry.ContainerName,
			"resourceName", entry.ResourceName, "deviceIDs", entry.DeviceIDs, "allocated", entry.AllocResp)
		allocResp := &pluginapi.ContainerAllocateResponse{}
		err := proto.Unmarshal(entry.AllocResp, allocResp)
		if err != nil {
			logger.Error(err, "Can't unmarshal allocResp", "podUID", entry.PodUID, "containerName", entry.ContainerName, "resourceName", entry.ResourceName)
			continue
		}
		pdev.insert(entry.PodUID, entry.ContainerName, entry.ResourceName, entry.DeviceIDs, allocResp)
	}
}

// Returns combined container runtime settings to consume the container's allocated devices.
func (pdev *podDevices) deviceRunContainerOptions(logger klog.Logger, podUID, contName string) *DeviceRunContainerOptions {
	pdev.RLock()
	defer pdev.RUnlock()

	containers, exists := pdev.devs[podUID]
	if !exists {
		return nil
	}
	resources, exists := containers[contName]
	if !exists {
		return nil
	}
	opts := &DeviceRunContainerOptions{}
	hasCommittedDevices := false
	// Maps to detect duplicate settings.
	devsMap := make(map[string]string)
	mountsMap := make(map[string]string)
	envsMap := make(map[string]string)
	annotationsMap := make(map[string]string)
	// Keep track of all CDI devices requested for the container.
	allCDIDevices := sets.New[string]()
	// Loops through AllocationResponses of all cached device resources.
	for _, devices := range resources {
		if devices.state != allocationCommitted {
			continue
		}
		hasCommittedDevices = true
		resp := devices.allocResp
		// Each Allocate response has the following artifacts.
		// Environment variables
		// Mount points
		// Device files
		// Container annotations
		// CDI device IDs
		// These artifacts are per resource per container.
		// Updates RunContainerOptions.Envs.
		for k, v := range resp.Envs {
			if e, ok := envsMap[k]; ok {
				logger.V(4).Info("Skip existing env", "envKey", k, "envValue", v)
				if e != v {
					logger.Error(nil, "Environment variable has conflicting setting", "envKey", k, "expected", v, "got", e)
				}
				continue
			}
			logger.V(4).Info("Add env", "envKey", k, "envValue", v)
			envsMap[k] = v
			opts.Envs = append(opts.Envs, kubecontainer.EnvVar{Name: k, Value: v})
		}

		// Updates RunContainerOptions.Devices.
		for _, dev := range resp.Devices {
			if d, ok := devsMap[dev.ContainerPath]; ok {
				logger.V(4).Info("Skip existing device", "containerPath", dev.ContainerPath, "hostPath", dev.HostPath)
				if d != dev.HostPath {
					logger.Error(nil, "Container device has conflicting mapping host devices",
						"containerPath", dev.ContainerPath, "got", d, "expected", dev.HostPath)
				}
				continue
			}
			logger.V(4).Info("Add device", "containerPath", dev.ContainerPath, "hostPath", dev.HostPath)
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
				logger.V(4).Info("Skip existing mount", "containerPath", mount.ContainerPath, "hostPath", mount.HostPath)
				if m != mount.HostPath {
					logger.Error(nil, "Container mount has conflicting mapping host mounts",
						"containerPath", mount.ContainerPath, "conflictingPath", m, "hostPath", mount.HostPath)
				}
				continue
			}
			logger.V(4).Info("Add mount", "containerPath", mount.ContainerPath, "hostPath", mount.HostPath)
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

		// Updates for Annotations
		for k, v := range resp.Annotations {
			if e, ok := annotationsMap[k]; ok {
				logger.V(4).Info("Skip existing annotation", "annotationKey", k, "annotationValue", v)
				if e != v {
					logger.Error(nil, "Annotation has conflicting setting", "annotationKey", k, "expected", e, "got", v)
				}
				continue
			}
			logger.V(4).Info("Add annotation", "annotationKey", k, "annotationValue", v)
			annotationsMap[k] = v
			opts.Annotations = append(opts.Annotations, kubecontainer.Annotation{Name: k, Value: v})
		}

		// Updates for CDI devices.
		cdiDevices := getCDIDeviceInfo(logger, resp, allCDIDevices)
		opts.CDIDevices = append(opts.CDIDevices, cdiDevices...)
	}
	if !hasCommittedDevices {
		return nil
	}

	return opts
}

// getCDIDeviceInfo returns CDI devices from an allocate response
func getCDIDeviceInfo(logger klog.Logger, resp *pluginapi.ContainerAllocateResponse, knownCDIDevices sets.Set[string]) []kubecontainer.CDIDevice {
	var cdiDevices []kubecontainer.CDIDevice
	for _, cdiDevice := range resp.CdiDevices {
		if knownCDIDevices.Has(cdiDevice.Name) {
			logger.V(4).Info("Skip existing CDI Device", "name", cdiDevice.Name)
			continue
		}
		logger.V(4).Info("Add CDI device", "name", cdiDevice.Name)
		knownCDIDevices.Insert(cdiDevice.Name)

		device := kubecontainer.CDIDevice{
			Name: cdiDevice.Name,
		}
		cdiDevices = append(cdiDevices, device)
	}

	return cdiDevices
}

// getContainerDevices returns the devices assigned to the provided container for all ResourceNames
func (pdev *podDevices) getContainerDevices(podUID, contName string) ResourceDeviceInstances {
	pdev.RLock()
	defer pdev.RUnlock()

	if _, podExists := pdev.devs[podUID]; !podExists {
		return nil
	}
	if _, contExists := pdev.devs[podUID][contName]; !contExists {
		return nil
	}
	resDev := NewResourceDeviceInstances()
	hasCommittedAllocation := false
	for resource, allocateInfo := range pdev.devs[podUID][contName] {
		if allocateInfo.state != allocationCommitted {
			continue
		}
		hasCommittedAllocation = true
		if len(allocateInfo.deviceIds) == 0 {
			continue
		}
		devicePluginMap := make(map[string]*pluginapi.Device)
		for numaid, devlist := range allocateInfo.deviceIds {
			for _, devID := range devlist {
				var topology *pluginapi.TopologyInfo
				if numaid != nodeWithoutTopology {
					NUMANodes := []*pluginapi.NUMANode{{ID: numaid}}
					if pDev, ok := devicePluginMap[devID]; ok && pDev.Topology != nil {
						if nodes := pDev.Topology.GetNodes(); nodes != nil {
							NUMANodes = append(NUMANodes, nodes...)
						}
					}

					// ID and Healthy are not relevant here.
					topology = &pluginapi.TopologyInfo{Nodes: NUMANodes}
				}
				devicePluginMap[devID] = &pluginapi.Device{
					Topology: topology,
				}
			}
		}
		resDev[resource] = devicePluginMap
	}
	if !hasCommittedAllocation {
		return nil
	}
	return resDev
}

// DeviceInstances is a mapping device name -> plugin device data
type DeviceInstances map[string]*pluginapi.Device

// ResourceDeviceInstances is a mapping resource name -> DeviceInstances
type ResourceDeviceInstances map[string]DeviceInstances

// NewResourceDeviceInstances returns a new ResourceDeviceInstances
func NewResourceDeviceInstances() ResourceDeviceInstances {
	return make(ResourceDeviceInstances)
}

// Clone returns a clone of ResourceDeviceInstances
func (rdev ResourceDeviceInstances) Clone() ResourceDeviceInstances {
	clone := NewResourceDeviceInstances()
	for resourceName, resourceDevs := range rdev {
		clone[resourceName] = maps.Clone(resourceDevs)
	}
	return clone
}

// Filter takes a condition set expressed as map[string]sets.Set[string] and returns a new
// ResourceDeviceInstances with only the devices matching the condition set.
func (rdev ResourceDeviceInstances) Filter(cond map[string]sets.Set[string]) ResourceDeviceInstances {
	filtered := NewResourceDeviceInstances()
	for resourceName, filterIDs := range cond {
		if _, exists := rdev[resourceName]; !exists {
			continue
		}
		filtered[resourceName] = DeviceInstances{}
		for instanceID, instance := range rdev[resourceName] {
			if filterIDs.Has(instanceID) {
				filtered[resourceName][instanceID] = instance
			}
		}
	}
	return filtered
}
