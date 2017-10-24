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
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"sync"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
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
// Handler defines the functions used to manage and access device plugin resources.
type Handler interface {
	// Start starts device plugin registration service.
	Start() error
	// Devices returns all of registered devices keyed by resourceName.
	Devices() map[string][]pluginapi.Device
	// Allocate attempts to allocate all of required extended resources for
	// the input container, issues an Allocate rpc request for each of such
	// resources, processes their AllocateResponses, and updates the cached
	// containerDevices on success.
	Allocate(pod *v1.Pod, container *v1.Container, activePods []*v1.Pod) error
	// GetDeviceRunContainerOptions checks whether we have cached containerDevices
	// for the passed-in <pod, container> and returns its DeviceRunContainerOptions
	// for the found one. An empty struct is returned in case no cached state is found.
	GetDeviceRunContainerOptions(pod *v1.Pod, container *v1.Container) *DeviceRunContainerOptions
}

// HandlerImpl implements the actual functionality to manage device plugin resources.
type HandlerImpl struct {
	// TODO: consider to change this to RWMutex.
	sync.Mutex
	devicePluginManager Manager
	// devicePluginManagerMonitorCallback is used for testing only.
	devicePluginManagerMonitorCallback MonitorCallback
	// allDevices contains all of registered resourceNames and their exported device IDs.
	allDevices map[string]sets.String
	// allocatedDevices contains allocated deviceIds, keyed by resourceName.
	allocatedDevices map[string]sets.String
	// podDevices contains pod to allocated device mapping.
	podDevices podDevices
}

// NewHandlerImpl creates a HandlerImpl to manage device plugin resources.
// updateCapacityFunc is called to update ContainerManager capacity when
// device capacity changes.
func NewHandlerImpl(updateCapacityFunc func(v1.ResourceList)) (*HandlerImpl, error) {
	glog.V(2).Infof("Creating Device Plugin Handler")
	handler := &HandlerImpl{
		allDevices:       make(map[string]sets.String),
		allocatedDevices: make(map[string]sets.String),
		podDevices:       make(podDevices),
	}

	deviceManagerMonitorCallback := func(resourceName string, added, updated, deleted []pluginapi.Device) {
		var capacity = v1.ResourceList{}
		kept := append(updated, added...)
		if _, ok := handler.allDevices[resourceName]; !ok {
			handler.allDevices[resourceName] = sets.NewString()
		}
		// For now, Handler only keeps track of healthy devices.
		// We can revisit this later when the need comes to track unhealthy devices here.
		for _, dev := range kept {
			if dev.Health == pluginapi.Healthy {
				handler.allDevices[resourceName].Insert(dev.ID)
			} else {
				handler.allDevices[resourceName].Delete(dev.ID)
			}
		}
		for _, dev := range deleted {
			handler.allDevices[resourceName].Delete(dev.ID)
		}
		capacity[v1.ResourceName(resourceName)] = *resource.NewQuantity(int64(handler.allDevices[resourceName].Len()), resource.DecimalSI)
		updateCapacityFunc(capacity)
	}

	mgr, err := NewManagerImpl(pluginapi.KubeletSocket, deviceManagerMonitorCallback)
	if err != nil {
		return nil, fmt.Errorf("Failed to initialize device plugin manager: %+v", err)
	}

	handler.devicePluginManager = mgr
	handler.devicePluginManagerMonitorCallback = deviceManagerMonitorCallback
	// Loads in allocatedDevices information from disk.
	err = handler.readCheckpoint()
	if err != nil {
		glog.Warningf("Continue after failing to read checkpoint file. Device allocation info may NOT be up-to-date. Err: %v", err)
	}
	return handler, nil
}

// Start starts device plugin registration service.
func (h *HandlerImpl) Start() error {
	return h.devicePluginManager.Start()
}

// Devices returns all of registered devices keyed by resourceName.
func (h *HandlerImpl) Devices() map[string][]pluginapi.Device {
	return h.devicePluginManager.Devices()
}

// Returns list of device Ids we need to allocate with Allocate rpc call.
// Returns empty list in case we don't need to issue the Allocate rpc call.
func (h *HandlerImpl) devicesToAllocate(podUID, contName, resource string, required int) (sets.String, error) {
	h.Lock()
	defer h.Unlock()
	needed := required
	// Gets list of devices that have already been allocated.
	// This can happen if a container restarts for example.
	devices := h.podDevices.containerDevices(podUID, contName, resource)
	if devices != nil {
		glog.V(3).Infof("Found pre-allocated devices for resource %s container %q in Pod %q: %v", resource, contName, podUID, devices.List())
		needed = needed - devices.Len()
		// A pod's resource is not expected to change once admitted by the API server,
		// so just fail loudly here. We can revisit this part if this no longer holds.
		if needed != 0 {
			return nil, fmt.Errorf("pod %v container %v changed request for resource %v from %v to %v", podUID, contName, resource, devices.Len(), required)
		}
	}
	if needed == 0 {
		// No change, no work.
		return nil, nil
	}
	devices = sets.NewString()
	// Needs to allocate additional devices.
	if h.allocatedDevices[resource] == nil {
		h.allocatedDevices[resource] = sets.NewString()
	}
	// Gets Devices in use.
	devicesInUse := h.allocatedDevices[resource]
	// Gets a list of available devices.
	available := h.allDevices[resource].Difference(devicesInUse)
	if int(available.Len()) < needed {
		return nil, fmt.Errorf("requested number of devices unavailable for %s. Requested: %d, Available: %d", resource, needed, available.Len())
	}
	allocated := available.UnsortedList()[:needed]
	// Updates h.allocatedDevices with allocated devices to prevent them
	// from being allocated to other pods/containers, given that we are
	// not holding lock during the rpc call.
	for _, device := range allocated {
		h.allocatedDevices[resource].Insert(device)
		devices.Insert(device)
	}
	return devices, nil
}

// Allocate attempts to allocate all of required extended resources for
// the input container, issues an Allocate rpc request for each of such
// resources, processes their AllocateResponses, and updates the cached
// containerDevices on success.
func (h *HandlerImpl) Allocate(pod *v1.Pod, container *v1.Container, activePods []*v1.Pod) error {
	podUID := string(pod.UID)
	contName := container.Name
	allocatedDevicesUpdated := false
	for k, v := range container.Resources.Limits {
		resource := string(k)
		needed := int(v.Value())
		glog.V(3).Infof("needs %d %s", needed, resource)
		if _, registeredResource := h.allDevices[resource]; !registeredResource {
			continue
		}
		// Updates allocatedDevices to garbage collect any stranded resources
		// before doing the device plugin allocation.
		if !allocatedDevicesUpdated {
			h.updateAllocatedDevices(activePods)
			allocatedDevicesUpdated = true
		}
		allocDevices, err := h.devicesToAllocate(podUID, contName, resource, needed)
		if err != nil {
			return err
		}
		if allocDevices == nil || len(allocDevices) <= 0 {
			continue
		}
		// devicePluginManager.Allocate involves RPC calls to device plugin, which
		// could be heavy-weight. Therefore we want to perform this operation outside
		// mutex lock. Note if Allcate call fails, we may leave container resources
		// partially allocated for the failed container. We rely on updateAllocatedDevices()
		// to garbage collect these resources later. Another side effect is that if
		// we have X resource A and Y resource B in total, and two containers, container1
		// and container2 both require X resource A and Y resource B. Both allocation
		// requests may fail if we serve them in mixed order.
		// TODO: may revisit this part later if we see inefficient resource allocation
		// in real use as the result of this.
		resp, err := h.devicePluginManager.Allocate(resource, allocDevices.UnsortedList())
		if err != nil {
			return err
		}

		// Update internal cached podDevices state.
		h.Lock()
		if oldDevices := h.podDevices.containerDevices(podUID, contName, resource); oldDevices != nil {
			freeDevices := oldDevices.Difference(allocDevices)
			glog.V(3).Infof("old %v alloc %v free devices %v for resource %s", oldDevices.UnsortedList(), allocDevices.UnsortedList(), freeDevices.UnsortedList(), resource)
			h.allocatedDevices[resource] = h.allocatedDevices[resource].Difference(freeDevices)
		}
		h.podDevices.insert(podUID, contName, resource, allocDevices, resp)
		h.Unlock()
	}

	// Checkpoints device to container allocation information.
	return h.writeCheckpoint()
}

// GetDeviceRunContainerOptions checks whether we have cached containerDevices
// for the passed-in <pod, container> and returns its DeviceRunContainerOptions
// for the found one. An empty struct is returned in case no cached state is found.
func (h *HandlerImpl) GetDeviceRunContainerOptions(pod *v1.Pod, container *v1.Container) *DeviceRunContainerOptions {
	h.Lock()
	defer h.Unlock()
	return h.podDevices.deviceRunContainerOptions(string(pod.UID), container.Name)
}

// updateAllocatedDevices gets a list of active pods and then frees any Devices that are bound to
// terminated pods. Returns error on failure.
func (h *HandlerImpl) updateAllocatedDevices(activePods []*v1.Pod) {
	h.Lock()
	defer h.Unlock()
	activePodUids := sets.NewString()
	for _, pod := range activePods {
		activePodUids.Insert(string(pod.UID))
	}
	allocatedPodUids := h.podDevices.pods()
	podsToBeRemoved := allocatedPodUids.Difference(activePodUids)
	if len(podsToBeRemoved) <= 0 {
		return
	}
	glog.V(5).Infof("pods to be removed: %v", podsToBeRemoved.List())
	h.podDevices.delete(podsToBeRemoved.List())
	// Regenerated allocatedDevices after we update pod allocation information.
	h.allocatedDevices = h.podDevices.devices()
}

// Checkpoints device to container allocation information to disk.
func (h *HandlerImpl) writeCheckpoint() error {
	h.Lock()
	defer h.Unlock()
	filepath := h.devicePluginManager.CheckpointFile()
	data := h.podDevices.toCheckpointData()
	dataJSON, err := json.Marshal(data)
	if err != nil {
		return err
	}
	return ioutil.WriteFile(filepath, dataJSON, 0644)
}

// Reads device to container allocation information from disk, and populates
// h.allocatedDevices accordingly.
func (h *HandlerImpl) readCheckpoint() error {
	h.Lock()
	defer h.Unlock()
	filepath := h.devicePluginManager.CheckpointFile()
	content, err := ioutil.ReadFile(filepath)
	if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to read checkpoint file %q: %v", filepath, err)
	}
	glog.V(2).Infof("Read checkpoint file %s\n", filepath)
	var data checkpointData
	if err := json.Unmarshal(content, &data); err != nil {
		return fmt.Errorf("failed to unmarshal checkpoint data: %v", err)
	}
	h.podDevices.fromCheckpointData(data)
	h.allocatedDevices = h.podDevices.devices()
	return nil
}
