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

package cm

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"path/filepath"
	"sync"

	"github.com/golang/glog"
	"golang.org/x/net/context"
	"google.golang.org/grpc"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/sets"
	v1helper "k8s.io/kubernetes/pkg/api/v1/helper"
	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/deviceplugin"
)

// podDevices represents a list of pod to device Id mappings.
type containerDevices map[string]sets.String
type podDevices map[string]containerDevices

func (pdev podDevices) pods() sets.String {
	ret := sets.NewString()
	for k := range pdev {
		ret.Insert(k)
	}
	return ret
}

func (pdev podDevices) insert(podUID, contName string, device string) {
	if _, exists := pdev[podUID]; !exists {
		pdev[podUID] = make(containerDevices)
	}
	if _, exists := pdev[podUID][contName]; !exists {
		pdev[podUID][contName] = sets.NewString()
	}
	pdev[podUID][contName].Insert(device)
}

func (pdev podDevices) getDevices(podUID, contName string) sets.String {
	containers, exists := pdev[podUID]
	if !exists {
		return nil
	}
	devices, exists := containers[contName]
	if !exists {
		return nil
	}
	return devices
}

func (pdev podDevices) delete(pods []string) {
	for _, uid := range pods {
		delete(pdev, uid)
	}
}

func (pdev podDevices) devices() sets.String {
	ret := sets.NewString()
	for _, containerDevices := range pdev {
		for _, deviceSet := range containerDevices {
			ret = ret.Union(deviceSet)
		}
	}
	return ret
}

type DevicePluginHandler interface {
	// Start starts device plugin registration service.
	Start() error
	// Devices returns all of registered devices keyed by resourceName.
	Devices() map[string][]*pluginapi.Device
	// Allocate attempts to allocate all of required extended resources for
	// the input container, issues an Allocate rpc request for each of such
	// resources, and returns their AllocateResponses on success.
	Allocate(pod *v1.Pod, container *v1.Container, activePods []*v1.Pod) ([]*pluginapi.AllocateResponse, error)
}

type DevicePluginHandlerImpl struct {
	sync.Mutex
	socketname string
	socketdir  string

	endpoints map[string]*deviceplugin.Endpoint // Key is ResourceName

	server *grpc.Server
	// deviceMonitorCallback is used for testing only.
	deviceMonitorCallback deviceplugin.MonitorCallback
	// allDevices contains all of registered resourceNames and their exported device IDs.
	allDevices map[string]sets.String
	// allocatedDevices contains pod to allocated device mapping, keyed by resourceName.
	allocatedDevices map[string]podDevices
}

// NewDevicePluginHandler create a DevicePluginHandler
// updateCapacityFunc is called to update ContainerManager capacity when
// device capacity changes.
func NewDevicePluginHandlerImpl(updateCapacityFunc func(v1.ResourceList)) (*DevicePluginHandlerImpl, error) {
	glog.V(2).Infof("Creating Device Plugin Handler")

	socketdir, socketname := filepath.Split(pluginapi.KubeletSocket)
	handler := &DevicePluginHandlerImpl{
		socketname:       socketname,
		socketdir:        socketdir,
		endpoints:        make(map[string]*deviceplugin.Endpoint),
		allDevices:       make(map[string]sets.String),
		allocatedDevices: make(map[string]podDevices),
	}

	deviceMonitorCallback := func(resourceName string, added, updated, deleted []*pluginapi.Device) {
		var capacity = v1.ResourceList{}
		kept := append(updated, added...)
		if _, ok := handler.allDevices[resourceName]; !ok {
			handler.allDevices[resourceName] = sets.NewString()
		}
		// For now, DevicePluginHandler only keeps track of healthy devices.
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

	handler.deviceMonitorCallback = deviceMonitorCallback
	// Loads in allocatedDevices information from disk.
	err := handler.readCheckpoint()
	if err != nil {
		glog.Warningf("Continue after failing to read checkpoint file. Device allocation info may NOT be up-to-date. Err: %v", err)
	}
	return handler, nil
}

func (h *DevicePluginHandlerImpl) Start() error {
	glog.V(2).Infof("Starting Device Plugin Handler")

	socketPath := filepath.Join(h.socketdir, h.socketname)
	os.MkdirAll(h.socketdir, 0755)

	// Removes all stale sockets in h.socketdir. Device plugins can monitor
	// this and use it as a signal to re-register with the new Kubelet.
	if err := removeContents(h.socketdir); err != nil {
		glog.Errorf("Fail to clean up stale contents under %s: %+v", h.socketdir, err)
	}

	if err := os.Remove(socketPath); err != nil && !os.IsNotExist(err) {
		glog.Errorf(deviceplugin.ErrRemoveSocket+" %+v", err)
		return err
	}

	s, err := net.Listen("unix", socketPath)
	if err != nil {
		glog.Errorf(deviceplugin.ErrListenSocket+" %+v", err)
		return err
	}

	h.server = grpc.NewServer([]grpc.ServerOption{}...)

	pluginapi.RegisterRegistrationServer(h.server, h)
	go h.server.Serve(s)

	return nil
}

func (h *DevicePluginHandlerImpl) Devices() map[string][]*pluginapi.Device {
	h.Lock()
	defer h.Unlock()

	devs := make(map[string][]*pluginapi.Device)
	for k, e := range h.endpoints {
		glog.V(3).Infof("Endpoint: %+v: %+v", k, e)
		devs[k] = e.GetDevices()
	}

	return devs
}

func (h *DevicePluginHandlerImpl) Allocate(pod *v1.Pod, container *v1.Container, activePods []*v1.Pod) ([]*pluginapi.AllocateResponse, error) {
	var ret []*pluginapi.AllocateResponse
	h.updateAllocatedDevices(activePods)
	for k, v := range container.Resources.Limits {
		resource := string(k)
		needed := int(v.Value())
		glog.V(3).Infof("needs %d %s", needed, resource)
		if !deviceplugin.IsDeviceName(k) || needed == 0 {
			continue
		}
		h.Lock()
		// Gets list of devices that have already been allocated.
		// This can happen if a container restarts for example.
		if h.allocatedDevices[resource] == nil {
			h.allocatedDevices[resource] = make(podDevices)
		}
		devices := h.allocatedDevices[resource].getDevices(string(pod.UID), container.Name)
		if devices != nil {
			glog.V(3).Infof("Found pre-allocated devices for resource %s container %q in Pod %q: %v", resource, container.Name, pod.UID, devices.List())
			needed = needed - devices.Len()
		}
		// Get Devices in use.
		devicesInUse := h.allocatedDevices[resource].devices()
		// Get a list of available devices.
		available := h.allDevices[resource].Difference(devicesInUse)
		if int(available.Len()) < needed {
			h.Unlock()
			return nil, fmt.Errorf("requested number of devices unavailable for %s. Requested: %d, Available: %d", resource, needed, available.Len())
		}
		allocated := available.UnsortedList()[:needed]
		for _, device := range allocated {
			// Update internal allocated device cache.
			h.allocatedDevices[resource].insert(string(pod.UID), container.Name, device)
		}
		e, ok := h.endpoints[resource]
		h.Unlock()
		// Allocate involves RPC calls to device plugin, which
		// could be heavy-weight. Therefore we want to perform this operation outside
		// mutex lock. Note if Allcate call fails, we may leave container resources
		// partially allocated for the failed container. We rely on updateAllocatedDevices()
		// to garbage collect these resources later. Another side effect is that if
		// we have X resource A and Y resource B in total, and two containers, container1
		// and container2 both require X resource A and Y resource B. Both allocation
		// requests may fail if we serve them in mixed order.
		// TODO: may revisit this part later if we see inefficient resource allocation
		// in real use as the result of this.
		if !ok {
			return nil, fmt.Errorf("Unknown Device Plugin %s", resource)
		}
		resp, err := e.Allocate(append(devices.UnsortedList(), allocated...))
		if err != nil {
			return nil, err
		}
		ret = append(ret, resp)
	}
	// Checkpoints device to container allocation information.
	if err := h.writeCheckpoint(); err != nil {
		return nil, err
	}
	return ret, nil
}

// Register registers a device plugin.
func (h *DevicePluginHandlerImpl) Register(ctx context.Context,
	r *pluginapi.RegisterRequest) (*pluginapi.Empty, error) {
	glog.V(2).Infof("Got request for Device Plugin %s", r.ResourceName)
	if r.Version != pluginapi.Version {
		return &pluginapi.Empty{}, fmt.Errorf(deviceplugin.ErrUnsuportedVersion)
	}

	if r.ResourceName == "" {
		return &pluginapi.Empty{}, fmt.Errorf(deviceplugin.ErrEmptyResourceName)
	}
	if !v1helper.IsExtendedResourceName(v1.ResourceName(r.ResourceName)) {
		return &pluginapi.Empty{}, fmt.Errorf(deviceplugin.ErrInvalidResourceName)
	}

	// TODO: for now, always accepts newest device plugin. Later may consider to
	// add some policies here, e.g., verify whether an old device plugin with the
	// same resource name is still alive to determine whether we want to accept
	// the new registration.
	go h.addEndpoint(r)

	return &pluginapi.Empty{}, nil
}

// updateAllocatedDevices updates the list of GPUs in use.
// It gets a list of active pods and then frees any GPUs that are bound to
// terminated pods. Returns error on failure.
func (h *DevicePluginHandlerImpl) updateAllocatedDevices(activePods []*v1.Pod) {
	h.Lock()
	defer h.Unlock()
	activePodUids := sets.NewString()
	for _, pod := range activePods {
		activePodUids.Insert(string(pod.UID))
	}
	for _, podDevs := range h.allocatedDevices {
		allocatedPodUids := podDevs.pods()
		podsToBeRemoved := allocatedPodUids.Difference(activePodUids)
		glog.V(5).Infof("pods to be removed: %v", podsToBeRemoved.List())
		podDevs.delete(podsToBeRemoved.List())
	}
}

type checkpointEntry struct {
	PodUID        string
	ContainerName string
	ResourceName  string
	DeviceID      string
}

// checkpointData struct is used to store pod to device allocation information
// in a checkpoint file.
// TODO: add version control when we need to change checkpoint format.
type checkpointData struct {
	Entries []checkpointEntry
}

// Checkpoints device to container allocation information to disk.
func (h *DevicePluginHandlerImpl) writeCheckpoint() error {
	filepath := checkpointFile(h.socketdir)
	var data checkpointData
	for resourceName, podDev := range h.allocatedDevices {
		for podUID, conDev := range podDev {
			for conName, devs := range conDev {
				for _, devId := range devs.UnsortedList() {
					data.Entries = append(data.Entries, checkpointEntry{podUID, conName, resourceName, devId})
				}
			}
		}
	}
	dataJson, err := json.Marshal(data)
	if err != nil {
		return err
	}
	return ioutil.WriteFile(filepath, dataJson, 0644)
}

// Reads device to container allocation information from disk, and populates
// h.allocatedDevices accordingly.
func (h *DevicePluginHandlerImpl) readCheckpoint() error {
	filepath := checkpointFile(h.socketdir)
	content, err := ioutil.ReadFile(filepath)
	if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to read checkpoint file %q: %v", filepath, err)
	}
	glog.V(2).Infof("Read checkpoint file %s\n", filepath)
	var data checkpointData
	if err := json.Unmarshal(content, &data); err != nil {
		return fmt.Errorf("failed to unmarshal checkpoint data: %v", err)
	}
	for _, entry := range data.Entries {
		glog.V(2).Infof("Get checkpoint entry: %v %v %v %v\n", entry.PodUID, entry.ContainerName, entry.ResourceName, entry.DeviceID)
		if h.allocatedDevices[entry.ResourceName] == nil {
			h.allocatedDevices[entry.ResourceName] = make(podDevices)
		}
		h.allocatedDevices[entry.ResourceName].insert(entry.PodUID, entry.ContainerName, entry.DeviceID)
	}
	return nil
}

func removeContents(dir string) error {
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer d.Close()
	names, err := d.Readdirnames(-1)
	if err != nil {
		return err
	}
	for _, name := range names {
		filePath := filepath.Join(dir, name)
		if filePath == checkpointFile(dir) {
			continue
		}
		stat, err := os.Stat(filePath)
		if err != nil {
			glog.Errorf("Failed to stat file %v: %v", filePath, err)
			continue
		}
		if stat.IsDir() {
			continue
		}
		err = os.RemoveAll(filePath)
		if err != nil {
			return err
		}
	}
	return nil
}

// checkpointFile returns device plugin checkpoint file path.
func checkpointFile(dir string) string {
	return filepath.Join(dir, "kubelet_internal_checkpoint")
}

func (h *DevicePluginHandlerImpl) addEndpoint(r *pluginapi.RegisterRequest) {
	// Stops existing endpoint if there is any.
	h.Lock()
	old, ok := h.endpoints[r.ResourceName]
	h.Unlock()
	if ok && old != nil {
		old.Stop()
	}

	socketPath := filepath.Join(h.socketdir, r.Endpoint)
	e, err := deviceplugin.NewEndpoint(socketPath, r.ResourceName, h.deviceMonitorCallback)
	if err != nil {
		glog.Errorf("Failed to dial device plugin with request %v: %v", r, err)
		return
	}

	stream, err := e.List()
	if err != nil {
		glog.Errorf("Failed to List devices for plugin %v: %v", r.ResourceName, err)
		return
	}

	go func() {
		e.ListAndWatch(stream)

		h.Lock()
		if old, ok := h.endpoints[r.ResourceName]; ok && old == e {
			delete(h.endpoints, r.ResourceName)
		}
		glog.V(2).Infof("Unregistered endpoint %v", e)
		h.Lock()
	}()

	h.Lock()
	h.endpoints[r.ResourceName] = e
	glog.V(2).Infof("Registered endpoint %v", e)
	h.Unlock()
}
