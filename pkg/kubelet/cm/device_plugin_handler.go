/*
Copyright 2016 The Kubernetes Authors.

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
	"fmt"
	"sync"
	"time"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/device-plugin"

	kubetypes "k8s.io/apimachinery/pkg/types"
	v1alpha1 "k8s.io/kubernetes/pkg/kubelet/apis/cri/v1alpha1/runtime"
	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/device-plugin/v1alpha1"
)

// KillPodFunc is the function that is used to kill the pod
// when a device assigned to a Pod's container becomes unhealthy
type KillPodFunc func(*v1.Pod, v1.PodStatus, int64)

// DevicePluginHandler is the structure in charge of interfacing
// with Kubelet and the Device Plugin manager
type DevicePluginHandler struct {
	devicePluginManager *deviceplugin.Manager

	// pod2Dev is map[podUUId]map[Container name]
	pod2Dev map[kubetypes.UID]map[string][]*pluginapi.Device
	dev2Pod map[string]*v1.Pod

	killFunc KillPodFunc

	mutex sync.Mutex
}

// NewDevicePluginHandler create a DevicePluginHandler
func NewDevicePluginHandler( /*devCapacity []v1.Device, */ pods []*v1.Pod,
	k KillPodFunc, socketPath string) (*DevicePluginHandler, error) {

	hdlr := &DevicePluginHandler{
		pod2Dev:  make(map[kubetypes.UID]map[string][]*pluginapi.Device),
		dev2Pod:  make(map[string]*v1.Pod),
		killFunc: k,
	}

	// devices := FromAPIToPluginDevices(devCapacity)

	// devs, used, unused, unhealthy
	// This adds the used pods to the hdlr's internal state
	//available, used := hdlr.reconcile(pods, devices)

	mgr, err := deviceplugin.NewManager(socketPath, nil, nil,
		hdlr.monitorCallback)
	if err != nil {
		return nil, fmt.Errorf("Failed to initialize device plugin with error: %+v", err)
	}

	hdlr.devicePluginManager = mgr

	// go hdlr.deallocate(used)

	return hdlr, nil
}

// Devices is the map of devices that are known by the Device
// Plugin manager with the Kind of the devices as key
func (h *DevicePluginHandler) Devices() map[string][]*pluginapi.Device {
	return /*FromPluginToAPI(*/ h.devicePluginManager.Devices() /*)*/
}

// AvailableDevices is the map of devices that are available to be
// consumed
func (h *DevicePluginHandler) AvailableDevices() map[string][]*pluginapi.Device {
	return /*FromPluginToAPI(*/ h.devicePluginManager.Available() /*)*/
}

// AllocateDevices is the call that you can use to allocate Devices for a
// container
func (h *DevicePluginHandler) AllocateDevices(p *v1.Pod, ctr *v1.Container,
	config *v1alpha1.ContainerConfig) error {

	// For now copy limits into requests
	// TODO define what behavior is expected here with OIR
	for key, v := range ctr.Resources.Limits {
		if isDevice, _ := deviceplugin.DeviceName(key); !isDevice {
			continue
		}

		ctr.Resources.Requests[key] = v
	}

	for key, v := range ctr.Resources.Requests {
		isDevice, name := deviceplugin.DeviceName(key)
		if !isDevice {
			continue
		}

		err := h.allocate(p, ctr, name, int(v.Value()), config)
		if err != nil {
			return err
		}
	}

	return nil
}

// Deallocate is the call that you can use to deallocate the devices assigned
// to a container
func (h *DevicePluginHandler) DeallocateDevices(p *v1.Pod, ctr string) {
	devs := h.pod2Dev[p.UID][ctr]

	go func() {
		h.deallocate(devs)

		h.mutex.Lock()
		defer h.mutex.Unlock()

		for _, d := range devs {
			k := deviceplugin.DeviceKey(d)

			if _, ok := h.dev2Pod[k]; ok {
				delete(h.dev2Pod, k)
			}
		}
	}()
}

func (h *DevicePluginHandler) deallocate(devs []*pluginapi.Device) {
	m := h.devicePluginManager

	for {
		err := m.Deallocate(devs)
		if err != nil {
			glog.Infof("Request to deallocate devs %+v was stopped by: %+v", devs, err)

			time.Sleep(5 * time.Second)
			continue
		}

		return
	}
}

// DevicesForCtr returns the devices that were assigned to a container
func (h *DevicePluginHandler) DevicesForCtr(uid kubetypes.UID,
	name string) []*container.Device {
	return FromPluginToContainerDevices(h.pod2Dev[uid][name])
}

// Stop stops the device plugin manager
func (h *DevicePluginHandler) Stop() {
	h.devicePluginManager.Stop()
}
