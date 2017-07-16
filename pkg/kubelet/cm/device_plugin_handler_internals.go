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
	"time"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/device-plugin"

	v1alpha1 "k8s.io/kubernetes/pkg/kubelet/apis/cri/v1alpha1/runtime"
	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/device-plugin/v1alpha1"
)

/*
func (h *DevicePluginHandler) reconcile(pods []*v1.Pod, devCapacity []*pluginapi.Device) (available,
	used []*pluginapi.Device) {

	// build unused list from used list
	for _, p := range pods {
		for _, ctr := range p.Status.ContainerStatuses {
			ctrDevs := FromAPIToPluginDevices(ctr.Devices)
			used = append(used, ctrDevs...)

			h.addDev(p, ctr.Name, ctrDevs)
		}
		for _, ctr := range p.Status.InitContainerStatuses {
			ctrDevs := FromAPIToPluginDevices(ctr.Devices)
			used = append(used, ctrDevs...)

			h.addDev(p, ctr.Name, ctrDevs)
		}
	}

	for _, d := range devCapacity {
		if _, ok := deviceplugin.HasDevice(d, used); ok {
			continue
		}

		available = append(available, d)
	}

	return available, used
}*/

func (h *DevicePluginHandler) allocate(p *v1.Pod, ctr *v1.Container, name string, ndevs int,
	c *v1alpha1.ContainerConfig) error {

	devs, err := h.shimAllocate(name, ndevs, c)
	if err != nil {
		h.DeallocateDevices(p, ctr.Name)
		return err
	}

	h.addDev(p, ctr.Name, devs)

	return nil
}

func (h *DevicePluginHandler) addDev(p *v1.Pod, ctr string, devs []*pluginapi.Device) {
	if _, ok := h.pod2Dev[p.UID]; !ok {
		h.pod2Dev[p.UID] = make(map[string][]*pluginapi.Device)
	}

	ctr2Dev := h.pod2Dev[p.UID]
	ctr2Dev[ctr] = append(ctr2Dev[ctr], devs...)

	h.mutex.Lock()
	defer h.mutex.Unlock()

	for _, dev := range devs {
		h.dev2Pod[deviceplugin.DeviceKey(dev)] = p
	}
}

// TODO understand why pod isn't killed by Container Runtime during
// the grace period
func (h *DevicePluginHandler) monitorCallback(dev *pluginapi.Device) {
	glog.Infof("Request to kill Unhealthy dev: %+v", dev)

	p, ok := h.dev2Pod[deviceplugin.DeviceKey(dev)]
	if !ok {
		glog.Infof("Device is not in use by any pod")
		return
	}

	status := v1.PodStatus{
		Phase:   v1.PodPending,
		Message: fmt.Sprintf("device %s/%s became unhealthy", dev.Kind, dev.Name),
		Reason:  "Unhealthy Device",
	}

	for {
		if h.killFunc != nil {
			h.killFunc(p, status, int64(10))
			return
		}

		glog.Infof("Waiting for Kill function to be set")
		time.Sleep(5 * time.Second)
	}
}
