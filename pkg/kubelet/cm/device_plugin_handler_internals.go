package cm

import (
	"fmt"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/kubelet/device-plugin"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	v1alpha1 "k8s.io/kubernetes/pkg/kubelet/apis/cri/v1alpha1/runtime"
	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/device-plugin/v1alpha1"
)

func (h *DevicePluginHandler) reconcile(pods []*v1.Pod, nodeName string,
	c clientset.Interface) (devs, available, unused []*pluginapi.Device, err error) {

	if c == nil {
		return nil, nil, nil, nil
	}

	// Get node status
	node, err := c.Core().Nodes().Get(nodeName, metav1.GetOptions{})
	if err != nil && !strings.Contains(err.Error(), "not found") {
		return nil, nil, nil, err
	}

	devs = FromAPIToPluginDevices(node.Status.DevCapacity)
	available = FromAPIToPluginDevices(node.Status.DevAvailable)

	// build unused list from used list
	used := h.devsFromPods(pods)
	for _, d := range devs {
		if _, ok := deviceplugin.HasDevice(d, used); ok {
			continue
		}

		unused = append(unused, d)
	}

	return devs, available, unused, nil
}

func (h *DevicePluginHandler) devsFromPods(pods []*v1.Pod) []*pluginapi.Device {
	var devs []*pluginapi.Device

	for _, p := range pods {
		for _, ctr := range p.Status.ContainerStatuses {
			ctrDevs := FromAPIToPluginDevices(ctr.Devices)
			devs = append(devs, ctrDevs...)

			h.addDev(p, ctr.Name, ctrDevs)
		}
		for _, ctr := range p.Status.InitContainerStatuses {
			ctrDevs := FromAPIToPluginDevices(ctr.Devices)
			devs = append(devs, ctrDevs...)

			h.addDev(p, ctr.Name, ctrDevs)
		}
	}

	return devs
}

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
		Phase:   v1.PodFailed,
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
