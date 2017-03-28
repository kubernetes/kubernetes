/*
Copyright 2014 The Kubernetes Authors.

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

package network

import (
	"fmt"
	"net"
	"strings"
	"sync"

	"github.com/golang/glog"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilsets "k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/network/hostport"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
	utilsysctl "k8s.io/kubernetes/pkg/util/sysctl"
)

const DefaultPluginName = "kubernetes.io/no-op"

// Called when the node's Pod CIDR is known when using the
// controller manager's --allocate-node-cidrs=true option
const NET_PLUGIN_EVENT_POD_CIDR_CHANGE = "pod-cidr-change"
const NET_PLUGIN_EVENT_POD_CIDR_CHANGE_DETAIL_CIDR = "pod-cidr"

// Plugin capabilities
const (
	// Indicates the plugin handles Kubernetes bandwidth shaping annotations internally
	NET_PLUGIN_CAPABILITY_SHAPING int = 1
)

// Plugin is an interface to network plugins for the kubelet
type NetworkPlugin interface {
	// Init initializes the plugin.  This will be called exactly once
	// before any other methods are called.
	Init(host Host, hairpinMode componentconfig.HairpinMode, nonMasqueradeCIDR string, mtu int) error

	// Called on various events like:
	// NET_PLUGIN_EVENT_POD_CIDR_CHANGE
	Event(name string, details map[string]interface{})

	// Name returns the plugin's name. This will be used when searching
	// for a plugin by name, e.g.
	Name() string

	// Returns a set of NET_PLUGIN_CAPABILITY_*
	Capabilities() utilsets.Int

	// SetUpPod is the method called after the infra container of
	// the pod has been created but before the other containers of the
	// pod are launched.
	// TODO: rename podInfraContainerID to sandboxID
	SetUpPod(namespace string, name string, podInfraContainerID kubecontainer.ContainerID, annotations map[string]string) error

	// TearDownPod is the method called before a pod's infra container will be deleted
	// TODO: rename podInfraContainerID to sandboxID
	TearDownPod(namespace string, name string, podInfraContainerID kubecontainer.ContainerID) error

	// Status is the method called to obtain the ipv4 or ipv6 addresses of the container
	// TODO: rename podInfraContainerID to sandboxID
	GetPodNetworkStatus(namespace string, name string, podInfraContainerID kubecontainer.ContainerID) (*PodNetworkStatus, error)

	// NetworkStatus returns error if the network plugin is in error state
	Status() error
}

// PodNetworkStatus stores the network status of a pod (currently just the primary IP address)
// This struct represents version "v1beta1"
type PodNetworkStatus struct {
	metav1.TypeMeta `json:",inline"`

	// IP is the primary ipv4/ipv6 address of the pod. Among other things it is the address that -
	//   - kube expects to be reachable across the cluster
	//   - service endpoints are constructed with
	//   - will be reported in the PodStatus.PodIP field (will override the IP reported by docker)
	IP net.IP `json:"ip" description:"Primary IP address of the pod"`
}

// LegacyHost implements the methods required by network plugins that
// were directly invoked by the kubelet. Implementations of this interface
// that do not wish to support these features can simply return false
// to SupportsLegacyFeatures.
type LegacyHost interface {
	// Get the pod structure by its name, namespace
	// Only used for hostport management and bw shaping
	GetPodByName(namespace, name string) (*v1.Pod, bool)

	// GetKubeClient returns a client interface
	// Only used in testing
	GetKubeClient() clientset.Interface

	// GetContainerRuntime returns the container runtime that implements the containers (e.g. docker/rkt)
	// Only used for hostport management
	GetRuntime() kubecontainer.Runtime

	// SupportsLegacyFeatures returns true if the network host support GetPodByName, KubeClient interface and kubelet
	// runtime interface. These interfaces will no longer be implemented by CRI shims.
	// This function helps network plugins to choose their behavior based on runtime.
	SupportsLegacyFeatures() bool
}

// Host is an interface that plugins can use to access the kubelet.
// TODO(#35457): get rid of this backchannel to the kubelet. The scope of
// the back channel is restricted to host-ports/testing, and restricted
// to kubenet. No other network plugin wrapper needs it. Other plugins
// only require a way to access namespace information and port mapping
// information , which they can do directly through the embedded interfaces.
type Host interface {
	// NamespaceGetter is a getter for sandbox namespace information.
	NamespaceGetter

	// PortMappingGetter is a getter for sandbox port mapping information.
	PortMappingGetter

	// LegacyHost contains methods that trap back into the Kubelet. Dependence
	// *do not* add more dependencies in this interface. In a post-cri world,
	// network plugins will be invoked by the runtime shim, and should only
	// require GetNetNS and GetPodPortMappings.
	LegacyHost
}

// NamespaceGetter is an interface to retrieve namespace information for a given
// sandboxID. Typically implemented by runtime shims that are closely coupled to
// CNI plugin wrappers like kubenet.
type NamespaceGetter interface {
	// GetNetNS returns network namespace information for the given containerID.
	GetNetNS(containerID string) (string, error)
}

// PortMappingGetter is an interface to retrieve port mapping information for a given
// sandboxID. Typically implemented by runtime shims that are closely coupled to
// CNI plugin wrappers like kubenet.
type PortMappingGetter interface {
	// GetPodPortMappings returns sandbox port mappings information.
	GetPodPortMappings(containerID string) ([]*hostport.PortMapping, error)
}

// InitNetworkPlugin inits the plugin that matches networkPluginName. Plugins must have unique names.
func InitNetworkPlugin(plugins []NetworkPlugin, networkPluginName string, host Host, hairpinMode componentconfig.HairpinMode, nonMasqueradeCIDR string, mtu int) (NetworkPlugin, error) {
	if networkPluginName == "" {
		// default to the no_op plugin
		plug := &NoopNetworkPlugin{}
		if err := plug.Init(host, hairpinMode, nonMasqueradeCIDR, mtu); err != nil {
			return nil, err
		}
		return plug, nil
	}

	pluginMap := map[string]NetworkPlugin{}

	allErrs := []error{}
	for _, plugin := range plugins {
		name := plugin.Name()
		if errs := validation.IsQualifiedName(name); len(errs) != 0 {
			allErrs = append(allErrs, fmt.Errorf("network plugin has invalid name: %q: %s", name, strings.Join(errs, ";")))
			continue
		}

		if _, found := pluginMap[name]; found {
			allErrs = append(allErrs, fmt.Errorf("network plugin %q was registered more than once", name))
			continue
		}
		pluginMap[name] = plugin
	}

	chosenPlugin := pluginMap[networkPluginName]
	if chosenPlugin != nil {
		err := chosenPlugin.Init(host, hairpinMode, nonMasqueradeCIDR, mtu)
		if err != nil {
			allErrs = append(allErrs, fmt.Errorf("Network plugin %q failed init: %v", networkPluginName, err))
		} else {
			glog.V(1).Infof("Loaded network plugin %q", networkPluginName)
		}
	} else {
		allErrs = append(allErrs, fmt.Errorf("Network plugin %q not found.", networkPluginName))
	}

	return chosenPlugin, utilerrors.NewAggregate(allErrs)
}

func UnescapePluginName(in string) string {
	return strings.Replace(in, "~", "/", -1)
}

type NoopNetworkPlugin struct {
}

const sysctlBridgeCallIPTables = "net/bridge/bridge-nf-call-iptables"

func (plugin *NoopNetworkPlugin) Init(host Host, hairpinMode componentconfig.HairpinMode, nonMasqueradeCIDR string, mtu int) error {
	// Set bridge-nf-call-iptables=1 to maintain compatibility with older
	// kubernetes versions to ensure the iptables-based kube proxy functions
	// correctly.  Other plugins are responsible for setting this correctly
	// depending on whether or not they connect containers to Linux bridges
	// or use some other mechanism (ie, SDN vswitch).

	// Ensure the netfilter module is loaded on kernel >= 3.18; previously
	// it was built-in.
	utilexec.New().Command("modprobe", "br-netfilter").CombinedOutput()
	if err := utilsysctl.New().SetSysctl(sysctlBridgeCallIPTables, 1); err != nil {
		glog.Warningf("can't set sysctl %s: %v", sysctlBridgeCallIPTables, err)
	}

	return nil
}

func (plugin *NoopNetworkPlugin) Event(name string, details map[string]interface{}) {
}

func (plugin *NoopNetworkPlugin) Name() string {
	return DefaultPluginName
}

func (plugin *NoopNetworkPlugin) Capabilities() utilsets.Int {
	return utilsets.NewInt()
}

func (plugin *NoopNetworkPlugin) SetUpPod(namespace string, name string, id kubecontainer.ContainerID, annotations map[string]string) error {
	return nil
}

func (plugin *NoopNetworkPlugin) TearDownPod(namespace string, name string, id kubecontainer.ContainerID) error {
	return nil
}

func (plugin *NoopNetworkPlugin) GetPodNetworkStatus(namespace string, name string, id kubecontainer.ContainerID) (*PodNetworkStatus, error) {
	return nil, nil
}

func (plugin *NoopNetworkPlugin) Status() error {
	return nil
}

func getOnePodIP(execer utilexec.Interface, nsenterPath, netnsPath, interfaceName, addrType string) (net.IP, error) {
	// Try to retrieve ip inside container network namespace
	output, err := execer.Command(nsenterPath, fmt.Sprintf("--net=%s", netnsPath), "-F", "--",
		"ip", "-o", addrType, "addr", "show", "dev", interfaceName, "scope", "global").CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("Unexpected command output %s with error: %v", output, err)
	}

	lines := strings.Split(string(output), "\n")
	if len(lines) < 1 {
		return nil, fmt.Errorf("Unexpected command output %s", output)
	}
	fields := strings.Fields(lines[0])
	if len(fields) < 4 {
		return nil, fmt.Errorf("Unexpected address output %s ", lines[0])
	}
	ip, _, err := net.ParseCIDR(fields[3])
	if err != nil {
		return nil, fmt.Errorf("CNI failed to parse ip from output %s due to %v", output, err)
	}

	return ip, nil
}

// GetPodIP gets the IP of the pod by inspecting the network info inside the pod's network namespace.
func GetPodIP(execer utilexec.Interface, nsenterPath, netnsPath, interfaceName string) (net.IP, error) {
	ip, err := getOnePodIP(execer, nsenterPath, netnsPath, interfaceName, "-4")
	if err != nil {
		// Fall back to IPv6 address if no IPv4 address is present
		ip, err = getOnePodIP(execer, nsenterPath, netnsPath, interfaceName, "-6")
	}
	if err != nil {
		return nil, err
	}

	return ip, nil
}

type NoopPortMappingGetter struct{}

func (*NoopPortMappingGetter) GetPodPortMappings(containerID string) ([]*hostport.PortMapping, error) {
	return nil, nil
}

// The PluginManager wraps a kubelet network plugin and provides synchronization
// for a given pod's network operations.  Each pod's setup/teardown/status operations
// are synchronized against each other, but network operations of other pods can
// proceed in parallel.
type PluginManager struct {
	// Network plugin being wrapped
	plugin NetworkPlugin

	// Pod list and lock
	podsLock sync.Mutex
	pods     map[string]*podLock
}

func NewPluginManager(plugin NetworkPlugin) *PluginManager {
	return &PluginManager{
		plugin: plugin,
		pods:   make(map[string]*podLock),
	}
}

func (pm *PluginManager) PluginName() string {
	return pm.plugin.Name()
}

func (pm *PluginManager) Event(name string, details map[string]interface{}) {
	pm.plugin.Event(name, details)
}

func (pm *PluginManager) Status() error {
	return pm.plugin.Status()
}

type podLock struct {
	// Count of in-flight operations for this pod; when this reaches zero
	// the lock can be removed from the pod map
	refcount uint

	// Lock to synchronize operations for this specific pod
	mu sync.Mutex
}

// Lock network operations for a specific pod.  If that pod is not yet in
// the pod map, it will be added.  The reference count for the pod will
// be increased.
func (pm *PluginManager) podLock(fullPodName string) *sync.Mutex {
	pm.podsLock.Lock()
	defer pm.podsLock.Unlock()

	lock, ok := pm.pods[fullPodName]
	if !ok {
		lock = &podLock{}
		pm.pods[fullPodName] = lock
	}
	lock.refcount++
	return &lock.mu
}

// Unlock network operations for a specific pod.  The reference count for the
// pod will be decreased.  If the reference count reaches zero, the pod will be
// removed from the pod map.
func (pm *PluginManager) podUnlock(fullPodName string) {
	pm.podsLock.Lock()
	defer pm.podsLock.Unlock()

	lock, ok := pm.pods[fullPodName]
	if !ok {
		glog.Warningf("Unbalanced pod lock unref for %s", fullPodName)
		return
	} else if lock.refcount == 0 {
		// This should never ever happen, but handle it anyway
		delete(pm.pods, fullPodName)
		glog.Warningf("Pod lock for %s still in map with zero refcount", fullPodName)
		return
	}
	lock.refcount--
	lock.mu.Unlock()
	if lock.refcount == 0 {
		delete(pm.pods, fullPodName)
	}
}

func (pm *PluginManager) GetPodNetworkStatus(podNamespace, podName string, id kubecontainer.ContainerID) (*PodNetworkStatus, error) {
	fullPodName := kubecontainer.BuildPodFullName(podName, podNamespace)
	pm.podLock(fullPodName).Lock()
	defer pm.podUnlock(fullPodName)

	netStatus, err := pm.plugin.GetPodNetworkStatus(podNamespace, podName, id)
	if err != nil {
		return nil, fmt.Errorf("NetworkPlugin %s failed on the status hook for pod %q: %v", pm.plugin.Name(), fullPodName, err)
	}

	return netStatus, nil
}

func (pm *PluginManager) SetUpPod(podNamespace, podName string, id kubecontainer.ContainerID, annotations map[string]string) error {
	fullPodName := kubecontainer.BuildPodFullName(podName, podNamespace)
	pm.podLock(fullPodName).Lock()
	defer pm.podUnlock(fullPodName)

	glog.V(3).Infof("Calling network plugin %s to set up pod %q", pm.plugin.Name(), fullPodName)
	if err := pm.plugin.SetUpPod(podNamespace, podName, id, annotations); err != nil {
		return fmt.Errorf("NetworkPlugin %s failed to set up pod %q network: %v", pm.plugin.Name(), fullPodName, err)
	}

	return nil
}

func (pm *PluginManager) TearDownPod(podNamespace, podName string, id kubecontainer.ContainerID) error {
	fullPodName := kubecontainer.BuildPodFullName(podName, podNamespace)
	pm.podLock(fullPodName).Lock()
	defer pm.podUnlock(fullPodName)

	glog.V(3).Infof("Calling network plugin %s to tear down pod %q", pm.plugin.Name(), fullPodName)
	if err := pm.plugin.TearDownPod(podNamespace, podName, id); err != nil {
		return fmt.Errorf("NetworkPlugin %s failed to teardown pod %q network: %v", pm.plugin.Name(), fullPodName, err)
	}

	return nil
}
