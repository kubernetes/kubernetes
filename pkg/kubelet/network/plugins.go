/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
	utilsysctl "k8s.io/kubernetes/pkg/util/sysctl"
	"k8s.io/kubernetes/pkg/util/validation"
)

const DefaultPluginName = "kubernetes.io/no-op"

// Called when the node's Pod CIDR is known when using the
// controller manager's --allocate-node-cidrs=true option
const NET_PLUGIN_EVENT_POD_CIDR_CHANGE = "pod-cidr-change"
const NET_PLUGIN_EVENT_POD_CIDR_CHANGE_DETAIL_CIDR = "pod-cidr"

// Plugin is an interface to network plugins for the kubelet
type NetworkPlugin interface {
	// Init initializes the plugin.  This will be called exactly once
	// before any other methods are called.
	Init(host Host) error

	// Called on various events like:
	// NET_PLUGIN_EVENT_POD_CIDR_CHANGE
	Event(name string, details map[string]interface{})

	// Name returns the plugin's name. This will be used when searching
	// for a plugin by name, e.g.
	Name() string

	// SetUpPod is the method called after the infra container of
	// the pod has been created but before the other containers of the
	// pod are launched.
	SetUpPod(namespace string, name string, podInfraContainerID kubecontainer.DockerID) error

	// TearDownPod is the method called before a pod's infra container will be deleted
	TearDownPod(namespace string, name string, podInfraContainerID kubecontainer.DockerID) error

	// Status is the method called to obtain the ipv4 or ipv6 addresses of the container
	Status(namespace string, name string, podInfraContainerID kubecontainer.DockerID) (*PodNetworkStatus, error)
}

// PodNetworkStatus stores the network status of a pod (currently just the primary IP address)
// This struct represents version "v1beta1"
type PodNetworkStatus struct {
	unversioned.TypeMeta `json:",inline"`

	// IP is the primary ipv4/ipv6 address of the pod. Among other things it is the address that -
	//   - kube expects to be reachable across the cluster
	//   - service endpoints are constructed with
	//   - will be reported in the PodStatus.PodIP field (will override the IP reported by docker)
	IP net.IP `json:"ip" description:"Primary IP address of the pod"`
}

// Host is an interface that plugins can use to access the kubelet.
type Host interface {
	// Get the pod structure by its name, namespace
	GetPodByName(namespace, name string) (*api.Pod, bool)

	// GetKubeClient returns a client interface
	GetKubeClient() clientset.Interface

	// GetContainerRuntime returns the container runtime that implements the containers (e.g. docker/rkt)
	GetRuntime() kubecontainer.Runtime
}

// InitNetworkPlugin inits the plugin that matches networkPluginName. Plugins must have unique names.
func InitNetworkPlugin(plugins []NetworkPlugin, networkPluginName string, host Host) (NetworkPlugin, error) {
	if networkPluginName == "" {
		// default to the no_op plugin
		plug := &noopNetworkPlugin{}
		if err := plug.Init(host); err != nil {
			return nil, err
		}
		return plug, nil
	}

	pluginMap := map[string]NetworkPlugin{}

	allErrs := []error{}
	for _, plugin := range plugins {
		name := plugin.Name()
		if !validation.IsQualifiedName(name) {
			allErrs = append(allErrs, fmt.Errorf("network plugin has invalid name: %#v", plugin))
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
		err := chosenPlugin.Init(host)
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

type noopNetworkPlugin struct {
}

const sysctlBridgeCallIptables = "net/bridge/bridge-nf-call-iptables"

func (plugin *noopNetworkPlugin) Init(host Host) error {
	// Set bridge-nf-call-iptables=1 to maintain compatibility with older
	// kubernetes versions to ensure the iptables-based kube proxy functions
	// correctly.  Other plugins are responsible for setting this correctly
	// depending on whether or not they connect containers to Linux bridges
	// or use some other mechanism (ie, SDN vswitch).

	// Ensure the netfilter module is loaded on kernel >= 3.18; previously
	// it was built-in.
	utilexec.New().Command("modprobe", "br-netfilter").CombinedOutput()
	if err := utilsysctl.SetSysctl(sysctlBridgeCallIptables, 1); err != nil {
		glog.Warningf("can't set sysctl %s: %v", sysctlBridgeCallIptables, err)
	}

	return nil
}

func (plugin *noopNetworkPlugin) Event(name string, details map[string]interface{}) {
}

func (plugin *noopNetworkPlugin) Name() string {
	return DefaultPluginName
}

func (plugin *noopNetworkPlugin) SetUpPod(namespace string, name string, id kubecontainer.DockerID) error {
	return nil
}

func (plugin *noopNetworkPlugin) TearDownPod(namespace string, name string, id kubecontainer.DockerID) error {
	return nil
}

func (plugin *noopNetworkPlugin) Status(namespace string, name string, id kubecontainer.DockerID) (*PodNetworkStatus, error) {
	return nil, nil
}
