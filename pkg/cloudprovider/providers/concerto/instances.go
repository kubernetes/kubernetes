/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package concerto_cloud

import (
	"fmt"
	"net"
	"regexp"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

// NodeAddresses returns the addresses of the specified instance.
func (concerto *ConcertoCloud) NodeAddresses(name string) ([]api.NodeAddress, error) {
	ci, err := concerto.service.GetInstanceByName(name)
	if err != nil {
		return nil, fmt.Errorf("error getting Concerto instance by name '%s' : %v", name, err)
	}
	publicAddress := api.NodeAddress{
		Type:    api.NodeExternalIP,
		Address: net.ParseIP(ci.PublicIP).String(),
	}
	return []api.NodeAddress{publicAddress}, nil
}

// ExternalID returns the cloud provider ID of the specified instance (deprecated).
func (concerto *ConcertoCloud) ExternalID(name string) (string, error) {
	return concerto.InstanceID(name)
}

// InstanceID returns the cloud provider ID of the specified instance.
// Note that if the instance does not exist or is no longer running, we must return ("", cloudprovider.InstanceNotFound)
func (concerto *ConcertoCloud) InstanceID(name string) (string, error) {
	ci, err := concerto.service.GetInstanceByName(name)
	if err == cloudprovider.InstanceNotFound {
		return "", cloudprovider.InstanceNotFound
	}
	if err != nil {
		return "", fmt.Errorf("error getting Concerto instance by name '%s' : %v", name, err)
	}
	return ci.Id, nil
}

// List lists instances that match 'filter' which is a regular expression which must match the entire instance name (fqdn)
func (concerto *ConcertoCloud) List(filter string) ([]string, error) {
	regexp, err := regexp.Compile(filter)
	if err != nil {
		return nil, fmt.Errorf("error compiling regular expression '%s' : %v", filter, err)
	}
	instances, err := concerto.service.GetInstanceList()
	if err != nil {
		return nil, fmt.Errorf("error getting Concerto instance list : %v", err)
	}
	names := make([]string, 0)
	for _, instance := range instances {
		if regexp.MatchString(instance.Name) {
			names = append(names, instance.Name)
		}
	}
	return names, nil
}

// GetNodeResources gets the resources for a particular node
func (concerto *ConcertoCloud) GetNodeResources(name string) (*api.NodeResources, error) {
	ci, err := concerto.service.GetInstanceByName(name)
	if err != nil {
		return nil, fmt.Errorf("error getting Concerto instance by name '%s' : %v", name, err)
	}
	return makeNodeResources(ci.CPUs, ci.Memory), nil
}

// Returns the name of the node we are currently running on
func (concerto *ConcertoCloud) CurrentNodeName(hostname string) (string, error) {
	return hostname, nil
}

// NOT SUPPORTED in Concerto Cloud
func (concerto *ConcertoCloud) AddSSHKeyToAllInstances(user string, keyData []byte) error {
	return fmt.Errorf("Unsupported operation")
}

// Builds an api.NodeResources
// cpu is in cores, memory is in MiB
func makeNodeResources(cpu float64, memory int64) *api.NodeResources {
	return &api.NodeResources{
		Capacity: api.ResourceList{
			api.ResourceCPU:    *resource.NewMilliQuantity(int64(cpu*1000), resource.DecimalSI),
			api.ResourceMemory: *resource.NewQuantity(int64(memory*1024*1024), resource.BinarySI),
		},
	}
}
