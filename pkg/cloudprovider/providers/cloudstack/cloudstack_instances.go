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

package cloudstack

import (
	"errors"
	"fmt"
	"regexp"

	"github.com/golang/glog"
	"github.com/xanzy/go-cloudstack/cloudstack"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

// NodeAddresses returns the addresses of the specified instance.
func (cs *CSCloud) NodeAddresses(name types.NodeName) ([]v1.NodeAddress, error) {
	instance, count, err := cs.client.VirtualMachine.GetVirtualMachineByName(
		string(name),
		cloudstack.WithProject(cs.projectID),
	)
	if err != nil {
		if count == 0 {
			return nil, cloudprovider.InstanceNotFound
		}
		return nil, fmt.Errorf("error retrieving node addresses: %v", err)
	}

	if len(instance.Nic) == 0 {
		return nil, errors.New("instance does not have an internal IP")
	}

	addresses := []v1.NodeAddress{
		{Type: v1.NodeInternalIP, Address: instance.Nic[0].Ipaddress},
	}

	if instance.Publicip != "" {
		addresses = append(addresses, v1.NodeAddress{Type: v1.NodeExternalIP, Address: instance.Publicip})
	} else {
		// Since there is no sane way to determine the external IP if the host isn't
		// using static NAT, we will just fire a log message and omit the external IP.
		glog.V(4).Infof("Could not determine the public IP of host %v", name)
	}

	return addresses, nil
}

// ExternalID returns the cloud provider ID of the specified instance (deprecated).
func (cs *CSCloud) ExternalID(name types.NodeName) (string, error) {
	return cs.InstanceID(name)
}

// InstanceID returns the cloud provider ID of the specified instance.
func (cs *CSCloud) InstanceID(name types.NodeName) (string, error) {
	instance, count, err := cs.client.VirtualMachine.GetVirtualMachineByName(
		string(name),
		cloudstack.WithProject(cs.projectID),
	)
	if err != nil {
		if count == 0 {
			return "", cloudprovider.InstanceNotFound
		}
		return "", fmt.Errorf("error retrieving instance ID: %v", err)
	}

	return "/" + instance.Zonename + "/" + instance.Id, nil
}

// InstanceType returns the type of the specified instance.
func (cs *CSCloud) InstanceType(name types.NodeName) (string, error) {
	instance, count, err := cs.client.VirtualMachine.GetVirtualMachineByName(
		string(name),
		cloudstack.WithProject(cs.projectID),
	)
	if err != nil {
		if count == 0 {
			return "", cloudprovider.InstanceNotFound
		}
		return "", fmt.Errorf("error retrieving instance type: %v", err)
	}

	return instance.Serviceofferingname, nil
}

// List all instances that match 'filter' which is a regular expression that
// must match the entire instance name (fqdn).
func (cs *CSCloud) List(filter string) ([]string, error) {
	p := cs.client.VirtualMachine.NewListVirtualMachinesParams()

	if cs.projectID != "" {
		p.SetProjectid(cs.projectID)
	}

	l, err := cs.client.VirtualMachine.ListVirtualMachines(p)
	if err != nil {
		return nil, fmt.Errorf("error retrieving instances list: %v", err)
	}

	re := regexp.MustCompile(filter)

	var vms []string
	for _, vm := range l.VirtualMachines {
		if match := re.MatchString(vm.Name); match {
			vms = append(vms, vm.Name)
		}
	}

	return vms, nil
}

// AddSSHKeyToAllInstances is currently not implemented.
func (cs *CSCloud) AddSSHKeyToAllInstances(user string, keyData []byte) error {
	return errors.New("unimplemented")
}

// CurrentNodeName returns the name of the node we are currently running on.
func (cs *CSCloud) CurrentNodeName(hostname string) (types.NodeName, error) {
	return types.NodeName(hostname), nil
}
