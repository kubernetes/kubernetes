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

package openstack

import (
	"errors"

	"github.com/golang/glog"
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/servers"
	"github.com/gophercloud/gophercloud/pagination"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

type Instances struct {
	compute *gophercloud.ServiceClient
}

// Instances returns an implementation of Instances for OpenStack.
func (os *OpenStack) Instances() (cloudprovider.Instances, bool) {
	glog.V(4).Info("openstack.Instances() called")

	compute, err := os.NewComputeV2()
	if err != nil {
		return nil, false
	}

	glog.V(1).Info("Claiming to support Instances")

	return &Instances{compute}, true
}

func (i *Instances) List(name_filter string) ([]types.NodeName, error) {
	glog.V(4).Infof("openstack List(%v) called", name_filter)

	opts := servers.ListOpts{
		Name:   name_filter,
		Status: "ACTIVE",
	}
	pager := servers.List(i.compute, opts)

	ret := make([]types.NodeName, 0)
	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		sList, err := servers.ExtractServers(page)
		if err != nil {
			return false, err
		}
		for i := range sList {
			ret = append(ret, mapServerToNodeName(&sList[i]))
		}
		return true, nil
	})
	if err != nil {
		return nil, err
	}

	glog.V(3).Infof("Found %v instances matching %v: %v",
		len(ret), name_filter, ret)

	return ret, nil
}

// Implementation of Instances.CurrentNodeName
// Note this is *not* necessarily the same as hostname.
func (i *Instances) CurrentNodeName(hostname string) (types.NodeName, error) {
	md, err := getMetadata()
	if err != nil {
		return "", err
	}
	return types.NodeName(md.Name), nil
}

func (i *Instances) AddSSHKeyToAllInstances(user string, keyData []byte) error {
	return errors.New("unimplemented")
}

func (i *Instances) NodeAddresses(name types.NodeName) ([]v1.NodeAddress, error) {
	glog.V(4).Infof("NodeAddresses(%v) called", name)

	addrs, err := getAddressesByName(i.compute, name)
	if err != nil {
		return nil, err
	}

	glog.V(4).Infof("NodeAddresses(%v) => %v", name, addrs)
	return addrs, nil
}

// NodeAddressesByProviderID returns the node addresses of an instances with the specified unique providerID
// This method will not be called from the node that is requesting this ID. i.e. metadata service
// and other local methods cannot be used here
func (i *Instances) NodeAddressesByProviderID(providerID string) ([]v1.NodeAddress, error) {
	return []v1.NodeAddress{}, errors.New("unimplemented")
}

// ExternalID returns the cloud provider ID of the specified instance (deprecated).
func (i *Instances) ExternalID(name types.NodeName) (string, error) {
	srv, err := getServerByName(i.compute, name)
	if err != nil {
		if err == ErrNotFound {
			return "", cloudprovider.InstanceNotFound
		}
		return "", err
	}
	return srv.ID, nil
}

// InstanceID returns the kubelet's cloud provider ID.
func (os *OpenStack) InstanceID() (string, error) {
	return os.localInstanceID, nil
}

// InstanceID returns the cloud provider ID of the specified instance.
func (i *Instances) InstanceID(name types.NodeName) (string, error) {
	srv, err := getServerByName(i.compute, name)
	if err != nil {
		return "", err
	}
	// In the future it is possible to also return an endpoint as:
	// <endpoint>/<instanceid>
	return "/" + srv.ID, nil
}

// InstanceTypeByProviderID returns the cloudprovider instance type of the node with the specified unique providerID
// This method will not be called from the node that is requesting this ID. i.e. metadata service
// and other local methods cannot be used here
func (i *Instances) InstanceTypeByProviderID(providerID string) (string, error) {
	return "", errors.New("unimplemented")
}

// InstanceType returns the type of the specified instance.
func (i *Instances) InstanceType(name types.NodeName) (string, error) {
	return "", nil
}
