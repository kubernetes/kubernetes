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
	"fmt"
	"regexp"

	"github.com/golang/glog"
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/servers"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

type Instances struct {
	compute *gophercloud.ServiceClient
	opts    MetadataOpts
}

// Instances returns an implementation of Instances for OpenStack.
func (os *OpenStack) Instances() (cloudprovider.Instances, bool) {
	glog.V(4).Info("openstack.Instances() called")

	compute, err := os.NewComputeV2()
	if err != nil {
		return nil, false
	}

	glog.V(1).Info("Claiming to support Instances")

	return &Instances{
		compute: compute,
		opts:    os.metadataOpts,
	}, true
}

// Implementation of Instances.CurrentNodeName
// Note this is *not* necessarily the same as hostname.
func (i *Instances) CurrentNodeName(hostname string) (types.NodeName, error) {
	md, err := getMetadata(i.opts.SearchOrder)
	if err != nil {
		return "", err
	}
	return types.NodeName(md.Name), nil
}

func (i *Instances) AddSSHKeyToAllInstances(user string, keyData []byte) error {
	return cloudprovider.NotImplemented
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
	instanceID, err := instanceIDFromProviderID(providerID)

	if err != nil {
		return []v1.NodeAddress{}, err
	}

	server, err := servers.Get(i.compute, instanceID).Extract()

	if err != nil {
		return []v1.NodeAddress{}, err
	}

	addresses, err := nodeAddresses(server)
	if err != nil {
		return []v1.NodeAddress{}, err
	}

	return addresses, nil
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

// InstanceExistsByProviderID returns true if the instance with the given provider id still exists and is running.
// If false is returned with no error, the instance will be immediately deleted by the cloud controller manager.
func (i *Instances) InstanceExistsByProviderID(providerID string) (bool, error) {
	instanceID, err := instanceIDFromProviderID(providerID)
	if err != nil {
		return false, err
	}

	server, err := servers.Get(i.compute, instanceID).Extract()
	if err != nil {
		if isNotFound(err) {
			return false, nil
		}
		return false, err
	}

	if server.Status != "ACTIVE" {
		glog.Warningf("the instance %s is not active", instanceID)
		return false, nil
	}

	return true, nil
}

// InstanceID returns the kubelet's cloud provider ID.
func (os *OpenStack) InstanceID() (string, error) {
	if len(os.localInstanceID) == 0 {
		id, err := readInstanceID(os.metadataOpts.SearchOrder)
		if err != nil {
			return "", err
		}
		os.localInstanceID = id
	}
	return os.localInstanceID, nil
}

// InstanceID returns the cloud provider ID of the specified instance.
func (i *Instances) InstanceID(name types.NodeName) (string, error) {
	srv, err := getServerByName(i.compute, name)
	if err != nil {
		if err == ErrNotFound {
			return "", cloudprovider.InstanceNotFound
		}
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
	instanceID, err := instanceIDFromProviderID(providerID)

	if err != nil {
		return "", err
	}

	server, err := servers.Get(i.compute, instanceID).Extract()

	if err != nil {
		return "", err
	}

	return srvInstanceType(server)
}

// InstanceType returns the type of the specified instance.
func (i *Instances) InstanceType(name types.NodeName) (string, error) {
	srv, err := getServerByName(i.compute, name)

	if err != nil {
		return "", err
	}

	return srvInstanceType(srv)
}

func srvInstanceType(srv *servers.Server) (string, error) {
	keys := []string{"name", "id", "original_name"}
	for _, key := range keys {
		val, found := srv.Flavor[key]
		if found {
			flavor, ok := val.(string)
			if ok {
				return flavor, nil
			}
		}
	}
	return "", fmt.Errorf("flavor name/id not found")
}

// instanceIDFromProviderID splits a provider's id and return instanceID.
// A providerID is build out of '${ProviderName}:///${instance-id}'which contains ':///'.
// See cloudprovider.GetInstanceProviderID and Instances.InstanceID.
func instanceIDFromProviderID(providerID string) (instanceID string, err error) {
	// If Instances.InstanceID or cloudprovider.GetInstanceProviderID is changed, the regexp should be changed too.
	var providerIdRegexp = regexp.MustCompile(`^` + ProviderName + `:///([^/]+)$`)

	matches := providerIdRegexp.FindStringSubmatch(providerID)
	if len(matches) != 2 {
		return "", fmt.Errorf("ProviderID \"%s\" didn't match expected format \"openstack:///InstanceID\"", providerID)
	}
	return matches[1], nil
}
