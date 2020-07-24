// +build !providerless

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
	"context"
	"fmt"
	"regexp"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/servers"
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"

	"k8s.io/apimachinery/pkg/types"
	cloudprovider "k8s.io/cloud-provider"
)

var _ cloudprovider.Instances = (*Instances)(nil)

// Instances encapsulates an implementation of Instances for OpenStack.
type Instances struct {
	compute *gophercloud.ServiceClient
	opts    MetadataOpts
}

const (
	instanceShutoff = "SHUTOFF"
)

// Instances returns an implementation of Instances for OpenStack.
func (os *OpenStack) Instances() (cloudprovider.Instances, bool) {
	klog.V(4).Info("openstack.Instances() called")

	err := os.ensureCloudProviderWasInitialized()
	if err != nil {
		// cannot initialize cloud provider - return empty instances without compute instance,
		// it will be generated later with any call of an Instances' receiver.
		klog.Errorf("cannot initialize cloud provider, only limited functionality is available : %v", err)
		return &Instances{
			opts: os.metadataOpts,
		}, true
	}

	compute, err := os.NewComputeV2()
	if err != nil {
		klog.Errorf("unable to access compute v2 API : %v", err)
		return nil, false
	}

	klog.V(4).Info("Claiming to support Instances")

	return &Instances{
		compute: compute,
		opts:    os.metadataOpts,
	}, true
}

// InstancesV2 returns an implementation of InstancesV2 for OpenStack.
func (os *OpenStack) InstancesV2() (cloudprovider.InstancesV2, bool) {
	return nil, false
}

// CurrentNodeName implements Instances.CurrentNodeName
// Note this is *not* necessarily the same as hostname.
func (i *Instances) CurrentNodeName(ctx context.Context, hostname string) (types.NodeName, error) {
	md, err := getMetadata(i.opts.SearchOrder)
	if err != nil {
		return "", err
	}
	return types.NodeName(md.Name), nil
}

// AddSSHKeyToAllInstances is not implemented for OpenStack
func (i *Instances) AddSSHKeyToAllInstances(ctx context.Context, user string, keyData []byte) error {
	return cloudprovider.NotImplemented
}

// NodeAddresses implements Instances.NodeAddresses
func (i *Instances) NodeAddresses(ctx context.Context, name types.NodeName) ([]v1.NodeAddress, error) {
	klog.V(4).Infof("NodeAddresses(%v) called", name)

	// check if the node is local, in this case we can get its addresses from the metadata service
	// without additional requests to Nova.
	md, err := getMetadata(i.opts.SearchOrder)
	if err != nil {
		return nil, err
	}
	if localName := types.NodeName(md.Name); localName == name {
		localAddress, publicAddress, err := getNodeAddresses()
		if err != nil {
			return nil, err
		}

		addrs := []v1.NodeAddress{
			{
				Type:    v1.NodeHostName,
				Address: md.Name,
			},
		}

		if localAddress != "" {
			addrs = append(addrs, v1.NodeAddress{
				Type:    v1.NodeInternalIP,
				Address: localAddress,
			})
		}

		if publicAddress != "" {
			addrs = append(addrs, v1.NodeAddress{
				Type:    v1.NodeExternalIP,
				Address: publicAddress,
			})
		}

		klog.V(4).Infof("NodeAddresses(%v) => %v", name, addrs)
		return addrs, nil
	}

	addrs, err := getAddressesByName(i.compute, name)
	if err != nil {
		return nil, err
	}

	klog.V(4).Infof("NodeAddresses(%v) => %v", name, addrs)
	return addrs, nil
}

// NodeAddressesByProviderID returns the node addresses of an instances with the specified unique providerID
// This method will not be called from the node that is requesting this ID. i.e. metadata service
// and other local methods cannot be used here
func (i *Instances) NodeAddressesByProviderID(ctx context.Context, providerID string) ([]v1.NodeAddress, error) {
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

// InstanceExistsByProviderID returns true if the instance with the given provider id still exist.
// If false is returned with no error, the instance will be immediately deleted by the cloud controller manager.
func (i *Instances) InstanceExistsByProviderID(ctx context.Context, providerID string) (bool, error) {
	instanceID, err := instanceIDFromProviderID(providerID)
	if err != nil {
		return false, err
	}

	_, err = servers.Get(i.compute, instanceID).Extract()
	if err != nil {
		if isNotFound(err) {
			return false, nil
		}
		return false, err
	}

	return true, nil
}

// InstanceShutdownByProviderID returns true if the instances is in safe state to detach volumes
func (i *Instances) InstanceShutdownByProviderID(ctx context.Context, providerID string) (bool, error) {
	instanceID, err := instanceIDFromProviderID(providerID)
	if err != nil {
		return false, err
	}

	server, err := servers.Get(i.compute, instanceID).Extract()
	if err != nil {
		return false, err
	}

	// SHUTOFF is the only state where we can detach volumes immediately
	if server.Status == instanceShutoff {
		return true, nil
	}
	return false, nil
}

// InstanceMetadataByProviderID returns metadata of the specified instance.
func (i *Instances) InstanceMetadataByProviderID(ctx context.Context, providerID string) (*cloudprovider.InstanceMetadata, error) {
	if providerID == "" {
		return nil, fmt.Errorf("couldn't compute InstanceMetadata for empty providerID")
	}

	instanceID, err := instanceIDFromProviderID(providerID)
	if err != nil {
		return nil, err
	}
	srv, err := servers.Get(i.compute, instanceID).Extract()
	if err != nil {
		return nil, err
	}

	instanceType, err := srvInstanceType(srv)
	if err != nil {
		return nil, err
	}
	addresses, err := nodeAddresses(srv)
	if err != nil {
		return nil, err
	}

	return &cloudprovider.InstanceMetadata{
		ProviderID:    providerID,
		Type:          instanceType,
		NodeAddresses: addresses,
	}, nil
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
func (i *Instances) InstanceID(ctx context.Context, name types.NodeName) (string, error) {
	// check if the node is local, in this case we can get its ID from the metadata service
	// without additional requests to Nova.
	md, err := getMetadata(i.opts.SearchOrder)
	if err != nil {
		return "", err
	}
	localName := types.NodeName(md.Name)
	if localName == name {
		return md.UUID, nil
	}

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
func (i *Instances) InstanceTypeByProviderID(ctx context.Context, providerID string) (string, error) {
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
func (i *Instances) InstanceType(ctx context.Context, name types.NodeName) (string, error) {
	// check if the node is local, in this case we can get its type from the metadata service
	// without additional requests to Nova.
	md, err := getMetadata(i.opts.SearchOrder)
	if err != nil {
		return "", err
	}
	localName := types.NodeName(md.Name)
	if localName == name {
		return getIntanceType()
	}

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
	var providerIDRegexp = regexp.MustCompile(`^` + ProviderName + `:///([^/]+)$`)

	matches := providerIDRegexp.FindStringSubmatch(providerID)
	if len(matches) != 2 {
		return "", fmt.Errorf("ProviderID \"%s\" didn't match expected format \"openstack:///InstanceID\"", providerID)
	}
	return matches[1], nil
}
