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

package openstack

import (
	"github.com/gophercloud/gophercloud/openstack/compute/v2/servers"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

const AvailabilityZone = "availability_zone"

func (os *OpenStack) GetZone() (cloudprovider.Zone, error) {
	md, err := getMetadata()
	if err != nil {
		return cloudprovider.Zone{}, err
	}

	zone := cloudprovider.Zone{
		FailureDomain: md.AvailabilityZone,
		Region:        os.region,
	}
	glog.V(1).Infof("Current zone is %v", zone)

	return zone, nil
}

// GetZoneByProviderID implements Zones.GetZoneByProviderID
// This is particularly useful in external cloud providers where the kubelet
// does not initialize node data.
func (os *OpenStack) GetZoneByProviderID(providerID string) (cloudprovider.Zone, error) {
	instanceID, err := instanceIDFromProviderID(providerID)
	if err != nil {
		return cloudprovider.Zone{}, err
	}

	compute, err := os.NewComputeV2()
	if err != nil {
		return cloudprovider.Zone{}, err
	}

	srv, err := servers.Get(compute, instanceID).Extract()
	if err != nil {
		return cloudprovider.Zone{}, err
	}

	zone := cloudprovider.Zone{
		FailureDomain: srv.Metadata[AvailabilityZone],
		Region:        os.region,
	}
	glog.V(4).Infof("The instance %s in zone %v", srv.Name, zone)

	return zone, nil
}

// GetZoneByNodeName implements Zones.GetZoneByNodeName
// This is particularly useful in external cloud providers where the kubelet
// does not initialize node data.
func (os *OpenStack) GetZoneByNodeName(nodeName types.NodeName) (cloudprovider.Zone, error) {
	compute, err := os.NewComputeV2()
	if err != nil {
		return cloudprovider.Zone{}, err
	}

	srv, err := getServerByName(compute, nodeName)
	if err != nil {
		if err == ErrNotFound {
			return cloudprovider.Zone{}, cloudprovider.InstanceNotFound
		}
		return cloudprovider.Zone{}, err
	}

	zone := cloudprovider.Zone{
		FailureDomain: srv.Metadata[AvailabilityZone],
		Region:        os.region,
	}
	glog.V(4).Infof("The instance %s in zone %v", srv.Name, zone)

	return zone, nil
}
