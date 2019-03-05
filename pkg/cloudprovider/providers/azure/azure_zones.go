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

package azure

import (
	"context"
	"fmt"
	"os"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

// GetZone returns the Zone containing the current availability zone and locality region that the program is running in.
// If the node is not running with availability zones, then it will fall back to fault domain.
func (az *Cloud) GetZone(ctx context.Context) (cloudprovider.Zone, error) {
	if az.UseInstanceMetadata {
		metadata, err := az.metadata.GetMetadata()
		if err != nil {
			return cloudprovider.Zone{}, err
		}

		if metadata.Compute == nil {
			return cloudprovider.Zone{}, fmt.Errorf("failure of getting compute information from instance metadata")
		}

		return cloudprovider.Zone{
			FailureDomain: metadata.Compute.FaultDomain,
			Region:        az.Location,
		}, nil
	}
	// if UseInstanceMetadata is false, get Zone name by calling ARM
	hostname, err := os.Hostname()
	if err != nil {
		return cloudprovider.Zone{}, fmt.Errorf("failure getting hostname from kernel")
	}
	return az.vmSet.GetZoneByNodeName(string(hostname))
}

// GetZoneByProviderID implements Zones.GetZoneByProviderID
// This is particularly useful in external cloud providers where the kubelet
// does not initialize node data.
func (az *Cloud) GetZoneByProviderID(ctx context.Context, providerID string) (cloudprovider.Zone, error) {
	nodeName, err := az.vmSet.GetNodeNameByProviderID(providerID)
	if err != nil {
		return cloudprovider.Zone{}, err
	}

	return az.GetZoneByNodeName(ctx, nodeName)
}

// GetZoneByNodeName implements Zones.GetZoneByNodeName
// This is particularly useful in external cloud providers where the kubelet
// does not initialize node data.
func (az *Cloud) GetZoneByNodeName(ctx context.Context, nodeName types.NodeName) (cloudprovider.Zone, error) {
	return az.vmSet.GetZoneByNodeName(string(nodeName))
}
