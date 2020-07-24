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

package azure

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"

	"k8s.io/apimachinery/pkg/types"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/klog/v2"
	azcache "k8s.io/legacy-cloud-providers/azure/cache"
)

// makeZone returns the zone value in format of <region>-<zone-id>.
func (az *Cloud) makeZone(location string, zoneID int) string {
	return fmt.Sprintf("%s-%d", strings.ToLower(location), zoneID)
}

// isAvailabilityZone returns true if the zone is in format of <region>-<zone-id>.
func (az *Cloud) isAvailabilityZone(zone string) bool {
	return strings.HasPrefix(zone, fmt.Sprintf("%s-", az.Location))
}

// GetZoneID returns the ID of zone from node's zone label.
func (az *Cloud) GetZoneID(zoneLabel string) string {
	if !az.isAvailabilityZone(zoneLabel) {
		return ""
	}

	return strings.TrimPrefix(zoneLabel, fmt.Sprintf("%s-", az.Location))
}

// GetZone returns the Zone containing the current availability zone and locality region that the program is running in.
// If the node is not running with availability zones, then it will fall back to fault domain.
func (az *Cloud) GetZone(ctx context.Context) (cloudprovider.Zone, error) {
	if az.UseInstanceMetadata {
		metadata, err := az.metadata.GetMetadata(azcache.CacheReadTypeUnsafe)
		if err != nil {
			return cloudprovider.Zone{}, err
		}

		if metadata.Compute == nil {
			az.metadata.imsCache.Delete(metadataCacheKey)
			return cloudprovider.Zone{}, fmt.Errorf("failure of getting compute information from instance metadata")
		}

		zone := ""
		location := metadata.Compute.Location
		if metadata.Compute.Zone != "" {
			zoneID, err := strconv.Atoi(metadata.Compute.Zone)
			if err != nil {
				return cloudprovider.Zone{}, fmt.Errorf("failed to parse zone ID %q: %v", metadata.Compute.Zone, err)
			}
			zone = az.makeZone(location, zoneID)
		} else {
			klog.V(3).Infof("Availability zone is not enabled for the node, falling back to fault domain")
			zone = metadata.Compute.FaultDomain
		}

		return cloudprovider.Zone{
			FailureDomain: strings.ToLower(zone),
			Region:        strings.ToLower(location),
		}, nil
	}
	// if UseInstanceMetadata is false, get Zone name by calling ARM
	hostname, err := os.Hostname()
	if err != nil {
		return cloudprovider.Zone{}, fmt.Errorf("failure getting hostname from kernel")
	}
	return az.vmSet.GetZoneByNodeName(strings.ToLower(hostname))
}

// GetZoneByProviderID implements Zones.GetZoneByProviderID
// This is particularly useful in external cloud providers where the kubelet
// does not initialize node data.
func (az *Cloud) GetZoneByProviderID(ctx context.Context, providerID string) (cloudprovider.Zone, error) {
	if providerID == "" {
		return cloudprovider.Zone{}, errNodeNotInitialized
	}

	// Returns nil for unmanaged nodes because azure cloud provider couldn't fetch information for them.
	if az.IsNodeUnmanagedByProviderID(providerID) {
		klog.V(2).Infof("GetZoneByProviderID: omitting unmanaged node %q", providerID)
		return cloudprovider.Zone{}, nil
	}

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
	// Returns "" for unmanaged nodes because azure cloud provider couldn't fetch information for them.
	unmanaged, err := az.IsNodeUnmanaged(string(nodeName))
	if err != nil {
		return cloudprovider.Zone{}, err
	}
	if unmanaged {
		klog.V(2).Infof("GetZoneByNodeName: omitting unmanaged node %q", nodeName)
		return cloudprovider.Zone{}, nil
	}

	return az.vmSet.GetZoneByNodeName(string(nodeName))
}
