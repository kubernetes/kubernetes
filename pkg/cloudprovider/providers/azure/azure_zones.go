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
	"strconv"
	"strings"
	"sync"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

const (
	faultDomainURI     = "v1/InstanceInfo/FD"
	computeMetadataURI = "instance/compute"
)

var faultMutex = &sync.Mutex{}
var faultDomain *string

// makeZone returns the zone value in format of <region>-<zone-id>.
func (az *Cloud) makeZone(zoneID int) string {
	return fmt.Sprintf("%s-%d", strings.ToLower(az.Location), zoneID)
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
	computeInfo := ComputeMetadata{}
	err := az.metadata.Object(computeMetadataURI, &computeInfo)
	if err != nil {
		return cloudprovider.Zone{}, err
	}

	if computeInfo.Zone == "" {
		glog.V(3).Infof("Availability zone is not enabled for the node, falling back to fault domain")
		return az.getZoneFromFaultDomain()
	}

	zoneID, err := strconv.Atoi(computeInfo.Zone)
	if err != nil {
		return cloudprovider.Zone{}, fmt.Errorf("failed to parse zone ID %q: %v", computeInfo.Zone, err)
	}

	return cloudprovider.Zone{
		FailureDomain: az.makeZone(zoneID),
		Region:        az.Location,
	}, nil
}

// getZoneFromFaultDomain gets fault domain for the instance.
// Fault domain is the fallback when availability zone is not enabled for the node.
func (az *Cloud) getZoneFromFaultDomain() (cloudprovider.Zone, error) {
	faultMutex.Lock()
	defer faultMutex.Unlock()
	if faultDomain == nil {
		var err error
		faultDomain, err = az.fetchFaultDomain()
		if err != nil {
			return cloudprovider.Zone{}, err
		}
	}
	zone := cloudprovider.Zone{
		FailureDomain: *faultDomain,
		Region:        az.Location,
	}
	return zone, nil
}

// GetZoneByProviderID implements Zones.GetZoneByProviderID
// This is particularly useful in external cloud providers where the kubelet
// does not initialize node data.
func (az *Cloud) GetZoneByProviderID(ctx context.Context, providerID string) (cloudprovider.Zone, error) {
	// Returns nil for unmanaged nodes because azure cloud provider couldn't fetch information for them.
	if az.IsNodeUnmanagedByProviderID(providerID) {
		glog.V(2).Infof("GetZoneByProviderID: omitting unmanaged node %q", providerID)
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
		glog.V(2).Infof("GetZoneByNodeName: omitting unmanaged node %q", nodeName)
		return cloudprovider.Zone{}, nil
	}

	return az.vmSet.GetZoneByNodeName(string(nodeName))
}

func (az *Cloud) fetchFaultDomain() (*string, error) {
	faultDomain, err := az.metadata.Text(faultDomainURI)
	if err != nil {
		return nil, err
	}

	return &faultDomain, nil
}
