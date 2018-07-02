/*
Copyright 2017 The Kubernetes Authors.

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

package gce

import (
	"context"
	"strings"

	compute "google.golang.org/api/compute/v1"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/filter"
)

func newZonesMetricContext(request, region string) *metricContext {
	return newGenericMetricContext("zones", request, region, unusedMetricLabel, computeV1Version)
}

// GetZone creates a cloudprovider.Zone of the current zone and region
func (gce *GCECloud) GetZone(ctx context.Context) (cloudprovider.Zone, error) {
	return cloudprovider.Zone{
		FailureDomain: gce.localZone,
		Region:        gce.region,
	}, nil
}

// GetZoneByProviderID implements Zones.GetZoneByProviderID
// This is particularly useful in external cloud providers where the kubelet
// does not initialize node data.
func (gce *GCECloud) GetZoneByProviderID(ctx context.Context, providerID string) (cloudprovider.Zone, error) {
	_, zone, _, err := splitProviderID(providerID)
	if err != nil {
		return cloudprovider.Zone{}, err
	}
	region, err := GetGCERegion(zone)
	if err != nil {
		return cloudprovider.Zone{}, err
	}
	return cloudprovider.Zone{FailureDomain: zone, Region: region}, nil
}

// GetZoneByNodeName implements Zones.GetZoneByNodeName
// This is particularly useful in external cloud providers where the kubelet
// does not initialize node data.
func (gce *GCECloud) GetZoneByNodeName(ctx context.Context, nodeName types.NodeName) (cloudprovider.Zone, error) {
	instanceName := mapNodeNameToInstanceName(nodeName)
	instance, err := gce.getInstanceByName(instanceName)
	if err != nil {
		return cloudprovider.Zone{}, err
	}
	region, err := GetGCERegion(instance.Zone)
	if err != nil {
		return cloudprovider.Zone{}, err
	}
	return cloudprovider.Zone{FailureDomain: instance.Zone, Region: region}, nil
}

// ListZonesInRegion returns all zones in a GCP region
func (gce *GCECloud) ListZonesInRegion(region string) ([]*compute.Zone, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newZonesMetricContext("list", region)
	list, err := gce.c.Zones().List(ctx, filter.Regexp("region", gce.getRegionLink(region)))
	if err != nil {
		return nil, mc.Observe(err)
	}
	return list, mc.Observe(err)
}

func (gce *GCECloud) getRegionLink(region string) string {
	return gce.service.BasePath + strings.Join([]string{gce.projectID, "regions", region}, "/")
}
