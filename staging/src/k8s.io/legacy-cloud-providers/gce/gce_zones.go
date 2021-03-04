// +build !providerless

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
	"fmt"
	"strings"

	compute "google.golang.org/api/compute/v1"

	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud"
	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/filter"
	"k8s.io/apimachinery/pkg/types"
	cloudprovider "k8s.io/cloud-provider"
)

func newZonesMetricContext(request, region string) *metricContext {
	return newGenericMetricContext("zones", request, region, unusedMetricLabel, computeV1Version)
}

// GetZone creates a cloudprovider.Zone of the current zone and region
func (g *Cloud) GetZone(ctx context.Context) (cloudprovider.Zone, error) {
	return cloudprovider.Zone{
		FailureDomain: g.localZone,
		Region:        g.region,
	}, nil
}

// GetZoneByProviderID implements Zones.GetZoneByProviderID
// This is particularly useful in external cloud providers where the kubelet
// does not initialize node data.
func (g *Cloud) GetZoneByProviderID(ctx context.Context, providerID string) (cloudprovider.Zone, error) {
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
func (g *Cloud) GetZoneByNodeName(ctx context.Context, nodeName types.NodeName) (cloudprovider.Zone, error) {
	instanceName := mapNodeNameToInstanceName(nodeName)
	instance, err := g.getInstanceByName(instanceName)
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
func (g *Cloud) ListZonesInRegion(region string) ([]*compute.Zone, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newZonesMetricContext("list", region)
	// Use regex match instead of an exact regional link constructed from getRegionalLink below.
	// See comments in issue kubernetes/kubernetes#87905
	list, err := g.c.Zones().List(ctx, filter.Regexp("region", fmt.Sprintf(".*/regions/%s", region)))
	if err != nil {
		return nil, mc.Observe(err)
	}
	return list, mc.Observe(err)
}

func (g *Cloud) getRegionLink(region string) string {
	return g.service.BasePath + strings.Join([]string{g.projectID, "regions", region}, "/")
}
