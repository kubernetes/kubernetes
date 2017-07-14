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
	"fmt"
	"time"

	compute "google.golang.org/api/compute/v1"

	"k8s.io/kubernetes/pkg/cloudprovider"
	"strings"
)

func newZonesMetricContext(request, region string) *metricContext {
	return &metricContext{
		start:      time.Now(),
		attributes: []string{"zones_" + request, region, unusedMetricLabel},
	}
}

// GetZone creates a cloudprovider.Zone of the current zone and region
func (gce *GCECloud) GetZone() (cloudprovider.Zone, error) {
	return cloudprovider.Zone{
		FailureDomain: gce.localZone,
		Region:        gce.region,
	}, nil
}

// ListZonesInRegion returns all zones in a GCP region
func (gce *GCECloud) ListZonesInRegion(region string) ([]*compute.Zone, error) {
	mc := newZonesMetricContext("list", region)
	filter := fmt.Sprintf("region eq %v", gce.getRegionLink(region))
	list, err := gce.service.Zones.List(gce.projectID).Filter(filter).Do()
	if err != nil {
		return nil, mc.Observe(err)
	}
	return list.Items, mc.Observe(err)
}

func (gce *GCECloud) getRegionLink(region string) string {
	return gce.service.BasePath + strings.Join([]string{gce.projectID, "regions", region}, "/")
}
