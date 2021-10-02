//go:build !providerless
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
	"net/http"
	"path"
	"time"

	"google.golang.org/api/compute/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2"

	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/filter"
	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/meta"
	cloudprovider "k8s.io/cloud-provider"
)

func newRoutesMetricContext(request string) *metricContext {
	return newGenericMetricContext("routes", request, unusedMetricLabel, unusedMetricLabel, computeV1Version)
}

// ListRoutes in the cloud environment.
func (g *Cloud) ListRoutes(ctx context.Context, clusterName string) ([]*cloudprovider.Route, error) {
	timeoutCtx, cancel := context.WithTimeout(ctx, 1*time.Hour)
	defer cancel()

	mc := newRoutesMetricContext("list")
	prefix := truncateClusterName(clusterName)
	f := filter.Regexp("name", prefix+"-.*").AndRegexp("network", g.NetworkURL()).AndRegexp("description", k8sNodeRouteTag)
	routes, err := g.c.Routes().List(timeoutCtx, f)
	if err != nil {
		return nil, mc.Observe(err)
	}
	var croutes []*cloudprovider.Route
	for _, r := range routes {
		target := path.Base(r.NextHopInstance)
		// TODO: Should we lastComponent(target) this?
		targetNodeName := types.NodeName(target) // NodeName == Instance Name on GCE
		croutes = append(croutes, &cloudprovider.Route{
			Name:            r.Name,
			TargetNode:      targetNodeName,
			DestinationCIDR: r.DestRange,
		})
	}
	return croutes, mc.Observe(nil)
}

// CreateRoute in the cloud environment.
func (g *Cloud) CreateRoute(ctx context.Context, clusterName string, nameHint string, route *cloudprovider.Route) error {
	timeoutCtx, cancel := context.WithTimeout(ctx, 1*time.Hour)
	defer cancel()

	mc := newRoutesMetricContext("create")

	targetInstance, err := g.getInstanceByName(mapNodeNameToInstanceName(route.TargetNode))
	if err != nil {
		return mc.Observe(err)
	}
	cr := &compute.Route{
		// TODO(thockin): generate a unique name for node + route cidr. Don't depend on name hints.
		Name:            truncateClusterName(clusterName) + "-" + nameHint,
		DestRange:       route.DestinationCIDR,
		NextHopInstance: fmt.Sprintf("zones/%s/instances/%s", targetInstance.Zone, targetInstance.Name),
		Network:         g.NetworkURL(),
		Priority:        1000,
		Description:     k8sNodeRouteTag,
	}
	err = g.c.Routes().Insert(timeoutCtx, meta.GlobalKey(cr.Name), cr)
	if isHTTPErrorCode(err, http.StatusConflict) {
		klog.Infof("Route %q already exists.", cr.Name)
		err = nil
	}
	return mc.Observe(err)
}

// DeleteRoute from the cloud environment.
func (g *Cloud) DeleteRoute(ctx context.Context, clusterName string, route *cloudprovider.Route) error {
	timeoutCtx, cancel := context.WithTimeout(ctx, 1*time.Hour)
	defer cancel()

	mc := newRoutesMetricContext("delete")
	return mc.Observe(g.c.Routes().Delete(timeoutCtx, meta.GlobalKey(route.Name)))
}

func truncateClusterName(clusterName string) string {
	if len(clusterName) > 26 {
		return clusterName[:26]
	}
	return clusterName
}
