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

	"github.com/golang/glog"
	compute "google.golang.org/api/compute/v1"
	"k8s.io/apimachinery/pkg/types"

	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/filter"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/meta"
)

func newRoutesMetricContext(request string) *metricContext {
	return newGenericMetricContext("routes", request, unusedMetricLabel, unusedMetricLabel, computeV1Version)
}

// ListRoutes in the cloud environment.
func (gce *GCECloud) ListRoutes(ctx context.Context, clusterName string) ([]*cloudprovider.Route, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newRoutesMetricContext("list")
	prefix := truncateClusterName(clusterName)
	f := filter.Regexp("name", prefix+"-.*").AndRegexp("network", gce.NetworkURL()).AndRegexp("description", k8sNodeRouteTag)
	routes, err := gce.c.Routes().List(ctx, f)
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
func (gce *GCECloud) CreateRoute(ctx context.Context, clusterName string, nameHint string, route *cloudprovider.Route) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newRoutesMetricContext("create")

	targetInstance, err := gce.getInstanceByName(mapNodeNameToInstanceName(route.TargetNode))
	if err != nil {
		return mc.Observe(err)
	}
	cr := &compute.Route{
		Name:            truncateClusterName(clusterName) + "-" + nameHint,
		DestRange:       route.DestinationCIDR,
		NextHopInstance: fmt.Sprintf("zones/%s/instances/%s", targetInstance.Zone, targetInstance.Name),
		Network:         gce.NetworkURL(),
		Priority:        1000,
		Description:     k8sNodeRouteTag,
	}
	err = gce.c.Routes().Insert(ctx, meta.GlobalKey(cr.Name), cr)
	if isHTTPErrorCode(err, http.StatusConflict) {
		glog.Infof("Route %q already exists.", cr.Name)
		err = nil
	}
	return mc.Observe(err)
}

// DeleteRoute from the cloud environment.
func (gce *GCECloud) DeleteRoute(ctx context.Context, clusterName string, route *cloudprovider.Route) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newRoutesMetricContext("delete")
	return mc.Observe(gce.c.Routes().Delete(ctx, meta.GlobalKey(route.Name)))
}

func truncateClusterName(clusterName string) string {
	if len(clusterName) > 26 {
		return clusterName[:26]
	}
	return clusterName
}
