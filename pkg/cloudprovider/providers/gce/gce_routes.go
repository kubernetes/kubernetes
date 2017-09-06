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
	"net/http"
	"path"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/cloudprovider"

	"github.com/golang/glog"
	compute "google.golang.org/api/compute/v1"
)

func newRoutesMetricContext(request string) *metricContext {
	return newGenericMetricContext("routes", request, unusedMetricLabel, unusedMetricLabel, computeV1Version)
}

func (gce *GCECloud) ListRoutes(clusterName string) ([]*cloudprovider.Route, error) {
	var routes []*cloudprovider.Route
	pageToken := ""
	page := 0
	for ; page == 0 || (pageToken != "" && page < maxPages); page++ {
		mc := newRoutesMetricContext("list_page")
		listCall := gce.service.Routes.List(gce.NetworkProjectID())

		prefix := truncateClusterName(clusterName)
		// Filter for routes starting with clustername AND belonging to the
		// relevant gcp network AND having description = "k8s-node-route".
		filter := "(name eq " + prefix + "-.*) "
		filter = filter + "(network eq " + gce.NetworkURL() + ") "
		filter = filter + "(description eq " + k8sNodeRouteTag + ")"
		listCall = listCall.Filter(filter)
		if pageToken != "" {
			listCall = listCall.PageToken(pageToken)
		}
		res, err := listCall.Do()
		mc.Observe(err)
		if err != nil {
			glog.Errorf("Error getting routes from GCE: %v", err)
			return nil, err
		}
		pageToken = res.NextPageToken
		for _, r := range res.Items {
			target := path.Base(r.NextHopInstance)
			// TODO: Should we lastComponent(target) this?
			targetNodeName := types.NodeName(target) // NodeName == Instance Name on GCE
			routes = append(routes, &cloudprovider.Route{Name: r.Name, TargetNode: targetNodeName, DestinationCIDR: r.DestRange})
		}
	}
	if page >= maxPages {
		glog.Errorf("ListRoutes exceeded maxPages=%d for Routes.List; truncating.", maxPages)
	}
	return routes, nil
}

func (gce *GCECloud) CreateRoute(clusterName string, nameHint string, route *cloudprovider.Route) error {
	routeName := truncateClusterName(clusterName) + "-" + nameHint

	instanceName := mapNodeNameToInstanceName(route.TargetNode)
	targetInstance, err := gce.getInstanceByName(instanceName)
	if err != nil {
		return err
	}

	mc := newRoutesMetricContext("create")
	insertOp, err := gce.service.Routes.Insert(gce.NetworkProjectID(), &compute.Route{
		Name:            routeName,
		DestRange:       route.DestinationCIDR,
		NextHopInstance: fmt.Sprintf("zones/%s/instances/%s", targetInstance.Zone, targetInstance.Name),
		Network:         gce.NetworkURL(),
		Priority:        1000,
		Description:     k8sNodeRouteTag,
	}).Do()
	if err != nil {
		if isHTTPErrorCode(err, http.StatusConflict) {
			glog.Infof("Route %v already exists.", routeName)
			return nil
		} else {
			return mc.Observe(err)
		}
	}
	return gce.waitForGlobalOpInProject(insertOp, gce.NetworkProjectID(), mc)
}

func (gce *GCECloud) DeleteRoute(clusterName string, route *cloudprovider.Route) error {
	mc := newRoutesMetricContext("delete")
	deleteOp, err := gce.service.Routes.Delete(gce.NetworkProjectID(), route.Name).Do()
	if err != nil {
		return mc.Observe(err)
	}
	return gce.waitForGlobalOpInProject(deleteOp, gce.NetworkProjectID(), mc)
}

func truncateClusterName(clusterName string) string {
	if len(clusterName) > 26 {
		return clusterName[:26]
	}
	return clusterName
}
