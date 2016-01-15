/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/layer3/routers"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

func (os *OpenStack) Routes() (cloudprovider.Routes, bool) {
	return os, true
}

func (os *OpenStack) ListRoutes(clusterName string) ([]*cloudprovider.Route, error) {
	glog.V(4).Info("openstack.ListRoutes() called")

	var routes []*cloudprovider.Route
	router, err := routers.Get(os.network, os.routeOpts.RouterId).Extract()
	if err != nil {
		return nil, err
	}

	for _, r := range router.Routes {
		var target string
		if os.routeOpts.HostnameOverride {
			target = r.NextHop
		} else {
			server, err := getServerByAddress(os.compute, r.NextHop)
			if err == nil {
				target = server.Name
			}
		}
		route := cloudprovider.Route{
			Name:            r.DestinationCIDR,
			TargetInstance:  target,
			DestinationCIDR: r.DestinationCIDR,
		}
		routes = append(routes, &route)
	}
	return routes, err
}

// Create the described managed route
// route.Name will be ignored, although the cloud-provider may use nameHint
// to create a more user-meaningful name.
func (os *OpenStack) CreateRoute(clusterName string, nameHint string, route *cloudprovider.Route) error {
	glog.V(4).Info("openstack.CreateRoute() called")

	var addr string
	if os.routeOpts.HostnameOverride {
		addr = route.TargetInstance
		glog.V(4).Infof("Hostname %s overriden in CreateRoute", addr)
	} else {
		server, err := getServerByName(os.compute, route.TargetInstance)
		if err != nil {
			return err
		}
		addr, err = getAddressByName(os.compute, server.Name)
		if err != nil {
			return err
		}
	}
	router, err := routers.Get(os.network, os.routeOpts.RouterId).Extract()
	if err != nil {
		return err
	}
	routes := router.Routes
	routes = append(routes, routers.Route{DestinationCIDR: route.DestinationCIDR, NextHop: addr})
	opts := routers.UpdateOpts{Routes: routes}

	_, err = routers.Update(os.network, router.ID, opts).Extract()
	if err != nil {
		return err
	}
	glog.V(4).Infof("Route Created: %s %s %s", clusterName, nameHint, routes)
	return nil
}

// Delete the specified managed route
// Route should be as returned by ListRoutes
func (os *OpenStack) DeleteRoute(clusterName string, route *cloudprovider.Route) error {
	glog.V(4).Info("openstack.DleteRoute() called")

	var addr string
	if os.routeOpts.HostnameOverride {
		addr = route.TargetInstance
		glog.V(4).Infof("Hostname %s overriden in DeleteRoute", addr)
		glog.Infof("Hostname %s overriden in DeleteRoute", addr)
	} else {
		server, err := getServerByName(os.compute, route.TargetInstance)
		if err != nil {
			return err
		}
		addr, err = getAddressByName(os.compute, server.Name)
		if err != nil {
			return err
		}
	}
	router, err := routers.Get(os.network, os.routeOpts.RouterId).Extract()
	if err != nil {
		return err
	}

	index := -1
	for i, r := range router.Routes {
		if r.DestinationCIDR == route.DestinationCIDR && r.NextHop == addr {
			index = i
		}
	}

	routes := append(router.Routes[:index], router.Routes[index+1:]...)
	opts := routers.UpdateOpts{Routes: routes}

	_, err = routers.Update(os.network, router.ID, opts).Extract()
	if err != nil {
		return err
	}
	glog.V(4).Infof("Route deleted: %s %s", clusterName, route)
	return nil
}
