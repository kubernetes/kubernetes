/*
Copyright 2018 The Kubernetes Authors.

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

// Package app implements a server that runs a set of active
// components.  This includes node controllers, service and
// route controller, and so on.
//
package app

import (
	"net"
	"net/http"
	"strings"

	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/klog"
	cloudcontrollerconfig "k8s.io/kubernetes/cmd/cloud-controller-manager/app/config"
	cloudcontrollers "k8s.io/kubernetes/pkg/controller/cloud"
	routecontroller "k8s.io/kubernetes/pkg/controller/route"
	servicecontroller "k8s.io/kubernetes/pkg/controller/service"
)

func startCloudNodeController(ctx *cloudcontrollerconfig.CompletedConfig, cloud cloudprovider.Interface, stopCh <-chan struct{}) (http.Handler, bool, error) {
	// Start the CloudNodeController
	nodeController := cloudcontrollers.NewCloudNodeController(
		ctx.SharedInformers.Core().V1().Nodes(),
		// cloud node controller uses existing cluster role from node-controller
		ctx.ClientBuilder.ClientOrDie("node-controller"),
		cloud,
		ctx.ComponentConfig.NodeStatusUpdateFrequency.Duration)

	go nodeController.Run(stopCh)

	return nil, true, nil
}

func startCloudNodeLifecycleController(ctx *cloudcontrollerconfig.CompletedConfig, cloud cloudprovider.Interface, stopCh <-chan struct{}) (http.Handler, bool, error) {
	// Start the cloudNodeLifecycleController
	cloudNodeLifecycleController, err := cloudcontrollers.NewCloudNodeLifecycleController(
		ctx.SharedInformers.Core().V1().Nodes(),
		// cloud node lifecycle controller uses existing cluster role from node-controller
		ctx.ClientBuilder.ClientOrDie("node-controller"),
		cloud,
		ctx.ComponentConfig.KubeCloudShared.NodeMonitorPeriod.Duration,
	)
	if err != nil {
		klog.Warningf("failed to start cloud node lifecycle controller: %s", err)
		return nil, false, nil
	}

	go cloudNodeLifecycleController.Run(stopCh)

	return nil, true, nil
}

func startServiceController(ctx *cloudcontrollerconfig.CompletedConfig, cloud cloudprovider.Interface, stopCh <-chan struct{}) (http.Handler, bool, error) {
	// Start the service controller
	serviceController, err := servicecontroller.New(
		cloud,
		ctx.ClientBuilder.ClientOrDie("service-controller"),
		ctx.SharedInformers.Core().V1().Services(),
		ctx.SharedInformers.Core().V1().Nodes(),
		ctx.ComponentConfig.KubeCloudShared.ClusterName,
	)
	if err != nil {
		// This error shouldn't fail. It lives like this as a legacy.
		klog.Errorf("Failed to start service controller: %v", err)
		return nil, false, nil
	}

	go serviceController.Run(stopCh, int(ctx.ComponentConfig.ServiceController.ConcurrentServiceSyncs))

	return nil, true, nil
}

func startRouteController(ctx *cloudcontrollerconfig.CompletedConfig, cloud cloudprovider.Interface, stopCh <-chan struct{}) (http.Handler, bool, error) {
	if !ctx.ComponentConfig.KubeCloudShared.AllocateNodeCIDRs || !ctx.ComponentConfig.KubeCloudShared.ConfigureCloudRoutes {
		klog.Infof("Will not configure cloud provider routes for allocate-node-cidrs: %v, configure-cloud-routes: %v.", ctx.ComponentConfig.KubeCloudShared.AllocateNodeCIDRs, ctx.ComponentConfig.KubeCloudShared.ConfigureCloudRoutes)
		return nil, false, nil
	}

	// If CIDRs should be allocated for pods and set on the CloudProvider, then start the route controller
	routes, ok := cloud.Routes()
	if !ok {
		klog.Warning("configure-cloud-routes is set, but cloud provider does not support routes. Will not configure cloud provider routes.")
		return nil, false, nil
	}
	var clusterCIDR *net.IPNet
	var err error
	if len(strings.TrimSpace(ctx.ComponentConfig.KubeCloudShared.ClusterCIDR)) != 0 {
		_, clusterCIDR, err = net.ParseCIDR(ctx.ComponentConfig.KubeCloudShared.ClusterCIDR)
		if err != nil {
			klog.Warningf("Unsuccessful parsing of cluster CIDR %v: %v", ctx.ComponentConfig.KubeCloudShared.ClusterCIDR, err)
		}
	}

	routeController := routecontroller.New(
		routes,
		ctx.ClientBuilder.ClientOrDie("route-controller"),
		ctx.SharedInformers.Core().V1().Nodes(),
		ctx.ComponentConfig.KubeCloudShared.ClusterName,
		clusterCIDR,
	)
	go routeController.Run(stopCh, ctx.ComponentConfig.KubeCloudShared.RouteReconciliationPeriod.Duration)

	return nil, true, nil
}
