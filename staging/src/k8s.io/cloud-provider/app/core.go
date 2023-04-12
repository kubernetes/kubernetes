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
package app

import (
	"context"
	"fmt"
	"net"
	"strings"

	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/cloud-provider/app/config"
	cloudnodecontroller "k8s.io/cloud-provider/controllers/node"
	cloudnodelifecyclecontroller "k8s.io/cloud-provider/controllers/nodelifecycle"
	routecontroller "k8s.io/cloud-provider/controllers/route"
	servicecontroller "k8s.io/cloud-provider/controllers/service"
	controllermanagerapp "k8s.io/controller-manager/app"
	"k8s.io/controller-manager/controller"
	"k8s.io/klog/v2"
	netutils "k8s.io/utils/net"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
)

func startCloudNodeController(ctx context.Context, initContext ControllerInitContext, controlexContext controllermanagerapp.ControllerContext, completedConfig *config.CompletedConfig, cloud cloudprovider.Interface) (controller.Interface, bool, error) {
	// Start the CloudNodeController
	nodeController, err := cloudnodecontroller.NewCloudNodeController(
		completedConfig.SharedInformers.Core().V1().Nodes(),
		// cloud node controller uses existing cluster role from node-controller
		completedConfig.ClientBuilder.ClientOrDie(initContext.ClientName),
		cloud,
		completedConfig.ComponentConfig.NodeStatusUpdateFrequency.Duration,
		completedConfig.ComponentConfig.NodeController.ConcurrentNodeSyncs,
	)
	if err != nil {
		klog.Warningf("failed to start cloud node controller: %s", err)
		return nil, false, nil
	}

	go nodeController.Run(ctx.Done(), controlexContext.ControllerManagerMetrics)

	return nil, true, nil
}

func startCloudNodeLifecycleController(ctx context.Context, initContext ControllerInitContext, controlexContext controllermanagerapp.ControllerContext, completedConfig *config.CompletedConfig, cloud cloudprovider.Interface) (controller.Interface, bool, error) {
	// Start the cloudNodeLifecycleController
	cloudNodeLifecycleController, err := cloudnodelifecyclecontroller.NewCloudNodeLifecycleController(
		completedConfig.SharedInformers.Core().V1().Nodes(),
		// cloud node lifecycle controller uses existing cluster role from node-controller
		completedConfig.ClientBuilder.ClientOrDie(initContext.ClientName),
		cloud,
		completedConfig.ComponentConfig.KubeCloudShared.NodeMonitorPeriod.Duration,
	)
	if err != nil {
		klog.Warningf("failed to start cloud node lifecycle controller: %s", err)
		return nil, false, nil
	}

	go cloudNodeLifecycleController.Run(ctx, controlexContext.ControllerManagerMetrics)

	return nil, true, nil
}

func startServiceController(ctx context.Context, initContext ControllerInitContext, controlexContext controllermanagerapp.ControllerContext, completedConfig *config.CompletedConfig, cloud cloudprovider.Interface) (controller.Interface, bool, error) {
	// Start the service controller
	serviceController, err := servicecontroller.New(
		cloud,
		completedConfig.ClientBuilder.ClientOrDie(initContext.ClientName),
		completedConfig.SharedInformers.Core().V1().Services(),
		completedConfig.SharedInformers.Core().V1().Nodes(),
		completedConfig.ComponentConfig.KubeCloudShared.ClusterName,
		utilfeature.DefaultFeatureGate,
	)
	if err != nil {
		// This error shouldn't fail. It lives like this as a legacy.
		klog.Errorf("Failed to start service controller: %v", err)
		return nil, false, nil
	}

	go serviceController.Run(ctx, int(completedConfig.ComponentConfig.ServiceController.ConcurrentServiceSyncs), controlexContext.ControllerManagerMetrics)

	return nil, true, nil
}

func startRouteController(ctx context.Context, initContext ControllerInitContext, controlexContext controllermanagerapp.ControllerContext, completedConfig *config.CompletedConfig, cloud cloudprovider.Interface) (controller.Interface, bool, error) {
	if !completedConfig.ComponentConfig.KubeCloudShared.ConfigureCloudRoutes {
		klog.Infof("Will not configure cloud provider routes, --configure-cloud-routes: %v", completedConfig.ComponentConfig.KubeCloudShared.ConfigureCloudRoutes)
		return nil, false, nil
	}

	// If CIDRs should be allocated for pods and set on the CloudProvider, then start the route controller
	routes, ok := cloud.Routes()
	if !ok {
		klog.Warning("--configure-cloud-routes is set, but cloud provider does not support routes. Will not configure cloud provider routes.")
		return nil, false, nil
	}

	// failure: bad cidrs in config
	clusterCIDRs, dualStack, err := processCIDRs(completedConfig.ComponentConfig.KubeCloudShared.ClusterCIDR)
	if err != nil {
		return nil, false, err
	}

	// failure: more than one cidr but they are not configured as dual stack
	if len(clusterCIDRs) > 1 && !dualStack {
		return nil, false, fmt.Errorf("len of ClusterCIDRs==%v and they are not configured as dual stack (at least one from each IPFamily", len(clusterCIDRs))
	}

	// failure: more than cidrs is not allowed even with dual stack
	if len(clusterCIDRs) > 2 {
		return nil, false, fmt.Errorf("length of clusterCIDRs is:%v more than max allowed of 2", len(clusterCIDRs))
	}

	routeController := routecontroller.New(
		routes,
		completedConfig.ClientBuilder.ClientOrDie(initContext.ClientName),
		completedConfig.SharedInformers.Core().V1().Nodes(),
		completedConfig.ComponentConfig.KubeCloudShared.ClusterName,
		clusterCIDRs,
	)
	go routeController.Run(ctx, completedConfig.ComponentConfig.KubeCloudShared.RouteReconciliationPeriod.Duration, controlexContext.ControllerManagerMetrics)

	return nil, true, nil
}

// processCIDRs is a helper function that works on a comma separated cidrs and returns
// a list of typed cidrs
// a flag if cidrs represents a dual stack
// error if failed to parse any of the cidrs
func processCIDRs(cidrsList string) ([]*net.IPNet, bool, error) {
	cidrsSplit := strings.Split(strings.TrimSpace(cidrsList), ",")

	cidrs, err := netutils.ParseCIDRs(cidrsSplit)
	if err != nil {
		return nil, false, err
	}

	// if cidrs has an error then the previous call will fail
	// safe to ignore error checking on next call
	dualstack, _ := netutils.IsDualStackCIDRs(cidrs)

	return cidrs, dualstack, nil
}
