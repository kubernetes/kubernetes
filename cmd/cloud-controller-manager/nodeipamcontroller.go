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

// This file holds the code related with the sample nodeipamcontroller
// which demonstrates how cloud providers add external controllers to cloud-controller-manager

package main

import (
	"errors"
	"fmt"
	"net"
	"net/http"
	"strings"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/cloud-provider/app"
	cloudcontrollerconfig "k8s.io/cloud-provider/app/config"
	genericcontrollermanager "k8s.io/controller-manager/app"
	"k8s.io/klog/v2"
	nodeipamcontroller "k8s.io/kubernetes/pkg/controller/nodeipam"
	nodeipamconfig "k8s.io/kubernetes/pkg/controller/nodeipam/config"
	"k8s.io/kubernetes/pkg/controller/nodeipam/ipam"
	netutils "k8s.io/utils/net"
)

func startNodeIpamController(ccmconfig *cloudcontrollerconfig.CompletedConfig, nodeipamconfig nodeipamconfig.NodeIPAMControllerConfiguration, ctx genericcontrollermanager.ControllerContext, cloud cloudprovider.Interface) (http.Handler, bool, error) {
	var serviceCIDR *net.IPNet
	var secondaryServiceCIDR *net.IPNet

	// should we start nodeIPAM
	if !ccmconfig.ComponentConfig.KubeCloudShared.AllocateNodeCIDRs {
		return nil, false, nil
	}

	// failure: bad cidrs in config
	clusterCIDRs, dualStack, err := processCIDRs(ccmconfig.ComponentConfig.KubeCloudShared.ClusterCIDR)
	if err != nil {
		return nil, false, err
	}

	// failure: more than one cidr and dual stack is not enabled
	if len(clusterCIDRs) > 1 && !utilfeature.DefaultFeatureGate.Enabled(app.IPv6DualStack) {
		return nil, false, fmt.Errorf("len of ClusterCIDRs==%v and dualstack feature is not enabled", len(clusterCIDRs))
	}

	// failure: more than one cidr but they are not configured as dual stack
	if len(clusterCIDRs) > 1 && !dualStack {
		return nil, false, fmt.Errorf("len of ClusterCIDRs==%v and they are not configured as dual stack (at least one from each IPFamily", len(clusterCIDRs))
	}

	// failure: more than cidrs is not allowed even with dual stack
	if len(clusterCIDRs) > 2 {
		return nil, false, fmt.Errorf("len of clusters is:%v > more than max allowed of 2", len(clusterCIDRs))
	}

	// service cidr processing
	if len(strings.TrimSpace(nodeipamconfig.ServiceCIDR)) != 0 {
		_, serviceCIDR, err = net.ParseCIDR(nodeipamconfig.ServiceCIDR)
		if err != nil {
			klog.Warningf("Unsuccessful parsing of service CIDR %v: %v", nodeipamconfig.ServiceCIDR, err)
		}
	}

	if len(strings.TrimSpace(nodeipamconfig.SecondaryServiceCIDR)) != 0 {
		_, secondaryServiceCIDR, err = net.ParseCIDR(nodeipamconfig.SecondaryServiceCIDR)
		if err != nil {
			klog.Warningf("Unsuccessful parsing of service CIDR %v: %v", nodeipamconfig.SecondaryServiceCIDR, err)
		}
	}

	// the following checks are triggered if both serviceCIDR and secondaryServiceCIDR are provided
	if serviceCIDR != nil && secondaryServiceCIDR != nil {
		// should have dual stack flag enabled
		if !utilfeature.DefaultFeatureGate.Enabled(app.IPv6DualStack) {
			return nil, false, fmt.Errorf("secondary service cidr is provided and IPv6DualStack feature is not enabled")
		}

		// should be dual stack (from different IPFamilies)
		dualstackServiceCIDR, err := netutils.IsDualStackCIDRs([]*net.IPNet{serviceCIDR, secondaryServiceCIDR})
		if err != nil {
			return nil, false, fmt.Errorf("failed to perform dualstack check on serviceCIDR and secondaryServiceCIDR error:%v", err)
		}
		if !dualstackServiceCIDR {
			return nil, false, fmt.Errorf("serviceCIDR and secondaryServiceCIDR are not dualstack (from different IPfamiles)")
		}
	}

	var nodeCIDRMaskSizeIPv4, nodeCIDRMaskSizeIPv6 int
	if utilfeature.DefaultFeatureGate.Enabled(app.IPv6DualStack) {
		// only --node-cidr-mask-size-ipv4 and --node-cidr-mask-size-ipv6 supported with dual stack clusters.
		// --node-cidr-mask-size flag is incompatible with dual stack clusters.
		nodeCIDRMaskSizeIPv4, nodeCIDRMaskSizeIPv6, err = setNodeCIDRMaskSizesDualStack(nodeipamconfig)
	} else {
		// only --node-cidr-mask-size supported with single stack clusters.
		// --node-cidr-mask-size-ipv4 and --node-cidr-mask-size-ipv6 flags are incompatible with dual stack clusters.
		nodeCIDRMaskSizeIPv4, nodeCIDRMaskSizeIPv6, err = setNodeCIDRMaskSizes(nodeipamconfig)
	}

	if err != nil {
		return nil, false, err
	}

	// get list of node cidr mask sizes
	nodeCIDRMaskSizes := getNodeCIDRMaskSizes(clusterCIDRs, nodeCIDRMaskSizeIPv4, nodeCIDRMaskSizeIPv6)

	nodeIpamController, err := nodeipamcontroller.NewNodeIpamController(
		ctx.InformerFactory.Core().V1().Nodes(),
		cloud,
		ctx.ClientBuilder.ClientOrDie("node-controller"),
		clusterCIDRs,
		serviceCIDR,
		secondaryServiceCIDR,
		nodeCIDRMaskSizes,
		ipam.CIDRAllocatorType(ccmconfig.ComponentConfig.KubeCloudShared.CIDRAllocatorType),
	)
	if err != nil {
		return nil, true, err
	}
	go nodeIpamController.Run(ctx.Stop)
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

// setNodeCIDRMaskSizes returns the IPv4 and IPv6 node cidr mask sizes.
// If --node-cidr-mask-size not set, then it will return default IPv4 and IPv6 cidr mask sizes.
func setNodeCIDRMaskSizes(cfg nodeipamconfig.NodeIPAMControllerConfiguration) (int, int, error) {
	ipv4Mask, ipv6Mask := defaultNodeMaskCIDRIPv4, defaultNodeMaskCIDRIPv6
	// NodeCIDRMaskSizeIPv4 and NodeCIDRMaskSizeIPv6 can be used only for dual-stack clusters
	if cfg.NodeCIDRMaskSizeIPv4 != 0 || cfg.NodeCIDRMaskSizeIPv6 != 0 {
		return ipv4Mask, ipv6Mask, errors.New("usage of --node-cidr-mask-size-ipv4 and --node-cidr-mask-size-ipv6 are not allowed with non dual-stack clusters")
	}
	if cfg.NodeCIDRMaskSize != 0 {
		ipv4Mask = int(cfg.NodeCIDRMaskSize)
		ipv6Mask = int(cfg.NodeCIDRMaskSize)
	}
	return ipv4Mask, ipv6Mask, nil
}

// setNodeCIDRMaskSizesDualStack returns the IPv4 and IPv6 node cidr mask sizes to the value provided
// for --node-cidr-mask-size-ipv4 and --node-cidr-mask-size-ipv6 respectively. If value not provided,
// then it will return default IPv4 and IPv6 cidr mask sizes.
func setNodeCIDRMaskSizesDualStack(cfg nodeipamconfig.NodeIPAMControllerConfiguration) (int, int, error) {
	ipv4Mask, ipv6Mask := defaultNodeMaskCIDRIPv4, defaultNodeMaskCIDRIPv6
	// NodeCIDRMaskSize can be used only for single stack clusters
	if cfg.NodeCIDRMaskSize != 0 {
		return ipv4Mask, ipv6Mask, errors.New("usage of --node-cidr-mask-size is not allowed with dual-stack clusters")
	}
	if cfg.NodeCIDRMaskSizeIPv4 != 0 {
		ipv4Mask = int(cfg.NodeCIDRMaskSizeIPv4)
	}
	if cfg.NodeCIDRMaskSizeIPv6 != 0 {
		ipv6Mask = int(cfg.NodeCIDRMaskSizeIPv6)
	}
	return ipv4Mask, ipv6Mask, nil
}

// getNodeCIDRMaskSizes is a helper function that helps the generate the node cidr mask
// sizes slice based on the cluster cidr slice
func getNodeCIDRMaskSizes(clusterCIDRs []*net.IPNet, maskSizeIPv4, maskSizeIPv6 int) []int {
	nodeMaskCIDRs := make([]int, len(clusterCIDRs))

	for idx, clusterCIDR := range clusterCIDRs {
		if netutils.IsIPv6CIDR(clusterCIDR) {
			nodeMaskCIDRs[idx] = maskSizeIPv6
		} else {
			nodeMaskCIDRs[idx] = maskSizeIPv4
		}
	}
	return nodeMaskCIDRs
}
