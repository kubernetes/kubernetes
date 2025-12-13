//go:build windows
// +build windows

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

package winkernel

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
)

type loadbalancerType string

const (
	loadbalancerTypeClusterIP   loadbalancerType = "ClusterIP"
	loadbalancerTypeNodePort    loadbalancerType = "NodePort"
	loadbalancerTypeIngressIP   loadbalancerType = "IngressIP"
	loadbalancerTypeExternalIP  loadbalancerType = "ExternalIP"
	loadbalancerTypeHealthCheck loadbalancerType = "HealthCheck"
)

type loadbalancerConfig struct {
	hnsID                   string
	srcVip                  string
	vip                     string
	protocol                uint16
	internalPort            uint16
	externalPort            uint16
	winProxyOptimization    bool
	endpointsAvailableForLB bool
	lbFlags                 loadBalancerFlags
	loadbalancerType        loadbalancerType
	endpoints               []endpointInfo
	queriedLoadBalancers    map[loadBalancerIdentifier]*loadBalancerInfo
}

func (lbConfig *loadbalancerConfig) String() string {
	return fmt.Sprintf("LoadbalancerConfig: {hnsID: %s, srcVip: %s, vip: %s, protocol: %d, internalPort: %d, externalPort: %d, winProxyOptimization: %t, endpointsAvailableForLB: %t, lbFlags: %v, loadbalancerType: %s, endpoints: %v, queriedLoadBalancers: %v}",
		lbConfig.hnsID, lbConfig.srcVip, lbConfig.vip, lbConfig.protocol, lbConfig.internalPort, lbConfig.externalPort, lbConfig.winProxyOptimization, lbConfig.endpointsAvailableForLB, lbConfig.lbFlags, lbConfig.loadbalancerType, lbConfig.endpoints, lbConfig.queriedLoadBalancers)
}

func (proxier *Proxier) requiresUpdateLoadbalancer(lbHnsID string, endpointCount int) bool {
	return proxier.supportedFeatures.ModifyLoadbalancer && lbHnsID != "" && endpointCount > 0
}

// handleUpdateLoadbalancerFailure will handle the error returned by updatePolicy. If the error is due to unsupported feature,
// then it will set the supportedFeatures.ModifyLoadbalancer to false. return true means skip the iteration.
func (proxier *Proxier) handleUpdateLoadbalancerFailure(err error, hnsID, svcIP string, endpointCount int) (skipIteration bool) {
	if err != nil {
		if proxier.hcn.IsNotImplemented(err) {
			klog.Warning("Update loadbalancer policies is not implemented.", "hnsID", hnsID, "svcIP", svcIP, "endpointCount", endpointCount)
			proxier.supportedFeatures.ModifyLoadbalancer = false
		} else {
			klog.ErrorS(err, "Update loadbalancer policy failed", "hnsID", hnsID, "svcIP", svcIP, "endpointCount", endpointCount)
			skipIteration = true
		}
	}
	return skipIteration
}

// manageLoadbalancer handles the lifecycle of the load balancer.
// It first checks whether an update is needed and if updates are supported (i.e., the ModifyLoadbalancer flag is true based on the supported HNS version range).
// If so, it attempts to update the load balancer. If the update is rejected by HNS as unsupported, the ModifyLoadbalancer flag is set to false,
// and the function proceeds with deleting and recreating the load balancer.
// If the initial check determines that update criteria are not met, it directly proceeds with deletion (if necessary) and creation of the load balancer.
func (proxier *Proxier) manageLoadbalancer(lbConfig *loadbalancerConfig) (success bool) {
	klog.V(3).InfoS("manageLoadbalancer invoked", "LoadbalancerConfig", lbConfig)
	success = true
	if proxier.requiresUpdateLoadbalancer(lbConfig.hnsID, len(lbConfig.endpoints)) && lbConfig.endpointsAvailableForLB {
		hnsLoadBalancer, err := proxier.hns.updateLoadBalancer(
			lbConfig.hnsID,
			lbConfig.srcVip,
			lbConfig.vip,
			lbConfig.endpoints,
			lbConfig.lbFlags,
			lbConfig.protocol,
			lbConfig.internalPort,
			lbConfig.externalPort,
			lbConfig.queriedLoadBalancers,
		)
		if skipIteration := proxier.handleUpdateLoadbalancerFailure(err, lbConfig.hnsID, lbConfig.vip, len(lbConfig.endpoints)); skipIteration {
			return false
		}
		if proxier.supportedFeatures.ModifyLoadbalancer {
			klog.V(3).InfoS("Loadbalancer update successful.", "LoadbalancerType", lbConfig.loadbalancerType, "srcVip", lbConfig.srcVip, "vip", lbConfig.vip, "hnsID", hnsLoadBalancer.hnsID, "ExternalPort", lbConfig.externalPort, "InternalPort", lbConfig.internalPort, "protocol", lbConfig.protocol, "EndpointCount", len(lbConfig.endpoints))
		} else {
			klog.Warning("Loadbalancer update unsupported by hns", "LoadbalancerType", lbConfig.loadbalancerType, "srcVip", lbConfig.srcVip, "vip", lbConfig.vip, "hnsID", hnsLoadBalancer.hnsID, "ExternalPort", lbConfig.externalPort, "InternalPort", lbConfig.internalPort, "protocol", lbConfig.protocol, "EndpointCount", len(lbConfig.endpoints))
		}
	}

	if !proxier.requiresUpdateLoadbalancer(lbConfig.hnsID, len(lbConfig.endpoints)) {
		proxier.deleteExistingLoadBalancer(proxier.hns, lbConfig.winProxyOptimization, &lbConfig.hnsID, lbConfig.vip, lbConfig.protocol, lbConfig.internalPort, lbConfig.externalPort, lbConfig.endpoints, lbConfig.queriedLoadBalancers)
		if len(lbConfig.endpoints) > 0 && lbConfig.endpointsAvailableForLB {

			// If all endpoints are terminating, then no need to create Cluster IP LoadBalancer
			// Cluster IP LoadBalancer creation
			hnsLoadBalancer, err := proxier.hns.getLoadBalancer(
				lbConfig.endpoints,
				lbConfig.lbFlags,
				lbConfig.srcVip,
				lbConfig.vip,
				lbConfig.protocol,
				lbConfig.internalPort,
				lbConfig.externalPort,
				lbConfig.queriedLoadBalancers,
			)
			if err != nil {
				klog.ErrorS(err, "Loadbalancer policy creation failed", "LoadbalancerType", lbConfig.loadbalancerType, "srcVip", lbConfig.srcVip, "vip", lbConfig.vip, "ExternalPort", lbConfig.externalPort, "InternalPort", lbConfig.internalPort, "protocol", lbConfig.protocol, "EndpointCount", len(lbConfig.endpoints))
				return false
			}

			lbConfig.hnsID = hnsLoadBalancer.hnsID
			klog.V(3).InfoS("Loadbalancer create successful.", "LoadbalancerType", lbConfig.loadbalancerType, "srcVip", lbConfig.srcVip, "vip", lbConfig.vip, "hnsID", hnsLoadBalancer.hnsID, "ExternalPort", lbConfig.externalPort, "InternalPort", lbConfig.internalPort, "protocol", lbConfig.protocol, "EndpointCount", len(lbConfig.endpoints))
		} else {
			klog.V(3).InfoS("Skipped creating Hns LoadBalancer for cluster ip resources. Reason : No endpoints available", "LoadbalancerType", lbConfig.loadbalancerType, "srcVip", lbConfig.srcVip, "vip", lbConfig.vip, "ExternalPort", lbConfig.externalPort, "InternalPort", lbConfig.internalPort, "protocol", lbConfig.protocol, "EndpointCount", len(lbConfig.endpoints))
		}
	}
	return success
}

// manageClusterIPLoadbalancer manages the lifecycle of the ClusterIP load balancer.
func (proxier *Proxier) manageClusterIPLoadbalancer(srcVip string, svcInfo *serviceInfo, endpoints []endpointInfo, queriedLoadBalancers map[loadBalancerIdentifier]*loadBalancerInfo) (success bool) {
	success = true
	sessionAffinityClientIP := svcInfo.SessionAffinityType() == v1.ServiceAffinityClientIP
	lbConfig := loadbalancerConfig{
		loadbalancerType:        loadbalancerTypeClusterIP,
		hnsID:                   svcInfo.hnsID,
		srcVip:                  srcVip,
		vip:                     svcInfo.ClusterIP().String(),
		protocol:                Enum(svcInfo.Protocol()),
		internalPort:            uint16(svcInfo.targetPort),
		externalPort:            uint16(svcInfo.Port()),
		lbFlags:                 loadBalancerFlags{isDSR: proxier.isDSR, isIPv6: proxier.ipFamily == v1.IPv6Protocol, sessionAffinity: sessionAffinityClientIP},
		endpoints:               endpoints,
		winProxyOptimization:    svcInfo.winProxyOptimization,
		endpointsAvailableForLB: true,
		queriedLoadBalancers:    queriedLoadBalancers,
	}
	success = proxier.manageLoadbalancer(&lbConfig)
	svcInfo.hnsID = lbConfig.hnsID
	return success
}

// manageNodePortLoadbalancer manages the lifecycle of the NodePort load balancer.
func (proxier *Proxier) manageNodePortLoadbalancer(srcVip string, svcInfo *serviceInfo, endpoints []endpointInfo, queriedLoadBalancers map[loadBalancerIdentifier]*loadBalancerInfo, endpointsAvailableForLB bool) (success bool) {
	if svcInfo.NodePort() <= 0 {
		return true
	}
	success = true
	sessionAffinityClientIP := svcInfo.SessionAffinityType() == v1.ServiceAffinityClientIP
	lbConfig := loadbalancerConfig{
		loadbalancerType:        loadbalancerTypeNodePort,
		hnsID:                   svcInfo.nodePorthnsID,
		srcVip:                  srcVip,
		vip:                     "",
		protocol:                Enum(svcInfo.Protocol()),
		internalPort:            uint16(svcInfo.targetPort),
		externalPort:            uint16(svcInfo.NodePort()),
		lbFlags:                 loadBalancerFlags{isVipExternalIP: true, isDSR: svcInfo.localTrafficDSR, localRoutedVIP: true, sessionAffinity: sessionAffinityClientIP, isIPv6: proxier.ipFamily == v1.IPv6Protocol},
		endpoints:               endpoints,
		winProxyOptimization:    svcInfo.winProxyOptimization,
		endpointsAvailableForLB: endpointsAvailableForLB,
		queriedLoadBalancers:    queriedLoadBalancers,
	}
	success = proxier.manageLoadbalancer(&lbConfig)
	svcInfo.nodePorthnsID = lbConfig.hnsID
	return success
}

// manageExternalIPLoadbalancers manages the lifecycle of the ExternalIP load balancers.
func (proxier *Proxier) manageExternalIPLoadbalancers(srcVip string, svcInfo *serviceInfo, endpoints []endpointInfo, queriedLoadBalancers map[loadBalancerIdentifier]*loadBalancerInfo, endpointsAvailableForLB bool) (success bool) {
	success = true
	sessionAffinityClientIP := svcInfo.SessionAffinityType() == v1.ServiceAffinityClientIP
	lbConfig := loadbalancerConfig{
		loadbalancerType:        loadbalancerTypeExternalIP,
		srcVip:                  srcVip,
		protocol:                Enum(svcInfo.Protocol()),
		internalPort:            uint16(svcInfo.targetPort),
		externalPort:            uint16(svcInfo.Port()),
		lbFlags:                 loadBalancerFlags{isVipExternalIP: true, isDSR: svcInfo.localTrafficDSR, sessionAffinity: sessionAffinityClientIP, isIPv6: proxier.ipFamily == v1.IPv6Protocol},
		endpoints:               endpoints,
		winProxyOptimization:    svcInfo.winProxyOptimization,
		endpointsAvailableForLB: endpointsAvailableForLB,
		queriedLoadBalancers:    queriedLoadBalancers,
	}

	// Create a Load Balancer Policy for each external IP
	for _, externalIP := range svcInfo.externalIPs {
		lbConfig.hnsID = externalIP.hnsID
		lbConfig.vip = externalIP.ip
		success = proxier.manageLoadbalancer(&lbConfig)
		externalIP.hnsID = lbConfig.hnsID
		if !success {
			klog.Warning("Failed to manage ExternalIP loadbalancer", "hnsID", lbConfig.hnsID, "vip", lbConfig.vip)
			return false
		}
	}

	return success
}

// manageIngressIPLoadbalancers manages the lifecycle of the IngressIP load balancers.
func (proxier *Proxier) manageIngressIPLoadbalancers(srcVip string, svcInfo *serviceInfo, endpoints []endpointInfo, queriedLoadBalancers map[loadBalancerIdentifier]*loadBalancerInfo, endpointsAvailableForLB bool) (success bool) {
	var gatewayHnsendpoint *endpointInfo = nil
	var gwEndpoints []endpointInfo
	success = true
	sessionAffinityClientIP := svcInfo.SessionAffinityType() == v1.ServiceAffinityClientIP
	healthPort := proxier.healthzPort

	if svcInfo.HealthCheckNodePort() != 0 {
		healthPort = svcInfo.HealthCheckNodePort()
	}

	if proxier.forwardHealthCheckVip && endpointsAvailableForLB {
		gatewayHnsendpoint, _ = proxier.hns.getEndpointByName(proxier.rootHnsEndpointName)
	}

	if gatewayHnsendpoint != nil {
		gwEndpoints = append(gwEndpoints, *gatewayHnsendpoint)
	}

	lbConfigIngressIP := loadbalancerConfig{
		loadbalancerType:        loadbalancerTypeIngressIP,
		srcVip:                  srcVip,
		protocol:                Enum(svcInfo.Protocol()),
		internalPort:            uint16(svcInfo.targetPort),
		externalPort:            uint16(svcInfo.Port()),
		lbFlags:                 loadBalancerFlags{isVipExternalIP: true, isDSR: svcInfo.preserveDIP || svcInfo.localTrafficDSR, useMUX: svcInfo.preserveDIP, preserveDIP: svcInfo.preserveDIP, sessionAffinity: sessionAffinityClientIP, isIPv6: proxier.ipFamily == v1.IPv6Protocol},
		endpoints:               endpoints,
		winProxyOptimization:    svcInfo.winProxyOptimization,
		endpointsAvailableForLB: endpointsAvailableForLB,
		queriedLoadBalancers:    queriedLoadBalancers,
	}

	lbConfigHealthCheck := loadbalancerConfig{
		loadbalancerType:        loadbalancerTypeHealthCheck,
		srcVip:                  srcVip,
		protocol:                Enum(svcInfo.Protocol()),
		internalPort:            uint16(healthPort),
		externalPort:            uint16(healthPort),
		lbFlags:                 loadBalancerFlags{isDSR: false, useMUX: svcInfo.preserveDIP, preserveDIP: svcInfo.preserveDIP, isIPv6: proxier.ipFamily == v1.IPv6Protocol},
		endpoints:               gwEndpoints,
		winProxyOptimization:    svcInfo.winProxyOptimization,
		endpointsAvailableForLB: endpointsAvailableForLB,
		queriedLoadBalancers:    queriedLoadBalancers,
	}

	// Create a Load Balancer Policy for each Ingress IP
	for _, lbIngressIP := range svcInfo.loadBalancerIngressIPs {
		lbConfigIngressIP.hnsID = lbIngressIP.hnsID
		lbConfigIngressIP.vip = lbIngressIP.ip
		success = proxier.manageLoadbalancer(&lbConfigIngressIP)
		lbIngressIP.hnsID = lbConfigIngressIP.hnsID
		if !success {
			klog.Warning("Failed to manage IngressIP loadbalancer", "hnsID", lbConfigIngressIP.hnsID, "vip", lbConfigIngressIP.vip)
			return false
		}

		lbConfigHealthCheck.hnsID = lbIngressIP.healthCheckHnsID
		lbConfigHealthCheck.vip = lbIngressIP.ip
		success = proxier.manageLoadbalancer(&lbConfigHealthCheck)
		lbIngressIP.healthCheckHnsID = lbConfigHealthCheck.hnsID
		if !success {
			klog.Warning("Failed to manage IngressIP HealthCheck loadbalancer", "hnsID", lbConfigHealthCheck.hnsID, "vip", lbConfigHealthCheck.vip)
			return false
		}
	}

	return success
}
