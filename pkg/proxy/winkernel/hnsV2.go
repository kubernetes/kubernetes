//go:build windows
// +build windows

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

package winkernel

import (
	"encoding/json"
	"fmt"

	"github.com/Microsoft/hcsshim/hcn"
	"k8s.io/klog/v2"

	"strings"
)

type hnsV2 struct{}

var (
	// LoadBalancerFlagsIPv6 enables IPV6.
	LoadBalancerFlagsIPv6 hcn.LoadBalancerFlags = 2
)

func (hns hnsV2) getNetworkByName(name string) (*hnsNetworkInfo, error) {
	hnsnetwork, err := hcn.GetNetworkByName(name)
	if err != nil {
		klog.ErrorS(err, "Error getting network by name")
		return nil, err
	}

	var remoteSubnets []*remoteSubnetInfo
	for _, policy := range hnsnetwork.Policies {
		if policy.Type == hcn.RemoteSubnetRoute {
			policySettings := hcn.RemoteSubnetRoutePolicySetting{}
			err = json.Unmarshal(policy.Settings, &policySettings)
			if err != nil {
				return nil, fmt.Errorf("failed to unmarshal Remote Subnet policy settings")
			}
			rs := &remoteSubnetInfo{
				destinationPrefix: policySettings.DestinationPrefix,
				isolationID:       policySettings.IsolationId,
				providerAddress:   policySettings.ProviderAddress,
				drMacAddress:      policySettings.DistributedRouterMacAddress,
			}
			remoteSubnets = append(remoteSubnets, rs)
		}
	}

	return &hnsNetworkInfo{
		id:            hnsnetwork.Id,
		name:          hnsnetwork.Name,
		networkType:   string(hnsnetwork.Type),
		remoteSubnets: remoteSubnets,
	}, nil
}

func (hns hnsV2) getAllEndpointsByNetwork(networkName string) (map[string]*(endpointsInfo), error) {
	hcnnetwork, err := hcn.GetNetworkByName(networkName)
	if err != nil {
		klog.ErrorS(err, "failed to get HNS network by name", "name", networkName)
		return nil, err
	}
	endpoints, err := hcn.ListEndpointsOfNetwork(hcnnetwork.Id)
	if err != nil {
		return nil, fmt.Errorf("failed to list endpoints: %w", err)
	}
	endpointInfos := make(map[string]*(endpointsInfo))
	for _, ep := range endpoints {
		// Add to map with key endpoint ID or IP address
		// Storing this is expensive in terms of memory, however there is a bug in Windows Server 2019 that can cause two endpoints to be created with the same IP address.
		// TODO: Store by IP only and remove any lookups by endpoint ID.
		endpointInfos[ep.Id] = &endpointsInfo{
			ip:         ep.IpConfigurations[0].IpAddress,
			isLocal:    uint32(ep.Flags&hcn.EndpointFlagsRemoteEndpoint) == 0,
			macAddress: ep.MacAddress,
			hnsID:      ep.Id,
			hns:        hns,
			// only ready and not terminating endpoints were added to HNS
			ready:       true,
			serving:     true,
			terminating: false,
		}
		endpointInfos[ep.IpConfigurations[0].IpAddress] = endpointInfos[ep.Id]
	}
	klog.V(3).InfoS("Queried endpoints from network", "network", networkName)
	return endpointInfos, nil
}

func (hns hnsV2) getEndpointByID(id string) (*endpointsInfo, error) {
	hnsendpoint, err := hcn.GetEndpointByID(id)
	if err != nil {
		return nil, err
	}
	return &endpointsInfo{ //TODO: fill out PA
		ip:         hnsendpoint.IpConfigurations[0].IpAddress,
		isLocal:    uint32(hnsendpoint.Flags&hcn.EndpointFlagsRemoteEndpoint) == 0, //TODO: Change isLocal to isRemote
		macAddress: hnsendpoint.MacAddress,
		hnsID:      hnsendpoint.Id,
		hns:        hns,
	}, nil
}
func (hns hnsV2) getEndpointByIpAddress(ip string, networkName string) (*endpointsInfo, error) {
	hnsnetwork, err := hcn.GetNetworkByName(networkName)
	if err != nil {
		klog.ErrorS(err, "Error getting network by name")
		return nil, err
	}

	endpoints, err := hcn.ListEndpoints()
	if err != nil {
		return nil, fmt.Errorf("failed to list endpoints: %w", err)
	}
	for _, endpoint := range endpoints {
		equal := false
		if endpoint.IpConfigurations != nil && len(endpoint.IpConfigurations) > 0 {
			equal = endpoint.IpConfigurations[0].IpAddress == ip

			if !equal && len(endpoint.IpConfigurations) > 1 {
				equal = endpoint.IpConfigurations[1].IpAddress == ip
			}
		}
		if equal && strings.EqualFold(endpoint.HostComputeNetwork, hnsnetwork.Id) {
			return &endpointsInfo{
				ip:         ip,
				isLocal:    uint32(endpoint.Flags&hcn.EndpointFlagsRemoteEndpoint) == 0, //TODO: Change isLocal to isRemote
				macAddress: endpoint.MacAddress,
				hnsID:      endpoint.Id,
				hns:        hns,
			}, nil
		}
	}
	return nil, fmt.Errorf("Endpoint %v not found on network %s", ip, networkName)
}
func (hns hnsV2) getEndpointByName(name string) (*endpointsInfo, error) {
	hnsendpoint, err := hcn.GetEndpointByName(name)
	if err != nil {
		return nil, err
	}
	return &endpointsInfo{ //TODO: fill out PA
		ip:         hnsendpoint.IpConfigurations[0].IpAddress,
		isLocal:    uint32(hnsendpoint.Flags&hcn.EndpointFlagsRemoteEndpoint) == 0, //TODO: Change isLocal to isRemote
		macAddress: hnsendpoint.MacAddress,
		hnsID:      hnsendpoint.Id,
		hns:        hns,
	}, nil
}
func (hns hnsV2) createEndpoint(ep *endpointsInfo, networkName string) (*endpointsInfo, error) {
	hnsNetwork, err := hcn.GetNetworkByName(networkName)
	if err != nil {
		return nil, err
	}
	var flags hcn.EndpointFlags
	if !ep.isLocal {
		flags |= hcn.EndpointFlagsRemoteEndpoint
	}
	ipConfig := &hcn.IpConfig{
		IpAddress: ep.ip,
	}
	hnsEndpoint := &hcn.HostComputeEndpoint{
		IpConfigurations: []hcn.IpConfig{*ipConfig},
		MacAddress:       ep.macAddress,
		Flags:            flags,
		SchemaVersion: hcn.SchemaVersion{
			Major: 2,
			Minor: 0,
		},
	}

	var createdEndpoint *hcn.HostComputeEndpoint
	if !ep.isLocal {
		if len(ep.providerAddress) != 0 {
			policySettings := hcn.ProviderAddressEndpointPolicySetting{
				ProviderAddress: ep.providerAddress,
			}
			policySettingsJson, err := json.Marshal(policySettings)
			if err != nil {
				return nil, fmt.Errorf("PA Policy creation failed: %v", err)
			}
			paPolicy := hcn.EndpointPolicy{
				Type:     hcn.NetworkProviderAddress,
				Settings: policySettingsJson,
			}
			hnsEndpoint.Policies = append(hnsEndpoint.Policies, paPolicy)
		}
		createdEndpoint, err = hnsNetwork.CreateRemoteEndpoint(hnsEndpoint)
		if err != nil {
			return nil, err
		}
	} else {
		createdEndpoint, err = hnsNetwork.CreateEndpoint(hnsEndpoint)
		if err != nil {
			return nil, err
		}
	}
	return &endpointsInfo{
		ip:              createdEndpoint.IpConfigurations[0].IpAddress,
		isLocal:         uint32(createdEndpoint.Flags&hcn.EndpointFlagsRemoteEndpoint) == 0,
		macAddress:      createdEndpoint.MacAddress,
		hnsID:           createdEndpoint.Id,
		providerAddress: ep.providerAddress, //TODO get from createdEndpoint
		hns:             hns,
	}, nil
}
func (hns hnsV2) deleteEndpoint(hnsID string) error {
	hnsendpoint, err := hcn.GetEndpointByID(hnsID)
	if err != nil {
		return err
	}
	err = hnsendpoint.Delete()
	if err == nil {
		klog.V(3).InfoS("Remote endpoint resource deleted", "hnsID", hnsID)
	}
	return err
}

func (hns hnsV2) getAllLoadBalancers() (map[loadBalancerIdentifier]*loadBalancerInfo, error) {
	lbs, err := hcn.ListLoadBalancers()
	var id loadBalancerIdentifier
	if err != nil {
		return nil, err
	}
	loadBalancers := make(map[loadBalancerIdentifier]*(loadBalancerInfo))
	for _, lb := range lbs {
		portMap := lb.PortMappings[0]
		if len(lb.FrontendVIPs) == 0 {
			// Leave VIP uninitialized
			id = loadBalancerIdentifier{protocol: uint16(portMap.Protocol), internalPort: portMap.InternalPort, externalPort: portMap.ExternalPort, endpointsCount: len(lb.HostComputeEndpoints)}
		} else {
			id = loadBalancerIdentifier{protocol: uint16(portMap.Protocol), internalPort: portMap.InternalPort, externalPort: portMap.ExternalPort, vip: lb.FrontendVIPs[0], endpointsCount: len(lb.HostComputeEndpoints)}
		}
		loadBalancers[id] = &loadBalancerInfo{
			hnsID: lb.Id,
		}
	}
	klog.V(3).InfoS("Queried load balancers", "count", len(lbs))
	return loadBalancers, nil
}

func (hns hnsV2) getLoadBalancer(endpoints []endpointsInfo, flags loadBalancerFlags, sourceVip string, vip string, protocol uint16, internalPort uint16, externalPort uint16, previousLoadBalancers map[loadBalancerIdentifier]*loadBalancerInfo) (*loadBalancerInfo, error) {
	var id loadBalancerIdentifier
	vips := []string{}
	if len(vip) > 0 {
		id = loadBalancerIdentifier{protocol: protocol, internalPort: internalPort, externalPort: externalPort, vip: vip, endpointsCount: len(endpoints)}
		vips = append(vips, vip)
	} else {
		id = loadBalancerIdentifier{protocol: protocol, internalPort: internalPort, externalPort: externalPort, endpointsCount: len(endpoints)}
	}

	if lb, found := previousLoadBalancers[id]; found {
		klog.V(1).InfoS("Found cached Hns loadbalancer policy resource", "policies", lb)
		return lb, nil
	}

	lbPortMappingFlags := hcn.LoadBalancerPortMappingFlagsNone
	if flags.isILB {
		lbPortMappingFlags |= hcn.LoadBalancerPortMappingFlagsILB
	}
	if flags.useMUX {
		lbPortMappingFlags |= hcn.LoadBalancerPortMappingFlagsUseMux
	}
	if flags.preserveDIP {
		lbPortMappingFlags |= hcn.LoadBalancerPortMappingFlagsPreserveDIP
	}
	if flags.localRoutedVIP {
		lbPortMappingFlags |= hcn.LoadBalancerPortMappingFlagsLocalRoutedVIP
	}

	lbFlags := hcn.LoadBalancerFlagsNone
	if flags.isDSR {
		lbFlags |= hcn.LoadBalancerFlagsDSR
	}

	if flags.isIPv6 {
		lbFlags |= LoadBalancerFlagsIPv6
	}

	lbDistributionType := hcn.LoadBalancerDistributionNone

	if flags.sessionAffinity {
		lbDistributionType = hcn.LoadBalancerDistributionSourceIP
	}

	loadBalancer := &hcn.HostComputeLoadBalancer{
		SourceVIP: sourceVip,
		PortMappings: []hcn.LoadBalancerPortMapping{
			{
				Protocol:         uint32(protocol),
				InternalPort:     internalPort,
				ExternalPort:     externalPort,
				DistributionType: lbDistributionType,
				Flags:            lbPortMappingFlags,
			},
		},
		FrontendVIPs: vips,
		SchemaVersion: hcn.SchemaVersion{
			Major: 2,
			Minor: 0,
		},
		Flags: lbFlags,
	}

	for _, ep := range endpoints {
		loadBalancer.HostComputeEndpoints = append(loadBalancer.HostComputeEndpoints, ep.hnsID)
	}

	lb, err := loadBalancer.Create()

	if err != nil {
		return nil, err
	}

	klog.V(1).InfoS("Created Hns loadbalancer policy resource", "loadBalancer", lb)
	lbInfo := &loadBalancerInfo{
		hnsID: lb.Id,
	}
	// Add to map of load balancers
	previousLoadBalancers[id] = lbInfo
	return lbInfo, err
}

func (hns hnsV2) deleteLoadBalancer(hnsID string) error {
	lb, err := hcn.GetLoadBalancerByID(hnsID)
	if err != nil {
		// Return silently
		return nil
	}

	err = lb.Delete()
	return err
}
