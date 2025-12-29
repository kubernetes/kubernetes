//go:build windows

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
	"crypto/sha1"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/Microsoft/hnslib/hcn"

	"k8s.io/klog/v2"
)

type HostNetworkService interface {
	getNetworkByName(name string) (*hnsNetworkInfo, error)
	getAllEndpointsByNetwork(networkName string) (map[string]*endpointInfo, error)
	getEndpointByID(id string) (*endpointInfo, error)
	getEndpointByIpAddress(ip string, networkName string) (*endpointInfo, error)
	getEndpointByName(id string) (*endpointInfo, error)
	createEndpoint(ep *endpointInfo, networkName string) (*endpointInfo, error)
	deleteEndpoint(hnsID string) error
	getLoadBalancer(endpoints []endpointInfo, flags loadBalancerFlags, sourceVip string, vip string, protocol uint16, internalPort uint16, externalPort uint16, previousLoadBalancers map[loadBalancerIdentifier]*loadBalancerInfo) (*loadBalancerInfo, error)
	getAllLoadBalancers() (map[loadBalancerIdentifier]*loadBalancerInfo, error)
	updateLoadBalancer(hnsID string, sourceVip, vip string, endpoints []endpointInfo, flags loadBalancerFlags, protocol, internalPort, externalPort uint16, previousLoadBalancers map[loadBalancerIdentifier]*loadBalancerInfo) (*loadBalancerInfo, error)
	deleteLoadBalancer(hnsID string) error
}

type hns struct {
	hcn HcnService
}

var (
	// LoadBalancerFlagsIPv6 enables IPV6.
	LoadBalancerFlagsIPv6 hcn.LoadBalancerFlags = 2
	// LoadBalancerPortMappingFlagsVipExternalIP enables VipExternalIP.
	LoadBalancerPortMappingFlagsVipExternalIP hcn.LoadBalancerPortMappingFlags = 16
)

func getLoadBalancerPolicyFlags(flags loadBalancerFlags) (lbPortMappingFlags hcn.LoadBalancerPortMappingFlags, lbFlags hcn.LoadBalancerFlags) {
	lbPortMappingFlags = hcn.LoadBalancerPortMappingFlagsNone
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
	if flags.isVipExternalIP {
		lbPortMappingFlags |= LoadBalancerPortMappingFlagsVipExternalIP
	}
	lbFlags = hcn.LoadBalancerFlagsNone
	if flags.isDSR {
		lbFlags |= hcn.LoadBalancerFlagsDSR
	}
	if flags.isIPv6 {
		lbFlags |= LoadBalancerFlagsIPv6
	}
	return
}

func (hns hns) getNetworkByName(name string) (*hnsNetworkInfo, error) {
	hnsnetwork, err := hns.hcn.GetNetworkByName(name)
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

func (hns hns) getAllEndpointsByNetwork(networkName string) (map[string]*(endpointInfo), error) {
	hcnnetwork, err := hns.hcn.GetNetworkByName(networkName)
	if err != nil {
		klog.ErrorS(err, "failed to get HNS network by name", "name", networkName)
		return nil, err
	}
	endpoints, err := hns.hcn.ListEndpointsOfNetwork(hcnnetwork.Id)
	if err != nil {
		return nil, fmt.Errorf("failed to list endpoints: %w", err)
	}
	endpointInfos := make(map[string]*(endpointInfo))
	for _, ep := range endpoints {

		if len(ep.IpConfigurations) == 0 {
			klog.V(3).InfoS("No IpConfigurations found in endpoint info of queried endpoints", "endpoint", ep)
			continue
		}

		for index, ipConfig := range ep.IpConfigurations {

			if index > 1 {
				// Expecting only ipv4 and ipv6 ipaddresses
				// This is highly unlikely to happen, but if it does, we should log a warning
				// and break out of the loop
				klog.Warning("Endpoint ipconfiguration holds more than 2 IP addresses.", "hnsID", ep.Id, "IP", ipConfig.IpAddress, "ipConfigCount", len(ep.IpConfigurations))
				break
			}

			isLocal := uint32(ep.Flags&hcn.EndpointFlagsRemoteEndpoint) == 0

			if existingEp, ok := endpointInfos[ipConfig.IpAddress]; ok && isLocal {
				// If the endpoint is already part of the queried endpoints map and is local,
				// then we should not add it again to the map
				// This is to avoid overwriting the remote endpoint info with a local endpoint.
				klog.V(3).InfoS("Endpoint already exists in queried endpoints map; skipping.", "newLocalEndpoint", ep, "ipConfig", ipConfig, "existingEndpoint", existingEp)
				continue
			}

			// Add to map with key endpoint ID or IP address
			// Storing this is expensive in terms of memory, however there is a bug in Windows Server 2019 and 2022 that can cause two endpoints (local and remote) to be created with the same IP address.
			// TODO: Store by IP only and remove any lookups by endpoint ID.
			epInfo := &endpointInfo{
				ip:         ipConfig.IpAddress,
				isLocal:    isLocal,
				macAddress: ep.MacAddress,
				hnsID:      ep.Id,
				hns:        hns,
				// only ready and not terminating endpoints were added to HNS
				ready:       true,
				serving:     true,
				terminating: false,
			}
			endpointInfos[ep.Id] = epInfo
			endpointInfos[ipConfig.IpAddress] = epInfo
		}
	}
	klog.V(3).InfoS("Queried endpoints from network", "network", networkName, "count", len(endpointInfos))
	klog.V(5).InfoS("Queried endpoints details", "network", networkName, "endpointInfos", endpointInfos)
	return endpointInfos, nil
}

func (hns hns) getEndpointByID(id string) (*endpointInfo, error) {
	hnsendpoint, err := hns.hcn.GetEndpointByID(id)
	if err != nil {
		return nil, err
	}
	return &endpointInfo{ //TODO: fill out PA
		ip:         hnsendpoint.IpConfigurations[0].IpAddress,
		isLocal:    uint32(hnsendpoint.Flags&hcn.EndpointFlagsRemoteEndpoint) == 0, //TODO: Change isLocal to isRemote
		macAddress: hnsendpoint.MacAddress,
		hnsID:      hnsendpoint.Id,
		hns:        hns,
	}, nil
}
func (hns hns) getEndpointByIpAddress(ip string, networkName string) (*endpointInfo, error) {
	hnsnetwork, err := hns.hcn.GetNetworkByName(networkName)
	if err != nil {
		klog.ErrorS(err, "Error getting network by name")
		return nil, err
	}

	endpoints, err := hns.hcn.ListEndpoints()
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
			return &endpointInfo{
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
func (hns hns) getEndpointByName(name string) (*endpointInfo, error) {
	hnsendpoint, err := hns.hcn.GetEndpointByName(name)
	if err != nil {
		return nil, err
	}
	return &endpointInfo{ //TODO: fill out PA
		ip:         hnsendpoint.IpConfigurations[0].IpAddress,
		isLocal:    uint32(hnsendpoint.Flags&hcn.EndpointFlagsRemoteEndpoint) == 0, //TODO: Change isLocal to isRemote
		macAddress: hnsendpoint.MacAddress,
		hnsID:      hnsendpoint.Id,
		hns:        hns,
	}, nil
}
func (hns hns) createEndpoint(ep *endpointInfo, networkName string) (*endpointInfo, error) {
	hnsNetwork, err := hns.hcn.GetNetworkByName(networkName)
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
		createdEndpoint, err = hns.hcn.CreateRemoteEndpoint(hnsNetwork, hnsEndpoint)
		if err != nil {
			return nil, err
		}
	} else {
		createdEndpoint, err = hns.hcn.CreateEndpoint(hnsNetwork, hnsEndpoint)
		if err != nil {
			return nil, err
		}
	}
	return &endpointInfo{
		ip:              createdEndpoint.IpConfigurations[0].IpAddress,
		isLocal:         uint32(createdEndpoint.Flags&hcn.EndpointFlagsRemoteEndpoint) == 0,
		macAddress:      createdEndpoint.MacAddress,
		hnsID:           createdEndpoint.Id,
		providerAddress: ep.providerAddress, //TODO get from createdEndpoint
		hns:             hns,
	}, nil
}
func (hns hns) deleteEndpoint(hnsID string) error {
	hnsendpoint, err := hns.hcn.GetEndpointByID(hnsID)
	if err != nil {
		return err
	}
	err = hns.hcn.DeleteEndpoint(hnsendpoint)
	if err == nil {
		klog.V(3).InfoS("Remote endpoint resource deleted", "hnsID", hnsID)
	}
	return err
}

// findLoadBalancerID will construct a id from the provided loadbalancer fields
func findLoadBalancerID(endpoints []endpointInfo, vip string, protocol, internalPort, externalPort uint16) (loadBalancerIdentifier, error) {
	// Compute hash from backends (endpoint IDs)
	hash, err := hashEndpoints(endpoints)
	if err != nil {
		klog.V(2).ErrorS(err, "Error hashing endpoints", "endpoints", endpoints)
		return loadBalancerIdentifier{}, err
	}
	if len(vip) > 0 {
		return loadBalancerIdentifier{protocol: protocol, internalPort: internalPort, externalPort: externalPort, vip: vip, endpointsHash: hash}, nil
	}
	return loadBalancerIdentifier{protocol: protocol, internalPort: internalPort, externalPort: externalPort, endpointsHash: hash}, nil
}

func (hns hns) getAllLoadBalancers() (map[loadBalancerIdentifier]*loadBalancerInfo, error) {
	lbs, err := hns.hcn.ListLoadBalancers()
	var id loadBalancerIdentifier
	if err != nil {
		return nil, err
	}
	loadBalancers := make(map[loadBalancerIdentifier]*(loadBalancerInfo))
	for _, lb := range lbs {
		portMap := lb.PortMappings[0]
		// Compute hash from backends (endpoint IDs)
		hash, err := hashEndpoints(lb.HostComputeEndpoints)
		if err != nil {
			klog.V(2).ErrorS(err, "Error hashing endpoints", "policy", lb)
			return nil, err
		}
		if len(lb.FrontendVIPs) == 0 {
			// Leave VIP uninitialized
			id = loadBalancerIdentifier{protocol: uint16(portMap.Protocol), internalPort: portMap.InternalPort, externalPort: portMap.ExternalPort, endpointsHash: hash}
		} else {
			id = loadBalancerIdentifier{protocol: uint16(portMap.Protocol), internalPort: portMap.InternalPort, externalPort: portMap.ExternalPort, vip: lb.FrontendVIPs[0], endpointsHash: hash}
		}
		loadBalancers[id] = &loadBalancerInfo{
			hnsID: lb.Id,
		}
	}
	klog.V(3).InfoS("Queried load balancers", "count", len(lbs))
	return loadBalancers, nil
}

func (hns hns) getLoadBalancer(endpoints []endpointInfo, flags loadBalancerFlags, sourceVip string, vip string, protocol uint16, internalPort uint16, externalPort uint16, previousLoadBalancers map[loadBalancerIdentifier]*loadBalancerInfo) (*loadBalancerInfo, error) {
	var id loadBalancerIdentifier
	vips := []string{}
	id, lbIdErr := findLoadBalancerID(
		endpoints,
		vip,
		protocol,
		internalPort,
		externalPort,
	)

	if lbIdErr != nil {
		klog.V(2).ErrorS(lbIdErr, "Error hashing endpoints", "endpoints", endpoints)
		return nil, lbIdErr
	}

	if len(vip) > 0 {
		vips = append(vips, vip)
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
	if flags.isVipExternalIP {
		lbPortMappingFlags |= LoadBalancerPortMappingFlagsVipExternalIP
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

	lb, err := hns.hcn.CreateLoadBalancer(loadBalancer)

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

func (hns hns) updateLoadBalancer(hnsID string,
	sourceVip,
	vip string,
	endpoints []endpointInfo,
	flags loadBalancerFlags,
	protocol,
	internalPort,
	externalPort uint16,
	previousLoadBalancers map[loadBalancerIdentifier]*loadBalancerInfo) (*loadBalancerInfo, error) {
	klog.V(3).InfoS("Updating existing loadbalancer called", "hnsLbID", hnsID, "endpointCount", len(endpoints), "vip", vip, "sourceVip", sourceVip, "internalPort", internalPort, "externalPort", externalPort)

	var id loadBalancerIdentifier
	vips := []string{}
	// Compute hash from backends (endpoint IDs)
	hash, err := hashEndpoints(endpoints)
	if err != nil {
		klog.V(2).ErrorS(err, "Error hashing endpoints", "endpoints", endpoints)
		return nil, err
	}
	if len(vip) > 0 {
		id = loadBalancerIdentifier{protocol: protocol, internalPort: internalPort, externalPort: externalPort, vip: vip, endpointsHash: hash}
		vips = append(vips, vip)
	} else {
		id = loadBalancerIdentifier{protocol: protocol, internalPort: internalPort, externalPort: externalPort, endpointsHash: hash}
	}

	if lb, found := previousLoadBalancers[id]; found {
		klog.V(1).InfoS("Found cached Hns loadbalancer policy resource", "policies", lb)
		return lb, nil
	}

	lbPortMappingFlags, lbFlags := getLoadBalancerPolicyFlags(flags)

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

	lb, err := hns.hcn.UpdateLoadBalancer(loadBalancer, hnsID)

	if err != nil {
		klog.V(2).ErrorS(err, "Error updating existing loadbalancer", "hnsLbID", hnsID, "error", err, "endpoints", endpoints)
		return nil, err
	}

	klog.V(1).InfoS("Update loadbalancer is successful", "loadBalancer", lb)
	lbInfo := &loadBalancerInfo{
		hnsID: lb.Id,
	}
	// Add to map of load balancers
	previousLoadBalancers[id] = lbInfo
	return lbInfo, err
}

func (hns hns) deleteLoadBalancer(hnsID string) error {
	lb, err := hns.hcn.GetLoadBalancerByID(hnsID)
	if err != nil {
		// Return silently
		return nil
	}

	err = hns.hcn.DeleteLoadBalancer(lb)
	if err != nil {
		// There is a bug in Windows Server 2019, that can cause the delete call to fail sometimes. We retry one more time.
		// TODO: The logic in syncProxyRules  should be rewritten in the future to better stage and handle a call like this failing using the policyApplied fields.
		klog.V(1).ErrorS(err, "Error deleting Hns loadbalancer policy resource. Attempting one more time...", "loadBalancer", lb)
		return hns.hcn.DeleteLoadBalancer(lb)
	}
	return err
}

// Calculates a hash from the given endpoint IDs.
func hashEndpoints[T string | endpointInfo](endpoints []T) (hash [20]byte, err error) {
	var id string
	// Recover in case something goes wrong. Return error and null byte array.
	defer func() {
		if r := recover(); r != nil {
			err = r.(error)
			hash = [20]byte{}
		}
	}()

	// Iterate over endpoints, compute hash
	for _, ep := range endpoints {
		switch x := any(ep).(type) {
		case endpointInfo:
			id = strings.ToUpper(x.hnsID)
		case string:
			id = strings.ToUpper(x)
		}
		if len(id) > 0 {
			// We XOR the hashes of endpoints, since they are an unordered set.
			// This can cause collisions, but is sufficient since we are using other keys to identify the load balancer.
			hash = xor(hash, sha1.Sum(([]byte(id))))
		}
	}
	return
}

func xor(b1 [20]byte, b2 [20]byte) (xorbytes [20]byte) {
	for i := 0; i < 20; i++ {
		xorbytes[i] = b1[i] ^ b2[i]
	}
	return xorbytes
}
