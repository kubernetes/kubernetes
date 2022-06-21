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
	"os"
	"time"

	"github.com/Microsoft/hcsshim/hcn"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/proxy"

	"strings"
)

type IHCN interface {
	GetNetworkByName(networkName string) (*hcn.HostComputeNetwork, error)
	ListEndpointsOfNetwork(networkId string) ([]hcn.HostComputeEndpoint, error)
	GetEndpointByID(endpointId string) (*hcn.HostComputeEndpoint, error)
	ListEndpoints() ([]hcn.HostComputeEndpoint, error)
	GetEndpointByName(endpointName string) (*hcn.HostComputeEndpoint, error)
	ListLoadBalancers() ([]hcn.HostComputeLoadBalancer, error)
	GetLoadBalancerByID(loadBalancerId string) (*hcn.HostComputeLoadBalancer, error)
	CreateEndpoint(endpoint *hcn.HostComputeEndpoint, network *hcn.HostComputeNetwork) (*hcn.HostComputeEndpoint, error)
	CreateLoadBalancer(loadbalancer *hcn.HostComputeLoadBalancer) (*hcn.HostComputeLoadBalancer, error)
	CreateRemoteEndpoint(endpoint *hcn.HostComputeEndpoint, network *hcn.HostComputeNetwork) (*hcn.HostComputeEndpoint, error)
	DeleteLoadBalancer(loadbalancer *hcn.HostComputeLoadBalancer) error
	DeleteEndpoint(endpoint *hcn.HostComputeEndpoint) error
}

// The RealHCN structure represents the real hcn, so we can call the real hcn.methods whenever we need them.
// If you want to use hcnutils methods on the real hcn you should create it this way: hcnutils := NewHCNUtils(&RealHCN{})
// Otherwise, if you want to use it on the fakeHCN (for testing purposes), use it this way: hcnutils := NewHCNUtils(&hcntesting.FakeHCN{})
type RealHCN struct{}

type hcnutils struct {
	hcninstance IHCN
}

func NewHCNUtils(hcnImpl IHCN) *hcnutils {
	return &hcnutils{hcnImpl}
}

type IHCNUtils interface {
	getNetworkByName(name string) (*hnsNetworkInfo, error)
	getAllEndpointsByNetwork(networkName string) (map[string]*endpointsInfo, error)
	getEndpointByID(id string) (*endpointsInfo, error)
	getEndpointByIpAddress(ip string, networkName string) (*endpointsInfo, error)
	getEndpointByName(id string) (*endpointsInfo, error)
	createEndpoint(ep *endpointsInfo, networkName string) (*endpointsInfo, error)
	deleteEndpoint(hnsID string) error
	getLoadBalancer(endpoints []endpointsInfo, flags loadBalancerFlags, sourceVip string, vip string, protocol uint16, internalPort uint16, externalPort uint16, previousLoadBalancers map[loadBalancerIdentifier]*loadBalancerInfo) (*loadBalancerInfo, error)
	getAllLoadBalancers() (map[loadBalancerIdentifier]*loadBalancerInfo, error)
	deleteLoadBalancer(hnsID string) error
}

var (
	// LoadBalancerFlagsIPv6 enables IPV6.
	LoadBalancerFlagsIPv6 hcn.LoadBalancerFlags = 2
)

func (hns hcnutils) getNetworkByName(name string) (*hnsNetworkInfo, error) {
	hnsnetwork, err := hns.hcninstance.GetNetworkByName(name)
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

func (hns hcnutils) getAllEndpointsByNetwork(networkName string) (map[string]*(endpointsInfo), error) {
	hcnnetwork, err := hns.hcninstance.GetNetworkByName(networkName)
	if err != nil {
		klog.ErrorS(err, "failed to get HNS network by name", "name", networkName)
		return nil, err
	}
	endpoints, err := hns.hcninstance.ListEndpointsOfNetwork(hcnnetwork.Id)
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

func (hns hcnutils) getEndpointByID(id string) (*endpointsInfo, error) {
	hnsendpoint, err := hns.hcninstance.GetEndpointByID(id)
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
func (hns hcnutils) getEndpointByIpAddress(ip string, networkName string) (*endpointsInfo, error) {
	hnsnetwork, err := hns.hcninstance.GetNetworkByName(networkName)
	if err != nil {
		klog.ErrorS(err, "Error getting network by name")
		return nil, err
	}

	endpoints, err := hns.hcninstance.ListEndpoints()
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
func (hns hcnutils) getEndpointByName(name string) (*endpointsInfo, error) {
	hnsendpoint, err := hns.hcninstance.GetEndpointByName(name)
	if err != nil {
		return nil, err
	}
	return &endpointsInfo{ //TODO: fill out PA
		ip:         hnsendpoint.IpConfigurations[0].IpAddress,
		isLocal:    uint32(hnsendpoint.Flags&hcn.EndpointFlagsRemoteEndpoint) == 0, //TODO: Change isLocal to isRemote
		macAddress: hnsendpoint.MacAddress,
		hnsID:      hnsendpoint.Id,
		hns:        hns,
		name:       hnsendpoint.Name,
	}, nil
}
func (hns hcnutils) createEndpoint(ep *endpointsInfo, networkName string) (*endpointsInfo, error) {
	hnsNetwork, err := hns.hcninstance.GetNetworkByName(networkName)
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
		createdEndpoint, err = hns.hcninstance.CreateRemoteEndpoint(hnsEndpoint, hnsNetwork)
		if err != nil {
			return nil, err
		}
	} else {
		createdEndpoint, err = hns.hcninstance.CreateEndpoint(hnsEndpoint, hnsNetwork)
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
func (hns hcnutils) deleteEndpoint(hnsID string) error {
	hnsendpoint, err := hns.hcninstance.GetEndpointByID(hnsID)
	if err != nil {
		return err
	}
	err = hns.hcninstance.DeleteEndpoint(hnsendpoint)
	if err == nil {
		klog.V(3).InfoS("Remote endpoint resource deleted", "hnsID", hnsID)
	}
	return err
}

func (hns hcnutils) getAllLoadBalancers() (map[loadBalancerIdentifier]*loadBalancerInfo, error) {
	lbs, err := hns.hcninstance.ListLoadBalancers()
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

func (hns hcnutils) getLoadBalancer(endpoints []endpointsInfo, flags loadBalancerFlags, sourceVip string, vip string, protocol uint16, internalPort uint16, externalPort uint16, previousLoadBalancers map[loadBalancerIdentifier]*loadBalancerInfo) (*loadBalancerInfo, error) {
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

	lb, err := hns.hcninstance.CreateLoadBalancer(loadBalancer)

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

func (hns hcnutils) deleteLoadBalancer(hnsID string) error {
	lb, err := hns.hcninstance.GetLoadBalancerByID(hnsID)
	if err != nil {
		// Return silently
		return nil
	}

	err = hns.hcninstance.DeleteLoadBalancer(lb)
	return err
}

func (h *RealHCN) GetNetworkByName(networkName string) (*hcn.HostComputeNetwork, error) {
	return hcn.GetNetworkByName(networkName)
}

func (h *RealHCN) ListEndpointsOfNetwork(networkId string) ([]hcn.HostComputeEndpoint, error) {
	return hcn.ListEndpointsOfNetwork(networkId)
}

func (h *RealHCN) GetEndpointByID(endpointId string) (*hcn.HostComputeEndpoint, error) {
	return hcn.GetEndpointByID(endpointId)
}

func (h *RealHCN) ListEndpoints() ([]hcn.HostComputeEndpoint, error) {
	return hcn.ListEndpoints()
}

func (h *RealHCN) GetEndpointByName(endpointName string) (*hcn.HostComputeEndpoint, error) {
	return hcn.GetEndpointByName(endpointName)
}

func (h *RealHCN) ListLoadBalancers() ([]hcn.HostComputeLoadBalancer, error) {
	return hcn.ListLoadBalancers()
}

func (h *RealHCN) GetLoadBalancerByID(loadBalancerId string) (*hcn.HostComputeLoadBalancer, error) {
	return hcn.GetLoadBalancerByID(loadBalancerId)
}

func (h *RealHCN) CreateEndpoint(endpoint *hcn.HostComputeEndpoint, network *hcn.HostComputeNetwork) (*hcn.HostComputeEndpoint, error) {
	return network.CreateEndpoint(endpoint)
}

func (h *RealHCN) CreateRemoteEndpoint(endpoint *hcn.HostComputeEndpoint, network *hcn.HostComputeNetwork) (*hcn.HostComputeEndpoint, error) {
	return network.CreateRemoteEndpoint(endpoint)
}

func (h *RealHCN) CreateLoadBalancer(loadbalancer *hcn.HostComputeLoadBalancer) (*hcn.HostComputeLoadBalancer, error) {
	return loadbalancer.Create()
}

func (h *RealHCN) DeleteLoadBalancer(loadbalancer *hcn.HostComputeLoadBalancer) error {
	return loadbalancer.Delete()
}

func (h *RealHCN) DeleteEndpoint(endpoint *hcn.HostComputeEndpoint) error {
	return endpoint.Delete()
}

// Here are some helper methods for proxier.go

func getNetworkName(hnsNetworkName string) (string, error) {
	if len(hnsNetworkName) == 0 {
		klog.V(3).InfoS("Flag --network-name not set, checking environment variable")
		hnsNetworkName = os.Getenv("KUBE_NETWORK")
		if len(hnsNetworkName) == 0 {
			return "", fmt.Errorf("Environment variable KUBE_NETWORK and network-flag not initialized")
		}
	}
	return hnsNetworkName, nil
}

func getNetworkInfo(hns IHCNUtils, hnsNetworkName string) (*hnsNetworkInfo, error) {
	hnsNetworkInfo, err := hns.getNetworkByName(hnsNetworkName)
	for err != nil {
		klog.ErrorS(err, "Unable to find HNS Network specified, please check network name and CNI deployment", "hnsNetworkName", hnsNetworkName)
		time.Sleep(1 * time.Second)
		hnsNetworkInfo, err = hns.getNetworkByName(hnsNetworkName)
	}
	return hnsNetworkInfo, err
}

func isOverlay(hnsNetworkInfo *hnsNetworkInfo) bool {
	return strings.EqualFold(hnsNetworkInfo.networkType, NETWORK_TYPE_OVERLAY)
}

func (svcInfo *serviceInfo) cleanupAllPolicies(endpoints []proxy.Endpoint) {
	klog.V(3).InfoS("Service cleanup", "serviceInfo", svcInfo)
	// Skip the svcInfo.policyApplied check to remove all the policies
	svcInfo.deleteAllHnsLoadBalancerPolicy()
	// Cleanup Endpoints references
	for _, ep := range endpoints {
		epInfo, ok := ep.(*endpointsInfo)
		if ok {
			epInfo.Cleanup()
		}
	}
	if svcInfo.remoteEndpoint != nil {
		svcInfo.remoteEndpoint.Cleanup()
	}

	svcInfo.policyApplied = false
}

func (svcInfo *serviceInfo) deleteAllHnsLoadBalancerPolicy() {
	// Remove the Hns Policy corresponding to this service
	hns := svcInfo.hns
	hns.deleteLoadBalancer(svcInfo.hnsID)
	svcInfo.hnsID = ""

	hns.deleteLoadBalancer(svcInfo.nodePorthnsID)
	svcInfo.nodePorthnsID = ""

	for _, externalIP := range svcInfo.externalIPs {
		hns.deleteLoadBalancer(externalIP.hnsID)
		externalIP.hnsID = ""
	}
	for _, lbIngressIP := range svcInfo.loadBalancerIngressIPs {
		hns.deleteLoadBalancer(lbIngressIP.hnsID)
		lbIngressIP.hnsID = ""
		if lbIngressIP.healthCheckHnsID != "" {
			hns.deleteLoadBalancer(lbIngressIP.healthCheckHnsID)
			lbIngressIP.healthCheckHnsID = ""
		}
	}
}

func deleteAllHnsLoadBalancerPolicy() {
	lbalancers, err := hcn.ListLoadBalancers()
	if err != nil {
		return
	}
	for _, lbalancer := range lbalancers {
		klog.V(3).InfoS("Remove policy", "policies", lbalancer)
		err = lbalancer.Delete()
		if err != nil {
			klog.ErrorS(err, "Failed to delete policy list")
		}
	}
}

func getHnsNetworkInfo(hnsNetworkName string) (*hnsNetworkInfo, error) {
	hnsnetwork, err := hcn.GetNetworkByName(hnsNetworkName)
	if err != nil {
		klog.ErrorS(err, "Failed to get HNS Network by name")
		return nil, err
	}

	return &hnsNetworkInfo{
		id:          hnsnetwork.Id,
		name:        hnsnetwork.Name,
		networkType: string(hnsnetwork.Type),
	}, nil
}
