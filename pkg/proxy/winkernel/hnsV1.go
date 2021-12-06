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
	"strings"

	"github.com/Microsoft/hcsshim"
	"k8s.io/klog/v2"
	netutils "k8s.io/utils/net"
)

type HostNetworkService interface {
	getNetworkByName(name string) (*hnsNetworkInfo, error)
	getEndpointByID(id string) (*endpointsInfo, error)
	getEndpointByIpAddress(ip string, networkName string) (*endpointsInfo, error)
	createEndpoint(ep *endpointsInfo, networkName string) (*endpointsInfo, error)
	deleteEndpoint(hnsID string) error
	getLoadBalancer(endpoints []endpointsInfo, flags loadBalancerFlags, sourceVip string, vip string, protocol uint16, internalPort uint16, externalPort uint16) (*loadBalancerInfo, error)
	deleteLoadBalancer(hnsID string) error
}

// V1 HNS API
type hnsV1 struct{}

func (hns hnsV1) getNetworkByName(name string) (*hnsNetworkInfo, error) {
	hnsnetwork, err := hcsshim.GetHNSNetworkByName(name)
	if err != nil {
		klog.ErrorS(err, "failed to get HNS network by name", "name", name)
		return nil, err
	}

	return &hnsNetworkInfo{
		id:          hnsnetwork.Id,
		name:        hnsnetwork.Name,
		networkType: hnsnetwork.Type,
	}, nil
}

func (hns hnsV1) getEndpointByID(id string) (*endpointsInfo, error) {
	hnsendpoint, err := hcsshim.GetHNSEndpointByID(id)
	if err != nil {
		klog.ErrorS(err, "failed to get HNS endpoint by id", "id", id)
		return nil, err
	}
	return &endpointsInfo{
		ip:         hnsendpoint.IPAddress.String(),
		isLocal:    !hnsendpoint.IsRemoteEndpoint, // TODO: Change isLocal to isRemote
		macAddress: hnsendpoint.MacAddress,
		hnsID:      hnsendpoint.Id,
		hns:        hns,

		// only ready and not terminating endpoints were added to HNS
		ready:       true,
		serving:     true,
		terminating: false,
	}, nil
}

func (hns hnsV1) getEndpointByIpAddress(ip string, networkName string) (*endpointsInfo, error) {
	hnsnetwork, err := hcsshim.GetHNSNetworkByName(networkName)
	if err != nil {
		klog.ErrorS(err, "failed to get HNS network by name", "name", networkName)
		return nil, err
	}

	endpoints, err := hcsshim.HNSListEndpointRequest()
	if err != nil {
		return nil, fmt.Errorf("failed to list endpoints: %w", err)
	}
	for _, endpoint := range endpoints {
		equal := false
		if endpoint.IPAddress != nil {
			equal = endpoint.IPAddress.String() == ip
		}
		if equal && strings.EqualFold(endpoint.VirtualNetwork, hnsnetwork.Id) {
			return &endpointsInfo{
				ip:         endpoint.IPAddress.String(),
				isLocal:    !endpoint.IsRemoteEndpoint,
				macAddress: endpoint.MacAddress,
				hnsID:      endpoint.Id,
				hns:        hns,

				// only ready and not terminating endpoints were added to HNS
				ready:       true,
				serving:     true,
				terminating: false,
			}, nil
		}
	}

	return nil, fmt.Errorf("Endpoint %v not found on network %s", ip, networkName)
}

func (hns hnsV1) createEndpoint(ep *endpointsInfo, networkName string) (*endpointsInfo, error) {
	hnsNetwork, err := hcsshim.GetHNSNetworkByName(networkName)
	if err != nil {
		return nil, err
	}
	hnsEndpoint := &hcsshim.HNSEndpoint{
		MacAddress: ep.macAddress,
		IPAddress:  netutils.ParseIPSloppy(ep.ip),
	}

	var createdEndpoint *hcsshim.HNSEndpoint
	if !ep.isLocal {
		if len(ep.providerAddress) != 0 {
			paPolicy := hcsshim.PaPolicy{
				Type: hcsshim.PA,
				PA:   ep.providerAddress,
			}
			paPolicyJson, err := json.Marshal(paPolicy)
			if err != nil {
				return nil, err
			}
			hnsEndpoint.Policies = append(hnsEndpoint.Policies, paPolicyJson)
		}
		createdEndpoint, err = hnsNetwork.CreateRemoteEndpoint(hnsEndpoint)
		if err != nil {
			return nil, err
		}

	} else {
		createdEndpoint, err = hnsNetwork.CreateEndpoint(hnsEndpoint)
		if err != nil {
			return nil, fmt.Errorf("local endpoint creation failed: %w", err)
		}
	}
	return &endpointsInfo{
		ip:              createdEndpoint.IPAddress.String(),
		isLocal:         createdEndpoint.IsRemoteEndpoint,
		macAddress:      createdEndpoint.MacAddress,
		hnsID:           createdEndpoint.Id,
		providerAddress: ep.providerAddress, // TODO get from createdEndpoint
		hns:             hns,

		ready:       ep.ready,
		serving:     ep.serving,
		terminating: ep.terminating,
	}, nil
}

func (hns hnsV1) deleteEndpoint(hnsID string) error {
	hnsendpoint, err := hcsshim.GetHNSEndpointByID(hnsID)
	if err != nil {
		return err
	}
	_, err = hnsendpoint.Delete()
	if err == nil {
		klog.V(3).InfoS("Remote endpoint resource deleted id", "id", hnsID)
	}
	return err
}

func (hns hnsV1) getLoadBalancer(endpoints []endpointsInfo, flags loadBalancerFlags, sourceVip string, vip string, protocol uint16, internalPort uint16, externalPort uint16) (*loadBalancerInfo, error) {
	plists, err := hcsshim.HNSListPolicyListRequest()
	if err != nil {
		return nil, err
	}

	if flags.isDSR {
		klog.V(3).InfoS("DSR is not supported in V1. Using non DSR instead")
	}

	for _, plist := range plists {
		if len(plist.EndpointReferences) != len(endpoints) {
			continue
		}
		// Validate if input meets any of the policy lists
		elbPolicy := hcsshim.ELBPolicy{}
		if err = json.Unmarshal(plist.Policies[0], &elbPolicy); err != nil {
			continue
		}
		if elbPolicy.Protocol == protocol && elbPolicy.InternalPort == internalPort && elbPolicy.ExternalPort == externalPort && elbPolicy.ILB == flags.isILB {
			if len(vip) > 0 {
				if len(elbPolicy.VIPs) == 0 || elbPolicy.VIPs[0] != vip {
					continue
				}
			} else if len(elbPolicy.VIPs) != 0 {
				continue
			}
			klog.V(1).InfoS("Found existing Hns loadbalancer policy resource", "policies", plist)
			return &loadBalancerInfo{
				hnsID: plist.ID,
			}, nil
		}
	}

	var hnsEndpoints []hcsshim.HNSEndpoint
	for _, ep := range endpoints {
		endpoint, err := hcsshim.GetHNSEndpointByID(ep.hnsID)
		if err != nil {
			return nil, err
		}
		hnsEndpoints = append(hnsEndpoints, *endpoint)
	}
	lb, err := hcsshim.AddLoadBalancer(
		hnsEndpoints,
		flags.isILB,
		sourceVip,
		vip,
		protocol,
		internalPort,
		externalPort,
	)

	if err == nil {
		klog.V(1).InfoS("Hns loadbalancer policy resource", "policies", lb)
	} else {
		return nil, err
	}
	return &loadBalancerInfo{
		hnsID: lb.ID,
	}, err
}

func (hns hnsV1) deleteLoadBalancer(hnsID string) error {
	if len(hnsID) == 0 {
		// Return silently
		return nil
	}

	// Cleanup HNS policies
	hnsloadBalancer, err := hcsshim.GetPolicyListByID(hnsID)
	if err != nil {
		return err
	}
	klog.V(2).InfoS("Removing Policy", "policies", hnsloadBalancer)

	_, err = hnsloadBalancer.Delete()
	return err
}
