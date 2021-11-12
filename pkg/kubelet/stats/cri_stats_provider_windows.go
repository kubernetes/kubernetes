//go:build windows
// +build windows

/*
Copyright 2019 The Kubernetes Authors.

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

package stats

import (
	"time"

	"k8s.io/klog/v2"

	"github.com/Microsoft/hcsshim"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
)

// windowsNetworkStatsProvider creates an interface that allows for testing the logic without needing to create a container
type windowsNetworkStatsProvider interface {
	HNSListEndpointRequest() ([]hcsshim.HNSEndpoint, error)
	GetHNSEndpointStats(endpointName string) (*hcsshim.HNSEndpointStats, error)
}

// networkStats exposes the required functionality for hcsshim in this scenario
type networkStats struct{}

func (s networkStats) HNSListEndpointRequest() ([]hcsshim.HNSEndpoint, error) {
	return hcsshim.HNSListEndpointRequest()
}

func (s networkStats) GetHNSEndpointStats(endpointName string) (*hcsshim.HNSEndpointStats, error) {
	return hcsshim.GetHNSEndpointStats(endpointName)
}

// listContainerNetworkStats returns the network stats of all the running containers.
func (p *criStatsProvider) listContainerNetworkStats() (map[string]*statsapi.NetworkStats, error) {
	networkStatsProvider := newNetworkStatsProvider(p)

	endpoints, err := networkStatsProvider.HNSListEndpointRequest()
	if err != nil {
		klog.ErrorS(err, "Failed to fetch current HNS endpoints")
		return nil, err
	}

	networkStats := make(map[string]*statsapi.NetworkStats)
	for _, endpoint := range endpoints {
		endpointStats, err := networkStatsProvider.GetHNSEndpointStats(endpoint.Id)
		if err != nil {
			klog.V(2).InfoS("Failed to fetch statistics for endpoint, continue to get stats for other endpoints", "endpointId", endpoint.Id, "containers", endpoint.SharedContainers)
			continue
		}

		// only add the interface for each container if not already in the list
		for _, cId := range endpoint.SharedContainers {
			networkStat, found := networkStats[cId]
			if found && networkStat.Name != endpoint.Name {
				iStat := hcsStatToInterfaceStat(endpointStats, endpoint.Name)
				networkStat.Interfaces = append(networkStat.Interfaces, iStat)
				continue
			}
			networkStats[cId] = hcsStatsToNetworkStats(p.clock.Now(), endpointStats, endpoint.Name)
		}
	}

	return networkStats, nil
}

// hcsStatsToNetworkStats converts hcsshim.Statistics.Network to statsapi.NetworkStats
func hcsStatsToNetworkStats(timestamp time.Time, hcsStats *hcsshim.HNSEndpointStats, endpointName string) *statsapi.NetworkStats {
	result := &statsapi.NetworkStats{
		Time:       metav1.NewTime(timestamp),
		Interfaces: make([]statsapi.InterfaceStats, 0),
	}

	iStat := hcsStatToInterfaceStat(hcsStats, endpointName)

	// TODO: add support of multiple interfaces for getting default interface.
	result.Interfaces = append(result.Interfaces, iStat)
	result.InterfaceStats = iStat

	return result
}

func hcsStatToInterfaceStat(hcsStats *hcsshim.HNSEndpointStats, endpointName string) statsapi.InterfaceStats {
	iStat := statsapi.InterfaceStats{
		Name:    endpointName,
		RxBytes: &hcsStats.BytesReceived,
		TxBytes: &hcsStats.BytesSent,
	}
	return iStat
}

// newNetworkStatsProvider uses the real windows hcsshim if not provided otherwise if the interface is provided
// by the cristatsprovider in testing scenarios it uses that one
func newNetworkStatsProvider(p *criStatsProvider) windowsNetworkStatsProvider {
	var statsProvider windowsNetworkStatsProvider
	if p.windowsNetworkStatsProvider == nil {
		statsProvider = networkStats{}
	} else {
		statsProvider = p.windowsNetworkStatsProvider.(windowsNetworkStatsProvider)
	}
	return statsProvider
}
