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

package stats

import (
	"time"

	"github.com/Microsoft/hcsshim"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog"
	statsapi "k8s.io/kubernetes/pkg/kubelet/apis/stats/v1alpha1"
)

// listContainerNetworkStats returns the network stats of all the running containers.
func (p *criStatsProvider) listContainerNetworkStats() (map[string]*statsapi.NetworkStats, error) {
	containers, err := hcsshim.GetContainers(hcsshim.ComputeSystemQuery{
		Types: []string{"Container"},
	})
	if err != nil {
		return nil, err
	}

	stats := make(map[string]*statsapi.NetworkStats)
	for _, c := range containers {
		container, err := hcsshim.OpenContainer(c.ID)
		if err != nil {
			klog.Warningf("Failed to open container %q with error '%v', continue to get stats for other containers", c.ID, err)
			continue
		}

		cstats, err := container.Statistics()
		if err != nil {
			klog.Warningf("Failed to get statistics for container %q with error '%v', continue to get stats for other containers", c.ID, err)
			continue
		}

		if len(cstats.Network) > 0 {
			stats[c.ID] = hcsStatsToNetworkStats(cstats.Timestamp, cstats.Network)
		}
	}

	return stats, nil
}

// hcsStatsToNetworkStats converts hcsshim.Statistics.Network to statsapi.NetworkStats
func hcsStatsToNetworkStats(timestamp time.Time, hcsStats []hcsshim.NetworkStats) *statsapi.NetworkStats {
	result := &statsapi.NetworkStats{
		Time:       metav1.NewTime(timestamp),
		Interfaces: make([]statsapi.InterfaceStats, 0),
	}

	for _, stat := range hcsStats {
		iStat := hcsStatsToInterfaceStats(stat)
		if iStat != nil {
			result.Interfaces = append(result.Interfaces, *iStat)
		}
	}

	// TODO(feiskyer): add support of multiple interfaces for getting default interface.
	if len(result.Interfaces) > 0 {
		result.InterfaceStats = result.Interfaces[0]
	}

	return result
}

// hcsStatsToInterfaceStats converts hcsshim.NetworkStats to statsapi.InterfaceStats.
func hcsStatsToInterfaceStats(stat hcsshim.NetworkStats) *statsapi.InterfaceStats {
	return &statsapi.InterfaceStats{
		Name:    stat.EndpointId,
		RxBytes: &stat.BytesReceived,
		TxBytes: &stat.BytesSent,
	}
}
