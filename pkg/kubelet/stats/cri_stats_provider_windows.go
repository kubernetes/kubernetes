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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/runtime/v1alpha2"
	statsapi "k8s.io/kubernetes/pkg/kubelet/apis/stats/v1alpha1"
)

// listPodSandboxStats returns the stats of all the running pod sandboxes.
func (p *criStatsProvider) listPodSandboxStats() (map[string]*statsapi.PodStats, error) {
	stats, err := p.runtimeService.ListPodSandboxStats(&runtimeapi.PodSandboxStatsFilter{})
	if err != nil {
		return nil, err
	}

	result := make(map[string]*statsapi.PodStats)
	for i := range stats {
		stat := stats[i]
		podStat := &statsapi.PodStats{
			PodRef: statsapi.PodReference{
				Name:      stat.Metadata.Name,
				UID:       stat.Metadata.Uid,
				Namespace: stat.Metadata.Namespace,
			},
			Network: criNetworkUsageToNetworkStats(stat.Network),
		}
		result[stat.Id] = podStat
	}

	return result, nil
}

// criStatsToNetworkInterfaceStats converts runtimeapi.InterfaceStats to statsapi.InterfaceStats.
func criStatsToNetworkInterfaceStats(stats *runtimeapi.InterfaceStats) *statsapi.InterfaceStats {
	if stats == nil {
		return nil
	}
	iStats := &statsapi.InterfaceStats{
		Name: stats.Name,
	}
	if stats.RxBytes != nil {
		iStats.RxBytes = &stats.RxBytes.Value
	}
	if stats.RxErrors != nil {
		iStats.RxErrors = &stats.RxErrors.Value
	}
	if stats.TxBytes != nil {
		iStats.TxBytes = &stats.TxBytes.Value
	}
	if stats.TxErrors != nil {
		iStats.TxErrors = &stats.TxErrors.Value
	}
	return iStats
}

// criNetworkUsageToNetworkStats returns the statsapi.NetworkStats converted from
// the network usage from CRI.
func criNetworkUsageToNetworkStats(stats *runtimeapi.NetworkUsage) *statsapi.NetworkStats {
	if stats == nil {
		return nil
	}
	iStats := statsapi.NetworkStats{
		Time: metav1.NewTime(time.Unix(0, stats.Timestamp)),
	}
	defaultStats := criStatsToNetworkInterfaceStats(stats.Default)
	if defaultStats != nil {
		iStats.InterfaceStats = *defaultStats
	}
	for i := range stats.Interfaces {
		iStat := criStatsToNetworkInterfaceStats(stats.Interfaces[i])
		if iStat != nil {
			iStats.Interfaces = append(iStats.Interfaces, *iStat)
		}
	}
	return &iStats
}
