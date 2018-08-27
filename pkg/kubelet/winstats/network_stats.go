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

package winstats

import (
	"encoding/json"
	"os/exec"

	cadvisorapi "github.com/google/cadvisor/info/v1"
)

// netAdapterStat represents network statistics for an adapter.
type netAdapterStat struct {
	Name           string `json:"Name,omitempty"`
	ReceivedBytes  uint64 `json:"ReceivedBytes,omitempty"`
	ReceivedErrors uint64 `json:"ReceivedPacketErrors,omitempty"`
	SentBytes      uint64 `json:"SentBytes,omitempty"`
	SentErrors     uint64 `json:"OutboundPacketErrors,omitempty"`
}

// toCadvisorInterfaceStats converts netAdapterStat to cadvisorapi.InterfaceStats.
func (s *netAdapterStat) toCadvisorInterfaceStats() cadvisorapi.InterfaceStats {
	return cadvisorapi.InterfaceStats{
		Name:     s.Name,
		RxBytes:  s.ReceivedBytes,
		RxErrors: s.ReceivedErrors,
		TxBytes:  s.SentBytes,
		TxErrors: s.SentErrors,
	}
}

// getNetAdapterStats gets a list of network adapter statistics.
func getNetAdapterStats() ([]cadvisorapi.InterfaceStats, error) {
	rawOutput, err := exec.Command("powershell", "/c", " Get-NetAdapterStatistics | ConvertTo-Json").CombinedOutput()
	if err != nil {
		return nil, err
	}

	var stats []*netAdapterStat
	err = json.Unmarshal(rawOutput, &stats)
	if err != nil {
		return nil, err
	}

	result := make([]cadvisorapi.InterfaceStats, len(stats))
	for i := range stats {
		result[i] = stats[i].toCadvisorInterfaceStats()
	}

	return result, nil
}
