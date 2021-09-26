// Copyright 2019 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build !windows

package sysfs

import (
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestParseSlowRate(t *testing.T) {
	tests := []struct {
		rate string
		want uint64
	}{
		{
			rate: "2.5 Gb/sec (1X SDR)",
			want: 312500000,
		},
		{
			rate: "500 Gb/sec (4X HDR)",
			want: 62500000000,
		},
	}

	for _, tt := range tests {
		rate, err := parseRate(tt.rate)
		if err != nil {
			t.Fatal(err)
		}
		if rate != tt.want {
			t.Errorf("Result for InfiniBand rate not correct: want %v, have %v", tt.want, rate)
		}
	}
}

func TestInfiniBandClass(t *testing.T) {
	fs, err := NewFS(sysTestFixtures)
	if err != nil {
		t.Fatal(err)
	}

	got, err := fs.InfiniBandClass()
	if err != nil {
		t.Fatal(err)
	}

	var (
		port1LinkDowned                  uint64
		port1LinkErrorRecovery           uint64
		port1PortRcvConstraintErrors     uint64
		port1PortRcvData                 uint64 = 8884894436
		port1PortRcvErrors               uint64
		port1PortRcvPackets              uint64 = 87169372
		port1PortRcvRemotePhysicalErrors uint64
		port1PortRcvSwitchRelayErrors    uint64
		port1PortXmitConstraintErrors    uint64
		port1PortXmitData                uint64 = 106036453180
		port1PortXmitDiscards            uint64
		port1PortXmitPackets             uint64 = 85734114
		port1PortXmitWait                uint64 = 3599
		port1SymbolError                 uint64
		port1VL15Dropped                 uint64

		port2LinkDowned                  uint64
		port2LinkErrorRecovery           uint64
		port2PortRcvConstraintErrors     uint64
		port2PortRcvData                 uint64 = 9841747136
		port2PortRcvErrors               uint64
		port2PortRcvPackets              uint64 = 89332064
		port2PortRcvRemotePhysicalErrors uint64
		port2PortRcvSwitchRelayErrors    uint64
		port2PortXmitConstraintErrors    uint64
		port2PortXmitData                uint64 = 106161427560
		port2PortXmitDiscards            uint64
		port2PortXmitPackets             uint64 = 88622850
		port2PortXmitWait                uint64 = 3846
		port2SymbolError                 uint64
		port2VL15Dropped                 uint64
	)

	want := InfiniBandClass{
		"mlx4_0": InfiniBandDevice{
			Name:            "mlx4_0",
			BoardID:         "SM_1141000001000",
			FirmwareVersion: "2.31.5050",
			HCAType:         "MT4099",
			Ports: map[uint]InfiniBandPort{
				1: {
					Name:        "mlx4_0",
					Port:        1,
					State:       "ACTIVE",
					StateID:     4,
					PhysState:   "LinkUp",
					PhysStateID: 5,
					Rate:        5000000000,
					Counters: InfiniBandCounters{
						LinkDowned:                  &port1LinkDowned,
						LinkErrorRecovery:           &port1LinkErrorRecovery,
						PortRcvConstraintErrors:     &port1PortRcvConstraintErrors,
						PortRcvData:                 &port1PortRcvData,
						PortRcvErrors:               &port1PortRcvErrors,
						PortRcvPackets:              &port1PortRcvPackets,
						PortRcvRemotePhysicalErrors: &port1PortRcvRemotePhysicalErrors,
						PortRcvSwitchRelayErrors:    &port1PortRcvSwitchRelayErrors,
						PortXmitConstraintErrors:    &port1PortXmitConstraintErrors,
						PortXmitData:                &port1PortXmitData,
						PortXmitDiscards:            &port1PortXmitDiscards,
						PortXmitPackets:             &port1PortXmitPackets,
						PortXmitWait:                &port1PortXmitWait,
						SymbolError:                 &port1SymbolError,
						VL15Dropped:                 &port1VL15Dropped,
					},
				},
				2: {
					Name:        "mlx4_0",
					Port:        2,
					State:       "ACTIVE",
					StateID:     4,
					PhysState:   "LinkUp",
					PhysStateID: 5,
					Rate:        5000000000,
					Counters: InfiniBandCounters{
						LinkDowned:                  &port2LinkDowned,
						LinkErrorRecovery:           &port2LinkErrorRecovery,
						PortRcvConstraintErrors:     &port2PortRcvConstraintErrors,
						PortRcvData:                 &port2PortRcvData,
						PortRcvErrors:               &port2PortRcvErrors,
						PortRcvPackets:              &port2PortRcvPackets,
						PortRcvRemotePhysicalErrors: &port2PortRcvRemotePhysicalErrors,
						PortRcvSwitchRelayErrors:    &port2PortRcvSwitchRelayErrors,
						PortXmitConstraintErrors:    &port2PortXmitConstraintErrors,
						PortXmitData:                &port2PortXmitData,
						PortXmitDiscards:            &port2PortXmitDiscards,
						PortXmitPackets:             &port2PortXmitPackets,
						PortXmitWait:                &port2PortXmitWait,
						SymbolError:                 &port2SymbolError,
						VL15Dropped:                 &port2VL15Dropped,
					},
				},
			},
		},
	}

	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("unexpected InfiniBand class (-want +got):\n%s", diff)
	}
}
