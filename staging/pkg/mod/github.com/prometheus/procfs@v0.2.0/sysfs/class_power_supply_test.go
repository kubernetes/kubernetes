// Copyright 2018 The Prometheus Authors
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

func TestPowerSupplyClass(t *testing.T) {
	fs, err := NewFS(sysTestFixtures)
	if err != nil {
		t.Fatalf("failed to open filesystem: %v", err)
	}

	got, err := fs.PowerSupplyClass()
	if err != nil {
		t.Fatalf("failed to parse power supply class: %v", err)
	}

	var (
		acOnline             int64
		bat0Capacity         int64 = 98
		bat0CycleCount       int64
		bat0EnergyFull       int64 = 50060000
		bat0EnergyFullDesign int64 = 47520000
		bat0EnergyNow        int64 = 49450000
		bat0PowerNow         int64 = 4830000
		bat0Present          int64 = 1
		bat0VoltageMinDesign int64 = 10800000
		bat0VoltageNow       int64 = 12229000
	)

	want := PowerSupplyClass{
		"AC": {
			Name:   "AC",
			Type:   "Mains",
			Online: &acOnline,
		},
		"BAT0": {
			Name:             "BAT0",
			Capacity:         &bat0Capacity,
			CapacityLevel:    "Normal",
			CycleCount:       &bat0CycleCount,
			EnergyFull:       &bat0EnergyFull,
			EnergyFullDesign: &bat0EnergyFullDesign,
			EnergyNow:        &bat0EnergyNow,
			Manufacturer:     "LGC",
			ModelName:        "LNV-45N1",
			PowerNow:         &bat0PowerNow,
			Present:          &bat0Present,
			SerialNumber:     "38109",
			Status:           "Discharging",
			Technology:       "Li-ion",
			Type:             "Battery",
			VoltageMinDesign: &bat0VoltageMinDesign,
			VoltageNow:       &bat0VoltageNow,
		},
	}

	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("unexpected power supply class (-want +got):\n%s", diff)
	}
}
