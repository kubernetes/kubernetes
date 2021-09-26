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
	"reflect"
	"testing"

	"github.com/prometheus/procfs/internal/util"
)

func TestClassThermalZoneStats(t *testing.T) {
	fs, err := NewFS(sysTestFixtures)
	if err != nil {
		t.Fatal(err)
	}

	thermalTest, err := fs.ClassThermalZoneStats()
	if err != nil {
		t.Fatal(err)
	}

	enabled := util.ParseBool("enabled")
	passive := uint64(0)

	classThermalZoneStats := []ClassThermalZoneStats{
		{
			Name:    "0",
			Type:    "bcm2835_thermal",
			Policy:  "step_wise",
			Temp:    49925,
			Mode:    nil,
			Passive: nil,
		},
		{
			Name:    "1",
			Type:    "acpitz",
			Policy:  "step_wise",
			Temp:    -44000,
			Mode:    enabled,
			Passive: &passive,
		},
	}

	if !reflect.DeepEqual(classThermalZoneStats, thermalTest) {
		t.Errorf("Result not correct: want %v, have %v", classThermalZoneStats, thermalTest)
	}
}
