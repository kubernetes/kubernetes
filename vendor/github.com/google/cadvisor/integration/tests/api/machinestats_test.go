// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package api

import (
	"testing"
	"time"

	"github.com/google/cadvisor/integration/framework"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMachineStatsIsReturned(t *testing.T) {
	fm := framework.New(t)
	defer fm.Cleanup()

	machineStats, err := fm.Cadvisor().ClientV2().MachineStats()
	if err != nil {
		t.Fatal(err)
	}

	as := assert.New(t)
	for _, stat := range machineStats {
		as.NotEqual(stat.Timestamp, time.Time{})
		as.True(stat.Cpu.Usage.Total > 0)
		as.True(len(stat.Cpu.Usage.PerCpu) > 0)
		if stat.CpuInst != nil {
			as.True(stat.CpuInst.Usage.Total > 0)
		}
		as.True(stat.Memory.Usage > 0)
		for _, nStat := range stat.Network.Interfaces {
			as.NotEqual(nStat.Name, "")
			as.NotEqual(nStat.RxBytes, 0)
		}
		for _, fsStat := range stat.Filesystem {
			as.NotEqual(fsStat.Device, "")
			as.NotNil(fsStat.Capacity)
			as.NotNil(fsStat.Usage)
			as.NotNil(fsStat.ReadsCompleted)
			require.NotEmpty(t, fsStat.Type)
			if fsStat.Type == "vfs" {
				as.NotEmpty(fsStat.InodesFree)
			}
		}
	}
}
