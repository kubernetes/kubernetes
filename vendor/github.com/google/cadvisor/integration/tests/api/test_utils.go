// Copyright 2014 Google Inc. All Rights Reserved.
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

	info "github.com/google/cadvisor/info/v1"

	"github.com/stretchr/testify/assert"
)

// Checks that expected and actual are within delta of each other.
func inDelta(t *testing.T, expected, actual, delta uint64, description string) {
	var diff uint64
	if expected > actual {
		diff = expected - actual
	} else {
		diff = actual - expected
	}
	if diff > delta {
		t.Errorf("%s (%d and %d) are not within %d of each other", description, expected, actual, delta)
	}
}

// Checks that CPU stats are valid.
func checkCpuStats(t *testing.T, stat info.CpuStats) {
	assert := assert.New(t)

	assert.NotEqual(0, stat.Usage.Total, "Total CPU usage should not be zero")
	assert.NotEmpty(stat.Usage.PerCpu, "Per-core usage should not be empty")
	totalUsage := uint64(0)
	for _, usage := range stat.Usage.PerCpu {
		totalUsage += usage
	}
	inDelta(t, stat.Usage.Total, totalUsage, uint64((5 * time.Millisecond).Nanoseconds()), "Per-core CPU usage")
	inDelta(t, stat.Usage.Total, stat.Usage.User+stat.Usage.System, uint64((500 * time.Millisecond).Nanoseconds()), "User + system CPU usage")
	// TODO(rjnagal): Add verification for cpu load.
}

func checkMemoryStats(t *testing.T, stat info.MemoryStats) {
	assert := assert.New(t)

	assert.NotEqual(0, stat.Usage, "Memory usage should not be zero")
	assert.NotEqual(0, stat.WorkingSet, "Memory working set should not be zero")
	if stat.WorkingSet > stat.Usage {
		t.Errorf("Memory working set (%d) should be at most equal to memory usage (%d)", stat.WorkingSet, stat.Usage)
	}
	// TODO(vmarmol): Add checks for ContainerData and HierarchicalData
}
