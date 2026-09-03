//go:build windows

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

package pidlimit

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCurrentProcessCount(t *testing.T) {
	count, err := currentProcessCount()
	require.NoError(t, err)
	// At minimum the test process itself must be enumerated.
	assert.Greater(t, count, int64(0))
}

func TestStats(t *testing.T) {
	stats, err := Stats()
	require.NoError(t, err)
	require.NotNil(t, stats)
	require.NotNil(t, stats.NumOfRunningProcesses)
	assert.Greater(t, *stats.NumOfRunningProcesses, int64(0))

	// Windows has no kernel pid_max knob, so there is no ceiling to report.
	// MaxPID must stay nil: fabricating it (e.g. mirroring the process count)
	// would make pid.available (MaxPID - NumOfRunningProcesses) always resolve
	// to 0 and permanently pressure the node, so pid eviction stays unsupported
	// on Windows.
	assert.Nil(t, stats.MaxPID)
}
