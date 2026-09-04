//go:build windows

/*
Copyright 2017 The Kubernetes Authors.

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
	"math"
	"time"

	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/winstats"
)

// Stats provides basic information about max and current process count.
//
// Windows has no system-wide PID ceiling analogous to Linux kernel.pid_max:
// PID accounting is per-Job-Object rather than against a single global limit.
// We therefore report the live system process count and a sentinel MaxPID so that
// the summary node Rlimit and the PID eviction signal are well-defined on Windows.
// The sentinel ceiling is large enough that NodePIDPressure never fires from
// process count alone; a real per-Job-Object limit would require plumbing the
// kubelet's own Job Object limits through to this package (tracked separately).
func Stats() (*statsapi.RlimitStats, error) {
	info, err := winstats.GetPerformanceInfo()
	if err != nil {
		return nil, err
	}

	numProcs := int64(info.ProcessCount)
	// No kernel.pid_max equivalent on Windows; use a sentinel so the
	// eviction signal's capacity/available computation stays well-defined.
	maxPID := int64(math.MaxInt64)

	return &statsapi.RlimitStats{
		MaxPID:                &maxPID,
		NumOfRunningProcesses: &numProcs,
		Time:                  v1.NewTime(time.Now()),
	}, nil
}
