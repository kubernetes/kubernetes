//go:build !windows

/*
Copyright The Kubernetes Authors.

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

package memorymanager

import (
	"fmt"
	"runtime"

	cadvisorapi "github.com/google/cadvisor/info/v1"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
)

// policyTypeBestEffort is defined per-platform (see also
// policy_best_effort_windows.go) so it is available on every platform while the
// BestEffort policy implementation itself stays Windows-only.
const policyTypeBestEffort policyType = "BestEffort"

// newBestEffortPolicy rejects the BestEffort memory manager policy on
// non-Windows platforms. The BestEffort policy exists only on Windows (its
// implementation in policy_best_effort_windows.go carries a windows build tag);
// this variant satisfies NewManager's call at compile time and returns the
// "not available" error at runtime.
func newBestEffortPolicy(logger klog.Logger, machineInfo *cadvisorapi.MachineInfo, nodeAllocatableReservation v1.ResourceList, reservedMemory []kubeletconfig.MemoryReservation, affinity topologymanager.Store) (Policy, error) {
	return nil, fmt.Errorf("policy %q is not available for platform %q", policyTypeBestEffort, runtime.GOOS)
}
