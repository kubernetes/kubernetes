// Copyright 2015 The rkt Authors
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

package kvm

import (
	"runtime"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
)

// The algorithm/reasoning:
// When running the VM takes a small amount of memory out of the container
//  (systemMemOverhead below)
// The VM also has a transient startup need - some extra 'space' really only
//  needed to 'get going' before we get to the workload.
//
// So, we have two rules:
// - we always add on the system overhead
// - we bump up to the minimum required for boot if we need to
//
// From those, and the app memory asks, calculate how much memory to hand to
//  the VM.
const (
	minMem            = 512 // MB - minimum we need in VM
	systemMemOverhead = 128 // MB - overhead we need for VM
)

// findResources finds value of last isolator for particular type.
func findResources(isolators types.Isolators) (mem, cpus int64) {
	for _, i := range isolators {
		switch v := i.Value().(type) {
		case *types.ResourceMemory:
			mem = v.Limit().Value()
			// Convert bytes into megabytes
			mem /= 1024 * 1024
		case *types.ResourceCPU:
			cpus = v.Limit().Value()
		}
	}
	return mem, cpus
}

// GetAppsResources returns values specified by user in pod-manifest.
// Function expects a podmanifest apps.
// Return aggregate quantity of mem (in MB) and cpus.
func GetAppsResources(apps schema.AppList) (totalCpus, totalMem int64) {
	cpusSpecified := false
	for i := range apps {
		ra := &apps[i]
		app := ra.App
		mem, cpus := findResources(app.Isolators)
		cpusSpecified = cpusSpecified || cpus != 0
		totalCpus += cpus
		totalMem += mem
	}
	// In case when number of specified cpus is greater than
	// number or when cpus aren't specified, we set number
	// of logical cpus as a limit.
	availableCpus := int64(runtime.NumCPU())
	if !cpusSpecified || totalCpus > availableCpus {
		totalCpus = availableCpus
	}

	// Add an overhead for the VM system
	totalMem += systemMemOverhead

	// Always ensure we have at least the minimum RAM needed
	if totalMem < minMem {
		totalMem = minMem
	}

	return totalCpus, totalMem
}
