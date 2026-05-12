/*
Copyright 2025 The Kubernetes Authors.

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

package resourceslice

import (
	"fmt"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/util/sets"
)

// validateDriverResources identifies problems that cannot be caught by
// the server-side validation and that would prevent using the published
// ResourceSlices, like incorrect cross-references. We do validation here
// so any issues can be discovered as early as possible. This is also
// checked in the allocator before allocating any devices.
func validateDriverResources(resources *DriverResources) error {
	for poolName, pool := range resources.Pools {
		if err := validatePool(poolName, pool); err != nil {
			return err
		}
	}
	return nil
}

// validatePool checks that there aren't any pool-wide issues that
// can't be caught in the API-server per-ResourceSlice validation.
//
// This logic is very similar to what we do in the allocator when we
// gather the pools. We might want to see if there is a good way to
// put this logic in one place.
func validatePool(name string, pool Pool) error {
	counterSets := make(map[string]resourceapi.CounterSet)
	for _, slice := range pool.Slices {
		for _, counterSet := range slice.SharedCounters {
			if _, found := counterSets[counterSet.Name]; found {
				return fmt.Errorf("found duplicate counter set %q in pool %q", counterSet.Name, name)
			}
			counterSets[counterSet.Name] = counterSet
		}
	}

	devices := sets.New[string]()
	for _, slice := range pool.Slices {
		for _, device := range slice.Devices {
			if _, found := devices[device.Name]; found {
				return fmt.Errorf("found duplicate device %q in pool %q", device.Name, name)
			}
			devices.Insert(device.Name)

			for _, dcc := range device.ConsumesCounters {
				counterSet, found := counterSets[dcc.CounterSet]
				if !found {
					return fmt.Errorf("counter set %q referenced by device %q not found", dcc.CounterSet, device.Name)
				}
				for counterName := range dcc.Counters {
					if _, found := counterSet.Counters[counterName]; !found {
						return fmt.Errorf("counter %q referenced by device %q not found in counter set %q", counterName, device.Name, counterSet.Name)
					}
				}
			}
		}
	}
	return nil
}
