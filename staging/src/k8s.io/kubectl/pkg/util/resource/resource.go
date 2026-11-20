/*
Copyright 2018 The Kubernetes Authors.

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

package resource

import (
	"fmt"
	"strconv"
	"strings"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/sets"
)

// ExtractContainerResourceValue extracts the value of a resource
// in an already known container
func ExtractContainerResourceValue(fs *corev1.ResourceFieldSelector, container *corev1.Container) (string, error) {
	divisor := resource.Quantity{}
	if divisor.Cmp(fs.Divisor) == 0 {
		divisor = resource.MustParse("1")
	} else {
		divisor = fs.Divisor
	}

	switch fs.Resource {
	case "limits.cpu":
		return convertResourceCPUToString(container.Resources.Limits.Cpu(), divisor)
	case "limits.memory":
		return convertResourceMemoryToString(container.Resources.Limits.Memory(), divisor)
	case "limits.ephemeral-storage":
		return convertResourceEphemeralStorageToString(container.Resources.Limits.StorageEphemeral(), divisor)
	case "requests.cpu":
		return convertResourceCPUToString(container.Resources.Requests.Cpu(), divisor)
	case "requests.memory":
		return convertResourceMemoryToString(container.Resources.Requests.Memory(), divisor)
	case "requests.ephemeral-storage":
		return convertResourceEphemeralStorageToString(container.Resources.Requests.StorageEphemeral(), divisor)
	}
	// handle extended standard resources with dynamic names
	// example: requests.hugepages-<pageSize> or limits.hugepages-<pageSize>
	if strings.HasPrefix(fs.Resource, "requests.") {
		resourceName := corev1.ResourceName(strings.TrimPrefix(fs.Resource, "requests."))
		if IsHugePageResourceName(resourceName) {
			return convertResourceHugePagesToString(container.Resources.Requests.Name(resourceName, resource.BinarySI), divisor)
		}
	}
	if strings.HasPrefix(fs.Resource, "limits.") {
		resourceName := corev1.ResourceName(strings.TrimPrefix(fs.Resource, "limits."))
		if IsHugePageResourceName(resourceName) {
			return convertResourceHugePagesToString(container.Resources.Limits.Name(resourceName, resource.BinarySI), divisor)
		}
	}
	return "", fmt.Errorf("Unsupported container resource : %v", fs.Resource)
}

// convertResourceToString divides value by divisor at the given scale and returns the
// ceiling, formatted as a base 10 integer.
//
// Both operands are read through the checked accessors so a quantity that does not fit
// in an int64 is reported rather than silently saturating at the rail, and the division
// is done in integer arithmetic: float64 carries a 53 bit mantissa, so converting values
// above 2^53 loses precision before the ceiling is taken.
func convertResourceToString(value *resource.Quantity, divisor resource.Quantity, scale resource.Scale) (string, error) {
	v, ok := value.AsScaledInt64(scale)
	if !ok {
		return "", fmt.Errorf("value %s is too large to be represented at scale %d", value.String(), scale)
	}
	d, ok := divisor.AsScaledInt64(scale)
	if !ok {
		return "", fmt.Errorf("divisor %s is too large to be represented at scale %d", divisor.String(), scale)
	}
	if d == 0 {
		// Callers substitute 1 for an unset divisor, so this is unreachable today.
		// Guard anyway: unlike the float division this replaces, integer division by
		// zero panics.
		return "", fmt.Errorf("divisor must not be zero")
	}

	// Integer ceiling. Computing v+d-1 first would overflow near the rails, so round up
	// from the truncated quotient instead, and only when the exact result is positive.
	q := v / d
	if v%d != 0 && (v > 0) == (d > 0) {
		q++
	}
	return strconv.FormatInt(q, 10), nil
}

// convertResourceCPUToString converts cpu value to the format of divisor and returns
// ceiling of the value.
func convertResourceCPUToString(cpu *resource.Quantity, divisor resource.Quantity) (string, error) {
	return convertResourceToString(cpu, divisor, resource.Milli)
}

// convertResourceMemoryToString converts memory value to the format of divisor and returns
// ceiling of the value.
func convertResourceMemoryToString(memory *resource.Quantity, divisor resource.Quantity) (string, error) {
	return convertResourceToString(memory, divisor, 0)
}

// convertResourceHugePagesToString converts hugepages value to the format of divisor and returns
// ceiling of the value.
func convertResourceHugePagesToString(hugePages *resource.Quantity, divisor resource.Quantity) (string, error) {
	return convertResourceToString(hugePages, divisor, 0)
}

// convertResourceEphemeralStorageToString converts ephemeral storage value to the format of divisor and returns
// ceiling of the value.
func convertResourceEphemeralStorageToString(ephemeralStorage *resource.Quantity, divisor resource.Quantity) (string, error) {
	return convertResourceToString(ephemeralStorage, divisor, 0)
}

var standardContainerResources = sets.New(
	string(corev1.ResourceCPU),
	string(corev1.ResourceMemory),
	string(corev1.ResourceEphemeralStorage),
)

// IsStandardContainerResourceName returns true if the container can make a resource request
// for the specified resource
func IsStandardContainerResourceName(str string) bool {
	return standardContainerResources.Has(str) || IsHugePageResourceName(corev1.ResourceName(str))
}

// IsHugePageResourceName returns true if the resource name has the huge page
// resource prefix.
func IsHugePageResourceName(name corev1.ResourceName) bool {
	return strings.HasPrefix(string(name), corev1.ResourceHugePagesPrefix)
}
