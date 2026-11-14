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
	"math"
	"math/big"
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

// convertQuantityToString converts a quantity value to the format of divisor and returns
// the ceiling of the value as a decimal string.
func convertQuantityToString(q *resource.Quantity, divisor resource.Quantity) (string, error) {
	if q == nil || q.IsZero() || q.Sign() <= 0 {
		return "0", nil
	}
	if divisor.IsZero() || divisor.Sign() <= 0 {
		return "0", nil
	}

	qDec := q.AsDec()
	divDec := divisor.AsDec()

	qBig := new(big.Int).Set(qDec.UnscaledBig())
	divBig := new(big.Int).Set(divDec.UnscaledBig())

	// Scales are int32 and reach math.MinInt32, so the differences below need
	// more room than an int has where it is 32 bits wide, such as linux/386.
	qScale := int64(qDec.Scale())
	divScale := int64(divDec.Scale())

	// Bound the work before aligning scales. Aligning multiplies one side by
	// 10^(scale difference), and a value written as 1e1000000 would make that
	// a million digits wide, so compare magnitudes first: the bit length gives
	// the decimal exponent without materializing anything.
	// A bit length pins the digit count only to within one, so each side is a
	// range and a shortcut is taken only where the ranges cannot overlap.
	qLow, qHigh := decimalExponentBounds(qBig, qScale)
	divLow, divHigh := decimalExponentBounds(divBig, divScale)
	switch {
	case qLow-divHigh > maxResultDigits:
		// Past what an int64 holds however the estimate landed, so saturate
		// instead of building the intermediate.
		return strconv.FormatInt(math.MaxInt64, 10), nil
	case divLow-qHigh > 1:
		// The divisor is larger however the estimate landed, so the ceiling
		// of this positive ratio is 1.
		return "1", nil
	}

	sDiff := divScale - qScale

	if sDiff > 0 {
		exp := new(big.Int).Exp(big.NewInt(10), big.NewInt(sDiff), nil)
		qBig.Mul(qBig, exp)
	} else if sDiff < 0 {
		exp := new(big.Int).Exp(big.NewInt(10), big.NewInt(-sDiff), nil)
		divBig.Mul(divBig, exp)
	}

	if divBig.Sign() <= 0 {
		return "0", nil
	}

	// ceil(N / D) = (N + D - 1) / D
	tmp := new(big.Int).Add(qBig, divBig)
	tmp.Sub(tmp, big.NewInt(1))
	res := new(big.Int).Quo(tmp, divBig)

	// Saturate whatever the shortcut could not rule out, so every input maps
	// to an int64 and the result stays monotonic across the boundary.
	if !res.IsInt64() {
		return strconv.FormatInt(math.MaxInt64, 10), nil
	}

	return res.String(), nil
}

// convertResourceCPUToString converts cpu value to the format of divisor and returns
// ceiling of the value.

// maxResultDigits is the number of decimal digits math.MaxInt64 has. A ratio
// wider than this cannot be expressed as an int64, so it saturates.
const maxResultDigits = 19

// decimalExponentBounds brackets the power of ten of unscaled/10^scale. It
// reads the bit length rather than the digits so that measuring a value like
// 1e1000000 stays cheap, and returns a range because 2^(bits-1) <= v < 2^bits
// pins the digit count only to within one. The bounds carry an extra digit of
// slack so that a padded coefficient, which a parsed quantity never has but
// NewDecimalQuantity allows, cannot push a caller onto the wrong side of a
// shortcut.
func decimalExponentBounds(unscaled *big.Int, scale int64) (low, high int64) {
	if unscaled.Sign() == 0 {
		return 0, 0
	}
	const log10of2 = 0.301029995663981195
	bits := int64(unscaled.BitLen())
	low = int64(float64(bits-1)*log10of2) - scale
	high = int64(float64(bits)*log10of2) + 2 - scale
	return low, high
}

func convertResourceCPUToString(cpu *resource.Quantity, divisor resource.Quantity) (string, error) {
	return convertQuantityToString(cpu, divisor)
}

// convertResourceMemoryToString converts memory value to the format of divisor and returns
// ceiling of the value.
func convertResourceMemoryToString(memory *resource.Quantity, divisor resource.Quantity) (string, error) {
	return convertQuantityToString(memory, divisor)
}

// convertResourceHugePagesToString converts hugepages value to the format of divisor and returns
// ceiling of the value.
func convertResourceHugePagesToString(hugePages *resource.Quantity, divisor resource.Quantity) (string, error) {
	return convertQuantityToString(hugePages, divisor)
}

// convertResourceEphemeralStorageToString converts ephemeral storage value to the format of divisor and returns
// ceiling of the value.
func convertResourceEphemeralStorageToString(ephemeralStorage *resource.Quantity, divisor resource.Quantity) (string, error) {
	return convertQuantityToString(ephemeralStorage, divisor)
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
