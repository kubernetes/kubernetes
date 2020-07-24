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

package helpers

import (
	"fmt"

	"k8s.io/apimachinery/pkg/api/resource"
)

const (
	// GB - GigaByte size
	GB = 1000 * 1000 * 1000
	// GiB - GibiByte size
	GiB = 1024 * 1024 * 1024

	// MB - MegaByte size
	MB = 1000 * 1000
	// MiB - MebiByte size
	MiB = 1024 * 1024

	// KB - KiloByte size
	KB = 1000
	// KiB - KibiByte size
	KiB = 1024
)

// RoundUpToGiB rounds up given quantity upto chunks of GiB
func RoundUpToGiB(size resource.Quantity) (int64, error) {
	requestBytes, ok := size.AsInt64()
	if !ok {
		return 0, fmt.Errorf("quantity %s is too great, overflows int64", size.String())
	}
	return roundUpSize(requestBytes, GiB), nil
}

// RoundUpToMB rounds up given quantity to chunks of MB
func RoundUpToMB(size resource.Quantity) (int64, error) {
	requestBytes, ok := size.AsInt64()
	if !ok {
		return 0, fmt.Errorf("quantity %s is too great, overflows int64", size.String())
	}
	return roundUpSize(requestBytes, MB), nil
}

// RoundUpToMiB rounds up given quantity upto chunks of MiB
func RoundUpToMiB(size resource.Quantity) (int64, error) {
	requestBytes, ok := size.AsInt64()
	if !ok {
		return 0, fmt.Errorf("quantity %s is too great, overflows int64", size.String())
	}
	return roundUpSize(requestBytes, MiB), nil
}

// RoundUpToKB rounds up given quantity to chunks of KB
func RoundUpToKB(size resource.Quantity) (int64, error) {
	requestBytes, ok := size.AsInt64()
	if !ok {
		return 0, fmt.Errorf("quantity %s is too great, overflows int64", size.String())
	}
	return roundUpSize(requestBytes, KB), nil
}

// RoundUpToKiB rounds up given quantity upto chunks of KiB
func RoundUpToKiB(size resource.Quantity) (int64, error) {
	requestBytes, ok := size.AsInt64()
	if !ok {
		return 0, fmt.Errorf("quantity %s is too great, overflows int64", size.String())
	}
	return roundUpSize(requestBytes, KiB), nil
}

// RoundUpToGiBInt rounds up given quantity upto chunks of GiB. It returns an
// int instead of an int64 and an error if there's overflow
func RoundUpToGiBInt(size resource.Quantity) (int, error) {
	requestBytes, ok := size.AsInt64()
	if !ok {
		return 0, fmt.Errorf("quantity %s is too great, overflows int64", size.String())
	}
	return roundUpSizeInt(requestBytes, GiB)
}

// RoundUpToMBInt rounds up given quantity to chunks of MB. It returns an
// int instead of an int64 and an error if there's overflow
func RoundUpToMBInt(size resource.Quantity) (int, error) {
	requestBytes, ok := size.AsInt64()
	if !ok {
		return 0, fmt.Errorf("quantity %s is too great, overflows int64", size.String())
	}
	return roundUpSizeInt(requestBytes, MB)
}

// RoundUpToMiBInt rounds up given quantity upto chunks of MiB. It returns an
// int instead of an int64 and an error if there's overflow
func RoundUpToMiBInt(size resource.Quantity) (int, error) {
	requestBytes, ok := size.AsInt64()
	if !ok {
		return 0, fmt.Errorf("quantity %s is too great, overflows int64", size.String())
	}
	return roundUpSizeInt(requestBytes, MiB)
}

// RoundUpToKBInt rounds up given quantity to chunks of KB. It returns an
// int instead of an int64 and an error if there's overflow
func RoundUpToKBInt(size resource.Quantity) (int, error) {
	requestBytes, ok := size.AsInt64()
	if !ok {
		return 0, fmt.Errorf("quantity %s is too great, overflows int64", size.String())
	}
	return roundUpSizeInt(requestBytes, KB)
}

// RoundUpToKiBInt rounds up given quantity upto chunks of KiB. It returns an
// int instead of an int64 and an error if there's overflow
func RoundUpToKiBInt(size resource.Quantity) (int, error) {
	requestBytes := size.Value()
	return roundUpSizeInt(requestBytes, KiB)
}

// RoundUpToGiBInt32 rounds up given quantity up to chunks of GiB. It returns an
// int32 instead of an int64 and an error if there's overflow
func RoundUpToGiBInt32(size resource.Quantity) (int32, error) {
	requestBytes, ok := size.AsInt64()
	if !ok {
		return 0, fmt.Errorf("quantity %s is too great, overflows int64", size.String())
	}
	return roundUpSizeInt32(requestBytes, GiB)
}

// roundUpSizeInt calculates how many allocation units are needed to accommodate
// a volume of given size. It returns an int instead of an int64 and an error if
// there's overflow
func roundUpSizeInt(volumeSizeBytes int64, allocationUnitBytes int64) (int, error) {
	roundedUp := roundUpSize(volumeSizeBytes, allocationUnitBytes)
	roundedUpInt := int(roundedUp)
	if int64(roundedUpInt) != roundedUp {
		return 0, fmt.Errorf("capacity %v is too great, casting results in integer overflow", roundedUp)
	}
	return roundedUpInt, nil
}

// roundUpSizeInt32 calculates how many allocation units are needed to accommodate
// a volume of given size. It returns an int32 instead of an int64 and an error if
// there's overflow
func roundUpSizeInt32(volumeSizeBytes int64, allocationUnitBytes int64) (int32, error) {
	roundedUp := roundUpSize(volumeSizeBytes, allocationUnitBytes)
	roundedUpInt32 := int32(roundedUp)
	if int64(roundedUpInt32) != roundedUp {
		return 0, fmt.Errorf("quantity %v is too great, overflows int32", roundedUp)
	}
	return roundedUpInt32, nil
}

// roundUpSize calculates how many allocation units are needed to accommodate
// a volume of given size. E.g. when user wants 1500MiB volume, while AWS EBS
// allocates volumes in gibibyte-sized chunks,
// RoundUpSize(1500 * 1024*1024, 1024*1024*1024) returns '2'
// (2 GiB is the smallest allocatable volume that can hold 1500MiB)
func roundUpSize(volumeSizeBytes int64, allocationUnitBytes int64) int64 {
	roundedUp := volumeSizeBytes / allocationUnitBytes
	if volumeSizeBytes%allocationUnitBytes > 0 {
		roundedUp++
	}
	return roundedUp
}
