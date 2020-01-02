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

// RoundUpToGB rounds up given quantity to chunks of GB
func RoundUpToGB(size resource.Quantity) int64 {
	requestBytes := size.Value()
	return roundUpSize(requestBytes, GB)
}

// RoundUpToGiB rounds up given quantity upto chunks of GiB
func RoundUpToGiB(size resource.Quantity) int64 {
	requestBytes := size.Value()
	return roundUpSize(requestBytes, GiB)
}

// RoundUpToMB rounds up given quantity to chunks of MB
func RoundUpToMB(size resource.Quantity) int64 {
	requestBytes := size.Value()
	return roundUpSize(requestBytes, MB)
}

// RoundUpToMiB rounds up given quantity upto chunks of MiB
func RoundUpToMiB(size resource.Quantity) int64 {
	requestBytes := size.Value()
	return roundUpSize(requestBytes, MiB)
}

// RoundUpToKB rounds up given quantity to chunks of KB
func RoundUpToKB(size resource.Quantity) int64 {
	requestBytes := size.Value()
	return roundUpSize(requestBytes, KB)
}

// RoundUpToKiB rounds up given quantity upto chunks of KiB
func RoundUpToKiB(size resource.Quantity) int64 {
	requestBytes := size.Value()
	return roundUpSize(requestBytes, KiB)
}

// RoundUpToGBInt rounds up given quantity to chunks of GB. It returns an
// int instead of an int64 and an error if there's overflow
func RoundUpToGBInt(size resource.Quantity) (int, error) {
	requestBytes := size.Value()
	return roundUpSizeInt(requestBytes, GB)
}

// RoundUpToGiBInt rounds up given quantity upto chunks of GiB. It returns an
// int instead of an int64 and an error if there's overflow
func RoundUpToGiBInt(size resource.Quantity) (int, error) {
	requestBytes := size.Value()
	return roundUpSizeInt(requestBytes, GiB)
}

// RoundUpToMBInt rounds up given quantity to chunks of MB. It returns an
// int instead of an int64 and an error if there's overflow
func RoundUpToMBInt(size resource.Quantity) (int, error) {
	requestBytes := size.Value()
	return roundUpSizeInt(requestBytes, MB)
}

// RoundUpToMiBInt rounds up given quantity upto chunks of MiB. It returns an
// int instead of an int64 and an error if there's overflow
func RoundUpToMiBInt(size resource.Quantity) (int, error) {
	requestBytes := size.Value()
	return roundUpSizeInt(requestBytes, MiB)
}

// RoundUpToKBInt rounds up given quantity to chunks of KB. It returns an
// int instead of an int64 and an error if there's overflow
func RoundUpToKBInt(size resource.Quantity) (int, error) {
	requestBytes := size.Value()
	return roundUpSizeInt(requestBytes, KB)
}

// RoundUpToKiBInt rounds up given quantity upto chunks of KiB. It returns an
// int instead of an int64 and an error if there's overflow
func RoundUpToKiBInt(size resource.Quantity) (int, error) {
	requestBytes := size.Value()
	return roundUpSizeInt(requestBytes, KiB)
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
