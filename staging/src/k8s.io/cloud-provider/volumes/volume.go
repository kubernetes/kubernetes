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

package volumes

import (
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/api/resource"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	//TODO: modify when removing pkg/kubelet/apis
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
)

const (
	// ProvisionedVolumeName is the name of a volume in external cloud that is being provisioned and thus
	// should be ignored by rest of Kubernetes.
	ProvisionedVolumeName = "placeholder-for-provisioning"

	// GIB - GibiByte size
	GIB = 1024 * 1024 * 1024
)

// NewDeletedVolumeInUseError returns a new instance of DeletedVolumeInUseError
// error.
func NewDeletedVolumeInUseError(message string) error {
	return deletedVolumeInUseError(message)
}

type deletedVolumeInUseError string

var _ error = deletedVolumeInUseError("")

func (err deletedVolumeInUseError) Error() string {
	return string(err)
}

// DanglingAttachError is the error on attach indicates volume is attached to a different node
// than we expected.
type DanglingAttachError struct {
	msg         string
	CurrentNode k8stypes.NodeName
	DevicePath  string
}

func (err *DanglingAttachError) Error() string {
	return err.msg
}

// NewDanglingError returns DanglingAttachError with provided parameters
func NewDanglingError(msg string, node k8stypes.NodeName, devicePath string) error {
	return &DanglingAttachError{
		msg:         msg,
		CurrentNode: node,
		DevicePath:  devicePath,
	}
}

// RoundUpSize calculates how many allocation units are needed to accommodate
// a volume of given size. E.g. when user wants 1500MiB volume, while AWS EBS
// allocates volumes in gibibyte-sized chunks,
// RoundUpSize(1500 * 1024*1024, 1024*1024*1024) returns '2'
// (2 GiB is the smallest allocatable volume that can hold 1500MiB)
func RoundUpSize(volumeSizeBytes int64, allocationUnitBytes int64) int64 {
	roundedUp := volumeSizeBytes / allocationUnitBytes
	if volumeSizeBytes%allocationUnitBytes > 0 {
		roundedUp++
	}
	return roundedUp
}

// RoundUpToGiB rounds up given quantity upto chunks of GiB
func RoundUpToGiB(size resource.Quantity) int64 {
	requestBytes := size.Value()
	return RoundUpSize(requestBytes, GIB)
}

// StringToSet converts a string containing list separated by specified delimiter to a set
func stringToSet(str, delimiter string) (sets.String, error) {
	zonesSlice := strings.Split(str, delimiter)
	zonesSet := make(sets.String)
	for _, zone := range zonesSlice {
		trimmedZone := strings.TrimSpace(zone)
		if trimmedZone == "" {
			return make(sets.String), fmt.Errorf(
				"%q separated list (%q) must not contain an empty string",
				delimiter,
				str)
		}
		zonesSet.Insert(trimmedZone)
	}
	return zonesSet, nil
}

// LabelZonesToSet converts a PV label value from string containing a delimited list of zones to set
func LabelZonesToSet(labelZonesValue string) (sets.String, error) {
	return stringToSet(labelZonesValue, kubeletapis.LabelMultiZoneDelimiter)
}

// RoundUpSizeInt calculates how many allocation units are needed to accommodate
// a volume of given size. It returns an int instead of an int64 and an error if
// there's overflow
func RoundUpSizeInt(volumeSizeBytes int64, allocationUnitBytes int64) (int, error) {
	roundedUp := RoundUpSize(volumeSizeBytes, allocationUnitBytes)
	roundedUpInt := int(roundedUp)
	if int64(roundedUpInt) != roundedUp {
		return 0, fmt.Errorf("capacity %v is too great, casting results in integer overflow", roundedUp)
	}
	return roundedUpInt, nil
}

// RoundUpToGiBInt rounds up given quantity upto chunks of GiB. It returns an
// int instead of an int64 and an error if there's overflow
func RoundUpToGiBInt(size resource.Quantity) (int, error) {
	requestBytes := size.Value()
	return RoundUpSizeInt(requestBytes, GIB)
}
