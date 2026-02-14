//go:build !linux

/*
Copyright 2024 The Kubernetes Authors.

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

	v1 "k8s.io/api/core/v1"
	corehelper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
)

// HugepagesReader provides an interface for reading hugepage information from the OS.
type HugepagesReader interface {
	GetFreeHugepages(numaNodeID int, pageSizeBytes uint64) (uint64, error)
}

// stubHugepagesReader is a no-op implementation for non-Linux platforms
type stubHugepagesReader struct{}

// NewSysfsHugepagesReader returns a stub implementation on non-Linux platforms
func NewSysfsHugepagesReader() HugepagesReader {
	return &stubHugepagesReader{}
}

// GetFreeHugepages returns an error on non-Linux platforms as hugepages are Linux-specific
func (r *stubHugepagesReader) GetFreeHugepages(numaNodeID int, pageSizeBytes uint64) (uint64, error) {
	return 0, fmt.Errorf("hugepages are not supported on this platform")
}

// resourceNameToPageSize extracts the page size in bytes from a hugepage resource name.
// This function works on all platforms as it uses the core helper functions.
func resourceNameToPageSize(resourceName v1.ResourceName) (uint64, error) {
	// Check if this is a hugepages resource
	if !corehelper.IsHugePageResourceName(resourceName) {
		return 0, fmt.Errorf("%s is not a hugepage resource", resourceName)
	}

	// Extract the quantity from the resource name
	pageSize, err := corehelper.HugePageSizeFromResourceName(resourceName)
	if err != nil {
		return 0, fmt.Errorf("failed to extract page size from resource name %s: %w", resourceName, err)
	}

	return uint64(pageSize.Value()), nil
}
