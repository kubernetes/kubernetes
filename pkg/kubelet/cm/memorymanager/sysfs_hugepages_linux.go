//go:build linux

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
	"os"
	"path/filepath"
	"strconv"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	corehelper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
)

const (
	// sysfsNodePath is the base path for NUMA node information in sysfs
	sysfsNodePath = "/sys/devices/system/node"
	// freeHugepagesFile is the name of the file containing free hugepage count
	freeHugepagesFile = "free_hugepages"
)

// HugepagesReader provides an interface for reading hugepage information from the OS.
// This interface allows for easier testing by enabling mock implementations.
type HugepagesReader interface {
	// GetFreeHugepages returns the number of free hugepages for a given NUMA node
	// and hugepage size. The pageSize is in bytes.
	GetFreeHugepages(numaNodeID int, pageSizeBytes uint64) (uint64, error)
}

// sysfsHugepagesReader reads hugepage information from sysfs
type sysfsHugepagesReader struct {
	sysfsPath string
}

// NewSysfsHugepagesReader creates a new HugepagesReader that reads from sysfs
func NewSysfsHugepagesReader() HugepagesReader {
	return &sysfsHugepagesReader{
		sysfsPath: sysfsNodePath,
	}
}

// newSysfsHugepagesReaderWithPath creates a HugepagesReader with a custom sysfs path (for testing)
func newSysfsHugepagesReaderWithPath(path string) HugepagesReader {
	return &sysfsHugepagesReader{
		sysfsPath: path,
	}
}

// GetFreeHugepages reads the number of free hugepages for a specific NUMA node and page size
// from /sys/devices/system/node/node<N>/hugepages/hugepages-<size>kB/free_hugepages
func (r *sysfsHugepagesReader) GetFreeHugepages(numaNodeID int, pageSizeBytes uint64) (uint64, error) {
	// Convert page size from bytes to kB for the sysfs path
	pageSizeKB := pageSizeBytes / 1024

	// Build the path: /sys/devices/system/node/node<N>/hugepages/hugepages-<size>kB/free_hugepages
	freeHugepagesPath := filepath.Join(
		r.sysfsPath,
		fmt.Sprintf("node%d", numaNodeID),
		"hugepages",
		fmt.Sprintf("hugepages-%dkB", pageSizeKB),
		freeHugepagesFile,
	)

	data, err := os.ReadFile(freeHugepagesPath)
	if err != nil {
		return 0, fmt.Errorf("failed to read free hugepages from %s: %w", freeHugepagesPath, err)
	}

	freePages, err := strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
	if err != nil {
		return 0, fmt.Errorf("failed to parse free hugepages value from %s: %w", freeHugepagesPath, err)
	}

	return freePages, nil
}

// GetFreeHugepagesBytes returns the total free hugepage memory in bytes for a specific
// NUMA node and page size
func (r *sysfsHugepagesReader) GetFreeHugepagesBytes(numaNodeID int, pageSizeBytes uint64) (uint64, error) {
	freePages, err := r.GetFreeHugepages(numaNodeID, pageSizeBytes)
	if err != nil {
		return 0, err
	}
	return freePages * pageSizeBytes, nil
}

// resourceNameToPageSize extracts the page size in bytes from a hugepage resource name
// e.g., "hugepages-2Mi" -> 2097152
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

// pageSizeToResourceName converts a page size in bytes to a hugepage resource name
func pageSizeToResourceName(pageSizeBytes uint64) v1.ResourceName {
	quantity := resource.NewQuantity(int64(pageSizeBytes), resource.BinarySI)
	return corehelper.HugePageResourceName(*quantity)
}
