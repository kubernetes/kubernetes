// +build go1.7

package virtualmachinedisk

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

import (
	"encoding/xml"

	"github.com/Azure/azure-sdk-for-go/services/classic/management"
)

// DiskClient is used to perform operations on Azure Disks
type DiskClient struct {
	client management.Client
}

// CreateDiskParameters represents a disk
//
// https://msdn.microsoft.com/en-us/library/azure/jj157188.aspx
type CreateDiskParameters struct {
	XMLName   xml.Name            `xml:"http://schemas.microsoft.com/windowsazure Disk"`
	OS        OperatingSystemType `xml:",omitempty"`
	Label     string
	MediaLink string `xml:",omitempty"`
	Name      string
}

// UpdateDiskParameters represents a disk
//
// https://msdn.microsoft.com/en-us/library/azure/jj157188.aspx
type UpdateDiskParameters struct {
	XMLName         xml.Name `xml:"http://schemas.microsoft.com/windowsazure Disk"`
	Label           string   `xml:",omitempty"`
	Name            string
	ResizedSizeInGB int `xml:",omitempty"`
}

// ListDiskResponse represents a disk
//
// https://msdn.microsoft.com/en-us/library/azure/jj157188.aspx
type ListDiskResponse struct {
	XMLName xml.Name `xml:"http://schemas.microsoft.com/windowsazure Disks"`
	Disk    []DiskResponse
}

// DiskResponse represents a disk
//
// https://msdn.microsoft.com/en-us/library/azure/jj157188.aspx
type DiskResponse struct {
	XMLName             xml.Name `xml:"http://schemas.microsoft.com/windowsazure Disk"`
	AffinityGroup       string
	AttachedTo          Resource
	IsCorrupted         bool
	OS                  OperatingSystemType
	Location            string
	LogicalDiskSizeInGB int
	MediaLink           string
	Name                string
	SourceImageName     string
	CreatedTime         string
	IOType              IOType
}

// Resource describes the resource details a disk is currently attached to
type Resource struct {
	XMLName           xml.Name `xml:"http://schemas.microsoft.com/windowsazure AttachedTo"`
	DeploymentName    string
	HostedServiceName string
	RoleName          string
}

// IOType represents an IO type
type IOType string

// These constants represent the possible IO types
const (
	IOTypeProvisioned IOType = "Provisioned"
	IOTypeStandard    IOType = "Standard"
)

// OperatingSystemType represents an operating system type
type OperatingSystemType string

// These constants represent the valid operating system types
const (
	OperatingSystemTypeNull    OperatingSystemType = "NULL"
	OperatingSystemTypeLinux   OperatingSystemType = "Linux"
	OperatingSystemTypeWindows OperatingSystemType = "Windows"
)

// CreateDataDiskParameters represents a data disk
//
// https://msdn.microsoft.com/en-us/library/azure/jj157188.aspx
type CreateDataDiskParameters struct {
	XMLName             xml.Name        `xml:"http://schemas.microsoft.com/windowsazure DataVirtualHardDisk"`
	HostCaching         HostCachingType `xml:",omitempty"`
	DiskLabel           string          `xml:",omitempty"`
	DiskName            string          `xml:",omitempty"`
	Lun                 int             `xml:",omitempty"`
	LogicalDiskSizeInGB int             `xml:",omitempty"`
	MediaLink           string
	SourceMediaLink     string `xml:",omitempty"`
}

// UpdateDataDiskParameters represents a data disk
//
// https://msdn.microsoft.com/en-us/library/azure/jj157188.aspx
type UpdateDataDiskParameters struct {
	XMLName     xml.Name        `xml:"http://schemas.microsoft.com/windowsazure DataVirtualHardDisk"`
	HostCaching HostCachingType `xml:",omitempty"`
	DiskName    string
	Lun         int
	MediaLink   string
}

// DataDiskResponse represents a data disk
//
// https://msdn.microsoft.com/en-us/library/azure/jj157188.aspx
type DataDiskResponse struct {
	XMLName             xml.Name `xml:"http://schemas.microsoft.com/windowsazure DataVirtualHardDisk"`
	HostCaching         HostCachingType
	DiskLabel           string
	DiskName            string
	Lun                 int
	LogicalDiskSizeInGB int
	MediaLink           string
}

// HostCachingType represents a host caching type
type HostCachingType string

// These constants represent the valid host caching types
const (
	HostCachingTypeNone      HostCachingType = "None"
	HostCachingTypeReadOnly  HostCachingType = "ReadOnly"
	HostCachingTypeReadWrite HostCachingType = "ReadWrite"
)
