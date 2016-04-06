/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

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

package ovf

type Envelope struct {
	References []File `xml:"References>File"`

	// Package level meta-data
	Annotation         *AnnotationSection         `xml:"AnnotationSection"`
	Product            *ProductSection            `xml:"ProductSection"`
	Network            *NetworkSection            `xml:"NetworkSection"`
	Disk               *DiskSection               `xml:"DiskSection"`
	OperatingSystem    *OperatingSystemSection    `xml:"OperatingSystemSection"`
	Eula               *EulaSection               `xml:"EulaSection"`
	VirtualHardware    *VirtualHardwareSection    `xml:"VirtualHardwareSection"`
	ResourceAllocation *ResourceAllocationSection `xml:"ResourceAllocationSection"`
	DeploymentOption   *DeploymentOptionSection   `xml:"DeploymentOptionSection"`

	// Content: A VirtualSystem or a VirtualSystemCollection
	VirtualSystem *VirtualSystem `xml:"VirtualSystem"`
}

type VirtualSystem struct {
	Content

	Annotation      []AnnotationSection      `xml:"AnnotationSection"`
	Product         []ProductSection         `xml:"ProductSection"`
	OperatingSystem []OperatingSystemSection `xml:"OperatingSystemSection"`
	Eula            []EulaSection            `xml:"EulaSection"`
	VirtualHardware []VirtualHardwareSection `xml:"VirtualHardwareSection"`
}

type File struct {
	ID          string  `xml:"id,attr"`
	Href        string  `xml:"href,attr"`
	Size        uint    `xml:"size,attr"`
	Compression *string `xml:"compression,attr"`
	ChunkSize   *int    `xml:"chunkSize,attr"`
}

type Content struct {
	ID   string  `xml:"id,attr"`
	Info string  `xml:"Info"`
	Name *string `xml:"Name"`
}

type Section struct {
	Required *bool  `xml:"required,attr"`
	Info     string `xml:"Info"`
}

type AnnotationSection struct {
	Section

	Annotation string `xml:"Annotation"`
}

type ProductSection struct {
	Section

	Class    *string `xml:"class,attr"`
	Instance *string `xml:"instance,attr"`

	Product     string     `xml:"Product"`
	Vendor      string     `xml:"Vendor"`
	Version     string     `xml:"Version"`
	FullVersion string     `xml:"FullVersion"`
	ProductURL  string     `xml:"ProductUrl"`
	VendorURL   string     `xml:"VendorUrl"`
	AppURL      string     `xml:"AppUrl"`
	Property    []Property `xml:"Property"`
}

type Property struct {
	Key              string  `xml:"key,attr"`
	Type             string  `xml:"type,attr"`
	Qualifiers       *string `xml:"qualifiers,attr"`
	UserConfigurable *bool   `xml:"userConfigurable,attr"`
	Default          *string `xml:"value,attr"`
	Password         *bool   `xml:"password,attr"`

	Label       *string `xml:"Label"`
	Description *string `xml:"Description"`

	Values []PropertyConfigurationValue `xml:"Value"`
}

type PropertyConfigurationValue struct {
	Value         string  `xml:"value,attr"`
	Configuration *string `xml:"configuration,attr"`
}

type NetworkSection struct {
	Section

	Networks []Network `xml:"Network"`
}

type Network struct {
	Name string `xml:"name,attr"`

	Description string `xml:"Description"`
}

type DiskSection struct {
	Section

	Disks []VirtualDiskDesc `xml:"Disk"`
}

type VirtualDiskDesc struct {
	DiskID                  string  `xml:"diskId,attr"`
	FileRef                 *string `xml:"fileRef,attr"`
	Capacity                string  `xml:"capacity,attr"`
	CapacityAllocationUnits *string `xml:"capacityAllocationUnits,attr"`
	Format                  *string `xml:"format,attr"`
	PopulatedSize           *int    `xml:"populatedSize,attr"`
	ParentRef               *string `xml:"parentRef,attr"`
}

type OperatingSystemSection struct {
	Section

	ID      uint16  `xml:"id,attr"`
	Version *string `xml:"version,attr"`
	OSType  *string `xml:"osType,attr"`

	Description *string `xml:"Description"`
}

type EulaSection struct {
	Section

	License string `xml:"License"`
}

type VirtualHardwareSection struct {
	Section

	ID        *string `xml:"id,attr"`
	Transport *string `xml:"transport,attr"`

	System *VirtualSystemSettingData       `xml:"System"`
	Item   []ResourceAllocationSettingData `xml:"Item"`
}

type VirtualSystemSettingData struct {
	CIMVirtualSystemSettingData
}

type ResourceAllocationSettingData struct {
	CIMResourceAllocationSettingData

	Required      *bool   `xml:"required,attr"`
	Configuration *string `xml:"configuration,attr"`
	Bound         *string `xml:"bound,attr"`
}

type ResourceAllocationSection struct {
	Section

	Item []ResourceAllocationSettingData `xml:"Item"`
}

type DeploymentOptionSection struct {
	Section

	Configuration []DeploymentOptionConfiguration `xml:"Configuration"`
}

type DeploymentOptionConfiguration struct {
	ID      string `xml:"id,attr"`
	Default *bool  `xml:"default,attr"`

	Label       string `xml:"Label"`
	Description string `xml:"Description"`
}
