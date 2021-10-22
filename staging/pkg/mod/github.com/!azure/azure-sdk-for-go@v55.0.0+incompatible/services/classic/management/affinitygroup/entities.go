// +build go1.7

package affinitygroup

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

import (
	"encoding/xml"
)

// CreateAffinityGroupParams respresents the set of parameters required for
// creating an affinity group creation request to Azure.
//
// https://msdn.microsoft.com/en-us/library/azure/gg715317.aspx
type CreateAffinityGroupParams struct {
	XMLName     xml.Name `xml:"http://schemas.microsoft.com/windowsazure CreateAffinityGroup"`
	Name        string
	Label       string
	Description string `xml:",omitempty"`
	Location    string
}

// HostedService is a struct containing details about a hosted service that is
// part of an affinity group on Azure.
type HostedService struct {
	URL         string `xml:"Url"`
	ServiceName string
}

// StorageService is a struct containing details about a storage  service that is
// part of an affinity group on Azure.
type StorageService struct {
	URL         string `xml:"Url"`
	ServiceName string
}

// AffinityGroup respresents the properties of an affinity group on Azure.
//
// https://msdn.microsoft.com/en-us/library/azure/ee460789.aspx
type AffinityGroup struct {
	Name            string
	Label           string
	Description     string
	Location        string
	HostedServices  []HostedService
	StorageServices []StorageService
	Capabilities    []string
}

// ComputeCapabilities represents the sets of capabilities of an affinity group
// obtained from an affinity group list call to Azure.
type ComputeCapabilities struct {
	VirtualMachineRoleSizes []string
	WebWorkerRoleSizes      []string
}

// AffinityGroupListResponse represents the properties obtained for each
// affinity group listed off Azure.
//
// https://msdn.microsoft.com/en-us/library/azure/ee460797.aspx
type AffinityGroupListResponse struct {
	Name                string
	Label               string
	Description         string
	Location            string
	Capabilities        []string
	ComputeCapabilities ComputeCapabilities
}

// ListAffinityGroupsResponse contains all the affinity groups obtained from a
// call to the Azure API to list all affinity groups.
type ListAffinityGroupsResponse struct {
	AffinityGroups []AffinityGroupListResponse `xml:"AffinityGroup"`
}

// UpdateAffinityGroupParams if the set of parameters required to update an
// affinity group on Azure.
//
// https://msdn.microsoft.com/en-us/library/azure/gg715316.aspx
type UpdateAffinityGroupParams struct {
	XMLName     xml.Name `xml:"http://schemas.microsoft.com/windowsazure UpdateAffinityGroup"`
	Label       string   `xml:",omitempty"`
	Description string   `xml:",omitempty"`
}
