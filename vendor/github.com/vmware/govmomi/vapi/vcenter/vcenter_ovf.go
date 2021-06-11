/*
Copyright (c) 2018 VMware, Inc. All Rights Reserved.

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

package vcenter

import (
	"context"
	"fmt"
	"net/http"

	"github.com/vmware/govmomi/vapi/internal"
	"github.com/vmware/govmomi/vapi/rest"
	"github.com/vmware/govmomi/vim25/types"
)

// AdditionalParams are additional OVF parameters which can be specified for a deployment target.
// This structure is a union where based on Type, only one of each commented section will be set.
type AdditionalParams struct {
	Class string `json:"@class"`
	Type  string `json:"type"`

	// DeploymentOptionParams
	SelectedKey       string             `json:"selected_key,omitempty"`
	DeploymentOptions []DeploymentOption `json:"deployment_options,omitempty"`

	// ExtraConfigs
	ExtraConfig []ExtraConfig `json:"extra_configs,omitempty"`

	// PropertyParams
	Properties []Property `json:"properties,omitempty"`

	// SizeParams
	ApproximateSparseDeploymentSize int64 `json:"approximate_sparse_deployment_size,omitempty"`
	VariableDiskSize                bool  `json:"variable_disk_size,omitempty"`
	ApproximateDownloadSize         int64 `json:"approximate_download_size,omitempty"`
	ApproximateFlatDeploymentSize   int64 `json:"approximate_flat_deployment_size,omitempty"`

	// IpAllocationParams
	SupportedAllocationScheme   []string `json:"supported_allocation_scheme,omitempty"`
	SupportedIPProtocol         []string `json:"supported_ip_protocol,omitempty"`
	SupportedIPAllocationPolicy []string `json:"supported_ip_allocation_policy,omitempty"`
	IPAllocationPolicy          string   `json:"ip_allocation_policy,omitempty"`
	IPProtocol                  string   `json:"ip_protocol,omitempty"`

	// UnknownSections
	UnknownSections []UnknownSection `json:"unknown_sections,omitempty"`
}

const (
	ClassDeploymentOptionParams = "com.vmware.vcenter.ovf.deployment_option_params"
	ClassPropertyParams         = "com.vmware.vcenter.ovf.property_params"
	TypeDeploymentOptionParams  = "DeploymentOptionParams"
	TypeExtraConfigParams       = "ExtraConfigParams"
	TypeIPAllocationParams      = "IpAllocationParams"
	TypePropertyParams          = "PropertyParams"
	TypeSizeParams              = "SizeParams"
)

// DeploymentOption contains the information about a deployment option as defined in the OVF specification
type DeploymentOption struct {
	Key           string `json:"key,omitempty"`
	Label         string `json:"label,omitempty"`
	Description   string `json:"description,omitempty"`
	DefaultChoice bool   `json:"default_choice,omitempty"`
}

// ExtraConfig contains information about a vmw:ExtraConfig OVF element
type ExtraConfig struct {
	Key             string `json:"key,omitempty"`
	Value           string `json:"value,omitempty"`
	VirtualSystemID string `json:"virtual_system_id,omitempty"`
}

// Property contains information about a property in an OVF package
type Property struct {
	Category    string `json:"category,omitempty"`
	ClassID     string `json:"class_id,omitempty"`
	Description string `json:"description,omitempty"`
	ID          string `json:"id,omitempty"`
	InstanceID  string `json:"instance_id,omitempty"`
	Label       string `json:"label,omitempty"`
	Type        string `json:"type,omitempty"`
	UIOptional  bool   `json:"ui_optional,omitempty"`
	Value       string `json:"value,omitempty"`
}

// UnknownSection contains information about an unknown section in an OVF package
type UnknownSection struct {
	Tag  string `json:"tag,omitempty"`
	Info string `json:"info,omitempty"`
}

// NetworkMapping specifies the target network to use for sections of type ovf:NetworkSection in the OVF descriptor
type NetworkMapping struct {
	Key   string `json:"key"`
	Value string `json:"value"`
}

// StorageGroupMapping defines the storage deployment target and storage provisioning type for a section of type vmw:StorageGroupSection in the OVF descriptor
type StorageGroupMapping struct {
	Type             string `json:"type"`
	StorageProfileID string `json:"storage_profile_id,omitempty"`
	DatastoreID      string `json:"datastore_id,omitempty"`
	Provisioning     string `json:"provisioning,omitempty"`
}

// StorageMapping specifies the target storage to use for sections of type vmw:StorageGroupSection in the OVF descriptor
type StorageMapping struct {
	Key   string              `json:"key"`
	Value StorageGroupMapping `json:"value"`
}

// DeploymentSpec is the deployment specification for the deployment
type DeploymentSpec struct {
	Name                string             `json:"name,omitempty"`
	Annotation          string             `json:"annotation,omitempty"`
	AcceptAllEULA       bool               `json:"accept_all_EULA,omitempty"`
	NetworkMappings     []NetworkMapping   `json:"network_mappings,omitempty"`
	StorageMappings     []StorageMapping   `json:"storage_mappings,omitempty"`
	StorageProvisioning string             `json:"storage_provisioning,omitempty"`
	StorageProfileID    string             `json:"storage_profile_id,omitempty"`
	Locale              string             `json:"locale,omitempty"`
	Flags               []string           `json:"flags,omitempty"`
	AdditionalParams    []AdditionalParams `json:"additional_parameters,omitempty"`
	DefaultDatastoreID  string             `json:"default_datastore_id,omitempty"`
}

// Target is the target for the deployment
type Target struct {
	ResourcePoolID string `json:"resource_pool_id,omitempty"`
	HostID         string `json:"host_id,omitempty"`
	FolderID       string `json:"folder_id,omitempty"`
}

// Deploy contains the information to start the deployment of a library OVF
type Deploy struct {
	DeploymentSpec `json:"deployment_spec,omitempty"`
	Target         `json:"target,omitempty"`
}

// Error is a SERVER error
type Error struct {
	Class    string                    `json:"@class,omitempty"`
	Messages []rest.LocalizableMessage `json:"messages,omitempty"`
}

// ParseIssue is a parse issue struct
type ParseIssue struct {
	Category     string                  `json:"@classcategory,omitempty"`
	File         string                  `json:"file,omitempty"`
	LineNumber   int64                   `json:"line_number,omitempty"`
	ColumnNumber int64                   `json:"column_number,omitempty"`
	Message      rest.LocalizableMessage `json:"message,omitempty"`
}

// OVFError is a list of errors from create or deploy
type OVFError struct {
	Category string                   `json:"category,omitempty"`
	Error    *Error                   `json:"error,omitempty"`
	Issues   []ParseIssue             `json:"issues,omitempty"`
	Message  *rest.LocalizableMessage `json:"message,omitempty"`
}

// ResourceID is a managed object reference for a deployed resource.
type ResourceID struct {
	Type  string `json:"type,omitempty"`
	Value string `json:"id,omitempty"`
}

// DeploymentError is an error that occurs when deploying and OVF from
// a library item.
type DeploymentError struct {
	Errors []OVFError `json:"errors,omitempty"`
}

// Error implements the error interface
func (e *DeploymentError) Error() string {
	msg := ""
	if len(e.Errors) != 0 {
		err := e.Errors[0]
		if err.Message != nil {
			msg = err.Message.DefaultMessage
		} else if err.Error != nil && len(err.Error.Messages) != 0 {
			msg = err.Error.Messages[0].DefaultMessage
		}
	}
	if msg == "" {
		msg = fmt.Sprintf("%#v", e)
	}
	return "deploy error: " + msg
}

// LibraryTarget specifies a Library or Library item
type LibraryTarget struct {
	LibraryID     string `json:"library_id,omitempty"`
	LibraryItemID string `json:"library_item_id,omitempty"`
}

// CreateSpec info used to create an OVF package from a VM
type CreateSpec struct {
	Description string   `json:"description,omitempty"`
	Name        string   `json:"name,omitempty"`
	Flags       []string `json:"flags,omitempty"`
}

// OVF data used by CreateOVF
type OVF struct {
	Spec   CreateSpec    `json:"create_spec"`
	Source ResourceID    `json:"source"`
	Target LibraryTarget `json:"target"`
}

// CreateResult used for decoded a CreateOVF response
type CreateResult struct {
	Succeeded bool             `json:"succeeded,omitempty"`
	ID        string           `json:"ovf_library_item_id,omitempty"`
	Error     *DeploymentError `json:"error,omitempty"`
}

// Deployment is the results from issuing a library OVF deployment
type Deployment struct {
	Succeeded  bool             `json:"succeeded,omitempty"`
	ResourceID *ResourceID      `json:"resource_id,omitempty"`
	Error      *DeploymentError `json:"error,omitempty"`
}

// FilterRequest contains the information to start a vcenter filter call
type FilterRequest struct {
	Target `json:"target,omitempty"`
}

// FilterResponse returns information from the vcenter filter call
type FilterResponse struct {
	EULAs            []string           `json:"EULAs,omitempty"`
	AdditionalParams []AdditionalParams `json:"additional_params,omitempty"`
	Annotation       string             `json:"Annotation,omitempty"`
	Name             string             `json:"name,omitempty"`
	Networks         []string           `json:"Networks,omitempty"`
	StorageGroups    []string           `json:"storage_groups,omitempty"`
}

// Manager extends rest.Client, adding content library related methods.
type Manager struct {
	*rest.Client
}

// NewManager creates a new Manager instance with the given client.
func NewManager(client *rest.Client) *Manager {
	return &Manager{
		Client: client,
	}
}

// CreateOVF creates a library OVF item in content library from an existing VM
func (c *Manager) CreateOVF(ctx context.Context, ovf OVF) (string, error) {
	if ovf.Source.Type == "" {
		ovf.Source.Type = "VirtualMachine"
	}
	url := c.Resource(internal.VCenterOVFLibraryItem)
	var res CreateResult
	err := c.Do(ctx, url.Request(http.MethodPost, ovf), &res)
	if err != nil {
		return "", err
	}
	if res.Succeeded {
		return res.ID, nil
	}
	return "", res.Error
}

// DeployLibraryItem deploys a library OVF
func (c *Manager) DeployLibraryItem(ctx context.Context, libraryItemID string, deploy Deploy) (*types.ManagedObjectReference, error) {
	url := c.Resource(internal.VCenterOVFLibraryItem).WithID(libraryItemID).WithAction("deploy")
	var res Deployment
	err := c.Do(ctx, url.Request(http.MethodPost, deploy), &res)
	if err != nil {
		return nil, err
	}
	if res.Succeeded {
		ref := types.ManagedObjectReference(*res.ResourceID)
		return &ref, nil
	}
	return nil, res.Error
}

// FilterLibraryItem deploys a library OVF
func (c *Manager) FilterLibraryItem(ctx context.Context, libraryItemID string, filter FilterRequest) (FilterResponse, error) {
	url := c.Resource(internal.VCenterOVFLibraryItem).WithID(libraryItemID).WithAction("filter")
	var res FilterResponse
	return res, c.Do(ctx, url.Request(http.MethodPost, filter), &res)
}
