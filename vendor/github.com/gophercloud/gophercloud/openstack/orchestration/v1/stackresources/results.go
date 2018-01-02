package stackresources

import (
	"encoding/json"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Resource represents a stack resource.
type Resource struct {
	Attributes   map[string]interface{} `json:"attributes"`
	CreationTime time.Time              `json:"-"`
	Description  string                 `json:"description"`
	Links        []gophercloud.Link     `json:"links"`
	LogicalID    string                 `json:"logical_resource_id"`
	Name         string                 `json:"resource_name"`
	PhysicalID   string                 `json:"physical_resource_id"`
	RequiredBy   []interface{}          `json:"required_by"`
	Status       string                 `json:"resource_status"`
	StatusReason string                 `json:"resource_status_reason"`
	Type         string                 `json:"resource_type"`
	UpdatedTime  time.Time              `json:"-"`
}

func (r *Resource) UnmarshalJSON(b []byte) error {
	type tmp Resource
	var s struct {
		tmp
		CreationTime gophercloud.JSONRFC3339NoZ `json:"creation_time"`
		UpdatedTime  gophercloud.JSONRFC3339NoZ `json:"updated_time"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = Resource(s.tmp)

	r.CreationTime = time.Time(s.CreationTime)
	r.UpdatedTime = time.Time(s.UpdatedTime)

	return nil
}

// FindResult represents the result of a Find operation.
type FindResult struct {
	gophercloud.Result
}

// Extract returns a slice of Resource objects and is called after a
// Find operation.
func (r FindResult) Extract() ([]Resource, error) {
	var s struct {
		Resources []Resource `json:"resources"`
	}
	err := r.ExtractInto(&s)
	return s.Resources, err
}

// ResourcePage abstracts the raw results of making a List() request against the API.
// As OpenStack extensions may freely alter the response bodies of structures returned to the client, you may only safely access the
// data provided through the ExtractResources call.
type ResourcePage struct {
	pagination.SinglePageBase
}

// IsEmpty returns true if a page contains no Server results.
func (r ResourcePage) IsEmpty() (bool, error) {
	resources, err := ExtractResources(r)
	return len(resources) == 0, err
}

// ExtractResources interprets the results of a single page from a List() call, producing a slice of Resource entities.
func ExtractResources(r pagination.Page) ([]Resource, error) {
	var s struct {
		Resources []Resource `json:"resources"`
	}
	err := (r.(ResourcePage)).ExtractInto(&s)
	return s.Resources, err
}

// GetResult represents the result of a Get operation.
type GetResult struct {
	gophercloud.Result
}

// Extract returns a pointer to a Resource object and is called after a
// Get operation.
func (r GetResult) Extract() (*Resource, error) {
	var s struct {
		Resource *Resource `json:"resource"`
	}
	err := r.ExtractInto(&s)
	return s.Resource, err
}

// MetadataResult represents the result of a Metadata operation.
type MetadataResult struct {
	gophercloud.Result
}

// Extract returns a map object and is called after a
// Metadata operation.
func (r MetadataResult) Extract() (map[string]string, error) {
	var s struct {
		Meta map[string]string `json:"metadata"`
	}
	err := r.ExtractInto(&s)
	return s.Meta, err
}

// ResourceTypePage abstracts the raw results of making a ListTypes() request against the API.
// As OpenStack extensions may freely alter the response bodies of structures returned to the client, you may only safely access the
// data provided through the ExtractResourceTypes call.
type ResourceTypePage struct {
	pagination.SinglePageBase
}

// IsEmpty returns true if a ResourceTypePage contains no resource types.
func (r ResourceTypePage) IsEmpty() (bool, error) {
	rts, err := ExtractResourceTypes(r)
	return len(rts) == 0, err
}

// ResourceTypes represents the type that holds the result of ExtractResourceTypes.
// We define methods on this type to sort it before output
type ResourceTypes []string

func (r ResourceTypes) Len() int {
	return len(r)
}

func (r ResourceTypes) Swap(i, j int) {
	r[i], r[j] = r[j], r[i]
}

func (r ResourceTypes) Less(i, j int) bool {
	return r[i] < r[j]
}

// ExtractResourceTypes extracts and returns resource types.
func ExtractResourceTypes(r pagination.Page) (ResourceTypes, error) {
	var s struct {
		ResourceTypes ResourceTypes `json:"resource_types"`
	}
	err := (r.(ResourceTypePage)).ExtractInto(&s)
	return s.ResourceTypes, err
}

// TypeSchema represents a stack resource schema.
type TypeSchema struct {
	Attributes    map[string]interface{} `json:"attributes"`
	Properties    map[string]interface{} `json:"properties"`
	ResourceType  string                 `json:"resource_type"`
	SupportStatus map[string]interface{} `json:"support_status"`
}

// SchemaResult represents the result of a Schema operation.
type SchemaResult struct {
	gophercloud.Result
}

// Extract returns a pointer to a TypeSchema object and is called after a
// Schema operation.
func (r SchemaResult) Extract() (*TypeSchema, error) {
	var s *TypeSchema
	err := r.ExtractInto(&s)
	return s, err
}

// TemplateResult represents the result of a Template operation.
type TemplateResult struct {
	gophercloud.Result
}

// Extract returns the template and is called after a
// Template operation.
func (r TemplateResult) Extract() ([]byte, error) {
	if r.Err != nil {
		return nil, r.Err
	}
	template, err := json.MarshalIndent(r.Body, "", "  ")
	return template, err
}
