package stackresources

import (
	"fmt"
	"reflect"
	"time"

	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// Resource represents a stack resource.
type Resource struct {
	Links        []gophercloud.Link `mapstructure:"links"`
	LogicalID    string             `mapstructure:"logical_resource_id"`
	Name         string             `mapstructure:"resource_name"`
	PhysicalID   string             `mapstructure:"physical_resource_id"`
	RequiredBy   []interface{}      `mapstructure:"required_by"`
	Status       string             `mapstructure:"resource_status"`
	StatusReason string             `mapstructure:"resource_status_reason"`
	Type         string             `mapstructure:"resource_type"`
	UpdatedTime  time.Time          `mapstructure:"-"`
}

// FindResult represents the result of a Find operation.
type FindResult struct {
	gophercloud.Result
}

// Extract returns a slice of Resource objects and is called after a
// Find operation.
func (r FindResult) Extract() ([]Resource, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res struct {
		Res []Resource `mapstructure:"resources"`
	}

	if err := mapstructure.Decode(r.Body, &res); err != nil {
		return nil, err
	}

	resources := r.Body.(map[string]interface{})["resources"].([]interface{})

	for i, resourceRaw := range resources {
		resource := resourceRaw.(map[string]interface{})
		if date, ok := resource["updated_time"]; ok && date != nil {
			t, err := time.Parse(gophercloud.STACK_TIME_FMT, date.(string))
			if err != nil {
				return nil, err
			}
			res.Res[i].UpdatedTime = t
		}
	}

	return res.Res, nil
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
	if err != nil {
		return true, err
	}
	return len(resources) == 0, nil
}

// LastMarker returns the last container name in a ListResult.
func (r ResourcePage) LastMarker() (string, error) {
	resources, err := ExtractResources(r)
	if err != nil {
		return "", err
	}
	if len(resources) == 0 {
		return "", nil
	}
	return resources[len(resources)-1].PhysicalID, nil
}

// ExtractResources interprets the results of a single page from a List() call, producing a slice of Resource entities.
func ExtractResources(page pagination.Page) ([]Resource, error) {
	casted := page.(ResourcePage).Body

	var response struct {
		Resources []Resource `mapstructure:"resources"`
	}
	err := mapstructure.Decode(casted, &response)

	var resources []interface{}
	switch casted.(type) {
	case map[string]interface{}:
		resources = casted.(map[string]interface{})["resources"].([]interface{})
	case map[string][]interface{}:
		resources = casted.(map[string][]interface{})["resources"]
	default:
		return response.Resources, fmt.Errorf("Unknown type: %v", reflect.TypeOf(casted))
	}

	for i, resourceRaw := range resources {
		resource := resourceRaw.(map[string]interface{})
		if date, ok := resource["updated_time"]; ok && date != nil {
			t, err := time.Parse(gophercloud.STACK_TIME_FMT, date.(string))
			if err != nil {
				return nil, err
			}
			response.Resources[i].UpdatedTime = t
		}
	}

	return response.Resources, err
}

// GetResult represents the result of a Get operation.
type GetResult struct {
	gophercloud.Result
}

// Extract returns a pointer to a Resource object and is called after a
// Get operation.
func (r GetResult) Extract() (*Resource, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res struct {
		Res *Resource `mapstructure:"resource"`
	}

	if err := mapstructure.Decode(r.Body, &res); err != nil {
		return nil, err
	}

	resource := r.Body.(map[string]interface{})["resource"].(map[string]interface{})

	if date, ok := resource["updated_time"]; ok && date != nil {
		t, err := time.Parse(gophercloud.STACK_TIME_FMT, date.(string))
		if err != nil {
			return nil, err
		}
		res.Res.UpdatedTime = t
	}

	return res.Res, nil
}

// MetadataResult represents the result of a Metadata operation.
type MetadataResult struct {
	gophercloud.Result
}

// Extract returns a map object and is called after a
// Metadata operation.
func (r MetadataResult) Extract() (map[string]string, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res struct {
		Meta map[string]string `mapstructure:"metadata"`
	}

	if err := mapstructure.Decode(r.Body, &res); err != nil {
		return nil, err
	}

	return res.Meta, nil
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
	if err != nil {
		return true, err
	}
	return len(rts) == 0, nil
}

// ExtractResourceTypes extracts and returns resource types.
func ExtractResourceTypes(page pagination.Page) ([]string, error) {
	var response struct {
		ResourceTypes []string `mapstructure:"resource_types"`
	}

	err := mapstructure.Decode(page.(ResourceTypePage).Body, &response)
	return response.ResourceTypes, err
}

// TypeSchema represents a stack resource schema.
type TypeSchema struct {
	Attributes   map[string]interface{} `mapstructure:"attributes"`
	Properties   map[string]interface{} `mapstrucutre:"properties"`
	ResourceType string                 `mapstructure:"resource_type"`
}

// SchemaResult represents the result of a Schema operation.
type SchemaResult struct {
	gophercloud.Result
}

// Extract returns a pointer to a TypeSchema object and is called after a
// Schema operation.
func (r SchemaResult) Extract() (*TypeSchema, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res TypeSchema

	if err := mapstructure.Decode(r.Body, &res); err != nil {
		return nil, err
	}

	return &res, nil
}

// TypeTemplate represents a stack resource template.
type TypeTemplate struct {
	HeatTemplateFormatVersion string
	Outputs                   map[string]interface{}
	Parameters                map[string]interface{}
	Resources                 map[string]interface{}
}

// TemplateResult represents the result of a Template operation.
type TemplateResult struct {
	gophercloud.Result
}

// Extract returns a pointer to a TypeTemplate object and is called after a
// Template operation.
func (r TemplateResult) Extract() (*TypeTemplate, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res TypeTemplate

	if err := mapstructure.Decode(r.Body, &res); err != nil {
		return nil, err
	}

	return &res, nil
}
