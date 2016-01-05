package stacktemplates

import (
	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
)

// Template represents a stack template.
type Template struct {
	Description         string                 `mapstructure:"description"`
	HeatTemplateVersion string                 `mapstructure:"heat_template_version"`
	Parameters          map[string]interface{} `mapstructure:"parameters"`
	Resources           map[string]interface{} `mapstructure:"resources"`
}

// GetResult represents the result of a Get operation.
type GetResult struct {
	gophercloud.Result
}

// Extract returns a pointer to a Template object and is called after a
// Get operation.
func (r GetResult) Extract() (*Template, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res Template
	if err := mapstructure.Decode(r.Body, &res); err != nil {
		return nil, err
	}

	return &res, nil
}

// ValidatedTemplate represents the parsed object returned from a Validate request.
type ValidatedTemplate struct {
	Description string
	Parameters  map[string]interface{}
}

// ValidateResult represents the result of a Validate operation.
type ValidateResult struct {
	gophercloud.Result
}

// Extract returns a pointer to a ValidatedTemplate object and is called after a
// Validate operation.
func (r ValidateResult) Extract() (*ValidatedTemplate, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res ValidatedTemplate
	if err := mapstructure.Decode(r.Body, &res); err != nil {
		return nil, err
	}

	return &res, nil
}
