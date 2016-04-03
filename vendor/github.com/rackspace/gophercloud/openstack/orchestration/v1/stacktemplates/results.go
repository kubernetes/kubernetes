package stacktemplates

import (
	"encoding/json"
	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
)

// GetResult represents the result of a Get operation.
type GetResult struct {
	gophercloud.Result
}

// Extract returns the JSON template and is called after a Get operation.
func (r GetResult) Extract() ([]byte, error) {
	if r.Err != nil {
		return nil, r.Err
	}
	template, err := json.MarshalIndent(r.Body, "", "  ")
	if err != nil {
		return nil, err
	}
	return template, nil
}

// ValidatedTemplate represents the parsed object returned from a Validate request.
type ValidatedTemplate struct {
	Description     string                 `mapstructure:"Description"`
	Parameters      map[string]interface{} `mapstructure:"Parameters"`
	ParameterGroups map[string]interface{} `mapstructure:"ParameterGroups"`
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
