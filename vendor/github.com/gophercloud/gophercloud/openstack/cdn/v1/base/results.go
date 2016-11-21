package base

import "github.com/gophercloud/gophercloud"

// HomeDocument is a resource that contains all the resources for the CDN API.
type HomeDocument map[string]interface{}

// GetResult represents the result of a Get operation.
type GetResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts a home document resource.
func (r GetResult) Extract() (*HomeDocument, error) {
	var s HomeDocument
	err := r.ExtractInto(&s)
	return &s, err
}

// PingResult represents the result of a Ping operation.
type PingResult struct {
	gophercloud.ErrResult
}
