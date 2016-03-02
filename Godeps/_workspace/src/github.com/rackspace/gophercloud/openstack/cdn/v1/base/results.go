package base

import (
	"errors"

	"github.com/rackspace/gophercloud"
)

// HomeDocument is a resource that contains all the resources for the CDN API.
type HomeDocument map[string]interface{}

// GetResult represents the result of a Get operation.
type GetResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts a home document resource.
func (r GetResult) Extract() (*HomeDocument, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	submap, ok := r.Body.(map[string]interface{})["resources"]
	if !ok {
		return nil, errors.New("Unexpected HomeDocument structure")
	}
	casted := HomeDocument(submap.(map[string]interface{}))

	return &casted, nil
}

// PingResult represents the result of a Ping operation.
type PingResult struct {
	gophercloud.ErrResult
}
