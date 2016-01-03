package throttle

import (
	"github.com/mitchellh/mapstructure"

	"github.com/rackspace/gophercloud"
)

// ConnectionThrottle represents the connection throttle configuration for a
// particular load balancer.
type ConnectionThrottle struct {
	MaxConnections int
}

// CreateResult represents the result of a create operation.
type CreateResult struct {
	gophercloud.ErrResult
}

// DeleteResult represents the result of a delete operation.
type DeleteResult struct {
	gophercloud.ErrResult
}

// GetResult represents the result of a get operation.
type GetResult struct {
	gophercloud.Result
}

// Extract interprets a GetResult as a SP, if possible.
func (r GetResult) Extract() (*ConnectionThrottle, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var response struct {
		CT ConnectionThrottle `mapstructure:"connectionThrottle"`
	}

	err := mapstructure.Decode(r.Body, &response)

	return &response.CT, err
}
