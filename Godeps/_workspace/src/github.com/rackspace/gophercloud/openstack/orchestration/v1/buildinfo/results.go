package buildinfo

import (
	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
)

// Revision represents the API/Engine revision of a Heat deployment.
type Revision struct {
	Revision string `mapstructure:"revision"`
}

// BuildInfo represents the build information for a Heat deployment.
type BuildInfo struct {
	API    Revision `mapstructure:"api"`
	Engine Revision `mapstructure:"engine"`
}

// GetResult represents the result of a Get operation.
type GetResult struct {
	gophercloud.Result
}

// Extract returns a pointer to a BuildInfo object and is called after a
// Get operation.
func (r GetResult) Extract() (*BuildInfo, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res BuildInfo
	if err := mapstructure.Decode(r.Body, &res); err != nil {
		return nil, err
	}

	return &res, nil
}
