package buildinfo

import (
	"github.com/gophercloud/gophercloud"
)

// Revision represents the API/Engine revision of a Heat deployment.
type Revision struct {
	Revision string `json:"revision"`
}

// BuildInfo represents the build information for a Heat deployment.
type BuildInfo struct {
	API    Revision `json:"api"`
	Engine Revision `json:"engine"`
}

// GetResult represents the result of a Get operation.
type GetResult struct {
	gophercloud.Result
}

// Extract returns a pointer to a BuildInfo object and is called after a
// Get operation.
func (r GetResult) Extract() (*BuildInfo, error) {
	var s *BuildInfo
	err := r.ExtractInto(&s)
	return s, err
}
