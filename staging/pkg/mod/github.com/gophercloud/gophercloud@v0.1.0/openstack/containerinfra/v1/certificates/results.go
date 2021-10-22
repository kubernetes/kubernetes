package certificates

import (
	"github.com/gophercloud/gophercloud"
)

type commonResult struct {
	gophercloud.Result
}

// GetResult is the response of a Get operations.
type GetResult struct {
	commonResult
}

// CreateResult is the response of a Create operations.
type CreateResult struct {
	commonResult
}

// UpdateResult is the response of an Update operations.
type UpdateResult struct {
	gophercloud.ErrResult
}

// Extract is a function that accepts a result and extracts a certificate resource.
func (r commonResult) Extract() (*Certificate, error) {
	var s *Certificate
	err := r.ExtractInto(&s)
	return s, err
}

// Represents a Certificate
type Certificate struct {
	ClusterUUID string             `json:"cluster_uuid"`
	BayUUID     string             `json:"bay_uuid"`
	Links       []gophercloud.Link `json:"links"`
	PEM         string             `json:"pem"`
	CSR         string             `json:"csr"`
}
