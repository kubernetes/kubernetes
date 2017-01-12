package volumeactions

import "github.com/gophercloud/gophercloud"

// AttachResult contains the response body and error from a Get request.
type AttachResult struct {
	gophercloud.ErrResult
}

// BeginDetachingResult contains the response body and error from a Get request.
type BeginDetachingResult struct {
	gophercloud.ErrResult
}

// DetachResult contains the response body and error from a Get request.
type DetachResult struct {
	gophercloud.ErrResult
}

// ReserveResult contains the response body and error from a Get request.
type ReserveResult struct {
	gophercloud.ErrResult
}

// UnreserveResult contains the response body and error from a Get request.
type UnreserveResult struct {
	gophercloud.ErrResult
}

// TerminateConnectionResult contains the response body and error from a Get request.
type TerminateConnectionResult struct {
	gophercloud.ErrResult
}

type commonResult struct {
	gophercloud.Result
}

// Extract will get the Volume object out of the commonResult object.
func (r commonResult) Extract() (map[string]interface{}, error) {
	var s struct {
		ConnectionInfo map[string]interface{} `json:"connection_info"`
	}
	err := r.ExtractInto(&s)
	return s.ConnectionInfo, err
}

// InitializeConnectionResult contains the response body and error from a Get request.
type InitializeConnectionResult struct {
	commonResult
}

// ExtendSizeResult contains the response body and error from an ExtendSize request.
type ExtendSizeResult struct {
	gophercloud.ErrResult
}
