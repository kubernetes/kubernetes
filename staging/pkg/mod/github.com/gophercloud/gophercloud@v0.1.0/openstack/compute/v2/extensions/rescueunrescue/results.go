package rescueunrescue

import "github.com/gophercloud/gophercloud"

type commonResult struct {
	gophercloud.Result
}

// RescueResult is the response from a Rescue operation. Call its Extract
// method to retrieve adminPass for a rescued server.
type RescueResult struct {
	commonResult
}

// UnrescueResult is the response from an UnRescue operation. Call its ExtractErr
// method to determine if the call succeeded or failed.
type UnrescueResult struct {
	gophercloud.ErrResult
}

// Extract interprets any RescueResult as an AdminPass, if possible.
func (r RescueResult) Extract() (string, error) {
	var s struct {
		AdminPass string `json:"adminPass"`
	}
	err := r.ExtractInto(&s)
	return s.AdminPass, err
}
