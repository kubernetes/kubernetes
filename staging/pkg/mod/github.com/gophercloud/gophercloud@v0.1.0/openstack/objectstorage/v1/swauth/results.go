package swauth

import (
	"github.com/gophercloud/gophercloud"
)

// GetAuthResult contains the response from the Auth request. Call its Extract
// method to interpret it as an AuthResult.
type GetAuthResult struct {
	gophercloud.HeaderResult
}

// AuthResult contains the authentication information from a Swauth
// authentication request.
type AuthResult struct {
	Token      string `json:"X-Auth-Token"`
	StorageURL string `json:"X-Storage-Url"`
	CDNURL     string `json:"X-CDN-Management-Url"`
}

// Extract is a method that attempts to interpret any Swauth authentication
// response as a AuthResult struct.
func (r GetAuthResult) Extract() (*AuthResult, error) {
	var s *AuthResult
	err := r.ExtractInto(&s)
	return s, err
}
