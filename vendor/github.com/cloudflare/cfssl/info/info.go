// Package info contains the definitions for the info endpoint
package info

// Req is the request struct for an info API request.
type Req struct {
	Label   string `json:"label"`
	Profile string `json:"profile"`
}

// Resp is the response for an Info API request.
type Resp struct {
	Certificate  string   `json:"certificate"`
	Usage        []string `json:"usages"`
	ExpiryString string   `json:"expiry"`
}
