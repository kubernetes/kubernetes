package tokens

import (
	"time"

	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
)

// commonResult is the deferred result of a Create or a Get call.
type commonResult struct {
	gophercloud.Result
}

// Extract interprets a commonResult as a Token.
func (r commonResult) Extract() (*Token, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var response struct {
		Token struct {
			ExpiresAt string `mapstructure:"expires_at"`
		} `mapstructure:"token"`
	}

	var token Token

	// Parse the token itself from the stored headers.
	token.ID = r.Header.Get("X-Subject-Token")

	err := mapstructure.Decode(r.Body, &response)
	if err != nil {
		return nil, err
	}

	// Attempt to parse the timestamp.
	token.ExpiresAt, err = time.Parse(gophercloud.RFC3339Milli, response.Token.ExpiresAt)

	return &token, err
}

// CreateResult is the deferred response from a Create call.
type CreateResult struct {
	commonResult
}

// createErr quickly creates a CreateResult that reports an error.
func createErr(err error) CreateResult {
	return CreateResult{
		commonResult: commonResult{Result: gophercloud.Result{Err: err}},
	}
}

// GetResult is the deferred response from a Get call.
type GetResult struct {
	commonResult
}

// RevokeResult is the deferred response from a Revoke call.
type RevokeResult struct {
	commonResult
}

// Token is a string that grants a user access to a controlled set of services in an OpenStack provider.
// Each Token is valid for a set length of time.
type Token struct {
	// ID is the issued token.
	ID string

	// ExpiresAt is the timestamp at which this token will no longer be accepted.
	ExpiresAt time.Time
}
