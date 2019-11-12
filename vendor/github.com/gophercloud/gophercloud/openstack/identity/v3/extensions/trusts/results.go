package trusts

import "github.com/gophercloud/gophercloud"

type trustResult struct {
	gophercloud.Result
}

// CreateResult is the response from a Create operation. Call its Extract method
// to interpret it as a Trust.
type CreateResult struct {
	trustResult
}

// DeleteResult is the response from a Delete operation. Call its ExtractErr to
// determine if the request succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

// Extract interprets any trust result as a Trust.
func (r trustResult) Extract() (*Trust, error) {
	var s struct {
		Trust *Trust `json:"trust"`
	}
	err := r.ExtractInto(&s)
	return s.Trust, err
}

// TrusteeUser represents the trusted user ID of a trust.
type TrusteeUser struct {
	ID string `json:"id"`
}

// TrustorUser represents the trusting user ID of a trust.
type TrustorUser struct {
	ID string `json:"id"`
}

// Trust represents a delegated authorization request between two
// identities.
type Trust struct {
	ID                 string      `json:"id"`
	Impersonation      bool        `json:"impersonation"`
	TrusteeUser        TrusteeUser `json:"trustee_user"`
	TrustorUser        TrustorUser `json:"trustor_user"`
	RedelegatedTrustID string      `json:"redelegated_trust_id"`
	RedelegationCount  int         `json:"redelegation_count"`
}

// Role specifies a single role that is granted to a trustee.
type Role struct {
	ID   string `json:"id,omitempty"`
	Name string `json:"name,omitempty"`
}

// TokenExt represents an extension of the base token result.
type TokenExt struct {
	Trust Trust `json:"OS-TRUST:trust"`
}
