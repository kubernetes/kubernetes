package trusts

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

// TokenExt represents an extension of the base token result.
type TokenExt struct {
	Trust Trust `json:"OS-TRUST:trust"`
}
