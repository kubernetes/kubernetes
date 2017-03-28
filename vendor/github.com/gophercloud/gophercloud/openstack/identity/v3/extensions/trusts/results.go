package trusts

import "github.com/gophercloud/gophercloud/openstack/identity/v3/tokens"

type TrusteeUser struct {
	ID string `json:"id"`
}

type TrustorUser struct {
	ID string `json:"id"`
}

type Trust struct {
	ID                 string      `json:"id"`
	Impersonation      bool        `json:"impersonation"`
	TrusteeUser        TrusteeUser `json:"trustee_user"`
	TrustorUser        TrustorUser `json:"trustor_user"`
	RedelegatedTrustID string      `json:"redelegated_trust_id"`
	RedelegationCount  int         `json:"redelegation_count"`
}

type Token struct {
	tokens.Token
	Trust Trust `json:"OS-TRUST:trust"`
}

type TokenExt struct {
	Token Token `json:"token"`
}
