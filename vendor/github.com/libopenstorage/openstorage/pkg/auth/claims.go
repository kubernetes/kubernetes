package auth

import "fmt"

const (
	// default type is sub
	UsernameClaimTypeDefault UsernameClaimType = ""
	// UsernameClaimTypeSubject requests to use "sub" as the claims for the
	// ID of the user
	UsernameClaimTypeSubject UsernameClaimType = "sub"
	// UsernameClaimTypeEmail requests to use "name" as the claims for the
	// ID of the user
	UsernameClaimTypeEmail UsernameClaimType = "email"
	// UsernameClaimTypeName requests to use "name" as the claims for the
	// ID of the user
	UsernameClaimTypeName UsernameClaimType = "name"
)

var (
	// Required claim keys
	requiredClaims = []string{"iss", "sub", "exp", "iat", "name", "email"}
	// Custom claims for OpenStorage
	customClaims = []string{"roles", "groups"}
)

// Claims provides information about the claims in the token
// See https://openid.net/specs/openid-connect-core-1_0.html#IDToken
// for more information.
type Claims struct {
	// Issuer is the token issuer. For selfsigned token do not prefix
	// with `https://`.
	Issuer string `json:"iss"`
	// Subject identifier. Unique ID of this account
	Subject string `json:"sub" yaml:"sub"`
	// Account name
	Name string `json:"name" yaml:"name"`
	// Account email
	Email string `json:"email" yaml:"email"`
	// Roles of this account
	Roles []string `json:"roles,omitempty" yaml:"roles,omitempty"`
	// (optional) Groups in which this account is part of
	Groups []string `json:"groups,omitempty" yaml:"groups,omitempty"`
}

// UsernameClaimType holds the claims type to be use as the unique id for the user
type UsernameClaimType string

// utility function to get username
func getUsername(usernameClaim UsernameClaimType, claims *Claims) string {
	switch usernameClaim {
	case UsernameClaimTypeEmail:
		return claims.Email
	case UsernameClaimTypeName:
		return claims.Name
	}
	return claims.Subject
}

func validateUsername(usernameClaim UsernameClaimType, claims *Claims) error {
	switch usernameClaim {
	case UsernameClaimTypeEmail:
		if claims.Email == "" {
			return fmt.Errorf("System set to use the value of email as the username," +
				" therefore the value of email in the token cannot be empty")
		}
	case UsernameClaimTypeName:
		if claims.Name == "" {
			return fmt.Errorf("System set to use the value of name as the username," +
				" therefore the value of name in the token cannot be empty")

		}
	default:
		if claims.Subject == "" {
			return fmt.Errorf("System set to use the value of sub as the username," +
				" therefore the value of sub in the token cannot be empty")
		}
	}

	return nil
}
