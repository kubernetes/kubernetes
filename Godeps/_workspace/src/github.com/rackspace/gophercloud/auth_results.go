package gophercloud

import "time"

// AuthResults encapsulates the raw results from an authentication request. As OpenStack allows
// extensions to influence the structure returned in ways that Gophercloud cannot predict at
// compile-time, you should use type-safe accessors to work with the data represented by this type,
// such as ServiceCatalog and TokenID.
type AuthResults interface {
	// TokenID returns the token's ID value from the authentication response.
	TokenID() (string, error)

	// ExpiresAt retrieves the token's expiration time.
	ExpiresAt() (time.Time, error)
}
