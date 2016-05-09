package gophercloud

import "time"

// AuthResults [deprecated] is a leftover type from the v0.x days. It was
// intended to describe common functionality among identity service results, but
// is not actually used anywhere.
type AuthResults interface {
	// TokenID returns the token's ID value from the authentication response.
	TokenID() (string, error)

	// ExpiresAt retrieves the token's expiration time.
	ExpiresAt() (time.Time, error)
}
