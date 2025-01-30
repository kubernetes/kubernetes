package runhcs

import "net/url"

const (
	SafePipePrefix = `\\.\pipe\ProtectedPrefix\Administrators\`
)

// ShimSuccess is the byte stream returned on a successful operation.
var ShimSuccess = []byte{0, 'O', 'K', 0}

func SafePipePath(name string) string {
	// Use a pipe in the Administrators protected prefixed to prevent malicious
	// squatting.
	return SafePipePrefix + url.PathEscape(name)
}
