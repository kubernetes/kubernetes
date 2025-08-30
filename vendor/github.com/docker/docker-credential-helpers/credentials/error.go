package credentials

import (
	"errors"
	"strings"
)

const (
	// ErrCredentialsNotFound standardizes the not found error, so every helper returns
	// the same message and docker can handle it properly.
	errCredentialsNotFoundMessage = "credentials not found in native keychain"

	// ErrCredentialsMissingServerURL and ErrCredentialsMissingUsername standardize
	// invalid credentials or credentials management operations
	errCredentialsMissingServerURLMessage = "no credentials server URL"
	errCredentialsMissingUsernameMessage  = "no credentials username"
)

// errCredentialsNotFound represents an error
// raised when credentials are not in the store.
type errCredentialsNotFound struct{}

// Error returns the standard error message
// for when the credentials are not in the store.
func (errCredentialsNotFound) Error() string {
	return errCredentialsNotFoundMessage
}

// NotFound implements the [ErrNotFound][errdefs.ErrNotFound] interface.
//
// [errdefs.ErrNotFound]: https://pkg.go.dev/github.com/docker/docker@v24.0.1+incompatible/errdefs#ErrNotFound
func (errCredentialsNotFound) NotFound() {}

// NewErrCredentialsNotFound creates a new error
// for when the credentials are not in the store.
func NewErrCredentialsNotFound() error {
	return errCredentialsNotFound{}
}

// IsErrCredentialsNotFound returns true if the error
// was caused by not having a set of credentials in a store.
func IsErrCredentialsNotFound(err error) bool {
	var target errCredentialsNotFound
	return errors.As(err, &target)
}

// IsErrCredentialsNotFoundMessage returns true if the error
// was caused by not having a set of credentials in a store.
//
// This function helps to check messages returned by an
// external program via its standard output.
func IsErrCredentialsNotFoundMessage(err string) bool {
	return strings.TrimSpace(err) == errCredentialsNotFoundMessage
}

// errCredentialsMissingServerURL represents an error raised
// when the credentials object has no server URL or when no
// server URL is provided to a credentials operation requiring
// one.
type errCredentialsMissingServerURL struct{}

func (errCredentialsMissingServerURL) Error() string {
	return errCredentialsMissingServerURLMessage
}

// InvalidParameter implements the [ErrInvalidParameter][errdefs.ErrInvalidParameter]
// interface.
//
// [errdefs.ErrInvalidParameter]: https://pkg.go.dev/github.com/docker/docker@v24.0.1+incompatible/errdefs#ErrInvalidParameter
func (errCredentialsMissingServerURL) InvalidParameter() {}

// errCredentialsMissingUsername represents an error raised
// when the credentials object has no username or when no
// username is provided to a credentials operation requiring
// one.
type errCredentialsMissingUsername struct{}

func (errCredentialsMissingUsername) Error() string {
	return errCredentialsMissingUsernameMessage
}

// InvalidParameter implements the [ErrInvalidParameter][errdefs.ErrInvalidParameter]
// interface.
//
// [errdefs.ErrInvalidParameter]: https://pkg.go.dev/github.com/docker/docker@v24.0.1+incompatible/errdefs#ErrInvalidParameter
func (errCredentialsMissingUsername) InvalidParameter() {}

// NewErrCredentialsMissingServerURL creates a new error for
// errCredentialsMissingServerURL.
func NewErrCredentialsMissingServerURL() error {
	return errCredentialsMissingServerURL{}
}

// NewErrCredentialsMissingUsername creates a new error for
// errCredentialsMissingUsername.
func NewErrCredentialsMissingUsername() error {
	return errCredentialsMissingUsername{}
}

// IsCredentialsMissingServerURL returns true if the error
// was an errCredentialsMissingServerURL.
func IsCredentialsMissingServerURL(err error) bool {
	var target errCredentialsMissingServerURL
	return errors.As(err, &target)
}

// IsCredentialsMissingServerURLMessage checks for an
// errCredentialsMissingServerURL in the error message.
func IsCredentialsMissingServerURLMessage(err string) bool {
	return strings.TrimSpace(err) == errCredentialsMissingServerURLMessage
}

// IsCredentialsMissingUsername returns true if the error
// was an errCredentialsMissingUsername.
func IsCredentialsMissingUsername(err error) bool {
	var target errCredentialsMissingUsername
	return errors.As(err, &target)
}

// IsCredentialsMissingUsernameMessage checks for an
// errCredentialsMissingUsername in the error message.
func IsCredentialsMissingUsernameMessage(err string) bool {
	return strings.TrimSpace(err) == errCredentialsMissingUsernameMessage
}
