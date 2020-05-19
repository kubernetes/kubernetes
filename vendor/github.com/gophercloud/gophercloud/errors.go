package gophercloud

import (
	"fmt"
	"strings"
)

// BaseError is an error type that all other error types embed.
type BaseError struct {
	DefaultErrString string
	Info             string
}

func (e BaseError) Error() string {
	e.DefaultErrString = "An error occurred while executing a Gophercloud request."
	return e.choseErrString()
}

func (e BaseError) choseErrString() string {
	if e.Info != "" {
		return e.Info
	}
	return e.DefaultErrString
}

// ErrMissingInput is the error when input is required in a particular
// situation but not provided by the user
type ErrMissingInput struct {
	BaseError
	Argument string
}

func (e ErrMissingInput) Error() string {
	e.DefaultErrString = fmt.Sprintf("Missing input for argument [%s]", e.Argument)
	return e.choseErrString()
}

// ErrInvalidInput is an error type used for most non-HTTP Gophercloud errors.
type ErrInvalidInput struct {
	ErrMissingInput
	Value interface{}
}

func (e ErrInvalidInput) Error() string {
	e.DefaultErrString = fmt.Sprintf("Invalid input provided for argument [%s]: [%+v]", e.Argument, e.Value)
	return e.choseErrString()
}

// ErrMissingEnvironmentVariable is the error when environment variable is required
// in a particular situation but not provided by the user
type ErrMissingEnvironmentVariable struct {
	BaseError
	EnvironmentVariable string
}

func (e ErrMissingEnvironmentVariable) Error() string {
	e.DefaultErrString = fmt.Sprintf("Missing environment variable [%s]", e.EnvironmentVariable)
	return e.choseErrString()
}

// ErrMissingAnyoneOfEnvironmentVariables is the error when anyone of the environment variables
// is required in a particular situation but not provided by the user
type ErrMissingAnyoneOfEnvironmentVariables struct {
	BaseError
	EnvironmentVariables []string
}

func (e ErrMissingAnyoneOfEnvironmentVariables) Error() string {
	e.DefaultErrString = fmt.Sprintf(
		"Missing one of the following environment variables [%s]",
		strings.Join(e.EnvironmentVariables, ", "),
	)
	return e.choseErrString()
}

// ErrUnexpectedResponseCode is returned by the Request method when a response code other than
// those listed in OkCodes is encountered.
type ErrUnexpectedResponseCode struct {
	BaseError
	URL      string
	Method   string
	Expected []int
	Actual   int
	Body     []byte
}

func (e ErrUnexpectedResponseCode) Error() string {
	e.DefaultErrString = fmt.Sprintf(
		"Expected HTTP response code %v when accessing [%s %s], but got %d instead\n%s",
		e.Expected, e.Method, e.URL, e.Actual, e.Body,
	)
	return e.choseErrString()
}

// ErrDefault400 is the default error type returned on a 400 HTTP response code.
type ErrDefault400 struct {
	ErrUnexpectedResponseCode
}

// ErrDefault401 is the default error type returned on a 401 HTTP response code.
type ErrDefault401 struct {
	ErrUnexpectedResponseCode
}

// ErrDefault403 is the default error type returned on a 403 HTTP response code.
type ErrDefault403 struct {
	ErrUnexpectedResponseCode
}

// ErrDefault404 is the default error type returned on a 404 HTTP response code.
type ErrDefault404 struct {
	ErrUnexpectedResponseCode
}

// ErrDefault405 is the default error type returned on a 405 HTTP response code.
type ErrDefault405 struct {
	ErrUnexpectedResponseCode
}

// ErrDefault408 is the default error type returned on a 408 HTTP response code.
type ErrDefault408 struct {
	ErrUnexpectedResponseCode
}

// ErrDefault409 is the default error type returned on a 409 HTTP response code.
type ErrDefault409 struct {
	ErrUnexpectedResponseCode
}

// ErrDefault429 is the default error type returned on a 429 HTTP response code.
type ErrDefault429 struct {
	ErrUnexpectedResponseCode
}

// ErrDefault500 is the default error type returned on a 500 HTTP response code.
type ErrDefault500 struct {
	ErrUnexpectedResponseCode
}

// ErrDefault503 is the default error type returned on a 503 HTTP response code.
type ErrDefault503 struct {
	ErrUnexpectedResponseCode
}

func (e ErrDefault400) Error() string {
	e.DefaultErrString = fmt.Sprintf(
		"Bad request with: [%s %s], error message: %s",
		e.Method, e.URL, e.Body,
	)
	return e.choseErrString()
}
func (e ErrDefault401) Error() string {
	return "Authentication failed"
}
func (e ErrDefault403) Error() string {
	e.DefaultErrString = fmt.Sprintf(
		"Request forbidden: [%s %s], error message: %s",
		e.Method, e.URL, e.Body,
	)
	return e.choseErrString()
}
func (e ErrDefault404) Error() string {
	return "Resource not found"
}
func (e ErrDefault405) Error() string {
	return "Method not allowed"
}
func (e ErrDefault408) Error() string {
	return "The server timed out waiting for the request"
}
func (e ErrDefault429) Error() string {
	return "Too many requests have been sent in a given amount of time. Pause" +
		" requests, wait up to one minute, and try again."
}
func (e ErrDefault500) Error() string {
	return "Internal Server Error"
}
func (e ErrDefault503) Error() string {
	return "The service is currently unable to handle the request due to a temporary" +
		" overloading or maintenance. This is a temporary condition. Try again later."
}

// Err400er is the interface resource error types implement to override the error message
// from a 400 error.
type Err400er interface {
	Error400(ErrUnexpectedResponseCode) error
}

// Err401er is the interface resource error types implement to override the error message
// from a 401 error.
type Err401er interface {
	Error401(ErrUnexpectedResponseCode) error
}

// Err403er is the interface resource error types implement to override the error message
// from a 403 error.
type Err403er interface {
	Error403(ErrUnexpectedResponseCode) error
}

// Err404er is the interface resource error types implement to override the error message
// from a 404 error.
type Err404er interface {
	Error404(ErrUnexpectedResponseCode) error
}

// Err405er is the interface resource error types implement to override the error message
// from a 405 error.
type Err405er interface {
	Error405(ErrUnexpectedResponseCode) error
}

// Err408er is the interface resource error types implement to override the error message
// from a 408 error.
type Err408er interface {
	Error408(ErrUnexpectedResponseCode) error
}

// Err409er is the interface resource error types implement to override the error message
// from a 409 error.
type Err409er interface {
	Error409(ErrUnexpectedResponseCode) error
}

// Err429er is the interface resource error types implement to override the error message
// from a 429 error.
type Err429er interface {
	Error429(ErrUnexpectedResponseCode) error
}

// Err500er is the interface resource error types implement to override the error message
// from a 500 error.
type Err500er interface {
	Error500(ErrUnexpectedResponseCode) error
}

// Err503er is the interface resource error types implement to override the error message
// from a 503 error.
type Err503er interface {
	Error503(ErrUnexpectedResponseCode) error
}

// ErrTimeOut is the error type returned when an operations times out.
type ErrTimeOut struct {
	BaseError
}

func (e ErrTimeOut) Error() string {
	e.DefaultErrString = "A time out occurred"
	return e.choseErrString()
}

// ErrUnableToReauthenticate is the error type returned when reauthentication fails.
type ErrUnableToReauthenticate struct {
	BaseError
	ErrOriginal error
}

func (e ErrUnableToReauthenticate) Error() string {
	e.DefaultErrString = fmt.Sprintf("Unable to re-authenticate: %s", e.ErrOriginal)
	return e.choseErrString()
}

// ErrErrorAfterReauthentication is the error type returned when reauthentication
// succeeds, but an error occurs afterword (usually an HTTP error).
type ErrErrorAfterReauthentication struct {
	BaseError
	ErrOriginal error
}

func (e ErrErrorAfterReauthentication) Error() string {
	e.DefaultErrString = fmt.Sprintf("Successfully re-authenticated, but got error executing request: %s", e.ErrOriginal)
	return e.choseErrString()
}

// ErrServiceNotFound is returned when no service in a service catalog matches
// the provided EndpointOpts. This is generally returned by provider service
// factory methods like "NewComputeV2()" and can mean that a service is not
// enabled for your account.
type ErrServiceNotFound struct {
	BaseError
}

func (e ErrServiceNotFound) Error() string {
	e.DefaultErrString = "No suitable service could be found in the service catalog."
	return e.choseErrString()
}

// ErrEndpointNotFound is returned when no available endpoints match the
// provided EndpointOpts. This is also generally returned by provider service
// factory methods, and usually indicates that a region was specified
// incorrectly.
type ErrEndpointNotFound struct {
	BaseError
}

func (e ErrEndpointNotFound) Error() string {
	e.DefaultErrString = "No suitable endpoint could be found in the service catalog."
	return e.choseErrString()
}

// ErrResourceNotFound is the error when trying to retrieve a resource's
// ID by name and the resource doesn't exist.
type ErrResourceNotFound struct {
	BaseError
	Name         string
	ResourceType string
}

func (e ErrResourceNotFound) Error() string {
	e.DefaultErrString = fmt.Sprintf("Unable to find %s with name %s", e.ResourceType, e.Name)
	return e.choseErrString()
}

// ErrMultipleResourcesFound is the error when trying to retrieve a resource's
// ID by name and multiple resources have the user-provided name.
type ErrMultipleResourcesFound struct {
	BaseError
	Name         string
	Count        int
	ResourceType string
}

func (e ErrMultipleResourcesFound) Error() string {
	e.DefaultErrString = fmt.Sprintf("Found %d %ss matching %s", e.Count, e.ResourceType, e.Name)
	return e.choseErrString()
}

// ErrUnexpectedType is the error when an unexpected type is encountered
type ErrUnexpectedType struct {
	BaseError
	Expected string
	Actual   string
}

func (e ErrUnexpectedType) Error() string {
	e.DefaultErrString = fmt.Sprintf("Expected %s but got %s", e.Expected, e.Actual)
	return e.choseErrString()
}

func unacceptedAttributeErr(attribute string) string {
	return fmt.Sprintf("The base Identity V3 API does not accept authentication by %s", attribute)
}

func redundantWithTokenErr(attribute string) string {
	return fmt.Sprintf("%s may not be provided when authenticating with a TokenID", attribute)
}

func redundantWithUserID(attribute string) string {
	return fmt.Sprintf("%s may not be provided when authenticating with a UserID", attribute)
}

// ErrAPIKeyProvided indicates that an APIKey was provided but can't be used.
type ErrAPIKeyProvided struct{ BaseError }

func (e ErrAPIKeyProvided) Error() string {
	return unacceptedAttributeErr("APIKey")
}

// ErrTenantIDProvided indicates that a TenantID was provided but can't be used.
type ErrTenantIDProvided struct{ BaseError }

func (e ErrTenantIDProvided) Error() string {
	return unacceptedAttributeErr("TenantID")
}

// ErrTenantNameProvided indicates that a TenantName was provided but can't be used.
type ErrTenantNameProvided struct{ BaseError }

func (e ErrTenantNameProvided) Error() string {
	return unacceptedAttributeErr("TenantName")
}

// ErrUsernameWithToken indicates that a Username was provided, but token authentication is being used instead.
type ErrUsernameWithToken struct{ BaseError }

func (e ErrUsernameWithToken) Error() string {
	return redundantWithTokenErr("Username")
}

// ErrUserIDWithToken indicates that a UserID was provided, but token authentication is being used instead.
type ErrUserIDWithToken struct{ BaseError }

func (e ErrUserIDWithToken) Error() string {
	return redundantWithTokenErr("UserID")
}

// ErrDomainIDWithToken indicates that a DomainID was provided, but token authentication is being used instead.
type ErrDomainIDWithToken struct{ BaseError }

func (e ErrDomainIDWithToken) Error() string {
	return redundantWithTokenErr("DomainID")
}

// ErrDomainNameWithToken indicates that a DomainName was provided, but token authentication is being used instead.s
type ErrDomainNameWithToken struct{ BaseError }

func (e ErrDomainNameWithToken) Error() string {
	return redundantWithTokenErr("DomainName")
}

// ErrUsernameOrUserID indicates that neither username nor userID are specified, or both are at once.
type ErrUsernameOrUserID struct{ BaseError }

func (e ErrUsernameOrUserID) Error() string {
	return "Exactly one of Username and UserID must be provided for password authentication"
}

// ErrDomainIDWithUserID indicates that a DomainID was provided, but unnecessary because a UserID is being used.
type ErrDomainIDWithUserID struct{ BaseError }

func (e ErrDomainIDWithUserID) Error() string {
	return redundantWithUserID("DomainID")
}

// ErrDomainNameWithUserID indicates that a DomainName was provided, but unnecessary because a UserID is being used.
type ErrDomainNameWithUserID struct{ BaseError }

func (e ErrDomainNameWithUserID) Error() string {
	return redundantWithUserID("DomainName")
}

// ErrDomainIDOrDomainName indicates that a username was provided, but no domain to scope it.
// It may also indicate that both a DomainID and a DomainName were provided at once.
type ErrDomainIDOrDomainName struct{ BaseError }

func (e ErrDomainIDOrDomainName) Error() string {
	return "You must provide exactly one of DomainID or DomainName to authenticate by Username"
}

// ErrMissingPassword indicates that no password was provided and no token is available.
type ErrMissingPassword struct{ BaseError }

func (e ErrMissingPassword) Error() string {
	return "You must provide a password to authenticate"
}

// ErrScopeDomainIDOrDomainName indicates that a domain ID or Name was required in a Scope, but not present.
type ErrScopeDomainIDOrDomainName struct{ BaseError }

func (e ErrScopeDomainIDOrDomainName) Error() string {
	return "You must provide exactly one of DomainID or DomainName in a Scope with ProjectName"
}

// ErrScopeProjectIDOrProjectName indicates that both a ProjectID and a ProjectName were provided in a Scope.
type ErrScopeProjectIDOrProjectName struct{ BaseError }

func (e ErrScopeProjectIDOrProjectName) Error() string {
	return "You must provide at most one of ProjectID or ProjectName in a Scope"
}

// ErrScopeProjectIDAlone indicates that a ProjectID was provided with other constraints in a Scope.
type ErrScopeProjectIDAlone struct{ BaseError }

func (e ErrScopeProjectIDAlone) Error() string {
	return "ProjectID must be supplied alone in a Scope"
}

// ErrScopeEmpty indicates that no credentials were provided in a Scope.
type ErrScopeEmpty struct{ BaseError }

func (e ErrScopeEmpty) Error() string {
	return "You must provide either a Project or Domain in a Scope"
}

// ErrAppCredMissingSecret indicates that no Application Credential Secret was provided with Application Credential ID or Name
type ErrAppCredMissingSecret struct{ BaseError }

func (e ErrAppCredMissingSecret) Error() string {
	return "You must provide an Application Credential Secret"
}
