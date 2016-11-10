package tokens

import (
	"errors"
	"fmt"
)

func unacceptedAttributeErr(attribute string) error {
	return fmt.Errorf("The base Identity V3 API does not accept authentication by %s", attribute)
}

func redundantWithTokenErr(attribute string) error {
	return fmt.Errorf("%s may not be provided when authenticating with a TokenID", attribute)
}

func redundantWithUserID(attribute string) error {
	return fmt.Errorf("%s may not be provided when authenticating with a UserID", attribute)
}

var (
	// ErrAPIKeyProvided indicates that an APIKey was provided but can't be used.
	ErrAPIKeyProvided = unacceptedAttributeErr("APIKey")

	// ErrTenantIDProvided indicates that a TenantID was provided but can't be used.
	ErrTenantIDProvided = unacceptedAttributeErr("TenantID")

	// ErrTenantNameProvided indicates that a TenantName was provided but can't be used.
	ErrTenantNameProvided = unacceptedAttributeErr("TenantName")

	// ErrUsernameWithToken indicates that a Username was provided, but token authentication is being used instead.
	ErrUsernameWithToken = redundantWithTokenErr("Username")

	// ErrUserIDWithToken indicates that a UserID was provided, but token authentication is being used instead.
	ErrUserIDWithToken = redundantWithTokenErr("UserID")

	// ErrDomainIDWithToken indicates that a DomainID was provided, but token authentication is being used instead.
	ErrDomainIDWithToken = redundantWithTokenErr("DomainID")

	// ErrDomainNameWithToken indicates that a DomainName was provided, but token authentication is being used instead.s
	ErrDomainNameWithToken = redundantWithTokenErr("DomainName")

	// ErrUsernameOrUserID indicates that neither username nor userID are specified, or both are at once.
	ErrUsernameOrUserID = errors.New("Exactly one of Username and UserID must be provided for password authentication")

	// ErrDomainIDWithUserID indicates that a DomainID was provided, but unnecessary because a UserID is being used.
	ErrDomainIDWithUserID = redundantWithUserID("DomainID")

	// ErrDomainNameWithUserID indicates that a DomainName was provided, but unnecessary because a UserID is being used.
	ErrDomainNameWithUserID = redundantWithUserID("DomainName")

	// ErrDomainIDOrDomainName indicates that a username was provided, but no domain to scope it.
	// It may also indicate that both a DomainID and a DomainName were provided at once.
	ErrDomainIDOrDomainName = errors.New("You must provide exactly one of DomainID or DomainName to authenticate by Username")

	// ErrMissingPassword indicates that no password and no token were provided and no token is available.
	ErrMissingPassword = errors.New("You must provide a password or a token to authenticate")

	// ErrScopeDomainIDOrDomainName indicates that a domain ID or Name was required in a Scope, but not present.
	ErrScopeDomainIDOrDomainName = errors.New("You must provide exactly one of DomainID or DomainName in a Scope with ProjectName")

	// ErrScopeProjectIDOrProjectName indicates that both a ProjectID and a ProjectName were provided in a Scope.
	ErrScopeProjectIDOrProjectName = errors.New("You must provide at most one of ProjectID or ProjectName in a Scope")

	// ErrScopeProjectIDAlone indicates that a ProjectID was provided with other constraints in a Scope.
	ErrScopeProjectIDAlone = errors.New("ProjectID must be supplied alone in a Scope")

	// ErrScopeDomainName indicates that a DomainName was provided alone in a Scope.
	ErrScopeDomainName = errors.New("DomainName must be supplied with a ProjectName or ProjectID in a Scope.")

	// ErrScopeEmpty indicates that no credentials were provided in a Scope.
	ErrScopeEmpty = errors.New("You must provide either a Project or Domain in a Scope")
)
