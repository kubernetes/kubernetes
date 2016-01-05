package tokens

import (
	"errors"
	"fmt"
)

var (
	// ErrUserIDProvided is returned if you attempt to authenticate with a UserID.
	ErrUserIDProvided = unacceptedAttributeErr("UserID")

	// ErrAPIKeyProvided is returned if you attempt to authenticate with an APIKey.
	ErrAPIKeyProvided = unacceptedAttributeErr("APIKey")

	// ErrDomainIDProvided is returned if you attempt to authenticate with a DomainID.
	ErrDomainIDProvided = unacceptedAttributeErr("DomainID")

	// ErrDomainNameProvided is returned if you attempt to authenticate with a DomainName.
	ErrDomainNameProvided = unacceptedAttributeErr("DomainName")

	// ErrUsernameRequired is returned if you attempt to authenticate without a Username.
	ErrUsernameRequired = errors.New("You must supply a Username in your AuthOptions.")

	// ErrPasswordRequired is returned if you don't provide a password.
	ErrPasswordRequired = errors.New("Please supply a Password in your AuthOptions.")
)

func unacceptedAttributeErr(attribute string) error {
	return fmt.Errorf("The base Identity V2 API does not accept authentication by %s", attribute)
}
