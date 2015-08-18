// Package pkcs11uri provides helpers for parsing PKCS #11 URIs. These
// are currently specified in the draft RFC
// http://datatracker.ietf.org/doc/draft-pechanec-pkcs11uri/
//
// Note that the only supported pin source at this time is via a file.
package pkcs11uri

import (
	"io/ioutil"
	"net/url"
	"strings"

	"github.com/cloudflare/cfssl/errors"
	"github.com/cloudflare/cfssl/signer/pkcs11"
)

func setIfPresent(val url.Values, k string, target *string) {
	sv := val.Get(k)
	if sv != "" {
		*target = sv
	}
}

// ErrInvalidURI is returned if the PKCS #11 URI is invalid.
var ErrInvalidURI = errors.New(errors.PrivateKeyError, errors.ParseFailed)

// ParsePKCS11URI parses a PKCS #11 URI into a PKCS #11
// configuration. Note that the module path will override the module
// name if present.
func ParsePKCS11URI(uri string) (*pkcs11.Config, error) {
	u, err := url.Parse(uri)
	if err != nil || u.Scheme != "pkcs11" {
		return nil, ErrInvalidURI
	}

	c := new(pkcs11.Config)

	pk11PAttr, err := url.ParseQuery(u.Opaque)
	if err != nil {
		return nil, ErrInvalidURI
	}

	pk11QAttr, err := url.ParseQuery(u.RawQuery)
	if err != nil {
		return nil, ErrInvalidURI
	}
	setIfPresent(pk11PAttr, "token", &c.Token)
	setIfPresent(pk11QAttr, "module-name", &c.Module)
	setIfPresent(pk11QAttr, "module-path", &c.Module)
	setIfPresent(pk11QAttr, "pin-value", &c.PIN)
	setIfPresent(pk11PAttr, "slot-description", &c.Label)

	var pinSourceURI string
	setIfPresent(pk11QAttr, "pin-source", &pinSourceURI)
	if pinSourceURI == "" {
		return c, nil
	}

	pinURI, err := url.Parse(pinSourceURI)
	if pinURI.Opaque != "" && pinURI.Path == "" {
		pinURI.Path = pinURI.Opaque
	}
	if err != nil || pinURI.Scheme != "file" || pinURI.Path == "" {
		return nil, ErrInvalidURI
	}

	pin, err := ioutil.ReadFile(pinURI.Path)
	if err != nil {
		return nil, ErrInvalidURI
	}
	c.PIN = strings.TrimSpace(string(pin))

	return c, nil
}
