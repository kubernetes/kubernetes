// Package auth implements an interface for providing CFSSL
// authentication. This is meant to authenticate a client CFSSL to a
// remote CFSSL in order to prevent unauthorised use of the signature
// capabilities. This package provides both the interface and a
// standard HMAC-based implementation.
package auth

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io/ioutil"
	"os"
	"strings"
)

// An AuthenticatedRequest contains a request and authentication
// token. The Provider may determine whether to validate the timestamp
// and remote address.
type AuthenticatedRequest struct {
	// An Authenticator decides whether to use this field.
	Timestamp     int64  `json:"timestamp,omitempty"`
	RemoteAddress []byte `json:"remote_address,omitempty"`
	Token         []byte `json:"token"`
	Request       []byte `json:"request"`
}

// A Provider can generate tokens from a request and verify a
// request. The handling of additional authentication data (such as
// the IP address) is handled by the concrete type, as is any
// serialisation and state-keeping.
type Provider interface {
	Token(req []byte) (token []byte, err error)
	Verify(aReq *AuthenticatedRequest) bool
}

// Standard implements an HMAC-SHA-256 authentication provider. It may
// be supplied additional data at creation time that will be used as
// request || additional-data with the HMAC.
type Standard struct {
	key []byte
	ad  []byte
}

// New generates a new standard authentication provider from the key
// and additional data. The additional data will be used when
// generating a new token.
func New(key string, ad []byte) (*Standard, error) {
	if splitKey := strings.SplitN(key, ":", 2); len(splitKey) == 2 {
		switch splitKey[0] {
		case "env":
			key = os.Getenv(splitKey[1])
		case "file":
			data, err := ioutil.ReadFile(splitKey[1])
			if err != nil {
				return nil, err
			}
			key = string(data)
		default:
			return nil, fmt.Errorf("unknown key prefix: %s", splitKey[0])
		}
	}

	keyBytes, err := hex.DecodeString(key)
	if err != nil {
		return nil, err
	}

	return &Standard{keyBytes, ad}, nil
}

// Token generates a new authentication token from the request.
func (p Standard) Token(req []byte) (token []byte, err error) {
	h := hmac.New(sha256.New, p.key)
	h.Write(req)
	h.Write(p.ad)
	return h.Sum(nil), nil
}

// Verify determines whether an authenticated request is valid.
func (p Standard) Verify(ad *AuthenticatedRequest) bool {
	if ad == nil {
		return false
	}

	// Standard token generation returns no error.
	token, _ := p.Token(ad.Request)
	if len(ad.Token) != len(token) {
		return false
	}

	return hmac.Equal(token, ad.Token)
}
