/*
Copyright 2019 Portworx

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package auth

import (
	"fmt"
)

// TokenGenerator allows for the creation of tokens
type TokenGenerator interface {
	// GetToken returns a token which can be used for
	// authentication and communication from node to node.
	GetToken(opts *Options) (string, error)

	// Issuer returns the token issuer for this generator necessary
	// for registering the authenticator in the SDK.
	Issuer() string

	// GetAuthenticator returns an authenticator for this issuer used by the SDK
	GetAuthenticator() (Authenticator, error)
}

var errAuthDisabled = fmt.Errorf("No authentication set")

// Default no auth
type noauth struct{}

// NoAuth returns the default no auth implementation
func NoAuth() *noauth {
	return &noauth{}
}

// Check for interface implementation
var _ TokenGenerator = &noauth{}

func (na *noauth) Issuer() string {
	return ""
}

func (na *noauth) GetAuthenticator() (Authenticator, error) {
	return nil, errAuthDisabled
}

func (na *noauth) GetToken(opts *Options) (string, error) {
	return "", nil
}
