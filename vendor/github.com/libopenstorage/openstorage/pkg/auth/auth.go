/*
Package auth can be used for authentication and authorization
Copyright 2018 Portworx

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
	"context"

	"github.com/libopenstorage/openstorage/pkg/correlation"
)

func init() {
	correlation.RegisterComponent(correlation.ComponentAuth)
}

const (
	systemGuestRoleName = "system.guest"
)

var (
	systemTokenInst TokenGenerator = &noauth{}

	// Inst returns the instance of system token manager.
	// This function can be overridden for testing purposes
	InitSystemTokenManager = func(tg TokenGenerator) {
		systemTokenInst = tg
	}

	// SystemTokenManagerInst returns the systemTokenManager instance
	SystemTokenManagerInst = func() TokenGenerator {
		return systemTokenInst
	}
)

// Authenticator interface validates and extracts the claims from a raw token
type Authenticator interface {
	// AuthenticateToken validates the token and returns the claims
	AuthenticateToken(context.Context, string) (*Claims, error)

	// Username returns the unique id according to the configuration. Default
	// it will return the value for "sub" in the token claims, but it can be
	// configured to return the email or name as the unique id.
	Username(*Claims) string
}

// Enabled returns whether or not auth is enabled.
func Enabled() bool {
	return len(systemTokenInst.Issuer()) != 0
}
