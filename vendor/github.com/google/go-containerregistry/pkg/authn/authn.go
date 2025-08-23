// Copyright 2018 Google LLC All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package authn

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
)

// Authenticator is used to authenticate Docker transports.
type Authenticator interface {
	// Authorization returns the value to use in an http transport's Authorization header.
	Authorization() (*AuthConfig, error)
}

// ContextAuthenticator is like Authenticator, but allows for context to be passed in.
type ContextAuthenticator interface {
	// Authorization returns the value to use in an http transport's Authorization header.
	AuthorizationContext(context.Context) (*AuthConfig, error)
}

// Authorization calls AuthorizationContext with ctx if the given [Authenticator] implements [ContextAuthenticator],
// otherwise it calls Resolve with the given [Resource].
func Authorization(ctx context.Context, authn Authenticator) (*AuthConfig, error) {
	if actx, ok := authn.(ContextAuthenticator); ok {
		return actx.AuthorizationContext(ctx)
	}

	return authn.Authorization()
}

// AuthConfig contains authorization information for connecting to a Registry
// Inlined what we use from github.com/docker/cli/cli/config/types
type AuthConfig struct {
	Username string `json:"username,omitempty"`
	Password string `json:"password,omitempty"`
	Auth     string `json:"auth,omitempty"`

	// IdentityToken is used to authenticate the user and get
	// an access token for the registry.
	IdentityToken string `json:"identitytoken,omitempty"`

	// RegistryToken is a bearer token to be sent to a registry
	RegistryToken string `json:"registrytoken,omitempty"`
}

// This is effectively a copy of the type AuthConfig. This simplifies
// JSON unmarshalling since AuthConfig methods are not inherited
type authConfig AuthConfig

// UnmarshalJSON implements json.Unmarshaler
func (a *AuthConfig) UnmarshalJSON(data []byte) error {
	var shadow authConfig
	err := json.Unmarshal(data, &shadow)
	if err != nil {
		return err
	}

	*a = (AuthConfig)(shadow)

	if len(shadow.Auth) != 0 {
		var derr error
		a.Username, a.Password, derr = decodeDockerConfigFieldAuth(shadow.Auth)
		if derr != nil {
			err = fmt.Errorf("unable to decode auth field: %w", derr)
		}
	} else if len(a.Username) != 0 && len(a.Password) != 0 {
		a.Auth = encodeDockerConfigFieldAuth(shadow.Username, shadow.Password)
	}

	return err
}

// MarshalJSON implements json.Marshaler
func (a AuthConfig) MarshalJSON() ([]byte, error) {
	shadow := (authConfig)(a)
	shadow.Auth = encodeDockerConfigFieldAuth(shadow.Username, shadow.Password)
	return json.Marshal(shadow)
}

// decodeDockerConfigFieldAuth deserializes the "auth" field from dockercfg into a
// username and a password. The format of the auth field is base64(<username>:<password>).
//
// From https://github.com/kubernetes/kubernetes/blob/75e49ec824b183288e1dbaccfd7dbe77d89db381/pkg/credentialprovider/config.go
// Copyright 2014 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0
func decodeDockerConfigFieldAuth(field string) (username, password string, err error) {
	var decoded []byte
	// StdEncoding can only decode padded string
	// RawStdEncoding can only decode unpadded string
	if strings.HasSuffix(strings.TrimSpace(field), "=") {
		// decode padded data
		decoded, err = base64.StdEncoding.DecodeString(field)
	} else {
		// decode unpadded data
		decoded, err = base64.RawStdEncoding.DecodeString(field)
	}

	if err != nil {
		return
	}

	parts := strings.SplitN(string(decoded), ":", 2)
	if len(parts) != 2 {
		err = fmt.Errorf("must be formatted as base64(username:password)")
		return
	}

	username = parts[0]
	password = parts[1]

	return
}

func encodeDockerConfigFieldAuth(username, password string) string {
	return base64.StdEncoding.EncodeToString([]byte(username + ":" + password))
}
