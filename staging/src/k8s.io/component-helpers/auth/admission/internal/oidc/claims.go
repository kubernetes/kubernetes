/*
Copyright The Kubernetes Authors.

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

package oidc

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

type unverifiedClaims struct {
	Issuer string `json:"iss"`
}

// parseUnverifiedClaims reads an untrusted issuer hint from a compact JWT.
// Authorization still requires signature verification.
func parseUnverifiedClaims(token string) (unverifiedClaims, error) {
	if strings.HasPrefix(strings.TrimSpace(token), "{") {
		return unverifiedClaims{}, errors.New("token is not a compact JWT")
	}
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return unverifiedClaims{}, errors.New("malformed token")
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return unverifiedClaims{}, fmt.Errorf("decoding token payload: %w", err)
	}
	var claims unverifiedClaims
	if err := json.Unmarshal(payload, &claims); err != nil {
		return unverifiedClaims{}, fmt.Errorf("parsing token payload: %w", err)
	}
	return claims, nil
}
