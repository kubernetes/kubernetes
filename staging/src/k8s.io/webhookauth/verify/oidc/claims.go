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

// unverifiedClaims is the minimal, UNVERIFIED view of a JWT payload used only for
// the cheap issuer pre-check in AuthenticateToken. The token is never trusted on
// the strength of this parse: its signature and all standard claims are
// re-verified by go-oidc against the fetched keys for every request.
type unverifiedClaims struct {
	Issuer string `json:"iss"`
}

// parseUnverifiedClaims decodes the payload segment of a compact JWS WITHOUT
// verifying its signature. It exists solely for the issuer pre-check; the value
// it returns is never trusted.
func parseUnverifiedClaims(token string) (unverifiedClaims, error) {
	parts := strings.Split(token, ".")
	if len(parts) < 2 {
		return unverifiedClaims{}, errors.New("token is not a JWT")
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
