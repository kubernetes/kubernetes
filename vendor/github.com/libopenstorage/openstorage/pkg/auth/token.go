/*
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
	"encoding/json"
	"fmt"
	"strings"
	"time"

	jwt "github.com/dgrijalva/jwt-go"
	"github.com/grpc-ecosystem/go-grpc-middleware/util/metautils"
)

const (
	authorizationHeader = "authorization"
)

// Options provide any options to apply to the token
type Options struct {
	// Expiration time in Unix format as per JWT standard
	Expiration int64

	// IATSubtract is the time duration you would like to remove from
	// the token IAT (Issue At Time). This is useful as a guard against
	// NTP drift within a cluster. Without this option, your token may
	// be denied due to the IAT being greater than the current time.
	IATSubtract time.Duration
}

// TokenClaims returns the claims for the raw JWT token.
func TokenClaims(rawtoken string) (*Claims, error) {
	parts := strings.Split(rawtoken, ".")

	// There are supposed to be three parts for the token
	if len(parts) < 3 {
		return nil, fmt.Errorf("Token is invalid: %v", rawtoken)
	}

	// Access claims in the token
	claimBytes, err := jwt.DecodeSegment(parts[1])
	if err != nil {
		return nil, fmt.Errorf("Failed to decode claims: %v", err)
	}
	var claims *Claims

	// Unmarshal claims
	err = json.Unmarshal(claimBytes, &claims)
	if err != nil {
		return nil, fmt.Errorf("Unable to get information from the claims in the token: %v", err)
	}

	return claims, nil
}

// TokenIssuer returns the issuer for the raw JWT token.
func TokenIssuer(rawtoken string) (string, error) {
	claims, err := TokenClaims(rawtoken)
	if err != nil {
		return "", err
	}

	// Return issuer
	if len(claims.Issuer) != 0 {
		return claims.Issuer, nil
	} else {
		return "", fmt.Errorf("Issuer was not specified in the token")
	}
}

// IsJwtToken returns true if the provided string is a valid jwt token
func IsJwtToken(authstring string) bool {
	_, _, err := new(jwt.Parser).ParseUnverified(authstring, jwt.MapClaims{})
	return err == nil
}

// Token returns a signed JWT containing the claims provided
func Token(
	claims *Claims,
	signature *Signature,
	options *Options,
) (string, error) {

	mapclaims := jwt.MapClaims{
		"sub":   claims.Subject,
		"iss":   claims.Issuer,
		"email": claims.Email,
		"name":  claims.Name,
		"roles": claims.Roles,
		"iat":   time.Now().Add(-options.IATSubtract).Unix(),
		"exp":   options.Expiration,
	}
	if claims.Groups != nil {
		mapclaims["groups"] = claims.Groups
	}
	token := jwt.NewWithClaims(signature.Type, mapclaims)
	signedtoken, err := token.SignedString(signature.Key)
	if err != nil {
		return "", err
	}

	return signedtoken, nil
}

func IsGuest(ctx context.Context) bool {
	return metautils.ExtractIncoming(ctx).Get(authorizationHeader) == ""
}
