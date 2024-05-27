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
	"context"
	"encoding/json"
	"fmt"
	"strings"

	jwt "github.com/dgrijalva/jwt-go"
)

// JwtAuthConfig provides JwtAuthenticator the keys to validate the token
type JwtAuthConfig struct {
	// SharedSecret in byte array form
	SharedSecret []byte
	// RsaPublicPem is the contents of the RSA public key file
	RsaPublicPem []byte
	// ECDSPublicPem is the contents of the ECDS public key file
	ECDSPublicPem []byte
	// UsernameClaim has the location of the unique id for the user.
	// If empty, "sub" will be used for the user name unique id.
	UsernameClaim UsernameClaimType
}

// JwtAuthenticator definition. It contains the raw bytes of the keys and their
// objects as returned by the Jwt package
type JwtAuthenticator struct {
	config          JwtAuthConfig
	rsaKey          interface{}
	ecdsKey         interface{}
	sharedSecretKey interface{}
	usernameClaim   UsernameClaimType
}

// New returns a JwtAuthenticator
func NewJwtAuth(config *JwtAuthConfig) (*JwtAuthenticator, error) {

	if config == nil {
		return nil, fmt.Errorf("Must provide configuration")
	}

	// Check at least one is set
	if len(config.SharedSecret) == 0 &&
		len(config.RsaPublicPem) == 0 &&
		len(config.ECDSPublicPem) == 0 {
		return nil, fmt.Errorf("Server was passed empty authentication information with no shared secret or pem files set")
	}

	authenticator := &JwtAuthenticator{
		config:        *config,
		usernameClaim: config.UsernameClaim,
	}

	var err error
	if len(config.SharedSecret) != 0 {
		authenticator.sharedSecretKey = config.SharedSecret
	}
	if len(config.RsaPublicPem) != 0 {
		authenticator.rsaKey, err = jwt.ParseRSAPublicKeyFromPEM(config.RsaPublicPem)
		if err != nil {
			return nil, fmt.Errorf("Unable to parse rsa public key: %v", err)
		}
	}
	if len(config.ECDSPublicPem) != 0 {
		authenticator.ecdsKey, err = jwt.ParseECPublicKeyFromPEM(config.ECDSPublicPem)
		if err != nil {
			return nil, fmt.Errorf("Unable to parse ecds public key: %v", err)
		}
	}

	return authenticator, nil
}

// AuthenticateToken determines if a token is valid and if it is, returns
// the information in the claims.
func (j *JwtAuthenticator) AuthenticateToken(ctx context.Context, rawtoken string) (*Claims, error) {

	// Parse token
	token, err := jwt.Parse(rawtoken, func(token *jwt.Token) (interface{}, error) {

		// Verify Method
		if strings.HasPrefix(token.Method.Alg(), "RS") {
			// RS256, RS384, or RS512
			return j.rsaKey, nil
		} else if strings.HasPrefix(token.Method.Alg(), "ES") {
			// ES256, ES384, or ES512
			return j.ecdsKey, nil
		} else if strings.HasPrefix(token.Method.Alg(), "HS") {
			// HS256, HS384, or HS512
			return j.sharedSecretKey, nil
		}
		return nil, fmt.Errorf("Unknown token algorithm: %s", token.Method.Alg())
	})
	if err != nil {
		return nil, err
	}

	if !token.Valid {
		return nil, fmt.Errorf("Token failed validation")
	}

	// Get claims
	claims, ok := token.Claims.(jwt.MapClaims)
	if claims == nil || !ok {
		return nil, fmt.Errorf("No claims found in token")
	}

	// Check for required claims
	for _, requiredClaim := range requiredClaims {
		if _, ok := claims[requiredClaim]; !ok {
			// Claim missing
			return nil, fmt.Errorf("Required claim %v missing from token", requiredClaim)
		}
	}

	// Token now has been verified.
	// Claims holds all the authorization information.
	// Here we need to first decode it then unmarshal it from JSON
	parts := strings.Split(token.Raw, ".")
	claimBytes, err := jwt.DecodeSegment(parts[1])
	if err != nil {
		return nil, fmt.Errorf("Failed to decode claims: %v", err)
	}
	var sdkClaims Claims
	err = json.Unmarshal(claimBytes, &sdkClaims)
	if err != nil {
		return nil, fmt.Errorf("Unable to get sdkclaims: %v", err)
	}

	if err := validateUsername(j.usernameClaim, &sdkClaims); err != nil {
		return nil, err
	}

	return &sdkClaims, nil
}

func (j *JwtAuthenticator) Username(claims *Claims) string {
	return getUsername(j.usernameClaim, claims)
}
