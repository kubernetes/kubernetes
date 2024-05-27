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

	oidc "github.com/coreos/go-oidc"
	"github.com/libopenstorage/openstorage/pkg/grpcutil"
)

// OIDCAuthConfig configures an OIDC connection
type OIDCAuthConfig struct {
	// Issuer of the OIDC tokens
	// e.g. https://accounts.google.com
	Issuer string
	// ClientID is the client id provided by the OIDC
	ClientID string
	// SkipClientIDCheck skips a verification on tokens which are returned
	// from the OIDC without the client ID set
	SkipClientIDCheck bool
	// SkipIssuerCheck skips verification of the issuer URL.
	SkipIssuerCheck bool
	// UsernameClaim has the location of the unique id for the user.
	// If empty, "sub" will be used for the user name unique id.
	UsernameClaim UsernameClaimType
	// Namespace sets the namespace for all custom claims. For example
	// if the claims had the key: "https://mynamespace/roles", then
	// the namespace would be "https://mynamespace/".
	Namespace string
}

// OIDCAuthenticator is used to validate tokens with an OIDC
type OIDCAuthenticator struct {
	url           string
	provider      *oidc.Provider
	verifier      *oidc.IDTokenVerifier
	usernameClaim UsernameClaimType
	namespace     string
}

// NewOIDC returns a new OIDC authenticator
func NewOIDC(config *OIDCAuthConfig) (*OIDCAuthenticator, error) {
	ctx, cancel := grpcutil.WithDefaultTimeout(context.Background())
	defer cancel()

	p, err := oidc.NewProvider(ctx, config.Issuer)
	if err != nil {
		return nil, fmt.Errorf("Unable to communicate with OIDC provider %s: %v",
			config.Issuer,
			err)
	}

	v := p.Verifier(&oidc.Config{
		ClientID:          config.ClientID,
		SkipClientIDCheck: config.SkipClientIDCheck,
		SkipIssuerCheck:   config.SkipIssuerCheck,
	})
	return &OIDCAuthenticator{
		url:           config.Issuer,
		usernameClaim: config.UsernameClaim,
		namespace:     config.Namespace,
		provider:      p,
		verifier:      v,
	}, nil
}

// AuthenticateToken will verify the validity of the provided token with the OIDC
func (o *OIDCAuthenticator) AuthenticateToken(ctx context.Context, rawtoken string) (*Claims, error) {
	idToken, err := o.verifier.Verify(ctx, rawtoken)
	if err != nil {
		return nil, fmt.Errorf("Token failed validation: %v", err)
	}

	// Check for required claims
	var claims map[string]interface{}
	if err := idToken.Claims(&claims); err != nil {
		return nil, fmt.Errorf("Unable to get claim map from token: %v", err)
	}
	for _, requiredClaim := range requiredClaims {
		if _, ok := claims[requiredClaim]; !ok {
			// Claim missing
			return nil, fmt.Errorf("Required claim %v missing from token", requiredClaim)
		}
	}

	return o.parseClaims(claims)
}

// Username returns the configured unique id of the user
func (o *OIDCAuthenticator) Username(claims *Claims) string {
	return getUsername(o.usernameClaim, claims)
}

// This will let us unit test this function without having a real OIDC
func (o *OIDCAuthenticator) parseClaims(claims map[string]interface{}) (*Claims, error) {

	// If we have namespace set, then use it to get custom claims:
	if len(o.namespace) > 0 {
		for _, cc := range customClaims {
			// Check if there claims needed are under a namespace
			if v, ok := claims[o.namespace+cc]; ok {
				// Move it to the top of the json tree overwriting anything
				// there with the same name.
				claims[cc] = v
			}
		}
	}

	// Marshal into byte stream so that we can unmarshal into SDK Claims
	cbytes, err := json.Marshal(claims)
	if err != nil {
		return nil, fmt.Errorf("Internal error, unable to re-encode OIDC token claims: %v", err)
	}

	// Return claims
	var sdkClaims Claims
	if err := json.Unmarshal(cbytes, &sdkClaims); err != nil {
		return nil, fmt.Errorf("Unable to get claims from token: %v", err)
	}

	if err := validateUsername(o.usernameClaim, &sdkClaims); err != nil {
		return nil, err
	}

	return &sdkClaims, nil
}
