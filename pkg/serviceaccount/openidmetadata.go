/*
Copyright 2019 The Kubernetes Authors.

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

package serviceaccount

import (
	"context"
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rsa"
	"encoding/json"
	"fmt"
	"net/url"
	"sync/atomic"

	jose "gopkg.in/go-jose/go-jose.v2"

	"k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
)

const (
	// OpenIDConfigPath is the URL path at which the API server serves
	// an OIDC Provider Configuration Information document, corresponding
	// to the Kubernetes Service Account key issuer.
	// https://openid.net/specs/openid-connect-discovery-1_0.html
	OpenIDConfigPath = "/.well-known/openid-configuration"

	// JWKSPath is the URL path at which the API server serves a JWKS
	// containing the public keys that may be used to sign Kubernetes
	// Service Account keys.
	JWKSPath = "/openid/v1/jwks"
)

// OpenIDMetadataProvider returns pre-rendered responses for OIDC discovery endpoints.
type OpenIDMetadataProvider interface {
	GetConfigJSON() (json []byte, maxAge int)
	GetKeysetJSON() (json []byte, maxAge int)
}

type openidConfigProvider struct {
	issuerURL, jwksURI string
	pubKeyGetter       PublicKeysGetter
	config             atomic.Pointer[openidConfig]
}
type openidConfig struct {
	configJSON []byte
	keysetJSON []byte
}

func (p *openidConfigProvider) GetConfigJSON() ([]byte, int) {
	return p.config.Load().configJSON, p.pubKeyGetter.GetCacheAgeMaxSeconds()
}
func (p *openidConfigProvider) GetKeysetJSON() ([]byte, int) {
	return p.config.Load().keysetJSON, p.pubKeyGetter.GetCacheAgeMaxSeconds()
}
func (p *openidConfigProvider) Enqueue() {
	err := p.Update()
	if err != nil {
		klog.ErrorS(err, "failed to update openid config metadata")
	}
}
func (p *openidConfigProvider) Update() error {
	pubKeys := []PublicKey{}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	unfilteredPubKeys := p.pubKeyGetter.GetPublicKeys(ctx, "")
	for _, key := range unfilteredPubKeys {
		if !key.ExcludeFromOIDCDiscovery {
			pubKeys = append(pubKeys, key)
		}
	}

	if len(pubKeys) == 0 {
		return fmt.Errorf("no keys provided for validating keyset")
	}
	configJSON, err := openIDConfigJSON(p.issuerURL, p.jwksURI, pubKeys)
	if err != nil {
		return fmt.Errorf("could not marshal issuer discovery JSON, error: %w", err)
	}
	keysetJSON, err := openIDKeysetJSON(pubKeys)
	if err != nil {
		return fmt.Errorf("could not marshal issuer keys JSON, error: %w", err)
	}
	p.config.Store(&openidConfig{
		configJSON: configJSON,
		keysetJSON: keysetJSON,
	})
	return nil
}

// NewOpenIDMetadataProvider returns a provider for the OIDC discovery
// endpoints, or an error if they could not be constructed. Callers should note
// that this function may perform additional validation on inputs that is not
// backwards-compatible with all command-line validation. The recommendation is
// to log the error and skip installing the OIDC discovery endpoints.
func NewOpenIDMetadataProvider(issuerURL, jwksURI, defaultExternalAddress string, pubKeyGetter PublicKeysGetter) (OpenIDMetadataProvider, error) {
	if issuerURL == "" {
		return nil, fmt.Errorf("empty issuer URL")
	}
	if jwksURI == "" && defaultExternalAddress == "" {
		return nil, fmt.Errorf("either the JWKS URI or the default external address, or both, must be set")
	}
	if pubKeyGetter == nil {
		return nil, fmt.Errorf("no public key getter provided")
	}

	// Ensure the issuer URL meets the OIDC spec (this is the additional
	// validation the doc comment warns about).
	// https://openid.net/specs/openid-connect-discovery-1_0.html#ProviderMetadata
	iss, err := url.Parse(issuerURL)
	if err != nil {
		return nil, err
	}
	if iss.Scheme != "https" {
		return nil, fmt.Errorf("issuer URL must use https scheme, got: %s", issuerURL)
	}
	if iss.RawQuery != "" {
		return nil, fmt.Errorf("issuer URL may not include a query, got: %s", issuerURL)
	}
	if iss.Fragment != "" {
		return nil, fmt.Errorf("issuer URL may not include a fragment, got: %s", issuerURL)
	}

	// Either use the provided JWKS URI or default to ExternalAddress plus
	// the JWKS path.
	if jwksURI == "" {
		const msg = "attempted to build jwks_uri from external " +
			"address %s, but could not construct a valid URL. Error: %v"

		if defaultExternalAddress == "" {
			return nil, fmt.Errorf(msg, defaultExternalAddress,
				fmt.Errorf("empty address"))
		}

		u := &url.URL{
			Scheme: "https",
			Host:   defaultExternalAddress,
			Path:   JWKSPath,
		}
		jwksURI = u.String()

		// TODO(mtaufen): I think we can probably expect ExternalAddress is
		// at most just host + port and skip the sanity check, but want to be
		// careful until that is confirmed.

		// Sanity check that the jwksURI we produced is the valid URL we expect.
		// This is just in case ExternalAddress came in as something weird,
		// like a scheme + host + port, instead of just host + port.
		parsed, err := url.Parse(jwksURI)
		if err != nil {
			return nil, fmt.Errorf(msg, defaultExternalAddress, err)
		} else if u.Scheme != parsed.Scheme ||
			u.Host != parsed.Host ||
			u.Path != parsed.Path {
			return nil, fmt.Errorf(msg, defaultExternalAddress,
				fmt.Errorf("got %v, expected %v", parsed, u))
		}
	} else {
		// Double-check that jwksURI is an https URL
		if u, err := url.Parse(jwksURI); err != nil {
			return nil, err
		} else if u.Scheme != "https" {
			return nil, fmt.Errorf("jwksURI requires https scheme, parsed as: %v", u.String())
		}
	}

	provider := &openidConfigProvider{
		issuerURL:    issuerURL,
		jwksURI:      jwksURI,
		pubKeyGetter: pubKeyGetter,
	}
	// Register to be notified if public keys change
	pubKeyGetter.AddListener(provider)
	// Synchronously construct the config / keyset json once at startup to ensure a successful starting point
	if err := provider.Update(); err != nil {
		return nil, err
	}
	return provider, nil
}

// openIDMetadata provides a minimal subset of OIDC provider metadata:
// https://openid.net/specs/openid-connect-discovery-1_0.html#ProviderMetadata
type openIDMetadata struct {
	Issuer string `json:"issuer"` // REQUIRED in OIDC; meaningful to relying parties.
	// TODO(mtaufen): Since our goal is compatibility for relying parties that
	// need to validate ID tokens, but do not need to initiate login flows,
	// and since we aren't sure what to put in authorization_endpoint yet,
	// we will omit this field until someone files a bug.
	// AuthzEndpoint string   `json:"authorization_endpoint"`                // REQUIRED in OIDC; but useless to relying parties.
	JWKSURI       string   `json:"jwks_uri"`                              // REQUIRED in OIDC; meaningful to relying parties.
	ResponseTypes []string `json:"response_types_supported"`              // REQUIRED in OIDC
	SubjectTypes  []string `json:"subject_types_supported"`               // REQUIRED in OIDC
	SigningAlgs   []string `json:"id_token_signing_alg_values_supported"` // REQUIRED in OIDC
}

// openIDConfigJSON returns the JSON OIDC Discovery Doc for the service
// account issuer.
func openIDConfigJSON(iss, jwksURI string, keys []PublicKey) ([]byte, error) {
	keyset, errs := publicJWKSFromKeys(keys)
	if errs != nil {
		return nil, errs
	}

	metadata := openIDMetadata{
		Issuer:        iss,
		JWKSURI:       jwksURI,
		ResponseTypes: []string{"id_token"}, // Kubernetes only produces ID tokens
		SubjectTypes:  []string{"public"},   // https://openid.net/specs/openid-connect-core-1_0.html#SubjectIDTypes
		SigningAlgs:   getAlgs(keyset),      // REQUIRED by OIDC
	}

	metadataJSON, err := json.Marshal(metadata)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal service account issuer metadata: %v", err)
	}

	return metadataJSON, nil
}

// openIDKeysetJSON returns the JSON Web Key Set for the service account
// issuer's keys.
func openIDKeysetJSON(keys []PublicKey) ([]byte, error) {
	keyset, errs := publicJWKSFromKeys(keys)
	if errs != nil {
		return nil, errs
	}

	keysetJSON, err := json.Marshal(keyset)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal service account issuer JWKS: %v", err)
	}

	return keysetJSON, nil
}

func getAlgs(keys *jose.JSONWebKeySet) []string {
	algs := sets.NewString()
	for _, k := range keys.Keys {
		algs.Insert(k.Algorithm)
	}
	// Note: List returns a sorted slice.
	return algs.List()
}

type publicKeyGetter interface {
	Public() crypto.PublicKey
}

// publicJWKSFromKeys constructs a JSONWebKeySet from a list of keys. The key
// set will only contain the public keys associated with the input keys.
func publicJWKSFromKeys(in []PublicKey) (*jose.JSONWebKeySet, errors.Aggregate) {
	// Decode keys into a JWKS.
	var keys jose.JSONWebKeySet
	var errs []error
	for i, key := range in {
		pubkey, err := jwkFromPublicKey(key)
		if err != nil {
			errs = append(errs, fmt.Errorf("error constructing JWK for key #%d: %v", i, err))
			continue
		}

		if !pubkey.Valid() {
			errs = append(errs, fmt.Errorf("key #%d not valid", i))
			continue
		}
		keys.Keys = append(keys.Keys, *pubkey)
	}
	if len(errs) != 0 {
		return nil, errors.NewAggregate(errs)
	}
	return &keys, nil
}

func jwkFromPublicKey(publicKey PublicKey) (*jose.JSONWebKey, error) {
	alg, err := algorithmFromPublicKey(publicKey.PublicKey)
	if err != nil {
		return nil, err
	}

	jwk := &jose.JSONWebKey{
		Algorithm: string(alg),
		Key:       publicKey.PublicKey,
		KeyID:     publicKey.KeyID,
		Use:       "sig",
	}

	if !jwk.IsPublic() {
		return nil, fmt.Errorf("JWK was not a public key! JWK: %v", jwk)
	}

	return jwk, nil
}

func algorithmFromPublicKey(publicKey crypto.PublicKey) (jose.SignatureAlgorithm, error) {
	switch pk := publicKey.(type) {
	case *rsa.PublicKey:
		// IMPORTANT: If this function is updated to support additional key sizes,
		// signerFromRSAPrivateKey in serviceaccount/jwt.go must also be
		// updated to support the same key sizes. Today we only support RS256.
		return jose.RS256, nil
	case *ecdsa.PublicKey:
		switch pk.Curve {
		case elliptic.P256():
			return jose.ES256, nil
		case elliptic.P384():
			return jose.ES384, nil
		case elliptic.P521():
			return jose.ES512, nil
		default:
			return "", fmt.Errorf("unknown private key curve, must be 256, 384, or 521")
		}
	case jose.OpaqueSigner:
		return jose.SignatureAlgorithm(pk.Public().Algorithm), nil
	default:
		return "", fmt.Errorf("unknown public key type, must be *rsa.PublicKey, *ecdsa.PublicKey, or jose.OpaqueSigner")
	}
}
