/*
Copyright 2014 The Kubernetes Authors.

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
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	jose "gopkg.in/square/go-jose.v2"
	"gopkg.in/square/go-jose.v2/jwt"

	v1 "k8s.io/api/core/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	apiserverserviceaccount "k8s.io/apiserver/pkg/authentication/serviceaccount"
)

// ServiceAccountTokenGetter defines functions to retrieve a named service account and secret
type ServiceAccountTokenGetter interface {
	GetServiceAccount(namespace, name string) (*v1.ServiceAccount, error)
	GetPod(namespace, name string) (*v1.Pod, error)
	GetSecret(namespace, name string) (*v1.Secret, error)
	GetNode(name string) (*v1.Node, error)
}

type TokenGenerator interface {
	// GenerateToken generates a token which will identify the given
	// ServiceAccount. privateClaims is an interface that will be
	// serialized into the JWT payload JSON encoding at the root level of
	// the payload object. Public claims take precedent over private
	// claims i.e. if both claims and privateClaims have an "exp" field,
	// the value in claims will be used.
	GenerateToken(claims *jwt.Claims, privateClaims interface{}) (string, error)
}

// JWTTokenGenerator returns a TokenGenerator that generates signed JWT tokens, using the given privateKey.
// privateKey is a PEM-encoded byte array of a private RSA key.
func JWTTokenGenerator(iss string, privateKey interface{}) (TokenGenerator, error) {
	var signer jose.Signer
	var err error
	switch pk := privateKey.(type) {
	case *rsa.PrivateKey:
		signer, err = signerFromRSAPrivateKey(pk)
		if err != nil {
			return nil, fmt.Errorf("could not generate signer for RSA keypair: %v", err)
		}
	case *ecdsa.PrivateKey:
		signer, err = signerFromECDSAPrivateKey(pk)
		if err != nil {
			return nil, fmt.Errorf("could not generate signer for ECDSA keypair: %v", err)
		}
	case jose.OpaqueSigner:
		signer, err = signerFromOpaqueSigner(pk)
		if err != nil {
			return nil, fmt.Errorf("could not generate signer for OpaqueSigner: %v", err)
		}
	default:
		return nil, fmt.Errorf("unknown private key type %T, must be *rsa.PrivateKey, *ecdsa.PrivateKey, or jose.OpaqueSigner", privateKey)
	}

	return &jwtTokenGenerator{
		iss:    iss,
		signer: signer,
	}, nil
}

// keyIDFromPublicKey derives a key ID non-reversibly from a public key.
//
// The Key ID is field on a given on JWTs and JWKs that help relying parties
// pick the correct key for verification when the identity party advertises
// multiple keys.
//
// Making the derivation non-reversible makes it impossible for someone to
// accidentally obtain the real key from the key ID and use it for token
// validation.
func keyIDFromPublicKey(publicKey interface{}) (string, error) {
	publicKeyDERBytes, err := x509.MarshalPKIXPublicKey(publicKey)
	if err != nil {
		return "", fmt.Errorf("failed to serialize public key to DER format: %v", err)
	}

	hasher := crypto.SHA256.New()
	hasher.Write(publicKeyDERBytes)
	publicKeyDERHash := hasher.Sum(nil)

	keyID := base64.RawURLEncoding.EncodeToString(publicKeyDERHash)

	return keyID, nil
}

func signerFromRSAPrivateKey(keyPair *rsa.PrivateKey) (jose.Signer, error) {
	keyID, err := keyIDFromPublicKey(&keyPair.PublicKey)
	if err != nil {
		return nil, fmt.Errorf("failed to derive keyID: %v", err)
	}

	// IMPORTANT: If this function is updated to support additional key sizes,
	// algorithmForPublicKey in serviceaccount/openidmetadata.go must also be
	// updated to support the same key sizes. Today we only support RS256.

	// Wrap the RSA keypair in a JOSE JWK with the designated key ID.
	privateJWK := &jose.JSONWebKey{
		Algorithm: string(jose.RS256),
		Key:       keyPair,
		KeyID:     keyID,
		Use:       "sig",
	}

	signer, err := jose.NewSigner(
		jose.SigningKey{
			Algorithm: jose.RS256,
			Key:       privateJWK,
		},
		nil,
	)

	if err != nil {
		return nil, fmt.Errorf("failed to create signer: %v", err)
	}

	return signer, nil
}

func signerFromECDSAPrivateKey(keyPair *ecdsa.PrivateKey) (jose.Signer, error) {
	var alg jose.SignatureAlgorithm
	switch keyPair.Curve {
	case elliptic.P256():
		alg = jose.ES256
	case elliptic.P384():
		alg = jose.ES384
	case elliptic.P521():
		alg = jose.ES512
	default:
		return nil, fmt.Errorf("unknown private key curve, must be 256, 384, or 521")
	}

	keyID, err := keyIDFromPublicKey(&keyPair.PublicKey)
	if err != nil {
		return nil, fmt.Errorf("failed to derive keyID: %v", err)
	}

	// Wrap the ECDSA keypair in a JOSE JWK with the designated key ID.
	privateJWK := &jose.JSONWebKey{
		Algorithm: string(alg),
		Key:       keyPair,
		KeyID:     keyID,
		Use:       "sig",
	}

	signer, err := jose.NewSigner(
		jose.SigningKey{
			Algorithm: alg,
			Key:       privateJWK,
		},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create signer: %v", err)
	}

	return signer, nil
}

func signerFromOpaqueSigner(opaqueSigner jose.OpaqueSigner) (jose.Signer, error) {
	alg := jose.SignatureAlgorithm(opaqueSigner.Public().Algorithm)

	signer, err := jose.NewSigner(
		jose.SigningKey{
			Algorithm: alg,
			Key: &jose.JSONWebKey{
				Algorithm: string(alg),
				Key:       opaqueSigner,
				KeyID:     opaqueSigner.Public().KeyID,
				Use:       "sig",
			},
		},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create signer: %v", err)
	}

	return signer, nil
}

type jwtTokenGenerator struct {
	iss    string
	signer jose.Signer
}

func (j *jwtTokenGenerator) GenerateToken(claims *jwt.Claims, privateClaims interface{}) (string, error) {
	// claims are applied in reverse precedence
	return jwt.Signed(j.signer).
		Claims(privateClaims).
		Claims(claims).
		Claims(&jwt.Claims{
			Issuer: j.iss,
		}).
		CompactSerialize()
}

// JWTTokenAuthenticator authenticates tokens as JWT tokens produced by JWTTokenGenerator
// Token signatures are verified using each of the given public keys until one works (allowing key rotation)
// If lookup is true, the service account and secret referenced as claims inside the token are retrieved and verified with the provided ServiceAccountTokenGetter
func JWTTokenAuthenticator[PrivateClaims any](issuers []string, publicKeysGetter PublicKeysGetter, implicitAuds authenticator.Audiences, validator Validator[PrivateClaims]) authenticator.Token {
	issuersMap := make(map[string]bool)
	for _, issuer := range issuers {
		issuersMap[issuer] = true
	}
	return &jwtTokenAuthenticator[PrivateClaims]{
		issuers:      issuersMap,
		keysGetter:   publicKeysGetter,
		implicitAuds: implicitAuds,
		validator:    validator,
	}
}

// Listener is an interface to use to notify interested parties of a change.
type Listener interface {
	// Enqueue should be called when an input may have changed
	Enqueue()
}

// PublicKeysGetter returns public keys for a given key id.
type PublicKeysGetter interface {
	// AddListener is adds a listener to be notified of potential input changes.
	// This is a noop on static providers.
	AddListener(listener Listener)

	// GetCacheAgeMaxSeconds returns the seconds a call to GetPublicKeys() can be cached for.
	// If the results of GetPublicKeys() can be dynamic, this means a new key must be included in the results
	// for at least this long before it is used to sign new tokens.
	GetCacheAgeMaxSeconds() int

	// GetPublicKeys returns public keys to use for verifying a token with the given key id.
	// keyIDHint may be empty if the token did not have a kid header, or if all public keys are desired.
	GetPublicKeys(keyIDHint string) []PublicKey
}

type PublicKey struct {
	KeyID     string
	PublicKey interface{}
}

type staticPublicKeysGetter struct {
	allPublicKeys  []PublicKey
	publicKeysByID map[string][]PublicKey
}

// StaticPublicKeysGetter constructs an implementation of PublicKeysGetter
// which returns all public keys when key id is unspecified, and returns
// the public keys matching the keyIDFromPublicKey-derived key id when
// a key id is specified.
func StaticPublicKeysGetter(keys []interface{}) (PublicKeysGetter, error) {
	allPublicKeys := []PublicKey{}
	publicKeysByID := map[string][]PublicKey{}
	for _, key := range keys {
		if privateKey, isPrivateKey := key.(publicKeyGetter); isPrivateKey {
			// This is a private key. Extract its public key.
			key = privateKey.Public()
		}

		keyID, err := keyIDFromPublicKey(key)
		if err != nil {
			return nil, err
		}
		pk := PublicKey{PublicKey: key, KeyID: keyID}
		publicKeysByID[keyID] = append(publicKeysByID[keyID], pk)
		allPublicKeys = append(allPublicKeys, pk)
	}
	return &staticPublicKeysGetter{
		allPublicKeys:  allPublicKeys,
		publicKeysByID: publicKeysByID,
	}, nil
}

func (s staticPublicKeysGetter) AddListener(listener Listener) {
	// no-op, static key content never changes
}

func (s staticPublicKeysGetter) GetCacheAgeMaxSeconds() int {
	// hard-coded to match cache max-age set in OIDC discovery
	return 3600
}

func (s staticPublicKeysGetter) GetPublicKeys(keyID string) []PublicKey {
	if len(keyID) == 0 {
		return s.allPublicKeys
	}
	return s.publicKeysByID[keyID]
}

type jwtTokenAuthenticator[PrivateClaims any] struct {
	issuers      map[string]bool
	keysGetter   PublicKeysGetter
	validator    Validator[PrivateClaims]
	implicitAuds authenticator.Audiences
}

// Validator is called by the JWT token authenticator to apply domain specific
// validation to a token and extract user information.
// PrivateClaims is the struct that the authenticator should deserialize the JWT payload into, thus
// it should contain fields for any private claims that the Validator requires to validate the JWT.
type Validator[PrivateClaims any] interface {
	// Validate validates a token and returns user information or an error.
	// Validator can assume that the issuer and signature of a token are already
	// verified when this function is called.
	Validate(ctx context.Context, tokenData string, public *jwt.Claims, private *PrivateClaims) (*apiserverserviceaccount.ServiceAccountInfo, error)
}

func (j *jwtTokenAuthenticator[PrivateClaims]) AuthenticateToken(ctx context.Context, tokenData string) (*authenticator.Response, bool, error) {
	if !j.hasCorrectIssuer(tokenData) {
		return nil, false, nil
	}

	tok, err := jwt.ParseSigned(tokenData)
	if err != nil {
		return nil, false, nil
	}

	public := &jwt.Claims{}
	private := new(PrivateClaims)

	// Pick the key that has the same key ID as `tok`, if one exists.
	var kid string
	for _, header := range tok.Headers {
		if header.KeyID != "" {
			kid = header.KeyID
			break
		}
	}

	var (
		found   bool
		errlist []error
	)
	keys := j.keysGetter.GetPublicKeys(kid)
	if len(keys) == 0 {
		return nil, false, fmt.Errorf("invalid signature, no keys found")
	}
	for _, key := range keys {
		if err := tok.Claims(key.PublicKey, public, private); err != nil {
			errlist = append(errlist, err)
			continue
		}
		found = true
		break
	}

	if !found {
		return nil, false, utilerrors.NewAggregate(errlist)
	}

	// sanity check issuer since we parsed it out before signature validation
	if !j.issuers[public.Issuer] {
		return nil, false, fmt.Errorf("token issuer %q is invalid", public.Issuer)
	}

	tokenAudiences := authenticator.Audiences(public.Audience)
	if len(tokenAudiences) == 0 {
		// only apiserver audiences are allowed for legacy tokens
		audit.AddAuditAnnotation(ctx, "authentication.k8s.io/legacy-token", public.Subject)
		legacyTokensTotal.WithContext(ctx).Inc()
		tokenAudiences = j.implicitAuds
	}

	requestedAudiences, ok := authenticator.AudiencesFrom(ctx)
	if !ok {
		// default to apiserver audiences
		requestedAudiences = j.implicitAuds
	}

	auds := authenticator.Audiences(tokenAudiences).Intersect(requestedAudiences)
	if len(auds) == 0 && len(j.implicitAuds) != 0 {
		return nil, false, fmt.Errorf("token audiences %q is invalid for the target audiences %q", tokenAudiences, requestedAudiences)
	}

	// If we get here, we have a token with a recognized signature and
	// issuer string.
	sa, err := j.validator.Validate(ctx, tokenData, public, private)
	if err != nil {
		return nil, false, err
	}

	return &authenticator.Response{
		User:      sa.UserInfo(),
		Audiences: auds,
	}, true, nil
}

// hasCorrectIssuer returns true if tokenData is a valid JWT in compact
// serialization format and the "iss" claim matches the iss field of this token
// authenticator, and otherwise returns false.
//
// Note: go-jose currently does not allow access to unverified JWS payloads.
// See https://github.com/square/go-jose/issues/169
func (j *jwtTokenAuthenticator[PrivateClaims]) hasCorrectIssuer(tokenData string) bool {
	if strings.HasPrefix(strings.TrimSpace(tokenData), "{") {
		return false
	}
	parts := strings.Split(tokenData, ".")
	if len(parts) != 3 {
		return false
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return false
	}
	claims := struct {
		// WARNING: this JWT is not verified. Do not trust these claims.
		Issuer string `json:"iss"`
	}{}
	if err := json.Unmarshal(payload, &claims); err != nil {
		return false
	}
	return j.issuers[claims.Issuer]
}
