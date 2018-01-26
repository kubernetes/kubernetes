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
	"bytes"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rsa"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"k8s.io/api/core/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	apiserverserviceaccount "k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/apiserver/pkg/authentication/user"

	"github.com/golang/glog"
	jose "gopkg.in/square/go-jose.v2"
	"gopkg.in/square/go-jose.v2/jwt"
)

const LegacyIssuer = "kubernetes/serviceaccount"

type privateClaims struct {
	ServiceAccountName string `json:"kubernetes.io/serviceaccount/service-account.name"`
	ServiceAccountUID  string `json:"kubernetes.io/serviceaccount/service-account.uid"`
	SecretName         string `json:"kubernetes.io/serviceaccount/secret.name"`
	Namespace          string `json:"kubernetes.io/serviceaccount/namespace"`
}

// ServiceAccountTokenGetter defines functions to retrieve a named service account and secret
type ServiceAccountTokenGetter interface {
	GetServiceAccount(namespace, name string) (*v1.ServiceAccount, error)
	GetSecret(namespace, name string) (*v1.Secret, error)
}

type TokenGenerator interface {
	// GenerateToken generates a token which will identify the given ServiceAccount.
	// The returned token will be stored in the given (and yet-unpersisted) Secret.
	GenerateToken(serviceAccount v1.ServiceAccount, secret v1.Secret) (string, error)
}

// JWTTokenGenerator returns a TokenGenerator that generates signed JWT tokens, using the given privateKey.
// privateKey is a PEM-encoded byte array of a private RSA key.
// JWTTokenAuthenticator()
func JWTTokenGenerator(iss string, privateKey interface{}) TokenGenerator {
	return &jwtTokenGenerator{
		iss:        iss,
		privateKey: privateKey,
	}
}

type jwtTokenGenerator struct {
	iss        string
	privateKey interface{}
}

func (j *jwtTokenGenerator) GenerateToken(serviceAccount v1.ServiceAccount, secret v1.Secret) (string, error) {
	var alg jose.SignatureAlgorithm
	switch privateKey := j.privateKey.(type) {
	case *rsa.PrivateKey:
		alg = jose.RS256
	case *ecdsa.PrivateKey:
		switch privateKey.Curve {
		case elliptic.P256():
			alg = jose.ES256
		case elliptic.P384():
			alg = jose.ES384
		case elliptic.P521():
			alg = jose.ES512
		default:
			return "", fmt.Errorf("unknown private key curve, must be 256, 384, or 521")
		}
	default:
		return "", fmt.Errorf("unknown private key type %T, must be *rsa.PrivateKey or *ecdsa.PrivateKey", j.privateKey)
	}

	signer, err := jose.NewSigner(
		jose.SigningKey{
			Algorithm: alg,
			Key:       j.privateKey,
		},
		nil,
	)
	if err != nil {
		return "", err
	}

	return jwt.Signed(signer).
		Claims(&jwt.Claims{
			Issuer:  j.iss,
			Subject: apiserverserviceaccount.MakeUsername(serviceAccount.Namespace, serviceAccount.Name),
		}).
		Claims(&privateClaims{
			Namespace:          serviceAccount.Namespace,
			ServiceAccountName: serviceAccount.Name,
			ServiceAccountUID:  string(serviceAccount.UID),
			SecretName:         secret.Name,
		}).CompactSerialize()
}

// JWTTokenAuthenticator authenticates tokens as JWT tokens produced by JWTTokenGenerator
// Token signatures are verified using each of the given public keys until one works (allowing key rotation)
// If lookup is true, the service account and secret referenced as claims inside the token are retrieved and verified with the provided ServiceAccountTokenGetter
func JWTTokenAuthenticator(iss string, keys []interface{}, lookup bool, getter ServiceAccountTokenGetter) authenticator.Token {
	return &jwtTokenAuthenticator{
		iss:  iss,
		keys: keys,
		validator: &legacyValidator{
			lookup: lookup,
			getter: getter,
		},
	}
}

type jwtTokenAuthenticator struct {
	iss       string
	keys      []interface{}
	validator Validator
}

type Validator interface {
	Validate(tokenData string, public *jwt.Claims, private *privateClaims) error
}

var errMismatchedSigningMethod = errors.New("invalid signing method")

func (j *jwtTokenAuthenticator) AuthenticateToken(tokenData string) (user.Info, bool, error) {
	if !j.hasCorrectIssuer(tokenData) {
		return nil, false, nil
	}

	tok, err := jwt.ParseSigned(tokenData)
	if err != nil {
		return nil, false, nil
	}

	public := &jwt.Claims{}
	private := &privateClaims{}

	var (
		found   bool
		errlist []error
	)
	for _, key := range j.keys {
		if err := tok.Claims(key, public, private); err != nil {
			errlist = append(errlist, err)
			continue
		}
		found = true
		break
	}

	if !found {
		return nil, false, utilerrors.NewAggregate(errlist)
	}

	// If we get here, we have a token with a recognized signature and
	// issuer string.
	if err := j.validator.Validate(tokenData, public, private); err != nil {
		return nil, false, err
	}

	return UserInfo(private.Namespace, private.ServiceAccountName, private.ServiceAccountUID), true, nil

}

// hasCorrectIssuer returns true if tokenData is a valid JWT in compact
// serialization format and the "iss" claim matches the iss field of this token
// authenticator, and otherwise returns false.
//
// Note: go-jose currently does not allow access to unverified JWS payloads.
// See https://github.com/square/go-jose/issues/169
func (j *jwtTokenAuthenticator) hasCorrectIssuer(tokenData string) bool {
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
	if claims.Issuer != j.iss {
		return false
	}
	return true

}

type legacyValidator struct {
	lookup bool
	getter ServiceAccountTokenGetter
}

func (v *legacyValidator) Validate(tokenData string, public *jwt.Claims, private *privateClaims) error {

	// Make sure the claims we need exist
	if len(public.Subject) == 0 {
		return errors.New("sub claim is missing")
	}
	namespace := private.Namespace
	if len(namespace) == 0 {
		return errors.New("namespace claim is missing")
	}
	secretName := private.SecretName
	if len(secretName) == 0 {
		return errors.New("secretName claim is missing")
	}
	serviceAccountName := private.ServiceAccountName
	if len(serviceAccountName) == 0 {
		return errors.New("serviceAccountName claim is missing")
	}
	serviceAccountUID := private.ServiceAccountUID
	if len(serviceAccountUID) == 0 {
		return errors.New("serviceAccountUID claim is missing")
	}

	subjectNamespace, subjectName, err := apiserverserviceaccount.SplitUsername(public.Subject)
	if err != nil || subjectNamespace != namespace || subjectName != serviceAccountName {
		return errors.New("sub claim is invalid")
	}

	if v.lookup {
		// Make sure token hasn't been invalidated by deletion of the secret
		secret, err := v.getter.GetSecret(namespace, secretName)
		if err != nil {
			glog.V(4).Infof("Could not retrieve token %s/%s for service account %s/%s: %v", namespace, secretName, namespace, serviceAccountName, err)
			return errors.New("Token has been invalidated")
		}
		if secret.DeletionTimestamp != nil {
			glog.V(4).Infof("Token is deleted and awaiting removal: %s/%s for service account %s/%s", namespace, secretName, namespace, serviceAccountName)
			return errors.New("Token has been invalidated")
		}
		if bytes.Compare(secret.Data[v1.ServiceAccountTokenKey], []byte(tokenData)) != 0 {
			glog.V(4).Infof("Token contents no longer matches %s/%s for service account %s/%s", namespace, secretName, namespace, serviceAccountName)
			return errors.New("Token does not match server's copy")
		}

		// Make sure service account still exists (name and UID)
		serviceAccount, err := v.getter.GetServiceAccount(namespace, serviceAccountName)
		if err != nil {
			glog.V(4).Infof("Could not retrieve service account %s/%s: %v", namespace, serviceAccountName, err)
			return err
		}
		if serviceAccount.DeletionTimestamp != nil {
			glog.V(4).Infof("Service account has been deleted %s/%s", namespace, serviceAccountName)
			return fmt.Errorf("ServiceAccount %s/%s has been deleted", namespace, serviceAccountName)
		}
		if string(serviceAccount.UID) != serviceAccountUID {
			glog.V(4).Infof("Service account UID no longer matches %s/%s: %q != %q", namespace, serviceAccountName, string(serviceAccount.UID), serviceAccountUID)
			return fmt.Errorf("ServiceAccount UID (%s) does not match claim (%s)", serviceAccount.UID, serviceAccountUID)
		}
	}

	return nil
}
