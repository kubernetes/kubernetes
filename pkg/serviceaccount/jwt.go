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
	"errors"
	"fmt"

	"k8s.io/api/core/v1"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	apiserverserviceaccount "k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/apiserver/pkg/authentication/user"

	jwt "github.com/dgrijalva/jwt-go"
	"github.com/golang/glog"
)

const (
	Issuer = "kubernetes/serviceaccount"

	SubjectClaim            = "sub"
	IssuerClaim             = "iss"
	ServiceAccountNameClaim = "kubernetes.io/serviceaccount/service-account.name"
	ServiceAccountUIDClaim  = "kubernetes.io/serviceaccount/service-account.uid"
	SecretNameClaim         = "kubernetes.io/serviceaccount/secret.name"
	NamespaceClaim          = "kubernetes.io/serviceaccount/namespace"
)

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
func JWTTokenGenerator(privateKey interface{}) TokenGenerator {
	return &jwtTokenGenerator{privateKey}
}

type jwtTokenGenerator struct {
	privateKey interface{}
}

func (j *jwtTokenGenerator) GenerateToken(serviceAccount v1.ServiceAccount, secret v1.Secret) (string, error) {
	var method jwt.SigningMethod
	switch privateKey := j.privateKey.(type) {
	case *rsa.PrivateKey:
		method = jwt.SigningMethodRS256
	case *ecdsa.PrivateKey:
		switch privateKey.Curve {
		case elliptic.P256():
			method = jwt.SigningMethodES256
		case elliptic.P384():
			method = jwt.SigningMethodES384
		case elliptic.P521():
			method = jwt.SigningMethodES512
		default:
			return "", fmt.Errorf("unknown private key curve, must be 256, 384, or 521")
		}
	default:
		return "", fmt.Errorf("unknown private key type %T, must be *rsa.PrivateKey or *ecdsa.PrivateKey", j.privateKey)
	}

	token := jwt.New(method)

	claims, _ := token.Claims.(jwt.MapClaims)

	// Identify the issuer
	claims[IssuerClaim] = Issuer

	// Username
	claims[SubjectClaim] = apiserverserviceaccount.MakeUsername(serviceAccount.Namespace, serviceAccount.Name)

	// Persist enough structured info for the authenticator to be able to look up the service account and secret
	claims[NamespaceClaim] = serviceAccount.Namespace
	claims[ServiceAccountNameClaim] = serviceAccount.Name
	claims[ServiceAccountUIDClaim] = serviceAccount.UID
	claims[SecretNameClaim] = secret.Name

	// Sign and get the complete encoded token as a string
	return token.SignedString(j.privateKey)
}

// JWTTokenAuthenticator authenticates tokens as JWT tokens produced by JWTTokenGenerator
// Token signatures are verified using each of the given public keys until one works (allowing key rotation)
// If lookup is true, the service account and secret referenced as claims inside the token are retrieved and verified with the provided ServiceAccountTokenGetter
func JWTTokenAuthenticator(keys []interface{}, lookup bool, getter ServiceAccountTokenGetter) authenticator.Token {
	return &jwtTokenAuthenticator{keys, lookup, getter}
}

type jwtTokenAuthenticator struct {
	keys   []interface{}
	lookup bool
	getter ServiceAccountTokenGetter
}

var errMismatchedSigningMethod = errors.New("invalid signing method")

func (j *jwtTokenAuthenticator) AuthenticateToken(token string) (user.Info, bool, error) {
	var validationError error

	for i, key := range j.keys {
		// Attempt to verify with each key until we find one that works
		parsedToken, err := jwt.Parse(token, func(token *jwt.Token) (interface{}, error) {
			switch token.Method.(type) {
			case *jwt.SigningMethodRSA:
				if _, ok := key.(*rsa.PublicKey); ok {
					return key, nil
				}
				return nil, errMismatchedSigningMethod
			case *jwt.SigningMethodECDSA:
				if _, ok := key.(*ecdsa.PublicKey); ok {
					return key, nil
				}
				return nil, errMismatchedSigningMethod
			default:
				return nil, fmt.Errorf("Unexpected signing method: %v", token.Header["alg"])
			}
		})

		if err != nil {
			switch err := err.(type) {
			case *jwt.ValidationError:
				if (err.Errors & jwt.ValidationErrorMalformed) != 0 {
					// Not a JWT, no point in continuing
					return nil, false, nil
				}

				if (err.Errors & jwt.ValidationErrorSignatureInvalid) != 0 {
					// Signature error, perhaps one of the other keys will verify the signature
					// If not, we want to return this error
					glog.V(4).Infof("Signature error (key %d): %v", i, err)
					validationError = err
					continue
				}

				// This key doesn't apply to the given signature type
				// Perhaps one of the other keys will verify the signature
				// If not, we want to return this error
				if err.Inner == errMismatchedSigningMethod {
					glog.V(4).Infof("Mismatched key type (key %d): %v", i, err)
					validationError = err
					continue
				}
			}

			// Other errors should just return as errors
			return nil, false, err
		}

		// If we get here, we have a token with a recognized signature

		claims, _ := parsedToken.Claims.(jwt.MapClaims)

		// Make sure we issued the token
		iss, _ := claims[IssuerClaim].(string)
		if iss != Issuer {
			return nil, false, nil
		}

		// Make sure the claims we need exist
		sub, _ := claims[SubjectClaim].(string)
		if len(sub) == 0 {
			return nil, false, errors.New("sub claim is missing")
		}
		namespace, _ := claims[NamespaceClaim].(string)
		if len(namespace) == 0 {
			return nil, false, errors.New("namespace claim is missing")
		}
		secretName, _ := claims[SecretNameClaim].(string)
		if len(namespace) == 0 {
			return nil, false, errors.New("secretName claim is missing")
		}
		serviceAccountName, _ := claims[ServiceAccountNameClaim].(string)
		if len(serviceAccountName) == 0 {
			return nil, false, errors.New("serviceAccountName claim is missing")
		}
		serviceAccountUID, _ := claims[ServiceAccountUIDClaim].(string)
		if len(serviceAccountUID) == 0 {
			return nil, false, errors.New("serviceAccountUID claim is missing")
		}

		subjectNamespace, subjectName, err := apiserverserviceaccount.SplitUsername(sub)
		if err != nil || subjectNamespace != namespace || subjectName != serviceAccountName {
			return nil, false, errors.New("sub claim is invalid")
		}

		if j.lookup {
			// Make sure token hasn't been invalidated by deletion of the secret
			secret, err := j.getter.GetSecret(namespace, secretName)
			if err != nil {
				glog.V(4).Infof("Could not retrieve token %s/%s for service account %s/%s: %v", namespace, secretName, namespace, serviceAccountName, err)
				return nil, false, errors.New("Token has been invalidated")
			}
			if secret.DeletionTimestamp != nil {
				glog.V(4).Infof("Token is deleted and awaiting removal: %s/%s for service account %s/%s", namespace, secretName, namespace, serviceAccountName)
				return nil, false, errors.New("Token has been invalidated")
			}
			if bytes.Compare(secret.Data[v1.ServiceAccountTokenKey], []byte(token)) != 0 {
				glog.V(4).Infof("Token contents no longer matches %s/%s for service account %s/%s", namespace, secretName, namespace, serviceAccountName)
				return nil, false, errors.New("Token does not match server's copy")
			}

			// Make sure service account still exists (name and UID)
			serviceAccount, err := j.getter.GetServiceAccount(namespace, serviceAccountName)
			if err != nil {
				glog.V(4).Infof("Could not retrieve service account %s/%s: %v", namespace, serviceAccountName, err)
				return nil, false, err
			}
			if serviceAccount.DeletionTimestamp != nil {
				glog.V(4).Infof("Service account has been deleted %s/%s", namespace, serviceAccountName)
				return nil, false, fmt.Errorf("ServiceAccount %s/%s has been deleted", namespace, serviceAccountName)
			}
			if string(serviceAccount.UID) != serviceAccountUID {
				glog.V(4).Infof("Service account UID no longer matches %s/%s: %q != %q", namespace, serviceAccountName, string(serviceAccount.UID), serviceAccountUID)
				return nil, false, fmt.Errorf("ServiceAccount UID (%s) does not match claim (%s)", serviceAccount.UID, serviceAccountUID)
			}
		}

		return UserInfo(namespace, serviceAccountName, serviceAccountUID), true, nil
	}

	return nil, false, validationError
}
