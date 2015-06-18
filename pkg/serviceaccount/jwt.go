/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"crypto/rsa"
	"errors"
	"fmt"
	"io/ioutil"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authenticator"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/user"

	jwt "github.com/dgrijalva/jwt-go"
	"github.com/golang/glog"
)

const (
	ServiceAccountUsernamePrefix    = "system:serviceaccount"
	ServiceAccountUsernameSeparator = ":"

	Issuer = "kubernetes/serviceaccount"

	SubjectClaim            = "sub"
	IssuerClaim             = "iss"
	ServiceAccountNameClaim = "kubernetes.io/serviceaccount/service-account.name"
	ServiceAccountUIDClaim  = "kubernetes.io/serviceaccount/service-account.uid"
	SecretNameClaim         = "kubernetes.io/serviceaccount/secret.name"
	NamespaceClaim          = "kubernetes.io/serviceaccount/namespace"
)

type TokenGenerator interface {
	// GenerateToken generates a token which will identify the given ServiceAccount.
	// The returned token will be stored in the given (and yet-unpersisted) Secret.
	GenerateToken(serviceAccount api.ServiceAccount, secret api.Secret) (string, error)
}

// ReadPrivateKey is a helper function for reading an rsa.PrivateKey from a PEM-encoded file
func ReadPrivateKey(file string) (*rsa.PrivateKey, error) {
	data, err := ioutil.ReadFile(file)
	if err != nil {
		return nil, err
	}
	return jwt.ParseRSAPrivateKeyFromPEM(data)
}

// ReadPublicKey is a helper function for reading an rsa.PublicKey from a PEM-encoded file
// Reads public keys from both public and private key files
func ReadPublicKey(file string) (*rsa.PublicKey, error) {
	data, err := ioutil.ReadFile(file)
	if err != nil {
		return nil, err
	}

	if privateKey, err := jwt.ParseRSAPrivateKeyFromPEM(data); err == nil {
		return &privateKey.PublicKey, nil
	}

	return jwt.ParseRSAPublicKeyFromPEM(data)
}

// MakeUsername generates a username from the given namespace and ServiceAccount name.
// The resulting username can be passed to SplitUsername to extract the original namespace and ServiceAccount name.
func MakeUsername(namespace, name string) string {
	return strings.Join([]string{ServiceAccountUsernamePrefix, namespace, name}, ServiceAccountUsernameSeparator)
}

// SplitUsername returns the namespace and ServiceAccount name embedded in the given username,
// or an error if the username is not a valid name produced by MakeUsername
func SplitUsername(username string) (string, string, error) {
	if !strings.HasPrefix(username, ServiceAccountUsernamePrefix+ServiceAccountUsernameSeparator) {
		return "", "", fmt.Errorf("Username must be in the form %s", MakeUsername("namespace", "name"))
	}
	username = strings.TrimPrefix(username, ServiceAccountUsernamePrefix+ServiceAccountUsernameSeparator)
	parts := strings.Split(username, ServiceAccountUsernameSeparator)
	if len(parts) != 2 || len(parts[0]) == 0 || len(parts[1]) == 0 {
		return "", "", fmt.Errorf("Username must be in the form %s", MakeUsername("namespace", "name"))
	}
	return parts[0], parts[1], nil
}

// JWTTokenGenerator returns a TokenGenerator that generates signed JWT tokens, using the given privateKey.
// privateKey is a PEM-encoded byte array of a private RSA key.
// JWTTokenAuthenticator()
func JWTTokenGenerator(key *rsa.PrivateKey) TokenGenerator {
	return &jwtTokenGenerator{key}
}

type jwtTokenGenerator struct {
	key *rsa.PrivateKey
}

func (j *jwtTokenGenerator) GenerateToken(serviceAccount api.ServiceAccount, secret api.Secret) (string, error) {
	token := jwt.New(jwt.SigningMethodRS256)

	// Identify the issuer
	token.Claims[IssuerClaim] = Issuer

	// Username: `serviceaccount:<namespace>:<serviceaccount>`
	token.Claims[SubjectClaim] = MakeUsername(serviceAccount.Namespace, serviceAccount.Name)

	// Persist enough structured info for the authenticator to be able to look up the service account and secret
	token.Claims[NamespaceClaim] = serviceAccount.Namespace
	token.Claims[ServiceAccountNameClaim] = serviceAccount.Name
	token.Claims[ServiceAccountUIDClaim] = serviceAccount.UID
	token.Claims[SecretNameClaim] = secret.Name

	// Sign and get the complete encoded token as a string
	return token.SignedString(j.key)
}

// JWTTokenAuthenticator authenticates tokens as JWT tokens produced by JWTTokenGenerator
// Token signatures are verified using each of the given public keys until one works (allowing key rotation)
// If lookup is true, the service account and secret referenced as claims inside the token are retrieved and verified with the provided ServiceAccountTokenGetter
func JWTTokenAuthenticator(keys []*rsa.PublicKey, lookup bool, getter ServiceAccountTokenGetter) authenticator.Token {
	return &jwtTokenAuthenticator{keys, lookup, getter}
}

type jwtTokenAuthenticator struct {
	keys   []*rsa.PublicKey
	lookup bool
	getter ServiceAccountTokenGetter
}

func (j *jwtTokenAuthenticator) AuthenticateToken(token string) (user.Info, bool, error) {
	var validationError error

	for i, key := range j.keys {
		// Attempt to verify with each key until we find one that works
		parsedToken, err := jwt.Parse(token, func(token *jwt.Token) (interface{}, error) {
			if _, ok := token.Method.(*jwt.SigningMethodRSA); !ok {
				return nil, fmt.Errorf("Unexpected signing method: %v", token.Header["alg"])
			}
			return key, nil
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
			}

			// Other errors should just return as errors
			return nil, false, err
		}

		// If we get here, we have a token with a recognized signature

		// Make sure we issued the token
		iss, _ := parsedToken.Claims[IssuerClaim].(string)
		if iss != Issuer {
			return nil, false, nil
		}

		// Make sure the claims we need exist
		sub, _ := parsedToken.Claims[SubjectClaim].(string)
		if len(sub) == 0 {
			return nil, false, errors.New("sub claim is missing")
		}
		namespace, _ := parsedToken.Claims[NamespaceClaim].(string)
		if len(namespace) == 0 {
			return nil, false, errors.New("namespace claim is missing")
		}
		secretName, _ := parsedToken.Claims[SecretNameClaim].(string)
		if len(namespace) == 0 {
			return nil, false, errors.New("secretName claim is missing")
		}
		serviceAccountName, _ := parsedToken.Claims[ServiceAccountNameClaim].(string)
		if len(serviceAccountName) == 0 {
			return nil, false, errors.New("serviceAccountName claim is missing")
		}
		serviceAccountUID, _ := parsedToken.Claims[ServiceAccountUIDClaim].(string)
		if len(serviceAccountUID) == 0 {
			return nil, false, errors.New("serviceAccountUID claim is missing")
		}

		if j.lookup {
			// Make sure token hasn't been invalidated by deletion of the secret
			secret, err := j.getter.GetSecret(namespace, secretName)
			if err != nil {
				glog.V(4).Infof("Could not retrieve token %s/%s for service account %s/%s: %v", namespace, secretName, namespace, serviceAccountName, err)
				return nil, false, errors.New("Token has been invalidated")
			}
			if bytes.Compare(secret.Data[api.ServiceAccountTokenKey], []byte(token)) != 0 {
				glog.V(4).Infof("Token contents no longer matches %s/%s for service account %s/%s", namespace, secretName, namespace, serviceAccountName)
				return nil, false, errors.New("Token does not match server's copy")
			}

			// Make sure service account still exists (name and UID)
			serviceAccount, err := j.getter.GetServiceAccount(namespace, serviceAccountName)
			if err != nil {
				glog.V(4).Infof("Could not retrieve service account %s/%s: %v", namespace, serviceAccountName, err)
				return nil, false, err
			}
			if string(serviceAccount.UID) != serviceAccountUID {
				glog.V(4).Infof("Service account UID no longer matches %s/%s: %q != %q", namespace, serviceAccountName, string(serviceAccount.UID), serviceAccountUID)
				return nil, false, fmt.Errorf("ServiceAccount UID (%s) does not match claim (%s)", serviceAccount.UID, serviceAccountUID)
			}
		}

		return &user.DefaultInfo{
			Name:   sub,
			UID:    serviceAccountUID,
			Groups: []string{},
		}, true, nil
	}

	return nil, false, validationError
}
