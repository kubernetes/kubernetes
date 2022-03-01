// Copyright 2018 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package auth

import (
	"crypto/ecdsa"
	"crypto/rsa"
	"fmt"
	"io/ioutil"
	"time"

	jwt "github.com/form3tech-oss/jwt-go"
)

const (
	optSignMethod = "sign-method"
	optPublicKey  = "pub-key"
	optPrivateKey = "priv-key"
	optTTL        = "ttl"
)

var knownOptions = map[string]bool{
	optSignMethod: true,
	optPublicKey:  true,
	optPrivateKey: true,
	optTTL:        true,
}

var (
	// DefaultTTL will be used when a 'ttl' is not specified
	DefaultTTL = 5 * time.Minute
)

type jwtOptions struct {
	SignMethod jwt.SigningMethod
	PublicKey  []byte
	PrivateKey []byte
	TTL        time.Duration
}

// ParseWithDefaults will load options from the specified map or set defaults where appropriate
func (opts *jwtOptions) ParseWithDefaults(optMap map[string]string) error {
	if opts.TTL == 0 && optMap[optTTL] == "" {
		opts.TTL = DefaultTTL
	}

	return opts.Parse(optMap)
}

// Parse will load options from the specified map
func (opts *jwtOptions) Parse(optMap map[string]string) error {
	var err error
	if ttl := optMap[optTTL]; ttl != "" {
		opts.TTL, err = time.ParseDuration(ttl)
		if err != nil {
			return err
		}
	}

	if file := optMap[optPublicKey]; file != "" {
		opts.PublicKey, err = ioutil.ReadFile(file)
		if err != nil {
			return err
		}
	}

	if file := optMap[optPrivateKey]; file != "" {
		opts.PrivateKey, err = ioutil.ReadFile(file)
		if err != nil {
			return err
		}
	}

	// signing method is a required field
	method := optMap[optSignMethod]
	opts.SignMethod = jwt.GetSigningMethod(method)
	if opts.SignMethod == nil {
		return ErrInvalidAuthMethod
	}

	return nil
}

// Key will parse and return the appropriately typed key for the selected signature method
func (opts *jwtOptions) Key() (interface{}, error) {
	switch opts.SignMethod.(type) {
	case *jwt.SigningMethodRSA, *jwt.SigningMethodRSAPSS:
		return opts.rsaKey()
	case *jwt.SigningMethodECDSA:
		return opts.ecKey()
	case *jwt.SigningMethodHMAC:
		return opts.hmacKey()
	default:
		return nil, fmt.Errorf("unsupported signing method: %T", opts.SignMethod)
	}
}

func (opts *jwtOptions) hmacKey() (interface{}, error) {
	if len(opts.PrivateKey) == 0 {
		return nil, ErrMissingKey
	}
	return opts.PrivateKey, nil
}

func (opts *jwtOptions) rsaKey() (interface{}, error) {
	var (
		priv *rsa.PrivateKey
		pub  *rsa.PublicKey
		err  error
	)

	if len(opts.PrivateKey) > 0 {
		priv, err = jwt.ParseRSAPrivateKeyFromPEM(opts.PrivateKey)
		if err != nil {
			return nil, err
		}
	}

	if len(opts.PublicKey) > 0 {
		pub, err = jwt.ParseRSAPublicKeyFromPEM(opts.PublicKey)
		if err != nil {
			return nil, err
		}
	}

	if priv == nil {
		if pub == nil {
			// Neither key given
			return nil, ErrMissingKey
		}
		// Public key only, can verify tokens
		return pub, nil
	}

	// both keys provided, make sure they match
	if pub != nil && pub.E != priv.E && pub.N.Cmp(priv.N) != 0 {
		return nil, ErrKeyMismatch
	}

	return priv, nil
}

func (opts *jwtOptions) ecKey() (interface{}, error) {
	var (
		priv *ecdsa.PrivateKey
		pub  *ecdsa.PublicKey
		err  error
	)

	if len(opts.PrivateKey) > 0 {
		priv, err = jwt.ParseECPrivateKeyFromPEM(opts.PrivateKey)
		if err != nil {
			return nil, err
		}
	}

	if len(opts.PublicKey) > 0 {
		pub, err = jwt.ParseECPublicKeyFromPEM(opts.PublicKey)
		if err != nil {
			return nil, err
		}
	}

	if priv == nil {
		if pub == nil {
			// Neither key given
			return nil, ErrMissingKey
		}
		// Public key only, can verify tokens
		return pub, nil
	}

	// both keys provided, make sure they match
	if pub != nil && pub.Curve != priv.Curve &&
		pub.X.Cmp(priv.X) != 0 && pub.Y.Cmp(priv.Y) != 0 {
		return nil, ErrKeyMismatch
	}

	return priv, nil
}
