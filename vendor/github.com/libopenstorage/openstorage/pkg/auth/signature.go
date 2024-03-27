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
	"fmt"
	"io/ioutil"

	jwt "github.com/dgrijalva/jwt-go"
)

// Signature describes the signature type using definitions from
// the jwt package
type Signature struct {
	Type jwt.SigningMethod
	Key  interface{}
}

func NewSignatureSharedSecret(secret string) (*Signature, error) {
	return &Signature{
		Key:  []byte(secret),
		Type: jwt.SigningMethodHS256,
	}, nil
}

func NewSignatureRSAFromFile(filename string) (*Signature, error) {
	pem, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("Failed to read RSA file: %v", err)
	}
	return NewSignatureRSA(pem)
}

func NewSignatureRSA(pem []byte) (*Signature, error) {
	var err error
	signature := &Signature{}
	signature.Key, err = jwt.ParseRSAPrivateKeyFromPEM(pem)
	if err != nil {
		return nil, fmt.Errorf("Failed to parse RSA file: %v", err)
	}
	signature.Type = jwt.SigningMethodRS256
	return signature, nil
}

func NewSignatureECDSAFromFile(filename string) (*Signature, error) {
	pem, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("Failed to read ECDSA file: %v", err)
	}
	return NewSignatureECDSA(pem)
}

func NewSignatureECDSA(pem []byte) (*Signature, error) {
	var err error
	signature := &Signature{}
	signature.Key, err = jwt.ParseECPrivateKeyFromPEM(pem)
	if err != nil {
		return nil, fmt.Errorf("Failed to parse ECDSA file: %v", err)
	}
	signature.Type = jwt.SigningMethodES256
	return signature, nil
}
