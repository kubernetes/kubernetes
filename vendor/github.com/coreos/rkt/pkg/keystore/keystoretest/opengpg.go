// Copyright 2014 The rkt Authors
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

// Package keystoretest provides utilities for ACI keystore testing.
//go:generate go run ./keygen/keygen.go
package keystoretest

import (
	"bytes"
	"errors"
	"io"

	"golang.org/x/crypto/openpgp"
)

// A KeyDetails represents an openpgp.Entity and its key details.
type KeyDetails struct {
	Fingerprint       string
	ArmoredPublicKey  string
	ArmoredPrivateKey string
}

// NewMessageAndSignature generates a new random message signed by the given entity.
// NewMessageAndSignature returns message, signature and an error if any.
func NewMessageAndSignature(armoredPrivateKey string) (io.ReadSeeker, io.ReadSeeker, error) {
	entityList, err := openpgp.ReadArmoredKeyRing(bytes.NewBufferString(armoredPrivateKey))
	if err != nil {
		return nil, nil, err
	}
	if len(entityList) < 1 {
		return nil, nil, errors.New("empty entity list")
	}
	signature := &bytes.Buffer{}
	message := []byte("data")
	if err := openpgp.ArmoredDetachSign(signature, entityList[0], bytes.NewReader(message), nil); err != nil {
		return nil, nil, err
	}
	return bytes.NewReader(message), bytes.NewReader(signature.Bytes()), nil
}
