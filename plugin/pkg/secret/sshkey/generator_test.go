/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package sshkey

import (
	"crypto/ecdsa"
	"crypto/rsa"
	"reflect"
	"strconv"
	"testing"

	"golang.org/x/crypto/ssh"

	"k8s.io/kubernetes/pkg/api"
)

type req struct {
	keyType string
	keySize int
}

var reqs = []req{
	{},
	{keyType: "ecdsa"},
	{keySize: 19},
	{keySize: 521, keyType: "ecdsa"},
}

func TestGenerate(t *testing.T) {
	generator := New(nil)

	for _, r := range reqs {
		l := ""
		if r.keySize > 0 {
			l = strconv.Itoa(r.keySize)
		}
		genReq := api.GenerateSecretRequest{
			ObjectMeta: api.ObjectMeta{
				Annotations: map[string]string{
					KeyTypeAnnotation: r.keyType,
					KeySizeAnnotation: l,
				},
			},
		}
		vals, err := generator.GenerateValues(&genReq)
		if err != nil {
			t.Errorf("Unexpected error returned from secret generator: %v", err)
		}
		if len(vals) != 2 {
			t.Errorf("Wrong number of generated values")
		}
		keyType := r.keyType
		if keyType == "" {
			keyType = DefaultKeyType
		}
		generatedPrivateKey := vals[PrivateKeyAnnotation]
		priv, err := ssh.ParseRawPrivateKey(generatedPrivateKey)
		if err != nil {
			t.Errorf("Invalid generated private key: %v", err)
		}
		keySize := r.keySize
		var generatedKeySize int
		switch k := priv.(type) {
		case *rsa.PrivateKey:
			if keyType != "rsa" {
				t.Errorf("Wrong type of generated private key (rsa instead of %s)", keyType)
			}
			if keySize == 0 {
				keySize = DefaultRSAKeyLength
			}
			generatedKeySize = k.PublicKey.N.BitLen()
		case *ecdsa.PrivateKey:
			if keyType != "ecdsa" {
				t.Errorf("Wrong type of generated private key (ecdsa instead of %s)", keyType)
			}
			if keySize == 0 {
				keySize = DefaultECDSAKeyLength
			}
			generatedKeySize = k.PublicKey.Curve.Params().BitSize
		default:
			t.Errorf("Unknown generated private key type: %v", reflect.TypeOf(k))
		}
		if generatedKeySize != keySize {
			t.Errorf("Wrong generated key size (%d instead of %d)", generatedKeySize, keySize)
		}
	}
}
