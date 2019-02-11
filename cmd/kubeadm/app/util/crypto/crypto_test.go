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

package crypto

import (
	"testing"

	"github.com/lithammer/dedent"

	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

func TestEncryptAndDecryptData(t *testing.T) {
	key1, err := CreateRandBytes(kubeadmconstants.CertificateKeySize)
	if err != nil {
		t.Fatal(err)
	}
	key2, err := CreateRandBytes(kubeadmconstants.CertificateKeySize)
	if err != nil {
		t.Fatal(err)
	}
	testData := []byte("Lorem ipsum dolor sit amet, consectetur adipiscing elit.")

	tests := map[string]struct {
		encryptKey       []byte
		decryptKey       []byte
		data             []byte
		expectDecryptErr bool
	}{
		"can decrypt using the correct key": {
			encryptKey:       key1,
			decryptKey:       key1,
			data:             testData,
			expectDecryptErr: false,
		},
		"can't decrypt using incorrect key": {
			encryptKey:       key1,
			decryptKey:       key2,
			data:             testData,
			expectDecryptErr: true,
		},
		"can't decrypt without a key": {
			encryptKey:       key1,
			decryptKey:       []byte{},
			data:             testData,
			expectDecryptErr: true,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t2 *testing.T) {
			encryptedData, err := EncryptBytes(test.data, test.encryptKey)
			if err != nil {
				t2.Fatalf(dedent.Dedent(
					"EncryptBytes failed\nerror: %v"),
					err,
				)
			}

			decryptedData, err := DecryptBytes(encryptedData, test.decryptKey)
			if (err != nil) != test.expectDecryptErr {
				t2.Fatalf(dedent.Dedent(
					"DecryptBytes failed\nexpected error: %t\n\tgot: %t\nerror: %v"),
					test.expectDecryptErr,
					(err != nil),
					err,
				)
			}

			if (string(decryptedData) != string(test.data)) && !test.expectDecryptErr {
				t2.Fatalf(dedent.Dedent(
					"EncryptDecryptBytes failed\nexpected decryptedData equal to data\n\tgot: data=%q decryptedData=%q"),
					test.data,
					string(decryptedData),
				)
			}
		})
	}
}
