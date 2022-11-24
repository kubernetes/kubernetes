/*
Copyright 2022 The Kubernetes Authors.

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

package service_test

import (
	"bytes"
	"context"
	"testing"

	api "k8s.io/kms/apis/v2alpha1"
	"k8s.io/kms/encryption"
	"k8s.io/kms/service"
)

func TestService(t *testing.T) {
	kms, err := newRemoteKMS([]byte("hello world"))
	if err != nil {
		t.Fatal(err)
	}

	svc, err := service.NewKeyManagementService(kms)
	if err != nil {
		t.Fatal(err)
	}

	var keyID string
	var ciphertext []byte
	var ciphertextAnnotations map[string][]byte

	plaintext := []byte("lorem ipsum")
	t.Run("encryption and decryption", func(t *testing.T) {
		encryptResponse, err := svc.Encrypt(context.TODO(), &api.EncryptRequest{
			Plaintext: plaintext,
			Uid:       "123",
		})
		if err != nil {
			t.Fatal(err)
		}

		decryptResponse, err := svc.Decrypt(context.TODO(), &api.DecryptRequest{
			Ciphertext:  encryptResponse.Ciphertext,
			Uid:         "456",
			KeyId:       encryptResponse.KeyId,
			Annotations: encryptResponse.Annotations,
		})
		if err != nil {
			t.Fatal(err)
		}

		if !bytes.Equal(plaintext, decryptResponse.Plaintext) {
			t.Fatalf(
				"want: %s, have: %s",
				string(plaintext), string(decryptResponse.Plaintext),
			)
		}

		keyID = encryptResponse.KeyId
		ciphertext = encryptResponse.Ciphertext
		ciphertextAnnotations = encryptResponse.Annotations
	})

	t.Run("decrypt by other kms plugin", func(t *testing.T) {
		anotherSvc, err := service.NewKeyManagementService(kms)
		if err != nil {
			t.Fatal(err)
		}

		decryptResponse, err := anotherSvc.Decrypt(context.TODO(), &api.DecryptRequest{
			Ciphertext:  ciphertext,
			Uid:         "789",
			KeyId:       keyID,
			Annotations: ciphertextAnnotations,
		})
		if err != nil {
			t.Fatal(err)
		}

		if !bytes.Equal(plaintext, decryptResponse.Plaintext) {
			t.Errorf(
				"want: %s, have: %s",
				string(plaintext), string(decryptResponse.Plaintext),
			)
		}
	})

	t.Run("decrypt by remote kms", func(t *testing.T) {
		_, ct, err := kms.Encrypt(context.TODO(), plaintext)
		if err != nil {
			t.Fatal(err)
		}

		svc.Decrypt(context.TODO(), &api.DecryptRequest{
			Ciphertext: ct,
			Uid:        "135",
		})
	})
}

type remoteKMS struct {
	currentKeyID []byte
	cipher       *encryption.AESGCM
}

func newRemoteKMS(id []byte) (*remoteKMS, error) {
	key, err := encryption.NewKey()
	if err != nil {
		return nil, err
	}

	cipher, err := encryption.NewAESGCM(key)
	if err != nil {
		return nil, err
	}

	return &remoteKMS{
		cipher:       cipher,
		currentKeyID: id,
	}, nil
}

func (k *remoteKMS) Encrypt(ctx context.Context, pt []byte) ([]byte, []byte, error) {
	ct, err := k.cipher.Encrypt(ctx, pt)
	if err != nil {
		return nil, nil, err
	}

	return k.currentKeyID, ct, nil
}

func (k *remoteKMS) Decrypt(ctx context.Context, observedID, encryptedKey []byte) ([]byte, error) {
	pt, err := k.cipher.Decrypt(ctx, encryptedKey)
	if err != nil {
		return nil, err
	}

	return pt, nil
}
