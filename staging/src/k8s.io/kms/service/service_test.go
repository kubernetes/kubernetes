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
		_, ct, err := kms.Encrypt(plaintext)
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
	cipher, err := encryption.NewAESGCM()
	if err != nil {
		return nil, err
	}

	return &remoteKMS{
		cipher:       cipher,
		currentKeyID: id,
	}, nil
}

func (k *remoteKMS) Encrypt(pt []byte) ([]byte, []byte, error) {
	ct, err := k.cipher.Encrypt(pt)
	if err != nil {
		return nil, nil, err
	}

	return k.currentKeyID, ct, nil
}

func (k *remoteKMS) Decrypt(observedID, encryptedKey []byte) ([]byte, error) {
	pt, err := k.cipher.Decrypt(encryptedKey)
	if err != nil {
		return nil, err
	}

	return pt, nil
}
