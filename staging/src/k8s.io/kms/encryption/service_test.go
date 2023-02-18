/*
Copyright 2023 The Kubernetes Authors.

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

package encryption

import (
	"context"
	"encoding/base64"
	"errors"
	"reflect"
	"sync"
	"testing"
	"time"

	"k8s.io/kms/service"
)

func TestCopyResponseAndAddLocalKEKAnnotation(t *testing.T) {
	testCases := []struct {
		name  string
		input *service.EncryptResponse
		want  *service.EncryptResponse
	}{
		{
			name: "annotations is nil",
			input: &service.EncryptResponse{
				Ciphertext:  []byte("encryptedLocalKEK"),
				KeyID:       "keyID",
				Annotations: nil,
			},
			want: &service.EncryptResponse{
				KeyID: "keyID",
				Annotations: map[string][]byte{
					referenceKEKAnnotationKey: []byte("encryptedLocalKEK"),
				},
			},
		},
		{
			name: "remote KMS sent 1 annotation",
			input: &service.EncryptResponse{
				Ciphertext: []byte("encryptedLocalKEK"),
				KeyID:      "keyID",
				Annotations: map[string][]byte{
					"version.encryption.remote.io": []byte("1"),
				},
			},
			want: &service.EncryptResponse{
				KeyID: "keyID",
				Annotations: map[string][]byte{
					"version.encryption.remote.io": []byte("1"),
					referenceKEKAnnotationKey:      []byte("encryptedLocalKEK"),
				},
			},
		},
		{
			name: "remote KMS sent 2 annotations",
			input: &service.EncryptResponse{
				Ciphertext: []byte("encryptedLocalKEK"),
				KeyID:      "keyID",
				Annotations: map[string][]byte{
					"version.encryption.remote.io":     []byte("1"),
					"key-version.encryption.remote.io": []byte("2"),
				},
			},
			want: &service.EncryptResponse{
				KeyID: "keyID",
				Annotations: map[string][]byte{
					"version.encryption.remote.io":     []byte("1"),
					"key-version.encryption.remote.io": []byte("2"),
					referenceKEKAnnotationKey:          []byte("encryptedLocalKEK"),
				},
			},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			got := copyResponseAndAddLocalKEKAnnotation(tc.input)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("copyResponseAndAddLocalKEKAnnotation(%v) = %v, want %v", tc.input, got, tc.want)
			}
		})
	}
}

func TestAnnotationsWithoutReferenceKeys(t *testing.T) {
	testCases := []struct {
		name  string
		input map[string][]byte
		want  map[string][]byte
	}{
		{
			name:  "annotations is nil",
			input: nil,
			want:  nil,
		},
		{
			name:  "annotations is empty",
			input: map[string][]byte{},
			want:  nil,
		},
		{
			name: "annotations only contains reference keys",
			input: map[string][]byte{
				referenceKEKAnnotationKey: []byte("encryptedLocalKEK"),
			},
			want: nil,
		},
		{
			name: "annotations contains 1 reference key and 1 other key",
			input: map[string][]byte{
				referenceKEKAnnotationKey:      []byte("encryptedLocalKEK"),
				"version.encryption.remote.io": []byte("1"),
			},
			want: map[string][]byte{
				"version.encryption.remote.io": []byte("1"),
			},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			got := annotationsWithoutReferenceKeys(tc.input)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("annotationsWithoutReferenceKeys(%v) = %v, want %v", tc.input, got, tc.want)
			}
		})
	}
}

func TestValidateRemoteKMSResponse(t *testing.T) {
	testCases := []struct {
		name  string
		input *service.EncryptResponse
		want  error
	}{
		{
			name:  "annotations is nil",
			input: &service.EncryptResponse{},
			want:  nil,
		},
		{
			name: "annotation key contains reference suffix",
			input: &service.EncryptResponse{
				Annotations: map[string][]byte{
					"version.reference.encryption.k8s.io": []byte("1"),
				},
			},
			want: errInvalidKMSAnnotationKeySuffix,
		},
		{
			name: "no annotation key contains reference suffix",
			input: &service.EncryptResponse{
				Annotations: map[string][]byte{
					"version.encryption.remote.io":     []byte("1"),
					"key-version.encryption.remote.io": []byte("2"),
				},
			},
			want: nil,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			got := validateRemoteKMSResponse(tc.input)
			if got != tc.want {
				t.Errorf("validateRemoteKMSResponse(%v) = %v, want %v", tc.input, got, tc.want)
			}
		})
	}
}

var _ service.Service = &testRemoteService{}

type testRemoteService struct {
	mu sync.Mutex

	keyID    string
	disabled bool
}

func (s *testRemoteService) Encrypt(ctx context.Context, uid string, plaintext []byte) (*service.EncryptResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.disabled {
		return nil, errors.New("failed to encrypt")
	}
	return &service.EncryptResponse{
		KeyID:      s.keyID,
		Ciphertext: []byte(base64.StdEncoding.EncodeToString(plaintext)),
		Annotations: map[string][]byte{
			"version.encryption.remote.io": []byte("1"),
		},
	}, nil
}

func (s *testRemoteService) Decrypt(ctx context.Context, uid string, req *service.DecryptRequest) ([]byte, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.disabled {
		return nil, errors.New("failed to decrypt")
	}
	if len(req.Annotations) != 1 {
		return nil, errors.New("invalid annotations")
	}
	if v, ok := req.Annotations["version.encryption.remote.io"]; !ok || string(v) != "1" {
		return nil, errors.New("invalid version in annotations")
	}
	return base64.StdEncoding.DecodeString(string(req.Ciphertext))
}

func (s *testRemoteService) Status(ctx context.Context) (*service.StatusResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.disabled {
		return nil, errors.New("failed to get status")
	}
	return &service.StatusResponse{
		Version: "v2alpha1",
		Healthz: "ok",
		KeyID:   s.keyID,
	}, nil
}

func (s *testRemoteService) SetDisabledStatus(disabled bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.disabled = true
}

func TestEncrypt(t *testing.T) {
	remoteKMS := &testRemoteService{keyID: "test-key-id"}
	localKEKService := NewLocalKEKService(remoteKMS)

	validateResponse := func(got *service.EncryptResponse, t *testing.T) {
		if len(got.Annotations) != 2 {
			t.Fatalf("Encrypt() annotations = %v, want 2 annotations", got.Annotations)
		}
		if _, ok := got.Annotations[referenceKEKAnnotationKey]; !ok {
			t.Fatalf("Encrypt() annotations = %v, want %v", got.Annotations, referenceKEKAnnotationKey)
		}
		if got.KeyID != remoteKMS.keyID {
			t.Fatalf("Encrypt() keyID = %v, want %v", got.KeyID, remoteKMS.keyID)
		}
		if localKEKService.localTransformer == nil {
			t.Fatalf("Encrypt() localTransformer = %v, want non-nil", localKEKService.localTransformer)
		}
	}

	ctx := testContext(t)
	// local KEK is generated and encryption is successful
	got, err := localKEKService.Encrypt(ctx, "test-uid", []byte("test-plaintext"))
	if err != nil {
		t.Fatalf("Encrypt() error = %v", err)
	}
	validateResponse(got, t)

	// local KEK is used for encryption even when remote KMS is failing
	remoteKMS.SetDisabledStatus(true)
	if got, err = localKEKService.Encrypt(ctx, "test-uid", []byte("test-plaintext")); err != nil {
		t.Fatalf("Encrypt() error = %v", err)
	}
	validateResponse(got, t)
}

func TestEncryptError(t *testing.T) {
	remoteKMS := &testRemoteService{keyID: "test-key-id"}
	localKEKService := NewLocalKEKService(remoteKMS)

	ctx := testContext(t)

	localKEKGenerationPollTimeout = 5 * time.Second
	// first time local KEK generation fails because of remote KMS
	remoteKMS.SetDisabledStatus(true)
	_, err := localKEKService.Encrypt(ctx, "test-uid", []byte("test-plaintext"))
	if err == nil {
		t.Fatalf("Encrypt() error = %v, want non-nil", err)
	}
	if localKEKService.localTransformer != nil {
		t.Fatalf("Encrypt() localTransformer = %v, want nil", localKEKService.localTransformer)
	}

	remoteKMS.SetDisabledStatus(false)
}

func TestDecrypt(t *testing.T) {
	remoteKMS := &testRemoteService{keyID: "test-key-id"}
	localKEKService := NewLocalKEKService(remoteKMS)

	ctx := testContext(t)

	// local KEK is generated and encryption/decryption is successful
	got, err := localKEKService.Encrypt(ctx, "test-uid", []byte("test-plaintext"))
	if err != nil {
		t.Fatalf("Encrypt() error = %v", err)
	}
	if string(got.Ciphertext) == "test-plaintext" {
		t.Fatalf("Encrypt() ciphertext = %v, want it to be encrypted", got.Ciphertext)
	}
	decryptRequest := &service.DecryptRequest{
		Ciphertext:  got.Ciphertext,
		Annotations: got.Annotations,
		KeyID:       got.KeyID,
	}
	plaintext, err := localKEKService.Decrypt(ctx, "test-uid", decryptRequest)
	if err != nil {
		t.Fatalf("Decrypt() error = %v", err)
	}
	if string(plaintext) != "test-plaintext" {
		t.Fatalf("Decrypt() plaintext = %v, want %v", string(plaintext), "test-plaintext")
	}

	// local KEK is used for decryption even when remote KMS is failing
	remoteKMS.SetDisabledStatus(true)
	if _, err = localKEKService.Decrypt(ctx, "test-uid", decryptRequest); err != nil {
		t.Fatalf("Decrypt() error = %v", err)
	}
}

func TestDecryptError(t *testing.T) {
	remoteKMS := &testRemoteService{keyID: "test-key-id"}
	localKEKService := NewLocalKEKService(remoteKMS)

	ctx := testContext(t)

	got, err := localKEKService.Encrypt(ctx, "test-uid", []byte("test-plaintext"))
	if err != nil {
		t.Fatalf("Encrypt() error = %v", err)
	}
	decryptRequest := &service.DecryptRequest{
		Ciphertext:  got.Ciphertext,
		Annotations: got.Annotations,
		KeyID:       got.KeyID,
	}
	// local KEK for decryption not in cache and remote KMS is failing
	remoteKMS.SetDisabledStatus(true)
	// clear the cache
	localKEKService.transformers.Clear()
	if _, err = localKEKService.Decrypt(ctx, "test-uid", decryptRequest); err == nil {
		t.Fatalf("Decrypt() error = %v, want non-nil", err)
	}
}

func TestStatus(t *testing.T) {
	remoteKMS := &testRemoteService{keyID: "test-key-id"}
	localKEKService := NewLocalKEKService(remoteKMS)

	ctx := testContext(t)

	got, err := localKEKService.Status(ctx)
	if err != nil {
		t.Fatalf("Status() error = %v", err)
	}
	if got.Version != "v2alpha1" {
		t.Fatalf("Status() version = %v, want %v", got.Version, "v2alpha1")
	}
	if got.Healthz != "ok" {
		t.Fatalf("Status() healthz = %v, want %v", got.Healthz, "ok")
	}
	if got.KeyID != "test-key-id" {
		t.Fatalf("Status() keyID = %v, want %v", got.KeyID, "test-key-id")
	}

	// remote KMS is failing
	remoteKMS.SetDisabledStatus(true)
	if _, err = localKEKService.Status(ctx); err == nil {
		t.Fatalf("Status() error = %v, want non-nil", err)
	}
}

func testContext(t *testing.T) context.Context {
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	return ctx
}
