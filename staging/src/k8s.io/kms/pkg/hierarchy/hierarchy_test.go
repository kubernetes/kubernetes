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

package hierarchy

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kms/pkg/service"
	testingclock "k8s.io/utils/clock/testing"
)

func TestCopyResponseAndAddLocalKEKAnnotation(t *testing.T) {
	t.Parallel()
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
			t.Parallel()
			got := copyResponseAndAddLocalKEKAnnotation(tc.input)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("copyResponseAndAddLocalKEKAnnotation(%v) = %v, want %v", tc.input, got, tc.want)
			}
		})
	}
}

func TestAnnotationsWithoutReferenceKeys(t *testing.T) {
	t.Parallel()
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
			t.Parallel()
			got := annotationsWithoutReferenceKeys(tc.input)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("annotationsWithoutReferenceKeys(%v) = %v, want %v", tc.input, got, tc.want)
			}
		})
	}
}

func TestValidateRemoteKMSEncryptResponse(t *testing.T) {
	t.Parallel()
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
			t.Parallel()
			got := validateRemoteKMSEncryptResponse(tc.input)
			if got != tc.want {
				t.Errorf("validateRemoteKMSResponse(%v) = %v, want %v", tc.input, got, tc.want)
			}
		})
	}
}

func TestValidateRemoteKMSStatusResponse(t *testing.T) {
	t.Parallel()
	testCases := []struct {
		name    string
		input   *service.StatusResponse
		wantErr string
	}{
		{
			name: "keyID is empty",
			input: &service.StatusResponse{
				KeyID: "",
			},
			wantErr: "keyID is empty",
		},
		{
			name: "no error",
			input: &service.StatusResponse{
				KeyID: "keyID",
			},
			wantErr: "",
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := validateRemoteKMSStatusResponse(tc.input)
			if tc.wantErr != "" {
				if got == nil {
					t.Errorf("validateRemoteKMSStatusResponse(%v) = %v, want %v", tc.input, got, tc.wantErr)
				}
				if !strings.Contains(got.Error(), tc.wantErr) {
					t.Errorf("validateRemoteKMSStatusResponse(%v) = %v, want %v", tc.input, got, tc.wantErr)
				}
			} else {
				if got != nil {
					t.Errorf("validateRemoteKMSStatusResponse(%v) = %v, want %v", tc.input, got, tc.wantErr)
				}
			}
		})
	}
}

var _ service.Service = &testRemoteService{}

type testRemoteService struct {
	mu sync.Mutex

	keyID            string
	disabled         bool
	encryptCallCount int
	decryptCallCount int
}

func (s *testRemoteService) Encrypt(ctx context.Context, uid string, plaintext []byte) (*service.EncryptResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.encryptCallCount++
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

	s.decryptCallCount++
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

	resp := &service.StatusResponse{
		Version: "v2alpha1",
		Healthz: "ok",
		KeyID:   s.keyID,
	}
	if s.disabled {
		resp.Healthz = "remote KMS is disabled"
	}
	return resp, nil
}

func (s *testRemoteService) SetDisabledStatus(disabled bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.disabled = disabled
}

func (s *testRemoteService) SetKeyID(keyID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.keyID = keyID
}

func (s *testRemoteService) EncryptCallCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.encryptCallCount
}

func (s *testRemoteService) DecryptCallCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.decryptCallCount
}

func TestEncrypt(t *testing.T) {
	ctx := testContext(t)
	remoteKMS := &testRemoteService{keyID: "test-key-id"}
	localKEKService := NewLocalKEKService(ctx, remoteKMS)

	waitUntilReady(t, localKEKService)

	// local KEK is generated and encryption is successful
	got, err := localKEKService.Encrypt(ctx, "test-uid", []byte("test-plaintext"))
	if err != nil {
		t.Fatalf("Encrypt() error = %v", err)
	}
	validateEncryptResponse(t, got, remoteKMS.keyID, localKEKService)

	// local KEK is used for encryption even when remote KMS is failing
	remoteKMS.SetDisabledStatus(true)
	if got, err = localKEKService.Encrypt(ctx, "test-uid", []byte("test-plaintext")); err != nil {
		t.Fatalf("Encrypt() error = %v", err)
	}
	validateEncryptResponse(t, got, remoteKMS.keyID, localKEKService)
}

func TestEncryptError(t *testing.T) {
	ctx := testContext(t)
	remoteKMS := &testRemoteService{keyID: "test-key-id"}
	localKEKService := NewLocalKEKService(ctx, remoteKMS)

	// first time local KEK generation fails because of remote KMS
	remoteKMS.SetDisabledStatus(true)
	_, err := localKEKService.Encrypt(ctx, "test-uid", []byte("test-plaintext"))
	if err == nil {
		t.Fatalf("Encrypt() error = %v, want non-nil", err)
	}
	lk := localKEKService.getLocalKEK()
	if lk.transformer != nil {
		t.Fatalf("Encrypt() localKEKTracker = %v, want non-nil localKEK", lk)
	}

	remoteKMS.SetDisabledStatus(false)
}

func TestDecrypt(t *testing.T) {
	ctx := testContext(t)
	remoteKMS := &testRemoteService{keyID: "test-key-id"}
	localKEKService := NewLocalKEKService(ctx, remoteKMS)

	waitUntilReady(t, localKEKService)

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
	ctx := testContext(t)
	remoteKMS := &testRemoteService{keyID: "test-key-id"}
	localKEKService := NewLocalKEKService(ctx, remoteKMS)

	waitUntilReady(t, localKEKService)

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
	lk := localKEKService.localKEKTracker.Load()
	lk.transformer = nil
	localKEKService.localKEKTracker.Store(lk)

	// clear the cache
	localKEKService.transformers.Clear()
	if _, err = localKEKService.Decrypt(ctx, "test-uid", decryptRequest); err == nil {
		t.Fatalf("Decrypt() error = %v, want non-nil", err)
	}
}

func TestStatus(t *testing.T) {
	ctx := testContext(t)
	fakeClock := testingclock.NewFakeClock(time.Now())
	remoteKMS := &testRemoteService{keyID: "test-key-id"}
	localKEKService := newLocalKEKService(ctx, remoteKMS, 10, 5, 1*time.Second, 100*time.Millisecond, fakeClock)

	waitUntilReady(t, localKEKService)

	got, err := localKEKService.Status(ctx)
	if err != nil {
		t.Fatalf("Status() error = %v", err)
	}
	validateStatusResponse(t, got, "v2alpha1", "ok", "test-key-id")

	fakeClock.Step(1 * time.Second)
	// remote KMS is failing
	remoteKMS.SetDisabledStatus(true)
	// remote KMS keyID changed but local KEK not rotated because of remote KMS failure
	// the keyID in status should be the old keyID
	// the error should still be nil
	remoteKMS.SetKeyID("test-key-id-2")

	if got, err = localKEKService.Status(ctx); err != nil {
		t.Fatalf("Status() error = %v, want nil", err)
	}
	validateStatusResponse(t, got, "v2alpha1", "remote KMS is disabled", "test-key-id")

	fakeClock.Step(1 * time.Second)
	// wait for local KEK to expire and local KEK service ready to be false
	wait.PollImmediate(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		return !localKEKService.isReady.Load(), nil
	})

	// status response should include the localKEK unhealthy status
	if got, err = localKEKService.Status(ctx); err != nil {
		t.Fatalf("Status() error = %v, want nil", err)
	}
	validateStatusResponse(t, got, "v2alpha1", "remote KMS is disabled; localKEKService is not ready", "test-key-id")

	// remote KMS is functional again, local KEK is rotated
	remoteKMS.SetDisabledStatus(false)
	fakeClock.Step(1 * time.Second)
	waitUntilReady(t, localKEKService)
	if got, err = localKEKService.Status(ctx); err != nil {
		t.Fatalf("Status() error = %v, want nil", err)
	}
	validateStatusResponse(t, got, "v2alpha1", "ok", "test-key-id-2")
}

func TestRotationKeyUsage(t *testing.T) {
	ctx := testContext(t)

	var record sync.Map

	fakeClock := testingclock.NewFakeClock(time.Now())
	remoteKMS := &testRemoteService{keyID: "test-key-id"}
	localKEKService := newLocalKEKService(ctx, remoteKMS, 10, 5, 1*time.Minute, 100*time.Millisecond, fakeClock)
	waitUntilReady(t, localKEKService)
	lk := localKEKService.localKEKTracker.Load()
	encLocalKEK := lk.encKEK

	// check only single call for Encrypt to remote KMS
	if remoteKMS.EncryptCallCount() != 1 {
		t.Fatalf("Encrypt() remoteKMS.EncryptCallCount() = %v, want %v", remoteKMS.EncryptCallCount(), 1)
	}

	var wg sync.WaitGroup
	for i := 0; i < 6; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp, err := localKEKService.Encrypt(ctx, "test-uid", []byte(rand.String(32)))
			if err != nil {
				t.Fatalf("Encrypt() error = %v", err)
			}
			if v, ok := resp.Annotations[referenceKEKAnnotationKey]; !ok || !bytes.Equal(v, encLocalKEK) {
				t.Fatalf("Encrypt() annotations = %v, want %v", resp.Annotations, encLocalKEK)
			}
			record.Store(resp, nil)
		}()
	}
	wg.Wait()

	fakeClock.Step(30 * time.Second)
	rotated := false
	// wait for the local KEK to be rotated
	wait.PollImmediate(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		// local KEK must have been rotated after 5 usages
		lk = localKEKService.localKEKTracker.Load()
		rotated = !bytes.Equal(lk.encKEK, encLocalKEK)
		return rotated, nil
	})
	if !rotated {
		t.Fatalf("local KEK must have been rotated")
	}
	if remoteKMS.EncryptCallCount() != 2 {
		t.Fatalf("Encrypt() remoteKMS.EncryptCallCount() = %v, want %v", remoteKMS.EncryptCallCount(), 2)
	}

	// new local KEK must be used for encryption
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp, err := localKEKService.Encrypt(ctx, "test-uid", []byte(rand.String(32)))
			if err != nil {
				t.Fatalf("Encrypt() error = %v", err)
			}
			if v, ok := resp.Annotations[referenceKEKAnnotationKey]; !ok || !bytes.Equal(v, lk.encKEK) {
				t.Fatalf("Encrypt() annotations = %v, want %v", resp.Annotations, lk.encKEK)
			}
			record.Store(resp, nil)
		}()
	}
	wg.Wait()

	// check we can decrypt data encrypted with the old and new local KEKs
	record.Range(func(key, _ any) bool {
		k := key.(*service.EncryptResponse)
		decryptRequest := &service.DecryptRequest{
			Ciphertext:  k.Ciphertext,
			Annotations: k.Annotations,
			KeyID:       k.KeyID,
		}
		if _, err := localKEKService.Decrypt(ctx, "test-uid", decryptRequest); err != nil {
			t.Fatalf("Decrypt() error = %v", err)
		}
		return true
	})

	// Out of the 11 calls to Decrypt:
	// - 5 should be using the current local KEK
	// - 1 out of the 6 should generate a decrypt call to the remote KMS as the local KEK not in cache
	// - 5 out of the 6 should use the cached local KEK after 1st decrypt call to the remote KMS
	assertCallCount(t, remoteKMS, localKEKService)
}

func TestRotationKeyExpiry(t *testing.T) {
	ctx := testContext(t)

	var record sync.Map

	fakeClock := testingclock.NewFakeClock(time.Now())
	remoteKMS := &testRemoteService{keyID: "test-key-id"}
	localKEKService := newLocalKEKService(ctx, remoteKMS, 10, 5, 5*time.Second, 100*time.Millisecond, fakeClock)
	waitUntilReady(t, localKEKService)
	lk := localKEKService.localKEKTracker.Load()
	encLocalKEK := lk.encKEK

	var wg sync.WaitGroup
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp, err := localKEKService.Encrypt(ctx, "test-uid", []byte("test-plaintext"))
			if err != nil {
				t.Fatalf("Encrypt() error = %v", err)
			}
			if v, ok := resp.Annotations[referenceKEKAnnotationKey]; !ok || !bytes.Equal(v, encLocalKEK) {
				t.Fatalf("Encrypt() annotations = %v, want %v", resp.Annotations, encLocalKEK)
			}
			record.Store(resp, nil)
		}()
	}
	wg.Wait()

	// check local KEK has only been used 3 times and still under the suggested usage
	if lk.usage.Load() != 3 {
		t.Fatalf("local KEK usage = %v, want %v", lk.usage.Load(), 3)
	}

	// advance the clock to trigger key expiry
	fakeClock.Step(6 * time.Second)

	rotated := false
	// wait for the local KEK to be rotated due to key expiry
	wait.PollImmediate(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		// local KEK must have been rotated after the key max age
		t.Log("waiting for local KEK to be rotated")
		lk = localKEKService.localKEKTracker.Load()
		rotated = !bytes.Equal(lk.encKEK, encLocalKEK)
		return rotated, nil
	})
	if !rotated {
		t.Fatalf("local KEK must have been rotated")
	}

	// new local KEK must be used for encryption
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp, err := localKEKService.Encrypt(ctx, "test-uid", []byte("test-plaintext"))
			if err != nil {
				t.Fatalf("Encrypt() error = %v", err)
			}
			if v, ok := resp.Annotations[referenceKEKAnnotationKey]; !ok || !bytes.Equal(v, lk.encKEK) {
				t.Fatalf("Encrypt() annotations = %v, want %v", resp.Annotations, lk.encKEK)
			}
			record.Store(resp, nil)
		}()
	}
	wg.Wait()

	// check we can decrypt data encrypted with the old and new local KEKs
	record.Range(func(key, _ any) bool {
		k := key.(*service.EncryptResponse)
		decryptRequest := &service.DecryptRequest{
			Ciphertext:  k.Ciphertext,
			Annotations: k.Annotations,
			KeyID:       k.KeyID,
		}
		if _, err := localKEKService.Decrypt(ctx, "test-uid", decryptRequest); err != nil {
			t.Fatalf("Decrypt() error = %v", err)
		}
		return true
	})

	// Out of the 8 calls to Decrypt:
	// - 5 should be using the current local KEK
	// - 1 out of the 3 should generate a decrypt call to the remote KMS as the local KEK not in cache
	// - 2 out of the 3 should use the cached local KEK after 1st decrypt call to the remote KMS
	assertCallCount(t, remoteKMS, localKEKService)
}

func TestRotationRemoteKeyIDChanged(t *testing.T) {
	ctx := testContext(t)

	var record sync.Map

	fakeClock := testingclock.NewFakeClock(time.Now())
	remoteKMS := &testRemoteService{keyID: "test-key-id"}
	localKEKService := newLocalKEKService(ctx, remoteKMS, 10, 5, 1*time.Minute, 100*time.Millisecond, fakeClock)
	waitUntilReady(t, localKEKService)
	lk := localKEKService.localKEKTracker.Load()
	encLocalKEK := lk.encKEK

	var wg sync.WaitGroup
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp, err := localKEKService.Encrypt(ctx, "test-uid", []byte("test-plaintext"))
			if err != nil {
				t.Fatalf("Encrypt() error = %v", err)
			}
			if v, ok := resp.Annotations[referenceKEKAnnotationKey]; !ok || !bytes.Equal(v, encLocalKEK) {
				t.Fatalf("Encrypt() annotations = %v, want %v", resp.Annotations, encLocalKEK)
			}
			record.Store(resp, nil)
		}()
	}
	wg.Wait()

	// check local KEK has only been used 3 times and still under the suggested usage
	if lk.usage.Load() != 3 {
		t.Fatalf("local KEK usage = %v, want %v", lk.usage.Load(), 3)
	}

	fakeClock.Step(30 * time.Second)
	// change the remote key ID
	remoteKMS.SetKeyID("test-key-id-2")
	if _, err := localKEKService.Status(ctx); err != nil {
		t.Fatalf("Status() error = %v", err)
	}

	rotated := false
	// wait for the local KEK to be rotated due to remote key ID change
	wait.PollImmediate(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		lk = localKEKService.localKEKTracker.Load()
		rotated = !bytes.Equal(lk.encKEK, encLocalKEK)
		return rotated, nil
	})
	if !rotated {
		t.Fatalf("local KEK must have been rotated")
	}

	// new local KEK must be used for encryption
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp, err := localKEKService.Encrypt(ctx, "test-uid", []byte("test-plaintext"))
			if err != nil {
				t.Fatalf("Encrypt() error = %v", err)
			}
			if v, ok := resp.Annotations[referenceKEKAnnotationKey]; !ok || !bytes.Equal(v, lk.encKEK) {
				t.Fatalf("Encrypt() annotations = %v, want %v", resp.Annotations, lk.encKEK)
			}
			record.Store(resp, nil)
		}()
	}
	wg.Wait()

	// check we can decrypt data encrypted with the old and new local KEKs
	record.Range(func(key, _ any) bool {
		k := key.(*service.EncryptResponse)
		decryptRequest := &service.DecryptRequest{
			Ciphertext:  k.Ciphertext,
			Annotations: k.Annotations,
			KeyID:       k.KeyID,
		}
		if _, err := localKEKService.Decrypt(ctx, "test-uid", decryptRequest); err != nil {
			t.Fatalf("Decrypt() error = %v", err)
		}
		return true
	})

	// Out of the 8 calls to Decrypt:
	// - 5 should be using the current local KEK
	// - 1 out of the 3 should generate a decrypt call to the remote KMS as the local KEK not in cache
	// - 2 out of the 3 should use the cached local KEK after 1st decrypt call to the remote KMS
	assertCallCount(t, remoteKMS, localKEKService)
}

func testContext(t *testing.T) context.Context {
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	return ctx
}

func waitUntilReady(t *testing.T, s *LocalKEKService) {
	t.Helper()
	wait.PollImmediate(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		return s.isReady.Load(), nil
	})
}

func validateEncryptResponse(t *testing.T, got *service.EncryptResponse, wantKeyID string, localKEKService *LocalKEKService) {
	t.Helper()
	if len(got.Annotations) != 2 {
		t.Fatalf("Encrypt() annotations = %v, want 2 annotations", got.Annotations)
	}
	if _, ok := got.Annotations[referenceKEKAnnotationKey]; !ok {
		t.Fatalf("Encrypt() annotations = %v, want %v", got.Annotations, referenceKEKAnnotationKey)
	}
	if got.KeyID != wantKeyID {
		t.Fatalf("Encrypt() keyID = %v, want %v", got.KeyID, wantKeyID)
	}
	if localKEKService.localKEKTracker.Load() == nil {
		t.Fatalf("Encrypt() localKEKTracker = %v, want non-nil localKEK", localKEKService.localKEKTracker.Load())
	}
}

func validateStatusResponse(t *testing.T, got *service.StatusResponse, wantVersion, wantHealthz, wantKeyID string) {
	t.Helper()
	if got.Version != wantVersion {
		t.Fatalf("Status() version = %v, want %v", got.Version, wantVersion)
	}
	if !strings.EqualFold(got.Healthz, wantHealthz) {
		t.Fatalf("Status() healthz = %v, want %v", got.Healthz, wantHealthz)
	}
	if got.KeyID != wantKeyID {
		t.Fatalf("Status() keyID = %v, want %v", got.KeyID, wantKeyID)
	}
}

func assertCallCount(t *testing.T, remoteKMS *testRemoteService, localKEKService *LocalKEKService) {
	t.Helper()
	if remoteKMS.DecryptCallCount() != 1 {
		t.Fatalf("Decrypt() remoteKMS.DecryptCallCount() = %v, want %v", remoteKMS.DecryptCallCount(), 1)
	}
	if localKEKService.transformers.Len() != 1 {
		t.Fatalf("Decrypt() localKEKService.transformers.Len() = %v, want %v", localKEKService.transformers.Len(), 1)
	}
}
