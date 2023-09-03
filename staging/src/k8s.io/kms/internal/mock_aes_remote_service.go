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

package internal

import (
	"context"
	"crypto/aes"
	"errors"

	aestransformer "k8s.io/kms/pkg/encrypt/aes"
	"k8s.io/kms/pkg/service"
	"k8s.io/kms/pkg/value"
)

var _ service.Service = &mockAESRemoteService{}

const (
	mockAnnotationKey = "version.encryption.remote.io"
)

type mockAESRemoteService struct {
	keyID       string
	transformer value.Transformer
	dataCtx     value.DefaultContext
}

func (s *mockAESRemoteService) Encrypt(ctx context.Context, uid string, plaintext []byte) (*service.EncryptResponse, error) {
	out, err := s.transformer.TransformToStorage(ctx, plaintext, s.dataCtx)
	if err != nil {
		return nil, err
	}

	return &service.EncryptResponse{
		KeyID:      s.keyID,
		Ciphertext: out,
		Annotations: map[string][]byte{
			mockAnnotationKey: []byte("1"),
		},
	}, nil
}

func (s *mockAESRemoteService) Decrypt(ctx context.Context, uid string, req *service.DecryptRequest) ([]byte, error) {
	if len(req.Annotations) != 1 {
		return nil, errors.New("invalid annotations")
	}
	if v, ok := req.Annotations[mockAnnotationKey]; !ok || string(v) != "1" {
		return nil, errors.New("invalid version in annotations")
	}
	if req.KeyID != s.keyID {
		return nil, errors.New("invalid keyID")
	}
	from, _, err := s.transformer.TransformFromStorage(ctx, req.Ciphertext, s.dataCtx)
	if err != nil {
		return nil, err
	}
	return from, nil
}

func (s *mockAESRemoteService) Status(ctx context.Context) (*service.StatusResponse, error) {
	resp := &service.StatusResponse{
		Version: "v2beta1",
		Healthz: "ok",
		KeyID:   s.keyID,
	}
	return resp, nil
}

// NewMockAESService creates an instance of mockAESRemoteService.
func NewMockAESService(aesKey string, keyID string) (service.Service, error) {
	block, err := aes.NewCipher([]byte(aesKey))
	if err != nil {
		return nil, err
	}
	if len(keyID) == 0 {
		return nil, errors.New("invalid keyID")
	}
	return &mockAESRemoteService{
		transformer: aestransformer.NewGCMTransformer(block),
		keyID:       keyID,
		dataCtx:     value.DefaultContext([]byte{}),
	}, nil
}
