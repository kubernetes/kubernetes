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

package service

import (
	"context"

	"k8s.io/klog/v2"
	api "k8s.io/kms/apis/v2alpha1"
	"k8s.io/kms/encryption"
)

const (
	version              = "v2alpha1"
	encryptedLocalKEKKey = "encryptedLocalKEK"
)

// Service offers encryption and decryption cache upfront of an remote KMS.
type Service struct {
	managedKeys *encryption.ManagedCipher
}

var (
	_ api.KeyManagementServiceServer = (*Service)(nil)
)

// NewKeyManagementService creates an v2alpha1.KeyManagementServiceServer that
// can be used for encryption and decryption, if given an remote encryption
// service (remote KMS).
func NewKeyManagementService(remoteCipher encryption.EncrypterDecrypter) (api.KeyManagementServiceServer, error) {
	ctx := context.Background()
	mk, err := encryption.NewManagedCipher(ctx, remoteCipher)
	if err != nil {
		klog.V(2).Infof("create key management service: %w", err)
		return nil, err
	}

	klog.V(4).Infof("new key management service created")

	return &Service{
		managedKeys: mk,
	}, nil
}

// Status returns version data to verify the state of the service.
func (s *Service) Status(ctx context.Context, _ *api.StatusRequest) (*api.StatusResponse, error) {
	return &api.StatusResponse{
		Version: version,
		Healthz: "ok",
		KeyId:   s.managedKeys.CurrentKeyID(),
	}, nil
}

// Decrypt decrypts the given request. If no encrypted local KEK is given in
// the metadata section, the assumption is that the ciphertext is being
// decrypted directly by the remote KMS.
// Returns the assumed current key id. It is being synced if the
// local kek is unknown or not given at all.
func (s *Service) Decrypt(ctx context.Context, req *api.DecryptRequest) (*api.DecryptResponse, error) {
	klog.V(4).Infof("decrypt request (id: %q) received", req.Uid)

	encryptedLocalKEK, ok := req.Annotations[encryptedLocalKEKKey]
	if ok {
		pt, err := s.managedKeys.Decrypt(ctx, req.KeyId, encryptedLocalKEK, req.Ciphertext)
		if err != nil {
			klog.V(4).Infof("decrypt attempt (id: %q) failed: %w", req.Uid, err)
			return nil, err
		}

		klog.V(4).Infof("decrypt request (id: %q) succeeded", req.Uid)

		return &api.DecryptResponse{
			Plaintext: pt,
		}, nil
	}

	pt, err := s.managedKeys.DecryptRemotely(ctx, req.KeyId, req.Ciphertext)
	if err != nil {
		klog.V(4).Infof("decrypt remotely (id: %q) failed: %w", req.Uid, err)
	}

	return &api.DecryptResponse{
		Plaintext: pt,
	}, nil
}

// Encrypt encrypts the given plaintext with the currently used local KEK. The
// currently used local KEK is returned in encrypted form to be communicated in
// the metadata section to enable seamless decryption.
// The encrypted KEK must be sent along a future decryption request to decrypt
// the returned ciphertext.
// Returns also the assumed current key id. It is synchronized on local kek
// rotation.
func (s *Service) Encrypt(ctx context.Context, req *api.EncryptRequest) (*api.EncryptResponse, error) {
	klog.V(4).Infof("encrypt request received (id: %q)", req.Uid)

	remoteKeyID, encryptedLocalKEK, ct, err := s.managedKeys.Encrypt(ctx, req.Plaintext)
	if err != nil {
		klog.V(4).Infof("encrypt attempt (id: %q) failed: %w", req.Uid, err)
		return nil, err
	}

	klog.V(4).Infof("encrypt request (id: %q) succeeded", req.Uid)

	return &api.EncryptResponse{
		KeyId:      remoteKeyID,
		Ciphertext: ct,
		Annotations: map[string][]byte{
			encryptedLocalKEKKey: encryptedLocalKEK,
		},
	}, nil
}
