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
	"errors"

	"k8s.io/klog/v2"
	"k8s.io/kms/service"
)

var (
	// ErrNoCipher means that there is no remote kms given and therefore the keys in use can't be protected.
	ErrNoCipher = errors.New("no remote encryption service was specified")
	// EmptyContext is an empty slice of bytes.
	EmptyContext = DefaultContext([]byte{})
	// LocalKEKID is the key used to store the localKEK in the annotations.
	LocalKEKID = "kmsv2:local-kek"
)

// LocalKEKService adds an additional KEK layer to reduce calls to the remote
// KMS.
// The KEKs are stored as transformers in the local store. The encrypted
// form of the KEK is used to pick a transformer from the store. The KEKs should
// be encrypted by the remote KMS.
// There is a distinguished KEK (localKEK), that is generated and used by the
// LocalKEKService to encrypt.
type LocalKEKService struct {
	// remoteKMS is the remote kms that is used to encrypt the local KEKs.
	remoteKMS service.Service
	//  remoteKMSID is the ID that helps remoteKMS to decrypt localKEKID.
	remoteKMSID string
	// localKEKID is the localKEK in encrypted form.
	localKEKID []byte

	// transformers is a store that holds all known transformers.
	transformers Store
	// createTransformer creates a new transformer and appropriate keys.
	createTransformer CreateTransformer
	// createUID creates a new uid.
	createUID func() (string, error)
}

// NewLocalKEKService is being initialized with a key that is encrypted by the
// remoteService. In the current implementation, the localKEK Service needs to be
// restarted by the caller after security thresholds are met.
func NewLocalKEKService(
	ctx context.Context,
	remoteService service.Service,
	store Store,
	createTransformer CreateTransformer,
	createUID func() (string, error), // TODO add sensible defaults, use functional options
) (*LocalKEKService, error) {
	if remoteService == nil {
		klog.V(2).InfoS("can't create LocalKEKService without remoteService")
		return nil, ErrNoCipher
	}

	key, err := createTransformer.Key()
	if err != nil {
		klog.V(2).InfoS("create key", "err", err)
		return nil, err
	}

	transformer, err := createTransformer.Transformer(ctx, key)
	if err != nil {
		klog.V(2).InfoS("create new cipher", "err", err)
		return nil, err
	}

	uid, err := createUID()
	if err != nil {
		klog.V(2).InfoS("create new uid", "err", err)
		return nil, err
	}

	encRes, err := remoteService.Encrypt(ctx, uid, key)
	if err != nil {
		klog.V(2).InfoS("encrypt with remote", "err", err)
		return nil, err
	}

	store.Add(encRes.Ciphertext, transformer)

	return &LocalKEKService{
		remoteKMSID: encRes.KeyID,
		remoteKMS:   remoteService,
		localKEKID:  encRes.Ciphertext,

		transformers:      store,
		createTransformer: createTransformer,
		createUID:         createUID,
	}, nil
}

// getTransformer returns the transformer for the given keyID. If the keyID is
// not known, the key gets decrypted by the remoteKMS.
func (m *LocalKEKService) getTransformer(ctx context.Context, encKey []byte, uid, keyID string) (Transformer, error) {
	transformer, ok := m.transformers.Get(encKey)
	if ok {
		return transformer, nil
	}

	// Decrypt the unknown key with remote KMS. Plainkey must be treated with secrecy.
	plainKey, err := m.remoteKMS.Decrypt(ctx, uid, &service.DecryptRequest{
		Ciphertext: encKey,
		KeyID:      keyID,
	})
	if err != nil {
		klog.V(2).InfoS("decrypt key with remote key", "id", uid, "err", err)

		return nil, err
	}

	t, err := m.createTransformer.Transformer(ctx, plainKey)
	if err != nil {
		klog.V(2).InfoS("create transformer", "id", uid, "err", err)
		return nil, err
	}

	// Overwrite the plain key with 0s.
	copy(plainKey, make([]byte, len(plainKey)))

	m.transformers.Add(encKey, t)

	return t, nil
}

// Encrypt encrypts the plaintext with the localKEK.
func (m *LocalKEKService) Encrypt(ctx context.Context, uid string, pt []byte) (*service.EncryptResponse, error) {
	// It could happen that the localKEK is not available, if the store is an expiring cache.
	transformer, err := m.getTransformer(ctx, m.localKEKID, uid, m.remoteKMSID)
	if err != nil {
		klog.V(2).InfoS("encrypt plaintext", "id", uid, "err", err)
		return nil, err
	}

	ct, err := transformer.TransformToStorage(ctx, pt, EmptyContext)
	if err != nil {
		klog.V(2).InfoS("encrypt plaintext", "id", uid, "err", err)
		return nil, err
	}

	return &service.EncryptResponse{
		Ciphertext: ct,
		KeyID:      m.remoteKMSID,
		Annotations: map[string][]byte{
			LocalKEKID: m.localKEKID,
		},
	}, nil
}

// Decrypt attempts to decrypt the ciphertext with the localKEK, a KEK from the
// store, or the remote KMS.
func (m *LocalKEKService) Decrypt(ctx context.Context, uid string, req *service.DecryptRequest) ([]byte, error) {
	encKEK, ok := req.Annotations[LocalKEKID]
	if !ok {
		// If there is no local KEK ID in the annotations, we must delegate to remote KMS.
		pt, err := m.remoteKMS.Decrypt(ctx, uid, req)
		if err != nil {
			klog.V(2).InfoS("decrypt key with remote key", "id", uid, "err", err)

			return nil, err
		}

		return pt, nil
	}

	transformer, err := m.getTransformer(ctx, encKEK, uid, req.KeyID)
	if err != nil {
		klog.V(2).InfoS("decrypt ciphertext", "id", uid, "err", err)
		return nil, err
	}

	pt, _, err := transformer.TransformFromStorage(ctx, req.Ciphertext, EmptyContext)
	if err != nil {
		klog.V(2).InfoS("decrypt ciphertext with pulled key", "id", uid, "err", err)
		return nil, err
	}

	return pt, nil
}
