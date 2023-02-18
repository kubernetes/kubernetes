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
	"crypto/aes"
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"strings"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"
	aestransformer "k8s.io/kms/pkg/encrypt/aes"
	"k8s.io/kms/pkg/value"
	"k8s.io/kms/service"
	"k8s.io/utils/lru"
)

var (
	// emptyContext is an empty slice of bytes. This is passed as value.Context to the
	// GCM transformer. The grpc interface does not provide any additional authenticated data
	// to use with AEAD.
	emptyContext = value.DefaultContext([]byte{})
	// errInvalidKMSAnnotationKeySuffix is returned when the annotation key suffix is not allowed.
	errInvalidKMSAnnotationKeySuffix = fmt.Errorf("annotation keys are not allowed to use %s", referenceSuffix)

	// these are var instead of const so that we can set them during tests
	localKEKGenerationPollInterval = 1 * time.Second
	localKEKGenerationPollTimeout  = 5 * time.Minute
)

const (
	referenceSuffix = ".reference.encryption.k8s.io"
	// referenceKEKAnnotationKey is the key used to store the localKEK in the annotations.
	referenceKEKAnnotationKey = "encrypted-kek" + referenceSuffix
	numAnnotations            = 1
	cacheSize                 = 1_000
	// keyLength is the length of the local KEK in bytes.
	// This is the same length used for the DEKs generated in kube-apiserver.
	keyLength = 32
)

var _ service.Service = &LocalKEKService{}

// LocalKEKService adds an additional KEK layer to reduce calls to the remote
// KMS.
// The local KEK is generated once and stored in the LocalKEKService. This KEK
// is used for all encryption operations. For the decrypt operation, if the encrypted
// local KEK is not found in the cache, the remote KMS is used to decrypt the local KEK.
type LocalKEKService struct {
	// remoteKMS is the remote kms that is used to encrypt and decrypt the local KEKs.
	remoteKMS  service.Service
	remoteOnce sync.Once

	// transformers is a thread-safe LRU cache which caches decrypted DEKs indexed by their encrypted form.
	transformers *lru.Cache

	remoteKMSResponse   *service.EncryptResponse
	localTransformer    value.Transformer
	localTransformerErr error
}

// NewLocalKEKService is being initialized with a remote KMS service.
// In the current implementation, the localKEK Service needs to be
// restarted by the caller after security thresholds are met.
// TODO(aramase): handle rotation of local KEKs
//   - when the keyID in Status() no longer matches the keyID used during encryption
//   - when the local KEK has been used for a certain number of times
func NewLocalKEKService(remoteService service.Service) *LocalKEKService {
	return &LocalKEKService{
		remoteKMS:    remoteService,
		transformers: lru.New(cacheSize),
	}
}

func (m *LocalKEKService) getTransformerForEncryption(uid string) (value.Transformer, *service.EncryptResponse, error) {
	// Check if we have a local KEK
	//	- If exists, use the local KEK for encryption and return
	//  - Not exists, generate local KEK, encrypt with remote KEK,
	//	store it in cache encrypt the data and return. This can be
	// 	expensive but only 1 in N calls will incur this additional latency,
	// 	N being number of times local KEK is reused)
	m.remoteOnce.Do(func() {
		m.localTransformerErr = wait.PollImmediateWithContext(context.Background(), localKEKGenerationPollInterval, localKEKGenerationPollTimeout,
			func(ctx context.Context) (done bool, err error) {
				key, err := generateKey(keyLength)
				if err != nil {
					return false, fmt.Errorf("failed to generate local KEK: %w", err)
				}
				block, err := aes.NewCipher(key)
				if err != nil {
					return false, fmt.Errorf("failed to create cipher block: %w", err)
				}
				transformer := aestransformer.NewGCMTransformer(block)

				resp, err := m.remoteKMS.Encrypt(ctx, uid, key)
				if err != nil {
					klog.ErrorS(err, "failed to encrypt local KEK with remote KMS", "uid", uid)
					return false, nil
				}
				if err = validateRemoteKMSResponse(resp); err != nil {
					return false, fmt.Errorf("response annotations failed validation: %w", err)
				}
				m.remoteKMSResponse = copyResponseAndAddLocalKEKAnnotation(resp)
				m.localTransformer = transformer
				m.transformers.Add(base64.StdEncoding.EncodeToString(resp.Ciphertext), transformer)
				return true, nil
			})
	})
	return m.localTransformer, m.remoteKMSResponse, m.localTransformerErr
}

func copyResponseAndAddLocalKEKAnnotation(resp *service.EncryptResponse) *service.EncryptResponse {
	annotations := make(map[string][]byte, len(resp.Annotations)+numAnnotations)
	for s, bytes := range resp.Annotations {
		s := s
		bytes := bytes
		annotations[s] = bytes
	}
	annotations[referenceKEKAnnotationKey] = resp.Ciphertext

	return &service.EncryptResponse{
		// Ciphertext is not set on purpose - it is different per Encrypt call
		KeyID:       resp.KeyID,
		Annotations: annotations,
	}
}

// Encrypt encrypts the plaintext with the localKEK.
func (m *LocalKEKService) Encrypt(ctx context.Context, uid string, pt []byte) (*service.EncryptResponse, error) {
	transformer, resp, err := m.getTransformerForEncryption(uid)
	if err != nil {
		klog.V(2).InfoS("encrypt plaintext", "uid", uid, "err", err)
		return nil, err
	}

	ct, err := transformer.TransformToStorage(ctx, pt, emptyContext)
	if err != nil {
		klog.V(2).InfoS("encrypt plaintext", "uid", uid, "err", err)
		return nil, err
	}

	return &service.EncryptResponse{
		Ciphertext:  ct,
		KeyID:       resp.KeyID, // TODO what about rotation ??
		Annotations: resp.Annotations,
	}, nil
}

func (m *LocalKEKService) getTransformerForDecryption(ctx context.Context, uid string, req *service.DecryptRequest) (value.Transformer, error) {
	encKEK := req.Annotations[referenceKEKAnnotationKey]

	if _transformer, found := m.transformers.Get(base64.StdEncoding.EncodeToString(encKEK)); found {
		return _transformer.(value.Transformer), nil
	}

	key, err := m.remoteKMS.Decrypt(ctx, uid, &service.DecryptRequest{
		Ciphertext:  encKEK,
		KeyID:       req.KeyID,
		Annotations: annotationsWithoutReferenceKeys(req.Annotations),
	})
	if err != nil {
		return nil, err
	}

	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	transformer := aestransformer.NewGCMTransformer(block)

	// Overwrite the plain key with 0s.
	copy(key, make([]byte, len(key)))

	m.transformers.Add(encKEK, transformer)

	return transformer, nil
}

// Decrypt attempts to decrypt the ciphertext with the localKEK, a KEK from the
// store, or the remote KMS.
func (m *LocalKEKService) Decrypt(ctx context.Context, uid string, req *service.DecryptRequest) ([]byte, error) {
	if _, ok := req.Annotations[referenceKEKAnnotationKey]; !ok {
		return nil, fmt.Errorf("unable to find local KEK for request with uid %q", uid)
	}

	transformer, err := m.getTransformerForDecryption(ctx, uid, req)
	if err != nil {
		klog.V(2).InfoS("decrypt ciphertext", "uid", uid, "err", err)
		return nil, fmt.Errorf("failed to get transformer for decryption: %w", err)
	}

	pt, _, err := transformer.TransformFromStorage(ctx, req.Ciphertext, emptyContext)
	if err != nil {
		klog.V(2).InfoS("decrypt ciphertext with pulled key", "uid", uid, "err", err)
		return nil, err
	}

	return pt, nil
}

// Status returns the status of the remote KMS.
func (m *LocalKEKService) Status(ctx context.Context) (*service.StatusResponse, error) {
	// TODO(aramase): the response from the remote KMS is funneled through without any validation/action.
	// This needs to handle the case when remote KEK has changed. The local KEK needs to be rotated and
	// re-encrypted with the new remote KEK.
	return m.remoteKMS.Status(ctx)
}

func annotationsWithoutReferenceKeys(annotations map[string][]byte) map[string][]byte {
	if len(annotations) <= numAnnotations {
		return nil
	}

	m := make(map[string][]byte, len(annotations)-numAnnotations)
	for k, v := range annotations {
		k, v := k, v
		if strings.HasSuffix(k, referenceSuffix) {
			continue
		}
		m[k] = v
	}
	return m
}

func validateRemoteKMSResponse(resp *service.EncryptResponse) error {
	// validate annotations don't contain the reference implementation annotations
	for k := range resp.Annotations {
		if strings.HasSuffix(k, referenceSuffix) {
			return errInvalidKMSAnnotationKeySuffix
		}
	}
	return nil
}

// generateKey generates a random key using system randomness.
func generateKey(length int) (key []byte, err error) {
	key = make([]byte, length)
	if _, err = rand.Read(key); err != nil {
		return nil, err
	}

	return key, nil
}
