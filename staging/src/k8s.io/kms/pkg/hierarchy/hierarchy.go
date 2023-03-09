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
	"crypto/aes"
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"
	aestransformer "k8s.io/kms/pkg/encrypt/aes"
	"k8s.io/kms/pkg/service"
	"k8s.io/kms/pkg/value"
	"k8s.io/utils/clock"
	"k8s.io/utils/lru"
)

// localKEK is a struct that holds the local KEK and the remote KMS response.
type localKEK struct {
	encKEK            []byte
	usage             atomic.Uint64
	expiry            time.Time
	transformer       value.Transformer
	remoteKMSResponse *service.EncryptResponse
	generatedAt       time.Time
}

var (
	// emptyContext is an empty slice of bytes. This is passed as value.Context to the
	// GCM transformer. The grpc interface does not provide any additional authenticated data
	// to use with AEAD.
	emptyContext = value.DefaultContext([]byte{})
	// errInvalidKMSAnnotationKeySuffix is returned when the annotation key suffix is not allowed.
	errInvalidKMSAnnotationKeySuffix = fmt.Errorf("annotation keys are not allowed to use %s", referenceSuffix)
)

const (
	referenceSuffix = ".reference.encryption.k8s.io"
	// referenceKEKAnnotationKey is the key used to store the localKEK in the annotations.
	referenceKEKAnnotationKey = "encrypted-kek" + referenceSuffix
	numAnnotations            = 1
	cacheSize                 = 1_000

	// localKEKGenerationPollInterval is the interval at which the local KEK is checked for rotation.
	localKEKGenerationPollInterval = 1 * time.Minute

	// keyLength is the length of the local KEK in bytes.
	// This is the same length used for the DEKs generated in kube-apiserver.
	keyLength = 32
	// keyMaxUsage is the maximum number of times an AES GCM key can be used
	// with a random nonce: 2^32. The local KEK is a transformer that hold an
	// AES GCM key. It is based on recommendations from
	// https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-38d.pdf.
	// It is reduced by one to be comparable with a atomic.Uint32.
	// We picked a arbitrary number that is less than the max usage of the local KEK.
	keyMaxUsage = 1<<22 - 1
	// keySuggestedUsage is a threshold that triggers the rotation of a new local KEK. It means that half
	// the number of times a local KEK can be used has been reached.
	keySuggestedUsage = 1 << 21
	// keyMaxAge is the maximum age of a local KEK. It is not a cryptographic necessity.
	keyMaxAge = 7 * 24 * time.Hour
)

var _ service.Service = &LocalKEKService{}

// LocalKEKService adds an additional KEK layer to reduce calls to the remote KMS.
// The local KEK is generated at startup in a controller loop and stored in the
// LocalKEKService. This KEK is used for all encryption operations. For the decrypt
// operation, if the encrypted local KEK is not found in the cache, the remote KMS
// is used to decrypt the local KEK.
type LocalKEKService struct {
	mu sync.Mutex
	// remoteKMS is the remote kms that is used to encrypt and decrypt the local KEKs.
	remoteKMS service.Service
	// localKEKTracker is a atomic pointer to avoid locks. This is used to store the local KEK.
	localKEKTracker atomic.Pointer[localKEK]
	// transformers is a thread-safe LRU cache which caches decrypted DEKs indexed by their encrypted form.
	// The cache is only used for the decrypt operation.
	transformers *lru.Cache
	// isReady is an atomic boolean that indicates if the localKEK service is ready for encryption.
	isReady atomic.Bool

	keyMaxUsage       uint64
	keySuggestedUsage uint64
	keyMaxAge         time.Duration

	pollInterval time.Duration

	clock clock.Clock
}

// NewLocalKEKService is being initialized with a remote KMS service.
// The local KEK is generated in a controller loop. The local KEK is used for all
// encryption operations.
func NewLocalKEKService(ctx context.Context, remoteService service.Service) *LocalKEKService {
	return newLocalKEKService(ctx, remoteService, keyMaxUsage, keySuggestedUsage, keyMaxAge, localKEKGenerationPollInterval, clock.RealClock{})
}

func newLocalKEKService(ctx context.Context, remoteService service.Service, maxUsage, suggestedUsage uint64, maxAge, pollInterval time.Duration, clock clock.Clock) *LocalKEKService {
	localKEKService := &LocalKEKService{
		remoteKMS:         remoteService,
		transformers:      lru.New(cacheSize),
		keyMaxUsage:       maxUsage,
		keySuggestedUsage: suggestedUsage,
		keyMaxAge:         maxAge,
		pollInterval:      pollInterval,
		clock:             clock,
	}
	go localKEKService.run(ctx)
	return localKEKService
}

// Run method creates a new local KEK  when the following thresholds are met:
//   - the local KEK is used more often than keySuggestedUsage times or
//   - the local KEK is older than a localExpiry.
//
// this method starts the controller and blocks until the context is cancelled.
func (m *LocalKEKService) run(ctx context.Context) {
	// same as wait.UntilWithContext but with a custom clock
	wait.BackoffUntil(func() {
		lk := m.getLocalKEK()
		// this is the first time the local KEK is generated
		localKEKNotGenerated := lk.transformer == nil
		// the local KEK is used more often than keySuggestedUsage times
		localKEKUsageThresholdReached := lk.usage.Load() > m.keySuggestedUsage
		// the local KEK is older than the expiry
		localKEKExpired := m.clock.Now().After(lk.expiry)

		if localKEKNotGenerated || localKEKUsageThresholdReached || localKEKExpired {
			uid := string(uuid.NewUUID())
			err := m.generateLocalKEK(ctx, uid, "")
			if err == nil {
				m.isReady.Store(true)
				return
			}
			klog.V(2).ErrorS(err, "failed to generate local KEK", "uid", uid)
			// if the local KEK is expired and we cannot generate a new one, we set
			// isReady to false because we can no longer encrypt new data.
			if localKEKExpired {
				m.isReady.Store(false)
			}
		}
	}, wait.NewJitteredBackoffManager(m.pollInterval, 0, m.clock), true, ctx.Done())
}

// getTransformerForEncryption returns the local KEK as localTransformer, the corresponding
// remoteKMSResponse and a potential error.
// On every use the localUsage is incremented by one.
// It is assumed that only one encryption will happen with the returned transformer.
func (m *LocalKEKService) getTransformerForEncryption(uid string) (value.Transformer, *service.EncryptResponse, error) {
	lk := m.getLocalKEK()
	// localKEK is not initialized yet
	if lk.transformer == nil {
		return nil, nil, fmt.Errorf("local KEK is not initialized")
	}

	if m.clock.Now().After(lk.expiry) {
		return nil, nil, fmt.Errorf("local KEK has expired at %v", lk.expiry)
	}

	if counter := lk.usage.Add(1); counter >= m.keyMaxUsage {
		return nil, nil, fmt.Errorf("local KEK has reached maximum usage of %d", keyMaxUsage)
	}

	return lk.transformer, lk.remoteKMSResponse, nil
}

// Encrypt encrypts the plaintext with the localKEK.
func (m *LocalKEKService) Encrypt(ctx context.Context, uid string, pt []byte) (*service.EncryptResponse, error) {
	transformer, resp, err := m.getTransformerForEncryption(uid)
	if err != nil {
		klog.V(2).ErrorS(err, "failed to get transformer for encryption", "uid", uid)
		return nil, err
	}

	ct, err := transformer.TransformToStorage(ctx, pt, emptyContext)
	if err != nil {
		klog.V(2).ErrorS(err, "failed to encrypt data", "uid", uid)
		return nil, err
	}

	return &service.EncryptResponse{
		Ciphertext:  ct,
		KeyID:       resp.KeyID,
		Annotations: resp.Annotations,
	}, nil
}

// getTransformerForDecryption returns the transformer for the given encryptedKEK.
// - If the encryptedKEK is the current localKEK, the localKEK is returned.
// - If the encryptedKEK is not the current localKEK, the cache is checked.
// - If the encryptedKEK is not found in the cache, the remote KMS is used to decrypt the encryptedKEK.
func (m *LocalKEKService) getTransformerForDecryption(ctx context.Context, uid string, req *service.DecryptRequest) (value.Transformer, error) {
	encKEK := req.Annotations[referenceKEKAnnotationKey]

	// check if the key required for decryption is the current local KEK
	// that's being used for encryption
	lk := m.getLocalKEK()
	if lk.transformer != nil && bytes.Equal(lk.encKEK, encKEK) {
		return lk.transformer, nil
	}
	// check if the key required for decryption is already in the cache
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

	m.transformers.Add(base64.StdEncoding.EncodeToString(encKEK), transformer)

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
		klog.V(2).ErrorS(err, "failed to get transformer for decryption", "uid", uid)
		return nil, fmt.Errorf("failed to get transformer for decryption: %w", err)
	}

	pt, _, err := transformer.TransformFromStorage(ctx, req.Ciphertext, emptyContext)
	if err != nil {
		klog.V(2).ErrorS(err, "failed to decrypt data", "uid", uid)
		return nil, err
	}

	return pt, nil
}

// Status returns the status of the remote KMS.
func (m *LocalKEKService) Status(ctx context.Context) (*service.StatusResponse, error) {
	resp, err := m.remoteKMS.Status(ctx)
	if err != nil {
		return nil, err
	}
	if err := validateRemoteKMSStatusResponse(resp); err != nil {
		return nil, err
	}

	r := copyStatusResponse(resp)
	// if the remote KMS KeyID has changed, we need to rotate the local KEK
	lk := m.getLocalKEK()
	if lk.transformer != nil && r.KeyID != lk.remoteKMSResponse.KeyID {
		if err := m.rotateLocalKEK(ctx, r.KeyID); err != nil {
			klog.ErrorS(err, "failed to rotate local KEK", "expectedKeyID", r.KeyID, "currentKeyID", lk.remoteKMSResponse.KeyID)
			// if rotation fails, we will overwrite the keyID to the one we are currently using
			// for encryption as localKEKService is the authoritative source for the keyID.
			r.KeyID = lk.remoteKMSResponse.KeyID
			// TODO(aramase): we are currently not returning the error if rotation fails. We should
			// allow the failed state for an arbitrary time period and return the error if the state
			// is not eventually fixed.
		}
	}

	var aggregateHealthz []string
	if r.Healthz != "ok" {
		aggregateHealthz = append(aggregateHealthz, r.Healthz)
	}

	if !m.isReady.Load() {
		// if the localKEKService is not ready, we will set the healthz status to not ready
		klog.V(2).InfoS("localKEKService is not ready", "keyID", r.KeyID)
		aggregateHealthz = append(aggregateHealthz, "localKEKService is not ready")
	}
	if len(aggregateHealthz) > 0 {
		r.Healthz = strings.Join(aggregateHealthz, "; ")
	}

	return r, nil
}

// rotateLocalKEK rotates the local KEK by generating a new local KEK and encrypting it with the
// remote KMS.
func (m *LocalKEKService) rotateLocalKEK(ctx context.Context, expectedKeyID string) error {
	uid := string(uuid.NewUUID())
	if err := m.generateLocalKEK(ctx, uid, expectedKeyID); err != nil {
		klog.V(2).ErrorS(err, "failed to generate local KEK as part of rotation", "uid", uid)
		return fmt.Errorf("[uid=%s] failed to generate local KEK as part of rotation: %w", uid, err)
	}
	return nil
}

// generateLocalKEK generates a new local KEK and encrypts it with the remote KMS.
// if expectedKeyID is not empty, it will check if the keyID returned from the remote KMS matches
// the expected keyID. If the keyID does not match, it will continue using the existing local KEK
// and return an error.
func (m *LocalKEKService) generateLocalKEK(ctx context.Context, uid, expectedKeyID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	lk := m.getLocalKEK()
	// if the localKEK was generated in the last pollInterval duration, we will not generate a new
	// localKEK. This is to avoid regenerating a new localKEK for queued requests.
	if lk.transformer != nil && m.clock.Since(lk.generatedAt) < m.pollInterval {
		return nil
	}

	key, err := generateKey(keyLength)
	if err != nil {
		return fmt.Errorf("failed to generate local KEK: %w", err)
	}
	block, err := aes.NewCipher(key)
	if err != nil {
		return fmt.Errorf("failed to create cipher block: %w", err)
	}

	resp, err := m.remoteKMS.Encrypt(ctx, uid, key)
	if err != nil {
		return fmt.Errorf("failed to encrypt local KEK: %w", err)
	}
	if err = validateRemoteKMSEncryptResponse(resp); err != nil {
		return fmt.Errorf("invalid response from remote KMS: %w", err)
	}
	if expectedKeyID != "" && resp.KeyID != expectedKeyID {
		return fmt.Errorf("keyID returned from remote KMS does not match expected keyID")
	}

	now := m.clock.Now()
	m.localKEKTracker.Store(&localKEK{
		encKEK:            resp.Ciphertext,
		expiry:            now.Add(m.keyMaxAge),
		usage:             atomic.Uint64{},
		transformer:       aestransformer.NewGCMTransformer(block),
		remoteKMSResponse: copyResponseAndAddLocalKEKAnnotation(resp),
		generatedAt:       now,
	})

	return nil
}

func (m *LocalKEKService) getLocalKEK() *localKEK {
	lk := m.localKEKTracker.Load()
	if lk == nil {
		return &localKEK{}
	}
	return lk
}

// copyResponseAndAddLocalKEKAnnotation returns a copy of the remoteKMSResponse with the
// referenceKEKAnnotationKey added to the annotations.
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

// copyStatusResponse returns a copy of the remote KMS status response.
func copyStatusResponse(resp *service.StatusResponse) *service.StatusResponse {
	return &service.StatusResponse{
		Healthz: resp.Healthz,
		Version: resp.Version,
		KeyID:   resp.KeyID,
	}
}

// annotationsWithoutReferenceKeys returns a copy of the annotations without the reference implementation
// annotations.
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

// validateRemoteKMSEncryptResponse validates the EncryptResponse from the remote KMS.
func validateRemoteKMSEncryptResponse(resp *service.EncryptResponse) error {
	// validate annotations don't contain the reference implementation annotations
	for k := range resp.Annotations {
		if strings.HasSuffix(k, referenceSuffix) {
			return errInvalidKMSAnnotationKeySuffix
		}
	}
	return nil
}

// validateRemoteKMSStatusResponse validates the StatusResponse from the remote KMS.
func validateRemoteKMSStatusResponse(resp *service.StatusResponse) error {
	if len(resp.KeyID) == 0 {
		return fmt.Errorf("keyID is empty")
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
