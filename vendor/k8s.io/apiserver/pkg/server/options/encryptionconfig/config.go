/*
Copyright 2017 The Kubernetes Authors.

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

package encryptionconfig

import (
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	apiserverconfig "k8s.io/apiserver/pkg/apis/config"
	apiserverconfigv1 "k8s.io/apiserver/pkg/apis/config/v1"
	"k8s.io/apiserver/pkg/apis/config/validation"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/server/healthz"
	"k8s.io/apiserver/pkg/storage/value"
	aestransformer "k8s.io/apiserver/pkg/storage/value/encrypt/aes"
	"k8s.io/apiserver/pkg/storage/value/encrypt/envelope"
	envelopekmsv2 "k8s.io/apiserver/pkg/storage/value/encrypt/envelope/kmsv2"
	"k8s.io/apiserver/pkg/storage/value/encrypt/identity"
	"k8s.io/apiserver/pkg/storage/value/encrypt/secretbox"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
)

const (
	aesCBCTransformerPrefixV1    = "k8s:enc:aescbc:v1:"
	aesGCMTransformerPrefixV1    = "k8s:enc:aesgcm:v1:"
	secretboxTransformerPrefixV1 = "k8s:enc:secretbox:v1:"
	kmsTransformerPrefixV1       = "k8s:enc:kms:v1:"
	kmsTransformerPrefixV2       = "k8s:enc:kms:v2:"
	kmsPluginHealthzNegativeTTL  = 3 * time.Second
	kmsPluginHealthzPositiveTTL  = 20 * time.Second
	kmsAPIVersionV1              = "v1"
	kmsAPIVersionV2              = "v2"
	kmsReloadHealthCheckName     = "kms-providers"
)

type kmsPluginHealthzResponse struct {
	err      error
	received time.Time
}

type kmsPluginProbe struct {
	name         string
	ttl          time.Duration
	service      envelope.Service
	lastResponse *kmsPluginHealthzResponse
	l            *sync.Mutex
}

type kmsv2PluginProbe struct {
	name         string
	ttl          time.Duration
	service      envelopekmsv2.Service
	lastResponse *kmsPluginHealthzResponse
	l            *sync.Mutex
}

type kmsHealthChecker []healthz.HealthChecker

func (k kmsHealthChecker) Name() string {
	return kmsReloadHealthCheckName
}

func (k kmsHealthChecker) Check(req *http.Request) error {
	var errs []error

	for i := range k {
		checker := k[i]
		if err := checker.Check(req); err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", checker.Name(), err))
		}
	}

	return utilerrors.Reduce(utilerrors.NewAggregate(errs))
}

func (h *kmsPluginProbe) toHealthzCheck(idx int) healthz.HealthChecker {
	return healthz.NamedCheck(fmt.Sprintf("kms-provider-%d", idx), func(r *http.Request) error {
		return h.check()
	})
}

func (h *kmsv2PluginProbe) toHealthzCheck(idx int) healthz.HealthChecker {
	return healthz.NamedCheck(fmt.Sprintf("kms-provider-%d", idx), func(r *http.Request) error {
		return h.check(r.Context())
	})
}

// EncryptionConfiguration represents the parsed and normalized encryption configuration for the apiserver.
type EncryptionConfiguration struct {
	// Transformers is a list of value.Transformer that will be used to encrypt and decrypt data.
	Transformers map[schema.GroupResource]value.Transformer

	// HealthChecks is a list of healthz.HealthChecker that will be used to check the health of the encryption providers.
	HealthChecks []healthz.HealthChecker

	// EncryptionFileContentHash is the hash of the encryption config file.
	EncryptionFileContentHash string

	// KMSCloseGracePeriod is the duration we will wait before closing old transformers.
	// We wait for any in-flight requests to finish by using the duration which is longer than their timeout.
	KMSCloseGracePeriod time.Duration
}

// LoadEncryptionConfig parses and validates the encryption config specified by filepath.
// It may launch multiple go routines whose lifecycle is controlled by stopCh.
// If reload is true, or KMS v2 plugins are used with no KMS v1 plugins, the returned slice of health checkers will always be of length 1.
func LoadEncryptionConfig(filepath string, reload bool, stopCh <-chan struct{}) (*EncryptionConfiguration, error) {
	config, contentHash, err := loadConfig(filepath, reload)
	if err != nil {
		return nil, fmt.Errorf("error while parsing file: %w", err)
	}

	transformers, kmsHealthChecks, kmsUsed, err := getTransformerOverridesAndKMSPluginHealthzCheckers(config, stopCh)
	if err != nil {
		return nil, fmt.Errorf("error while building transformers: %w", err)
	}

	if reload || (kmsUsed.v2Used && !kmsUsed.v1Used) {
		kmsHealthChecks = []healthz.HealthChecker{kmsHealthChecker(kmsHealthChecks)}
	}

	// KMSTimeout is the duration we will wait before closing old transformers.
	// The way we calculate is as follows:
	// 1. Sum all timeouts across all KMS plugins. (check kmsPrefixTransformer for differences between v1 and v2)
	// 2. Multiply that by 2 (to allow for some buffer)
	// The reason we sum all timeout is because kmsHealthChecker() will run all health checks serially
	return &EncryptionConfiguration{
		Transformers:              transformers,
		HealthChecks:              kmsHealthChecks,
		EncryptionFileContentHash: contentHash,
		KMSCloseGracePeriod:       2 * kmsUsed.kmsTimeoutSum,
	}, err
}

func getTransformerOverridesAndKMSPluginHealthzCheckers(config *apiserverconfig.EncryptionConfiguration, stopCh <-chan struct{}) (map[schema.GroupResource]value.Transformer, []healthz.HealthChecker, *kmsState, error) {
	var kmsHealthChecks []healthz.HealthChecker
	transformers, probes, kmsUsed, err := getTransformerOverridesAndKMSPluginProbes(config, stopCh)
	if err != nil {
		return nil, nil, nil, err
	}
	for i := range probes {
		probe := probes[i]
		kmsHealthChecks = append(kmsHealthChecks, probe.toHealthzCheck(i))
	}

	return transformers, kmsHealthChecks, kmsUsed, nil
}

type healthChecker interface {
	toHealthzCheck(idx int) healthz.HealthChecker
}

func getTransformerOverridesAndKMSPluginProbes(config *apiserverconfig.EncryptionConfiguration, stopCh <-chan struct{}) (map[schema.GroupResource]value.Transformer, []healthChecker, *kmsState, error) {
	resourceToPrefixTransformer := map[schema.GroupResource][]value.PrefixTransformer{}
	var probes []healthChecker
	var kmsUsed kmsState

	// For each entry in the configuration
	for _, resourceConfig := range config.Resources {
		resourceConfig := resourceConfig

		transformers, p, used, err := prefixTransformersAndProbes(resourceConfig, stopCh)
		if err != nil {
			return nil, nil, nil, err
		}
		kmsUsed.v1Used = kmsUsed.v1Used || used.v1Used
		kmsUsed.v2Used = kmsUsed.v2Used || used.v2Used

		kmsUsed.kmsTimeoutSum += used.kmsTimeoutSum

		// For each resource, create a list of providers to use
		for _, resource := range resourceConfig.Resources {
			resource := resource
			gr := schema.ParseGroupResource(resource)
			resourceToPrefixTransformer[gr] = append(
				resourceToPrefixTransformer[gr], transformers...)
		}

		probes = append(probes, p...)
	}

	transformers := make(map[schema.GroupResource]value.Transformer, len(resourceToPrefixTransformer))
	for gr, transList := range resourceToPrefixTransformer {
		gr := gr
		transList := transList
		transformers[gr] = value.NewPrefixTransformers(fmt.Errorf("no matching prefix found"), transList...)
	}

	return transformers, probes, &kmsUsed, nil
}

// check encrypts and decrypts test data against KMS-Plugin's gRPC endpoint.
func (h *kmsPluginProbe) check() error {
	h.l.Lock()
	defer h.l.Unlock()

	if (time.Since(h.lastResponse.received)) < h.ttl {
		return h.lastResponse.err
	}

	p, err := h.service.Encrypt([]byte("ping"))
	if err != nil {
		h.lastResponse = &kmsPluginHealthzResponse{err: err, received: time.Now()}
		h.ttl = kmsPluginHealthzNegativeTTL
		return fmt.Errorf("failed to perform encrypt section of the healthz check for KMS Provider %s, error: %w", h.name, err)
	}

	if _, err := h.service.Decrypt(p); err != nil {
		h.lastResponse = &kmsPluginHealthzResponse{err: err, received: time.Now()}
		h.ttl = kmsPluginHealthzNegativeTTL
		return fmt.Errorf("failed to perform decrypt section of the healthz check for KMS Provider %s, error: %w", h.name, err)
	}

	h.lastResponse = &kmsPluginHealthzResponse{err: nil, received: time.Now()}
	h.ttl = kmsPluginHealthzPositiveTTL
	return nil
}

// check gets the healthz status of the KMSv2-Plugin using the Status() method.
func (h *kmsv2PluginProbe) check(ctx context.Context) error {
	h.l.Lock()
	defer h.l.Unlock()

	if (time.Since(h.lastResponse.received)) < h.ttl {
		return h.lastResponse.err
	}

	p, err := h.service.Status(ctx)
	if err != nil {
		h.lastResponse = &kmsPluginHealthzResponse{err: err, received: time.Now()}
		h.ttl = kmsPluginHealthzNegativeTTL
		return fmt.Errorf("failed to perform status section of the healthz check for KMS Provider %s, error: %w", h.name, err)
	}

	if err := isKMSv2ProviderHealthy(h.name, p); err != nil {
		h.lastResponse = &kmsPluginHealthzResponse{err: err, received: time.Now()}
		h.ttl = kmsPluginHealthzNegativeTTL
		return err
	}

	h.lastResponse = &kmsPluginHealthzResponse{err: nil, received: time.Now()}
	h.ttl = kmsPluginHealthzPositiveTTL
	return nil
}

// isKMSv2ProviderHealthy checks if the KMSv2-Plugin is healthy.
func isKMSv2ProviderHealthy(name string, response *envelopekmsv2.StatusResponse) error {
	var errs []error
	if response.Healthz != "ok" {
		errs = append(errs, fmt.Errorf("got unexpected healthz status: %s", response.Healthz))
	}
	if response.Version != envelopekmsv2.KMSAPIVersion {
		errs = append(errs, fmt.Errorf("expected KMSv2 API version %s, got %s", envelopekmsv2.KMSAPIVersion, response.Version))
	}
	if len(response.KeyID) == 0 {
		errs = append(errs, fmt.Errorf("expected KMSv2 KeyID to be set, got %s", response.KeyID))
	}

	if err := utilerrors.Reduce(utilerrors.NewAggregate(errs)); err != nil {
		return fmt.Errorf("kmsv2 Provider %s is not healthy, error: %w", name, err)
	}
	return nil
}

// loadConfig parses the encryption configuration file at filepath and returns the parsed config and hash of the file.
func loadConfig(filepath string, reload bool) (*apiserverconfig.EncryptionConfiguration, string, error) {
	f, err := os.Open(filepath)
	if err != nil {
		return nil, "", fmt.Errorf("error opening encryption provider configuration file %q: %w", filepath, err)
	}
	defer f.Close()

	data, err := io.ReadAll(f)
	if err != nil {
		return nil, "", fmt.Errorf("could not read contents: %w", err)
	}
	if len(data) == 0 {
		return nil, "", fmt.Errorf("encryption provider configuration file %q is empty", filepath)
	}

	scheme := runtime.NewScheme()
	codecs := serializer.NewCodecFactory(scheme)
	utilruntime.Must(apiserverconfig.AddToScheme(scheme))
	utilruntime.Must(apiserverconfigv1.AddToScheme(scheme))

	configObj, gvk, err := codecs.UniversalDecoder().Decode(data, nil, nil)
	if err != nil {
		return nil, "", err
	}
	config, ok := configObj.(*apiserverconfig.EncryptionConfiguration)
	if !ok {
		return nil, "", fmt.Errorf("got unexpected config type: %v", gvk)
	}

	return config, computeEncryptionConfigHash(data), validation.ValidateEncryptionConfiguration(config, reload).ToAggregate()
}

func prefixTransformersAndProbes(config apiserverconfig.ResourceConfiguration, stopCh <-chan struct{}) ([]value.PrefixTransformer, []healthChecker, *kmsState, error) {
	var transformers []value.PrefixTransformer
	var probes []healthChecker
	var kmsUsed kmsState

	for _, provider := range config.Providers {
		provider := provider
		var (
			transformer    value.PrefixTransformer
			transformerErr error
			probe          healthChecker
			used           *kmsState
		)

		switch {
		case provider.AESGCM != nil:
			transformer, transformerErr = aesPrefixTransformer(provider.AESGCM, aestransformer.NewGCMTransformer, aesGCMTransformerPrefixV1)

		case provider.AESCBC != nil:
			transformer, transformerErr = aesPrefixTransformer(provider.AESCBC, aestransformer.NewCBCTransformer, aesCBCTransformerPrefixV1)

		case provider.Secretbox != nil:
			transformer, transformerErr = secretboxPrefixTransformer(provider.Secretbox)

		case provider.KMS != nil:
			transformer, probe, used, transformerErr = kmsPrefixTransformer(provider.KMS, stopCh)
			if transformerErr == nil {
				probes = append(probes, probe)
				kmsUsed.v1Used = kmsUsed.v1Used || used.v1Used
				kmsUsed.v2Used = kmsUsed.v2Used || used.v2Used

				// calculate the maximum timeout for all KMS providers
				kmsUsed.kmsTimeoutSum += used.kmsTimeoutSum
			}

		case provider.Identity != nil:
			transformer = value.PrefixTransformer{
				Transformer: identity.NewEncryptCheckTransformer(),
				Prefix:      []byte{},
			}

		default:
			return nil, nil, nil, errors.New("provider does not contain any of the expected providers: KMS, AESGCM, AESCBC, Secretbox, Identity")
		}

		if transformerErr != nil {
			return nil, nil, nil, transformerErr
		}

		transformers = append(transformers, transformer)
	}

	return transformers, probes, &kmsUsed, nil
}

type blockTransformerFunc func(cipher.Block) value.Transformer

func aesPrefixTransformer(config *apiserverconfig.AESConfiguration, fn blockTransformerFunc, prefix string) (value.PrefixTransformer, error) {
	var result value.PrefixTransformer

	if len(config.Keys) == 0 {
		return result, fmt.Errorf("aes provider has no valid keys")
	}
	for _, key := range config.Keys {
		key := key
		if key.Name == "" {
			return result, fmt.Errorf("key with invalid name provided")
		}
		if key.Secret == "" {
			return result, fmt.Errorf("key %v has no provided secret", key.Name)
		}
	}

	keyTransformers := []value.PrefixTransformer{}

	for _, keyData := range config.Keys {
		keyData := keyData
		key, err := base64.StdEncoding.DecodeString(keyData.Secret)
		if err != nil {
			return result, fmt.Errorf("could not obtain secret for named key %s: %s", keyData.Name, err)
		}
		block, err := aes.NewCipher(key)
		if err != nil {
			return result, fmt.Errorf("error while creating cipher for named key %s: %s", keyData.Name, err)
		}

		// Create a new PrefixTransformer for this key
		keyTransformers = append(keyTransformers,
			value.PrefixTransformer{
				Transformer: fn(block),
				Prefix:      []byte(keyData.Name + ":"),
			})
	}

	// Create a prefixTransformer which can choose between these keys
	keyTransformer := value.NewPrefixTransformers(
		fmt.Errorf("no matching key was found for the provided AES transformer"), keyTransformers...)

	// Create a PrefixTransformer which shall later be put in a list with other providers
	result = value.PrefixTransformer{
		Transformer: keyTransformer,
		Prefix:      []byte(prefix),
	}
	return result, nil
}

func secretboxPrefixTransformer(config *apiserverconfig.SecretboxConfiguration) (value.PrefixTransformer, error) {
	var result value.PrefixTransformer

	if len(config.Keys) == 0 {
		return result, fmt.Errorf("secretbox provider has no valid keys")
	}
	for _, key := range config.Keys {
		key := key
		if key.Name == "" {
			return result, fmt.Errorf("key with invalid name provided")
		}
		if key.Secret == "" {
			return result, fmt.Errorf("key %v has no provided secret", key.Name)
		}
	}

	keyTransformers := []value.PrefixTransformer{}

	for _, keyData := range config.Keys {
		keyData := keyData
		key, err := base64.StdEncoding.DecodeString(keyData.Secret)
		if err != nil {
			return result, fmt.Errorf("could not obtain secret for named key %s: %s", keyData.Name, err)
		}

		if len(key) != 32 {
			return result, fmt.Errorf("expected key size 32 for secretbox provider, got %v", len(key))
		}

		keyArray := [32]byte{}
		copy(keyArray[:], key)

		// Create a new PrefixTransformer for this key
		keyTransformers = append(keyTransformers,
			value.PrefixTransformer{
				Transformer: secretbox.NewSecretboxTransformer(keyArray),
				Prefix:      []byte(keyData.Name + ":"),
			})
	}

	// Create a prefixTransformer which can choose between these keys
	keyTransformer := value.NewPrefixTransformers(
		fmt.Errorf("no matching key was found for the provided Secretbox transformer"), keyTransformers...)

	// Create a PrefixTransformer which shall later be put in a list with other providers
	result = value.PrefixTransformer{
		Transformer: keyTransformer,
		Prefix:      []byte(secretboxTransformerPrefixV1),
	}
	return result, nil
}

var (
	// The factory to create kms service. This is to make writing test easier.
	envelopeServiceFactory = envelope.NewGRPCService

	// The factory to create kmsv2 service.  Exported for integration tests.
	EnvelopeKMSv2ServiceFactory = envelopekmsv2.NewGRPCService
)

type kmsState struct {
	v1Used, v2Used bool
	kmsTimeoutSum  time.Duration
}

func kmsPrefixTransformer(config *apiserverconfig.KMSConfiguration, stopCh <-chan struct{}) (value.PrefixTransformer, healthChecker, *kmsState, error) {
	// we ignore the cancel func because this context should only be canceled when stopCh is closed
	ctx, _ := wait.ContextForChannel(stopCh)

	kmsName := config.Name
	switch config.APIVersion {
	case kmsAPIVersionV1:
		envelopeService, err := envelopeServiceFactory(ctx, config.Endpoint, config.Timeout.Duration)
		if err != nil {
			return value.PrefixTransformer{}, nil, nil, fmt.Errorf("could not configure KMSv1-Plugin's probe %q, error: %w", kmsName, err)
		}

		probe := &kmsPluginProbe{
			name:         kmsName,
			ttl:          kmsPluginHealthzNegativeTTL,
			service:      envelopeService,
			l:            &sync.Mutex{},
			lastResponse: &kmsPluginHealthzResponse{},
		}

		transformer := envelopePrefixTransformer(config, envelopeService, kmsTransformerPrefixV1)

		return transformer, probe, &kmsState{
			v1Used: true,
			// for v1 we will do encrypt and decrypt for health check. Since these are serial operations, we will double the timeout.
			kmsTimeoutSum: 2 * config.Timeout.Duration,
		}, nil

	case kmsAPIVersionV2:
		if !utilfeature.DefaultFeatureGate.Enabled(features.KMSv2) {
			return value.PrefixTransformer{}, nil, nil, fmt.Errorf("could not configure KMSv2 plugin %q, KMSv2 feature is not enabled", kmsName)
		}

		envelopeService, err := EnvelopeKMSv2ServiceFactory(ctx, config.Endpoint, config.Timeout.Duration)
		if err != nil {
			return value.PrefixTransformer{}, nil, nil, fmt.Errorf("could not configure KMSv2-Plugin's probe %q, error: %w", kmsName, err)
		}

		probe := &kmsv2PluginProbe{
			name:         kmsName,
			ttl:          kmsPluginHealthzNegativeTTL,
			service:      envelopeService,
			l:            &sync.Mutex{},
			lastResponse: &kmsPluginHealthzResponse{},
		}

		// using AES-GCM by default for encrypting data with KMSv2
		transformer := value.PrefixTransformer{
			Transformer: envelopekmsv2.NewEnvelopeTransformer(envelopeService, int(*config.CacheSize), aestransformer.NewGCMTransformer),
			Prefix:      []byte(kmsTransformerPrefixV2 + kmsName + ":"),
		}

		return transformer, probe, &kmsState{
			v2Used:        true,
			kmsTimeoutSum: config.Timeout.Duration,
		}, nil

	default:
		return value.PrefixTransformer{}, nil, nil, fmt.Errorf("could not configure KMS plugin %q, unsupported KMS API version %q", kmsName, config.APIVersion)
	}
}

func envelopePrefixTransformer(config *apiserverconfig.KMSConfiguration, envelopeService envelope.Service, prefix string) value.PrefixTransformer {
	baseTransformerFunc := func(block cipher.Block) value.Transformer {
		// v1.24: write using AES-CBC only but support reads via AES-CBC and AES-GCM (so we can move to AES-GCM)
		// v1.25: write using AES-GCM only but support reads via AES-GCM and fallback to AES-CBC for backwards compatibility
		// TODO(aramase): Post v1.25: We cannot drop CBC read support until we automate storage migration.
		// We could have a release note that hard requires users to perform storage migration.
		return unionTransformers{aestransformer.NewGCMTransformer(block), aestransformer.NewCBCTransformer(block)}
	}

	return value.PrefixTransformer{
		Transformer: envelope.NewEnvelopeTransformer(envelopeService, int(*config.CacheSize), baseTransformerFunc),
		Prefix:      []byte(prefix + config.Name + ":"),
	}
}

type unionTransformers []value.Transformer

func (u unionTransformers) TransformFromStorage(ctx context.Context, data []byte, dataCtx value.Context) (out []byte, stale bool, err error) {
	var errs []error
	for i := range u {
		transformer := u[i]
		result, stale, err := transformer.TransformFromStorage(ctx, data, dataCtx)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		// when i != 0, we have transformed the data from storage using the new transformer,
		// we want to issue a write to etcd even if the contents of the data haven't changed
		return result, stale || i != 0, nil
	}
	if err := utilerrors.Reduce(utilerrors.NewAggregate(errs)); err != nil {
		return nil, false, err
	}
	return nil, false, fmt.Errorf("unionTransformers: unable to transform from storage")
}

func (u unionTransformers) TransformToStorage(ctx context.Context, data []byte, dataCtx value.Context) (out []byte, err error) {
	return u[0].TransformToStorage(ctx, data, dataCtx)
}

// computeEncryptionConfigHash returns the expected hash for an encryption config file that has been loaded as bytes.
// We use a hash instead of the raw file contents when tracking changes to avoid holding any encryption keys in memory outside of their associated transformers.
// This hash must be used in-memory and not externalized to the process because it has no cross-release stability guarantees.
func computeEncryptionConfigHash(data []byte) string {
	return fmt.Sprintf("%x", sha256.Sum256(data))
}

var _ healthz.HealthChecker = &DynamicTransformers{}

// DynamicTransformers holds transformers that may be dynamically updated via a single external actor, likely a controller.
// This struct must avoid locks (even read write locks) as it is inline to all calls to storage.
type DynamicTransformers struct {
	transformTracker *atomic.Value
}

type transformTracker struct {
	transformerOverrides  map[schema.GroupResource]value.Transformer
	kmsPluginHealthzCheck healthz.HealthChecker
	closeTransformers     context.CancelFunc
	kmsCloseGracePeriod   time.Duration
}

// NewDynamicTransformers returns transformers, health checks for kms providers and an ability to close transformers.
func NewDynamicTransformers(
	transformerOverrides map[schema.GroupResource]value.Transformer,
	kmsPluginHealthzCheck healthz.HealthChecker,
	closeTransformers context.CancelFunc,
	kmsCloseGracePeriod time.Duration,
) *DynamicTransformers {
	dynamicTransformers := &DynamicTransformers{
		transformTracker: &atomic.Value{},
	}

	tracker := &transformTracker{
		transformerOverrides:  transformerOverrides,
		kmsPluginHealthzCheck: kmsPluginHealthzCheck,
		closeTransformers:     closeTransformers,
		kmsCloseGracePeriod:   kmsCloseGracePeriod,
	}
	dynamicTransformers.transformTracker.Store(tracker)

	return dynamicTransformers
}

// Check implements healthz.HealthChecker
func (d *DynamicTransformers) Check(req *http.Request) error {
	return d.transformTracker.Load().(*transformTracker).kmsPluginHealthzCheck.Check(req)
}

// Name implements healthz.HealthChecker
func (d *DynamicTransformers) Name() string {
	return kmsReloadHealthCheckName
}

// TransformerForResource returns the transformer for the given resource.
func (d *DynamicTransformers) TransformerForResource(resource schema.GroupResource) value.Transformer {
	return &resourceTransformer{
		resource:         resource,
		transformTracker: d.transformTracker,
	}
}

// Set sets the transformer overrides. This method is not go routine safe and must only be called by the same, single caller throughout the lifetime of this object.
func (d *DynamicTransformers) Set(
	transformerOverrides map[schema.GroupResource]value.Transformer,
	closeTransformers context.CancelFunc,
	kmsPluginHealthzCheck healthz.HealthChecker,
	kmsCloseGracePeriod time.Duration,
) {
	// store new values
	newTransformTracker := &transformTracker{
		transformerOverrides:  transformerOverrides,
		closeTransformers:     closeTransformers,
		kmsPluginHealthzCheck: kmsPluginHealthzCheck,
		kmsCloseGracePeriod:   kmsCloseGracePeriod,
	}

	// update new transformer overrides
	oldTransformTracker := d.transformTracker.Swap(newTransformTracker).(*transformTracker)

	// close old transformers once we wait for grpc request to finish any in-flight requests.
	// by the time we spawn this go routine, the new transformers have already been set and will be used for new requests.
	// if the server starts shutting down during sleep duration then the transformers will correctly closed early because their lifetime is tied to the api-server drain notifier.
	go func() {
		time.Sleep(oldTransformTracker.kmsCloseGracePeriod)
		oldTransformTracker.closeTransformers()
	}()
}

var _ value.Transformer = &resourceTransformer{}

type resourceTransformer struct {
	resource         schema.GroupResource
	transformTracker *atomic.Value
}

func (r *resourceTransformer) TransformFromStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, bool, error) {
	return r.transformer().TransformFromStorage(ctx, data, dataCtx)
}

func (r *resourceTransformer) TransformToStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, error) {
	return r.transformer().TransformToStorage(ctx, data, dataCtx)
}

func (r *resourceTransformer) transformer() value.Transformer {
	transformer := r.transformTracker.Load().(*transformTracker).transformerOverrides[r.resource]
	if transformer == nil {
		return identity.NewEncryptCheckTransformer()
	}
	return transformer
}

type ResourceTransformers interface {
	TransformerForResource(resource schema.GroupResource) value.Transformer
}

var _ ResourceTransformers = &DynamicTransformers{}
var _ ResourceTransformers = &StaticTransformers{}

type StaticTransformers map[schema.GroupResource]value.Transformer

// StaticTransformers
func (s StaticTransformers) TransformerForResource(resource schema.GroupResource) value.Transformer {
	transformer := s[resource]
	if transformer == nil {
		return identity.NewEncryptCheckTransformer()
	}
	return transformer
}
