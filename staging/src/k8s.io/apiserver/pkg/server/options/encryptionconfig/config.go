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
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"sync"
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
	return "kms-providers"
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

// LoadEncryptionConfig parses and validates the encryption config specified by filepath.
// It may launch multiple go routines whose lifecycle is controlled by stopCh.
// If reload is true, or KMS v2 plugins are used with no KMS v1 plugins, the returned slice of health checkers will always be of length 1.
func LoadEncryptionConfig(filepath string, reload bool, stopCh <-chan struct{}) (map[schema.GroupResource]value.Transformer, []healthz.HealthChecker, error) {
	config, err := loadConfig(filepath, reload)
	if err != nil {
		return nil, nil, fmt.Errorf("error while parsing file: %w", err)
	}

	transformers, kmsHealthChecks, kmsUsed, err := getTransformerOverridesAndKMSPluginHealthzCheckers(config, stopCh)
	if err != nil {
		return nil, nil, fmt.Errorf("error while building transformers: %w", err)
	}

	if reload || (kmsUsed.v2Used && !kmsUsed.v1Used) {
		kmsHealthChecks = []healthz.HealthChecker{kmsHealthChecker(kmsHealthChecks)}
	}

	return transformers, kmsHealthChecks, nil
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

func loadConfig(filepath string, reload bool) (*apiserverconfig.EncryptionConfiguration, error) {
	f, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("error opening encryption provider configuration file %q: %w", filepath, err)
	}
	defer f.Close()

	data, err := io.ReadAll(f)
	if err != nil {
		return nil, fmt.Errorf("could not read contents: %w", err)
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("encryption provider configuration file %q is empty", filepath)
	}

	scheme := runtime.NewScheme()
	codecs := serializer.NewCodecFactory(scheme)
	utilruntime.Must(apiserverconfig.AddToScheme(scheme))
	utilruntime.Must(apiserverconfigv1.AddToScheme(scheme))

	configObj, gvk, err := codecs.UniversalDecoder().Decode(data, nil, nil)
	if err != nil {
		return nil, err
	}
	config, ok := configObj.(*apiserverconfig.EncryptionConfiguration)
	if !ok {
		return nil, fmt.Errorf("got unexpected config type: %v", gvk)
	}

	return config, validation.ValidateEncryptionConfiguration(config, reload).ToAggregate()
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

		return transformer, probe, &kmsState{v1Used: true}, nil

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

		return transformer, probe, &kmsState{v2Used: true}, nil

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
