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
	"io/ioutil"
	"net/http"
	"os"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	apiserverconfig "k8s.io/apiserver/pkg/apis/config"
	apiserverconfigv1 "k8s.io/apiserver/pkg/apis/config/v1"
	"k8s.io/apiserver/pkg/apis/config/validation"
	"k8s.io/apiserver/pkg/server/healthz"
	"k8s.io/apiserver/pkg/storage/value"
	aestransformer "k8s.io/apiserver/pkg/storage/value/encrypt/aes"
	"k8s.io/apiserver/pkg/storage/value/encrypt/envelope"
	"k8s.io/apiserver/pkg/storage/value/encrypt/identity"
	"k8s.io/apiserver/pkg/storage/value/encrypt/secretbox"
)

const (
	aesCBCTransformerPrefixV1    = "k8s:enc:aescbc:v1:"
	aesGCMTransformerPrefixV1    = "k8s:enc:aesgcm:v1:"
	secretboxTransformerPrefixV1 = "k8s:enc:secretbox:v1:"
	kmsTransformerPrefixV1       = "k8s:enc:kms:v1:"
	kmsPluginHealthzNegativeTTL  = 3 * time.Second
	kmsPluginHealthzPositiveTTL  = 20 * time.Second
)

type kmsPluginHealthzResponse struct {
	err      error
	received time.Time
}

type kmsPluginProbe struct {
	name string
	ttl  time.Duration
	envelope.Service
	lastResponse *kmsPluginHealthzResponse
	l            *sync.Mutex
}

func (h *kmsPluginProbe) toHealthzCheck(idx int) healthz.HealthChecker {
	return healthz.NamedCheck(fmt.Sprintf("kms-provider-%d", idx), func(r *http.Request) error {
		return h.Check()
	})
}

// GetKMSPluginHealthzCheckers extracts KMSPluginProbes from the EncryptionConfig.
func GetKMSPluginHealthzCheckers(filepath string) ([]healthz.HealthChecker, error) {
	f, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("error opening encryption provider configuration file %q: %v", filepath, err)
	}
	defer f.Close()
	var result []healthz.HealthChecker
	probes, err := getKMSPluginProbes(f)
	if err != nil {
		return nil, err
	}

	for i, p := range probes {
		probe := p
		result = append(result, probe.toHealthzCheck(i))
	}
	return result, nil
}

func getKMSPluginProbes(reader io.Reader) ([]*kmsPluginProbe, error) {
	var result []*kmsPluginProbe

	configFileContents, err := ioutil.ReadAll(reader)
	if err != nil {
		return result, fmt.Errorf("could not read content of encryption provider configuration: %v", err)
	}

	config, err := loadConfig(configFileContents)
	if err != nil {
		return result, fmt.Errorf("error while parsing encryption provider configuration: %v", err)
	}

	for _, r := range config.Resources {
		for _, p := range r.Providers {
			if p.KMS != nil {
				s, err := envelope.NewGRPCService(p.KMS.Endpoint, p.KMS.Timeout.Duration)
				if err != nil {
					return nil, fmt.Errorf("could not configure KMS-Plugin's probe %q, error: %v", p.KMS.Name, err)
				}

				result = append(result, &kmsPluginProbe{
					name:         p.KMS.Name,
					ttl:          kmsPluginHealthzNegativeTTL,
					Service:      s,
					l:            &sync.Mutex{},
					lastResponse: &kmsPluginHealthzResponse{},
				})
			}
		}
	}

	return result, nil
}

// Check encrypts and decrypts test data against KMS-Plugin's gRPC endpoint.
func (h *kmsPluginProbe) Check() error {
	h.l.Lock()
	defer h.l.Unlock()

	if (time.Since(h.lastResponse.received)) < h.ttl {
		return h.lastResponse.err
	}

	p, err := h.Service.Encrypt([]byte("ping"))
	if err != nil {
		h.lastResponse = &kmsPluginHealthzResponse{err: err, received: time.Now()}
		h.ttl = kmsPluginHealthzNegativeTTL
		return fmt.Errorf("failed to perform encrypt section of the healthz check for KMS Provider %s, error: %v", h.name, err)
	}

	if _, err := h.Service.Decrypt(p); err != nil {
		h.lastResponse = &kmsPluginHealthzResponse{err: err, received: time.Now()}
		h.ttl = kmsPluginHealthzNegativeTTL
		return fmt.Errorf("failed to perform decrypt section of the healthz check for KMS Provider %s, error: %v", h.name, err)
	}

	h.lastResponse = &kmsPluginHealthzResponse{err: nil, received: time.Now()}
	h.ttl = kmsPluginHealthzPositiveTTL
	return nil
}

// GetTransformerOverrides returns the transformer overrides by reading and parsing the encryption provider configuration file
func GetTransformerOverrides(filepath string) (map[schema.GroupResource]value.Transformer, error) {
	f, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("error opening encryption provider configuration file %q: %v", filepath, err)
	}
	defer f.Close()

	result, err := parseEncryptionConfiguration(f)
	if err != nil {
		return nil, fmt.Errorf("error while parsing encryption provider configuration file %q: %v", filepath, err)
	}
	return result, nil
}

func parseEncryptionConfiguration(f io.Reader) (map[schema.GroupResource]value.Transformer, error) {
	configFileContents, err := ioutil.ReadAll(f)
	if err != nil {
		return nil, fmt.Errorf("could not read contents: %v", err)
	}

	config, err := loadConfig(configFileContents)
	if err != nil {
		return nil, fmt.Errorf("error while parsing file: %v", err)
	}

	resourceToPrefixTransformer := map[schema.GroupResource][]value.PrefixTransformer{}

	// For each entry in the configuration
	for _, resourceConfig := range config.Resources {
		transformers, err := prefixTransformers(&resourceConfig)
		if err != nil {
			return nil, err
		}

		// For each resource, create a list of providers to use
		for _, resource := range resourceConfig.Resources {
			gr := schema.ParseGroupResource(resource)
			resourceToPrefixTransformer[gr] = append(
				resourceToPrefixTransformer[gr], transformers...)
		}
	}

	result := map[schema.GroupResource]value.Transformer{}
	for gr, transList := range resourceToPrefixTransformer {
		result[gr] = value.NewMutableTransformer(value.NewPrefixTransformers(fmt.Errorf("no matching prefix found"), transList...))
	}
	return result, nil

}

func loadConfig(data []byte) (*apiserverconfig.EncryptionConfiguration, error) {
	scheme := runtime.NewScheme()
	codecs := serializer.NewCodecFactory(scheme)
	apiserverconfig.AddToScheme(scheme)
	apiserverconfigv1.AddToScheme(scheme)

	configObj, gvk, err := codecs.UniversalDecoder().Decode(data, nil, nil)
	if err != nil {
		return nil, err
	}
	config, ok := configObj.(*apiserverconfig.EncryptionConfiguration)
	if !ok {
		return nil, fmt.Errorf("got unexpected config type: %v", gvk)
	}

	return config, validation.ValidateEncryptionConfiguration(config).ToAggregate()
}

// The factory to create kms service. This is to make writing test easier.
var envelopeServiceFactory = envelope.NewGRPCService

func prefixTransformers(config *apiserverconfig.ResourceConfiguration) ([]value.PrefixTransformer, error) {
	var result []value.PrefixTransformer
	for _, provider := range config.Providers {
		var (
			transformer value.PrefixTransformer
			err         error
		)

		switch {
		case provider.AESGCM != nil:
			transformer, err = aesPrefixTransformer(provider.AESGCM, aestransformer.NewGCMTransformer, aesGCMTransformerPrefixV1)
		case provider.AESCBC != nil:
			transformer, err = aesPrefixTransformer(provider.AESCBC, aestransformer.NewCBCTransformer, aesCBCTransformerPrefixV1)
		case provider.Secretbox != nil:
			transformer, err = secretboxPrefixTransformer(provider.Secretbox)
		case provider.KMS != nil:
			var envelopeService envelope.Service
			envelopeService, err = envelopeServiceFactory(provider.KMS.Endpoint, provider.KMS.Timeout.Duration)
			if err != nil {
				return nil, fmt.Errorf("could not configure KMS plugin %q, error: %v", provider.KMS.Name, err)
			}

			transformer, err = envelopePrefixTransformer(provider.KMS, envelopeService, kmsTransformerPrefixV1)
		case provider.Identity != nil:
			transformer = value.PrefixTransformer{
				Transformer: identity.NewEncryptCheckTransformer(),
				Prefix:      []byte{},
			}
		default:
			return nil, errors.New("provider does not contain any of the expected providers: KMS, AESGCM, AESCBC, Secretbox, Identity")
		}

		if err != nil {
			return result, err
		}
		result = append(result, transformer)
	}
	return result, nil
}

type blockTransformerFunc func(cipher.Block) value.Transformer

func aesPrefixTransformer(config *apiserverconfig.AESConfiguration, fn blockTransformerFunc, prefix string) (value.PrefixTransformer, error) {
	var result value.PrefixTransformer

	if len(config.Keys) == 0 {
		return result, fmt.Errorf("aes provider has no valid keys")
	}
	for _, key := range config.Keys {
		if key.Name == "" {
			return result, fmt.Errorf("key with invalid name provided")
		}
		if key.Secret == "" {
			return result, fmt.Errorf("key %v has no provided secret", key.Name)
		}
	}

	keyTransformers := []value.PrefixTransformer{}

	for _, keyData := range config.Keys {
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
		if key.Name == "" {
			return result, fmt.Errorf("key with invalid name provided")
		}
		if key.Secret == "" {
			return result, fmt.Errorf("key %v has no provided secret", key.Name)
		}
	}

	keyTransformers := []value.PrefixTransformer{}

	for _, keyData := range config.Keys {
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

func envelopePrefixTransformer(config *apiserverconfig.KMSConfiguration, envelopeService envelope.Service, prefix string) (value.PrefixTransformer, error) {
	baseTransformerFunc := func(block cipher.Block) value.Transformer {
		// v1.24: write using AES-CBC only but support reads via AES-CBC and AES-GCM (so we can move to AES-GCM)
		// TODO(aramase): swap this ordering in v1.25
		return unionTransformers{aestransformer.NewCBCTransformer(block), aestransformer.NewGCMTransformer(block)}
	}

	envelopeTransformer, err := envelope.NewEnvelopeTransformer(envelopeService, int(*config.CacheSize), baseTransformerFunc)
	if err != nil {
		return value.PrefixTransformer{}, err
	}
	return value.PrefixTransformer{
		Transformer: envelopeTransformer,
		Prefix:      []byte(prefix + config.Name + ":"),
	}, nil
}

type unionTransformers []value.Transformer

func (u unionTransformers) TransformFromStorage(ctx context.Context, data []byte, dataCtx value.Context) (out []byte, stale bool, err error) {
	var errs []error
	for i, transformer := range u {
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
