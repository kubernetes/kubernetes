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
	"crypto/aes"
	"encoding/base64"
	"fmt"
	"io"
	"io/ioutil"
	"os"

	yaml "github.com/ghodss/yaml"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/storage/value"
	aestransformer "k8s.io/apiserver/pkg/storage/value/encrypt/aes"
	"k8s.io/apiserver/pkg/storage/value/encrypt/identity"
)

const (
	aesTransformerPrefixV1 = "k8s:enc:aes:v1:"
)

// GetTransformerOverrides returns the transformer overrides by reading and parsing the encryption provider configuration file
func GetTransformerOverrides(filepath string) (map[schema.GroupResource]value.Transformer, error) {
	f, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("error opening encryption provider configuration file %q: %v", filepath, err)
	}
	defer f.Close()

	result, err := ParseEncryptionConfiguration(f)
	if err != nil {
		return nil, fmt.Errorf("error while parsing encryption provider configuration file %q: %v", filepath, err)
	}
	return result, nil
}

// ParseEncryptionConfiguration parses configuration data and returns the transformer overrides
func ParseEncryptionConfiguration(f io.Reader) (map[schema.GroupResource]value.Transformer, error) {
	configFileContents, err := ioutil.ReadAll(f)
	if err != nil {
		return nil, fmt.Errorf("could not read contents: %v", err)
	}

	var config EncryptionConfig
	err = yaml.Unmarshal(configFileContents, &config)
	if err != nil {
		return nil, fmt.Errorf("error while parsing file: %v", err)
	}

	if config.Kind != "EncryptionConfig" && config.Kind != "" {
		return nil, fmt.Errorf("invalid configuration kind %q provided", config.Kind)
	}
	if config.Kind == "" {
		return nil, fmt.Errorf("invalid configuration file, missing Kind")
	}
	// TODO config.APIVersion is unchecked

	resourceToPrefixTransformer := map[schema.GroupResource][]value.PrefixTransformer{}

	// For each entry in the configuration
	for _, resourceConfig := range config.Resources {
		transformers, err := GetPrefixTransformers(&resourceConfig)
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

// GetPrefixTransformer constructs and returns the appropriate prefix transformers for the passed resource using its configuration
func GetPrefixTransformers(config *ResourceConfig) ([]value.PrefixTransformer, error) {
	var result []value.PrefixTransformer
	for _, provider := range config.Providers {
		found := false

		if provider.AES != nil {
			transformer, err := GetAESPrefixTransformer(provider.AES)
			found = true
			if err != nil {
				return result, err
			}
			result = append(result, transformer)
		}

		if provider.Identity != nil {
			if found == true {
				return result, fmt.Errorf("more than one provider specified in a single element, should split into different list elements")
			}
			found = true
			result = append(result, value.PrefixTransformer{
				Transformer: identity.NewEncryptCheckTransformer(),
				Prefix:      []byte{},
			})
		}

		if found == false {
			return result, fmt.Errorf("invalid provider configuration provided")
		}
	}
	return result, nil
}

// GetAESPrefixTransformer returns a prefix transformer from the provided configuration
func GetAESPrefixTransformer(config *AESConfig) (value.PrefixTransformer, error) {
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
				Transformer: aestransformer.NewGCMTransformer(block),
				Prefix:      []byte(keyData.Name + ":"),
			})
	}

	// Create a prefixTransformer which can choose between these keys
	keyTransformer := value.NewPrefixTransformers(
		fmt.Errorf("no matching key was found for the provided AES transformer"), keyTransformers...)

	// Create a PrefixTransformer which shall later be put in a list with other providers
	result = value.PrefixTransformer{
		Transformer: keyTransformer,
		Prefix:      []byte(aesTransformerPrefixV1),
	}
	return result, nil
}
