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

// EncryptionConfig stores the complete configuration for encryption providers.
type EncryptionConfig struct {
	// kind is the type of configuration file.
	Kind string `json:"kind"`
	// apiVersion is the API version this file has to be parsed as.
	APIVersion string `json:"apiVersion"`
	// resources is a list containing resources, and their corresponding encryption providers.
	Resources []ResourceConfig `json:"resources"`
}

// ResourceConfig stores per resource configuration.
type ResourceConfig struct {
	// resources is a list of kubernetes resources which have to be encrypted.
	Resources []string `json:"resources"`
	// providers is a list of transformers to be used for reading and writing the resources to disk.
	// eg: aes, identity.
	Providers []ProviderConfig `json:"providers"`
}

// ProviderConfig stores the provided configuration for an encryption provider.
type ProviderConfig struct {
	// aesgcm is the configuration for the AES-GCM transformer.
	AESGCM *AESConfig `json:"aesgcm,omitempty"`
	// aescbc is the configuration for the AES-CBC transformer.
	AESCBC *AESConfig `json:"aescbc,omitempty"`
	// secretbox is the configuration for the Secretbox based transformer.
	Secretbox *SecretboxConfig `json:"secretbox,omitempty"`
	// identity is the (empty) configuration for the identity transformer.
	Identity *IdentityConfig `json:"identity,omitempty"`
}

// AESConfig contains the API configuration for an AES transformer.
type AESConfig struct {
	// keys is a list of keys to be used for creating the AES transformer.
	Keys []Key `json:"keys"`
}

// SECRETBOXConfig contains the API configuration for an Secretbox transformer.
type SecretboxConfig struct {
	// keys is a list of keys to be used for creating the Secretbox transformer.
	Keys []Key `json:"keys"`
}

// Key contains name and secret of the provided key for AES transformer.
type Key struct {
	// name is the name of the key to be used while storing data to disk.
	Name string `json:"name"`
	// secret is the actual AES key, encoded in base64. It has to be 16, 24 or 32 bytes long.
	Secret string `json:"secret"`
}

// IdentityConfig is an empty struct to allow identity transformer in provider configuration.
type IdentityConfig struct{}
