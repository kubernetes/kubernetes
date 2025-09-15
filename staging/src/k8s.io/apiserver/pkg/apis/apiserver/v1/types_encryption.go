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

package v1

import (
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

/*
EncryptionConfiguration stores the complete configuration for encryption providers.
It also allows the use of wildcards to specify the resources that should be encrypted.
Use '*.<group>' to encrypt all resources within a group or '*.*' to encrypt all resources.
'*.' can be used to encrypt all resource in the core group.  '*.*' will encrypt all
resources, even custom resources that are added after API server start.
Use of wildcards that overlap within the same resource list or across multiple
entries are not allowed since part of the configuration would be ineffective.
Resource lists are processed in order, with earlier lists taking precedence.

Example:

	kind: EncryptionConfiguration
	apiVersion: apiserver.config.k8s.io/v1
	resources:
	- resources:
	  - events
	  providers:
	  - identity: {}  # do not encrypt events even though *.* is specified below
	- resources:
	  - secrets
	  - configmaps
	  - pandas.awesome.bears.example
	  providers:
	  - aescbc:
	      keys:
	      - name: key1
	        secret: c2VjcmV0IGlzIHNlY3VyZQ==
	- resources:
	  - '*.apps'
	  providers:
	  - aescbc:
	      keys:
	      - name: key2
	        secret: c2VjcmV0IGlzIHNlY3VyZSwgb3IgaXMgaXQ/Cg==
	- resources:
	  - '*.*'
	  providers:
	  - aescbc:
	      keys:
	      - name: key3
	        secret: c2VjcmV0IGlzIHNlY3VyZSwgSSB0aGluaw==
*/
type EncryptionConfiguration struct {
	metav1.TypeMeta
	// resources is a list containing resources, and their corresponding encryption providers.
	Resources []ResourceConfiguration `json:"resources"`
}

// ResourceConfiguration stores per resource configuration.
type ResourceConfiguration struct {
	// resources is a list of kubernetes resources which have to be encrypted. The resource names are derived from `resource` or `resource.group` of the group/version/resource.
	// eg: pandas.awesome.bears.example is a custom resource with 'group': awesome.bears.example, 'resource': pandas.
	// Use '*.*' to encrypt all resources and '*.<group>' to encrypt all resources in a specific group.
	// eg: '*.awesome.bears.example' will encrypt all resources in the group 'awesome.bears.example'.
	// eg: '*.' will encrypt all resources in the core group (such as pods, configmaps, etc).
	Resources []string `json:"resources"`
	// providers is a list of transformers to be used for reading and writing the resources to disk.
	// eg: aesgcm, aescbc, secretbox, identity, kms.
	Providers []ProviderConfiguration `json:"providers"`
}

// ProviderConfiguration stores the provided configuration for an encryption provider.
type ProviderConfiguration struct {
	// aesgcm is the configuration for the AES-GCM transformer.
	AESGCM *AESConfiguration `json:"aesgcm,omitempty"`
	// aescbc is the configuration for the AES-CBC transformer.
	AESCBC *AESConfiguration `json:"aescbc,omitempty"`
	// secretbox is the configuration for the Secretbox based transformer.
	Secretbox *SecretboxConfiguration `json:"secretbox,omitempty"`
	// identity is the (empty) configuration for the identity transformer.
	Identity *IdentityConfiguration `json:"identity,omitempty"`
	// kms contains the name, cache size and path to configuration file for a KMS based envelope transformer.
	KMS *KMSConfiguration `json:"kms,omitempty"`
}

// AESConfiguration contains the API configuration for an AES transformer.
type AESConfiguration struct {
	// keys is a list of keys to be used for creating the AES transformer.
	// Each key has to be 32 bytes long for AES-CBC and 16, 24 or 32 bytes for AES-GCM.
	Keys []Key `json:"keys"`
}

// SecretboxConfiguration contains the API configuration for an Secretbox transformer.
type SecretboxConfiguration struct {
	// keys is a list of keys to be used for creating the Secretbox transformer.
	// Each key has to be 32 bytes long.
	Keys []Key `json:"keys"`
}

// Key contains name and secret of the provided key for a transformer.
type Key struct {
	// name is the name of the key to be used while storing data to disk.
	Name string `json:"name"`
	// secret is the actual key, encoded in base64.
	Secret string `json:"secret"`
}

// String implements Stringer interface in a log safe way.
func (k Key) String() string {
	return fmt.Sprintf("Name: %s, Secret: [REDACTED]", k.Name)
}

// IdentityConfiguration is an empty struct to allow identity transformer in provider configuration.
type IdentityConfiguration struct{}

// KMSConfiguration contains the name, cache size and path to configuration file for a KMS based envelope transformer.
type KMSConfiguration struct {
	// apiVersion of KeyManagementService
	// +optional
	APIVersion string `json:"apiVersion"`
	// name is the name of the KMS plugin to be used.
	Name string `json:"name"`
	// cachesize is the maximum number of secrets which are cached in memory. The default value is 1000.
	// Set to a negative value to disable caching. This field is only allowed for KMS v1 providers.
	// +optional
	CacheSize *int32 `json:"cachesize,omitempty"`
	// endpoint is the gRPC server listening address, for example "unix:///var/run/kms-provider.sock".
	Endpoint string `json:"endpoint"`
	// timeout for gRPC calls to kms-plugin (ex. 5s). The default is 3 seconds.
	// +optional
	Timeout *metav1.Duration `json:"timeout,omitempty"`
}
