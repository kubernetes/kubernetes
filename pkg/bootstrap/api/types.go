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

package api

import (
	"k8s.io/api/core/v1"
)

const (
	// BootstrapTokenSecretPrefix is the prefix for bootstrap token names.
	// Bootstrap tokens secrets must be named in the form
	// `bootstrap-token-<token-id>`.  This is the prefix to be used before the
	// token ID.
	BootstrapTokenSecretPrefix = "bootstrap-token-"

	// SecretTypeBootstrapToken is used during the automated bootstrap process (first
	// implemented by kubeadm). It stores tokens that are used to sign well known
	// ConfigMaps. They may also eventually be used for authentication.
	SecretTypeBootstrapToken v1.SecretType = "bootstrap.kubernetes.io/token"

	// BootstrapTokenIDKey is the id of this token. This can be transmitted in the
	// clear and encoded in the name of the secret. It must be a random 6 character
	// string that matches the regexp `^([a-z0-9]{6})$`. Required.
	BootstrapTokenIDKey = "token-id"

	// BootstrapTokenSecretKey is the actual secret. It must be a random 16 character
	// string that matches the regexp `^([a-z0-9]{16})$`. Required.
	BootstrapTokenSecretKey = "token-secret"

	// BootstrapTokenExpirationKey is when this token should be expired and no
	// longer used. A controller will delete this resource after this time. This
	// is an absolute UTC time using RFC3339. If this cannot be parsed, the token
	// should be considered invalid. Optional.
	BootstrapTokenExpirationKey = "expiration"

	// BootstrapTokenDescriptionKey is a description in human-readable format that
	// describes what the bootstrap token is used for. Optional.
	BootstrapTokenDescriptionKey = "description"

	// BootstrapTokenUsagePrefix is the prefix for the other usage constants that specifies different
	// functions of a bootstrap token
	BootstrapTokenUsagePrefix = "usage-bootstrap-"

	// BootstrapTokenUsageSigningKey signals that this token should be used to
	// sign configs as part of the bootstrap process. Value must be "true". Any
	// other value is assumed to be false. Optional.
	BootstrapTokenUsageSigningKey = "usage-bootstrap-signing"

	// BootstrapTokenUsageAuthentication signals that this token should be used
	// as a bearer token to authenticate against the Kubernetes API. The bearer
	// token takes the form "<token-id>.<token-secret>" and authenticates as the
	// user "system:bootstrap:<token-id>" in the group "system:bootstrappers".
	// Value must be "true". Any other value is assumed to be false. Optional.
	BootstrapTokenUsageAuthentication = "usage-bootstrap-authentication"

	// ConfigMapClusterInfo defines the name for the ConfigMap where the information how to connect and trust the cluster exist
	ConfigMapClusterInfo = "cluster-info"

	// KubeConfigKey defines at which key in the Data object of the ConfigMap the KubeConfig object is stored
	KubeConfigKey = "kubeconfig"

	// JWSSignatureKeyPrefix defines what key prefix the JWS-signed tokens have
	JWSSignatureKeyPrefix = "jws-kubeconfig-"

	// BootstrapUserPrefix is the username prefix bootstrapping bearer tokens
	// authenticate as. The full username given is "system:bootstrap:<token-id>".
	BootstrapUserPrefix = "system:bootstrap:"

	// BootstrapGroup is the group bootstrapping bearer tokens authenticate in.
	BootstrapGroup = "system:bootstrappers"
)
