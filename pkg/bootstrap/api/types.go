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
	"k8s.io/client-go/pkg/api/v1"
)

const (
	// SecretTypeBootstrapToken is used during the automated bootstrap process (first
	// implemented by kubeadm). It stores tokens that are used to sign well known
	// ConfigMaps. They may also eventually be used for authentication.
	SecretTypeBootstrapToken v1.SecretType = "bootstrap.kubernetes.io/token"

	// BootstrapTokenIDKey is the id of this token. This can be transmitted in the
	// clear and encoded in the name of the secret. It should be a random 6
	// character string. Required
	BootstrapTokenIDKey = "token-id"

	// BootstrapTokenSecretKey is the actual secret. Typically this is a random 16
	// character string. Required.
	BootstrapTokenSecretKey = "token-secret"

	// BootstrapTokenExpirationKey is when this token should be expired and no
	// longer used. A controller will delete this resource after this time. This
	// is an absolute UTC time using RFC3339. If this cannot be parsed, the token
	// should be considered invalid. Optional.
	BootstrapTokenExpirationKey = "expiration"

	// BootstrapTokenUsageSigningKey signals that this token should be used to
	// sign configs as part of the bootstrap process. Value must be "true". Any
	// other value is assumed to be false. Optional.
	BootstrapTokenUsageSigningKey = "usage-bootstrap-signing"
)
