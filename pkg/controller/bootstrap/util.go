/*
Copyright 2016 The Kubernetes Authors.

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

package bootstrap

import (
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"

	bootstrapapi "k8s.io/cluster-bootstrap/token/api"
	bootstrapsecretutil "k8s.io/cluster-bootstrap/util/secrets"
)

func validateSecretForSigning(secret *v1.Secret) (tokenID, tokenSecret string, ok bool) {
	nameTokenID, ok := bootstrapsecretutil.ParseName(secret.Name)
	if !ok {
		klog.V(3).InfoS("Invalid secret name. Must be of form <bootstrap-secret-prefix><secret-id>.", "secretName", secret.Name)
		return "", "", false
	}

	tokenID = bootstrapsecretutil.GetData(secret, bootstrapapi.BootstrapTokenIDKey)
	if len(tokenID) == 0 {
		klog.V(3).InfoS("No key in Secret", "BootstrapTokenIDKey", bootstrapapi.BootstrapTokenIDKey, "secretNamespace", secret.Namespace, "secretName", secret.Name)
		return "", "", false
	}

	if nameTokenID != tokenID {
		klog.V(3).InfoS("Token ID doesn't match secret name", "tokenID", tokenID, "secretName", nameTokenID)
		return "", "", false
	}

	tokenSecret = bootstrapsecretutil.GetData(secret, bootstrapapi.BootstrapTokenSecretKey)
	if len(tokenSecret) == 0 {
		klog.V(3).InfoS("No key in Secret", "BootstrapTokenSecretKey", bootstrapapi.BootstrapTokenSecretKey, "secretNamespace", secret.Namespace, "secretName", secret.Name)
		return "", "", false
	}

	// Ensure this secret hasn't expired.  The TokenCleaner should remove this
	// but if that isn't working or it hasn't gotten there yet we should check
	// here.
	if bootstrapsecretutil.HasExpired(secret, time.Now()) {
		return "", "", false
	}

	// Make sure this secret can be used for signing
	okToSign := bootstrapsecretutil.GetData(secret, bootstrapapi.BootstrapTokenUsageSigningKey)
	if okToSign != "true" {
		return "", "", false
	}

	return tokenID, tokenSecret, true
}
