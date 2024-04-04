/*
Copyright 2024 The Kubernetes Authors.

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

package validation

import (
	"k8s.io/kubernetes/pkg/apis/core"
)

var (
	// we have multiple controllers reconciling the same secret,
	// resulting in unexpected outcomes such as the generation of new key pairs.
	// our goal is to prevent the generation of new key pairs by disallowing
	// deletions and permitting only updates, which appear to be 'safe'.
	//
	// thus we make an exception for the secrets in the following namespaces, during update
	// we allow the secret type to mutate from:
	//     ["SecretTypeTLS", core.SecretTypeOpaque] -> "kubernetes.io/tls"
	// some of our operators were accidentally creating secrets of type
	// "SecretTypeTLS", and this patch enables us to move these secrets
	// objects to the intended type in a ratcheting manner.
	//
	// we can drop this patch when we migrate all of the affected secret
	// objects to to intended type: https://issues.redhat.com/browse/API-1800
	whitelist = map[string]struct{}{
		"openshift-kube-apiserver-operator":          {},
		"openshift-kube-apiserver":                   {},
		"openshift-kube-controller-manager-operator": {},
		"openshift-config-managed":                   {},
	}
)

func openShiftValidateSecretUpdateIsTypeMutationAllowed(newSecret, oldSecret *core.Secret) bool {
	// initially, this check was stricter.
	// however, due to the platform's long history (spanning several years)
	// and the complexity of ensuring that resources were consistently created with only one type,
	// it is now permissible for (SecretTypeTLS, core.SecretTypeOpaque) type to transition to "kubernetes.io/tls".
	//
	// additionally, it should be noted that default values might also be applied in some cases.
	// (https://github.com/openshift/kubernetes/blob/258f1d5fb6491ba65fd8201c827e179432430627/pkg/apis/core/v1/defaults.go#L280-L284)
	if isOldSecretTypeMutationAllowed(oldSecret) && newSecret.Type == core.SecretTypeTLS {
		if _, ok := whitelist[oldSecret.Namespace]; ok {
			return true
		}
	}
	return false
}

func isOldSecretTypeMutationAllowed(oldSecret *core.Secret) bool {
	// core.SecretTypeOpaque seems safe because
	// https://github.com/kubernetes/kubernetes/blob/8628c3c4da6746b1dc967cc520b189a04ebd78d1/pkg/apis/core/validation/validation.go#L6393
	//
	// "SecretTypeTLS" is what kas-o used
	return oldSecret.Type == core.SecretTypeOpaque || oldSecret.Type == "SecretTypeTLS"
}
