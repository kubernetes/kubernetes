/*
Copyright 2015 The Kubernetes Authors.

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

package secret

import (
	"context"
	"crypto/tls"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/validation"
)

// strategy implements behavior for Secret objects
type strategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Secret
// objects via the REST API.
var Strategy = strategy{legacyscheme.Scheme, names.SimpleNameGenerator}

var _ = rest.RESTCreateStrategy(Strategy)

var _ = rest.RESTUpdateStrategy(Strategy)

func (strategy) NamespaceScoped() bool {
	return true
}

func (strategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	secret := obj.(*api.Secret)
	dropDisabledFields(secret, nil)
}

func (strategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	return validation.ValidateSecret(obj.(*api.Secret))
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (strategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return warningsForSecret(obj.(*api.Secret))
}

func (strategy) Canonicalize(obj runtime.Object) {
}

func (strategy) AllowCreateOnUpdate() bool {
	return false
}

func (strategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newSecret := obj.(*api.Secret)
	oldSecret := old.(*api.Secret)

	// this is weird, but consistent with what the validatedUpdate function used to do.
	if len(newSecret.Type) == 0 {
		newSecret.Type = oldSecret.Type
	}

	dropDisabledFields(newSecret, oldSecret)
}

func (strategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateSecretUpdate(obj.(*api.Secret), old.(*api.Secret))
}

// WarningsOnUpdate returns warnings for the given update.
func (strategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return warningsForSecret(obj.(*api.Secret))
}

func dropDisabledFields(secret *api.Secret, oldSecret *api.Secret) {
}

func (strategy) AllowUnconditionalUpdate() bool {
	return true
}

func warningsForSecret(secret *api.Secret) []string {
	var warnings []string
	if secret.Type == api.SecretTypeTLS {
		// Verify that the key matches the cert.
		_, err := tls.X509KeyPair(secret.Data[api.TLSCertKey], secret.Data[api.TLSPrivateKeyKey])
		if err != nil {
			warnings = append(warnings, err.Error())
		}
	}
	return warnings
}
