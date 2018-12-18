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

package podsecuritypolicy

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	psputil "k8s.io/kubernetes/pkg/api/podsecuritypolicy"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/apis/policy/validation"
)

// strategy implements behavior for PodSecurityPolicy objects
type strategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating PodSecurityPolicy
// objects via the REST API.
var Strategy = strategy{legacyscheme.Scheme, names.SimpleNameGenerator}

var _ = rest.RESTCreateStrategy(Strategy)

var _ = rest.RESTUpdateStrategy(Strategy)

func (strategy) NamespaceScoped() bool {
	return false
}

func (strategy) AllowCreateOnUpdate() bool {
	return false
}

func (strategy) AllowUnconditionalUpdate() bool {
	return true
}

func (strategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	psp := obj.(*policy.PodSecurityPolicy)

	psputil.DropDisabledFields(&psp.Spec, nil)
}

func (strategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newPsp := obj.(*policy.PodSecurityPolicy)
	oldPsp := old.(*policy.PodSecurityPolicy)

	psputil.DropDisabledFields(&newPsp.Spec, &oldPsp.Spec)
}

func (strategy) Canonicalize(obj runtime.Object) {
}

func (strategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	return validation.ValidatePodSecurityPolicy(obj.(*policy.PodSecurityPolicy))
}

func (strategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidatePodSecurityPolicyUpdate(old.(*policy.PodSecurityPolicy), obj.(*policy.PodSecurityPolicy))
}
