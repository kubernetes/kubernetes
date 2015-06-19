/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/fielderrors"
)

// strategy implements behavior for Secret objects
type strategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Secret
// objects via the REST API.
var Strategy = strategy{api.Scheme, api.SimpleNameGenerator}

var _ = rest.RESTCreateStrategy(Strategy)

var _ = rest.RESTUpdateStrategy(Strategy)

func (strategy) NamespaceScoped() bool {
	return true
}

func (strategy) PrepareForCreate(obj runtime.Object) {
}

func (strategy) Validate(ctx api.Context, obj runtime.Object) fielderrors.ValidationErrorList {
	return validation.ValidateSecret(obj.(*api.Secret))
}

func (strategy) AllowCreateOnUpdate() bool {
	return false
}

func (strategy) PrepareForUpdate(obj, old runtime.Object) {
}

func (strategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) fielderrors.ValidationErrorList {
	return validation.ValidateSecretUpdate(old.(*api.Secret), obj.(*api.Secret))
}

func (strategy) AllowUnconditionalUpdate() bool {
	return true
}

// Matcher returns a generic matcher for a given label and field selector.
func Matcher(label labels.Selector, field fields.Selector) generic.Matcher {
	return generic.MatcherFunc(func(obj runtime.Object) (bool, error) {
		sa, ok := obj.(*api.Secret)
		if !ok {
			return false, fmt.Errorf("not a secret")
		}
		fields := SelectableFields(sa)
		return label.Matches(labels.Set(sa.Labels)) && field.Matches(fields), nil
	})
}

// SelectableFields returns a label set that can be used for filter selection
func SelectableFields(obj *api.Secret) labels.Set {
	return labels.Set{
		"type": string(obj.Type),
	}
}
