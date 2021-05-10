/*
Copyright 2019 The Kubernetes Authors.

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

package serviceip

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/allocation"
	"k8s.io/kubernetes/pkg/apis/allocation/validation"
)

// serviceIPStrategy implements verification logic for Replication.
type serviceIPStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Replication ServiceIP objects.
var Strategy = serviceIPStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// Strategy should implement rest.RESTCreateStrategy
var _ rest.RESTCreateStrategy = Strategy

// Strategy should implement rest.RESTUpdateStrategy
var _ rest.RESTUpdateStrategy = Strategy

// NamespaceScoped returns false because all ServiceIPes is cluster scoped.
func (serviceIPStrategy) NamespaceScoped() bool {
	return false
}

// PrepareForCreate clears the status of an ServiceIP before creation.
func (serviceIPStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	_ = obj.(*allocation.ServiceIP)

}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (serviceIPStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newServiceIP := obj.(*allocation.ServiceIP)
	oldServiceIP := old.(*allocation.ServiceIP)

	_, _ = newServiceIP, oldServiceIP
}

// Validate validates a new ServiceIP.
func (serviceIPStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	serviceIP := obj.(*allocation.ServiceIP)
	err := validation.ValidateServiceIP(serviceIP)
	return err
}

// Canonicalize normalizes the object after validation.
func (serviceIPStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is false for ServiceIP; this means POST is needed to create one.
func (serviceIPStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (serviceIPStrategy) ValidateUpdate(ctx context.Context, new, old runtime.Object) field.ErrorList {
	// newServiceIP := new.(*allocation.ServiceIP)
	// oldServiceIP := old.(*allocation.ServiceIP)
	return nil
}

// AllowUnconditionalUpdate is the default update policy for ServiceIP objects.
func (serviceIPStrategy) AllowUnconditionalUpdate() bool {
	return true
}
