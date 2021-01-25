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

package iprange

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

// ipRangeStrategy implements verification logic for IPRange allocators.
type ipRangeStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Replication IPRange objects.
var Strategy = ipRangeStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// Strategy should implement rest.RESTCreateStrategy
var _ rest.RESTCreateStrategy = Strategy

// Strategy should implement rest.RESTUpdateStrategy
var _ rest.RESTUpdateStrategy = Strategy

// NamespaceScoped returns false because all IPRangees is cluster scoped.
func (ipRangeStrategy) NamespaceScoped() bool {
	return false
}

// PrepareForCreate clears the status of an IPRange before creation.
func (ipRangeStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	_ = obj.(*allocation.IPRange)

}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (ipRangeStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newIPRange := obj.(*allocation.IPRange)
	oldIPRange := old.(*allocation.IPRange)

	_, _ = newIPRange, oldIPRange
}

// Validate validates a new IPRange.
func (ipRangeStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	ipAddress := obj.(*allocation.IPRange)
	err := validation.ValidateIPRange(ipAddress)
	return err
}

// Canonicalize normalizes the object after validation.
func (ipRangeStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is false for IPRange; this means POST is needed to create one.
func (ipRangeStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (ipRangeStrategy) ValidateUpdate(ctx context.Context, new, old runtime.Object) field.ErrorList {
	return nil
}

// AllowUnconditionalUpdate is the default update policy for IPRange objects.
func (ipRangeStrategy) AllowUnconditionalUpdate() bool {
	return true
}
