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

package ipaddress

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

// ipAddressStrategy implements verification logic for Replication.
type ipAddressStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Replication IPAddress objects.
var Strategy = ipAddressStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// Strategy should implement rest.RESTCreateStrategy
var _ rest.RESTCreateStrategy = Strategy

// Strategy should implement rest.RESTUpdateStrategy
var _ rest.RESTUpdateStrategy = Strategy

// NamespaceScoped returns false because all IPAddresses is cluster scoped.
func (ipAddressStrategy) NamespaceScoped() bool {
	return false
}

// PrepareForCreate clears the status of an IPAddress before creation.
func (ipAddressStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	_ = obj.(*allocation.IPAddress)

}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (ipAddressStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newIPAddress := obj.(*allocation.IPAddress)
	oldIPAddress := old.(*allocation.IPAddress)

	_, _ = newIPAddress, oldIPAddress
}

// Validate validates a new IPAddress.
func (ipAddressStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	ipAddress := obj.(*allocation.IPAddress)
	err := validation.ValidateIPAddress(ipAddress)
	return err
}

// Canonicalize normalizes the object after validation.
func (ipAddressStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is false for IPAddress; this means POST is needed to create one.
func (ipAddressStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (ipAddressStrategy) ValidateUpdate(ctx context.Context, new, old runtime.Object) field.ErrorList {
	// newIPAddress := new.(*allocation.IPAddress)
	// oldIPAddress := old.(*allocation.IPAddress)
	return nil
}

// AllowUnconditionalUpdate is the default update policy for IPAddress objects.
func (ipAddressStrategy) AllowUnconditionalUpdate() bool {
	return true
}
