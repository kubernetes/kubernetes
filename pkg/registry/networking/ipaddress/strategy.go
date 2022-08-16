/*
Copyright 2022 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/apis/networking"
	"k8s.io/kubernetes/pkg/apis/networking/validation"
)

// ipAddressStrategy implements verification logic for Replication.
type ipAddressStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// noopNameGenerator does not generate names, it just returns the base.
type noopNameGenerator struct{}

func (noopNameGenerator) GenerateName(base string) string {
	return base
}

// Strategy is the default logic that applies when creating and updating Replication IPAddress objects.
var Strategy = ipAddressStrategy{legacyscheme.Scheme, noopNameGenerator{}}

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
	_ = obj.(*networking.IPAddress)

}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (ipAddressStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newIPAddress := obj.(*networking.IPAddress)
	oldIPAddress := old.(*networking.IPAddress)

	_, _ = newIPAddress, oldIPAddress
}

// Validate validates a new IPAddress.
func (ipAddressStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	ipAddress := obj.(*networking.IPAddress)
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
	newIPAddress := new.(*networking.IPAddress)
	oldIPAddress := old.(*networking.IPAddress)
	errList := validation.ValidateIPAddress(newIPAddress)
	errList = append(errList, validation.ValidateIPAddressUpdate(newIPAddress, oldIPAddress)...)
	return errList
}

// AllowUnconditionalUpdate is the default update policy for IPAddress objects.
func (ipAddressStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (ipAddressStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

// WarningsOnUpdate returns warnings for the given update.
func (ipAddressStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}
