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

// Package clustertrustbundle provides Registry interface and its RESTStorage
// implementation for storing ClusterTrustBundle objects.
package clustertrustbundle

import (
	"context"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/certificates"
	certvalidation "k8s.io/kubernetes/pkg/apis/certificates/validation"
)

// strategy implements behavior for ClusterTrustBundles.
type strategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the create, update, and delete strategy for ClusterTrustBundles.
var Strategy = strategy{legacyscheme.Scheme, names.SimpleNameGenerator}

var _ rest.RESTCreateStrategy = Strategy
var _ rest.RESTUpdateStrategy = Strategy
var _ rest.RESTDeleteStrategy = Strategy

func (strategy) NamespaceScoped() bool {
	return false
}

func (strategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {}

func (strategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	bundle := obj.(*certificates.ClusterTrustBundle)
	allErrs := certvalidation.ValidateClusterTrustBundle(bundle, certvalidation.ValidateClusterTrustBundleOptions{})
	return rest.ValidateDeclarativelyWithMigrationChecks(ctx, legacyscheme.Scheme, bundle, nil, allErrs, operation.Create)
}

func (strategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

func (strategy) Canonicalize(obj runtime.Object) {}

func (strategy) AllowCreateOnUpdate() bool {
	return false
}

func (s strategy) PrepareForUpdate(ctx context.Context, new, old runtime.Object) {}

func (s strategy) ValidateUpdate(ctx context.Context, new, old runtime.Object) field.ErrorList {
	newBundle := new.(*certificates.ClusterTrustBundle)
	oldBundle := old.(*certificates.ClusterTrustBundle)
	allErrs := certvalidation.ValidateClusterTrustBundleUpdate(newBundle, oldBundle)
	return rest.ValidateDeclarativelyWithMigrationChecks(ctx, legacyscheme.Scheme, newBundle, oldBundle, allErrs, operation.Update)
}

func (strategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (strategy) AllowUnconditionalUpdate() bool {
	return false
}
