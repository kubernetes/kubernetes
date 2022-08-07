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

package clustercidr

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/networking"
	"k8s.io/kubernetes/pkg/apis/networking/validation"
)

// clusterCIDRStrategy implements verification logic for ClusterCIDRs.
type clusterCIDRStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating clusterCIDR objects.
var Strategy = clusterCIDRStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// NamespaceScoped returns false because all clusterCIDRs do not need to be within a namespace.
func (clusterCIDRStrategy) NamespaceScoped() bool {
	return false
}

func (clusterCIDRStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {}

func (clusterCIDRStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {}

// Validate validates a new ClusterCIDR.
func (clusterCIDRStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	clusterCIDR := obj.(*networking.ClusterCIDR)
	return validation.ValidateClusterCIDR(clusterCIDR)
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (clusterCIDRStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

// Canonicalize normalizes the object after validation.
func (clusterCIDRStrategy) Canonicalize(obj runtime.Object) {}

// AllowCreateOnUpdate is false for ClusterCIDR; this means POST is needed to create one.
func (clusterCIDRStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (clusterCIDRStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	validationErrorList := validation.ValidateClusterCIDR(obj.(*networking.ClusterCIDR))
	updateErrorList := validation.ValidateClusterCIDRUpdate(obj.(*networking.ClusterCIDR), old.(*networking.ClusterCIDR))
	return append(validationErrorList, updateErrorList...)
}

// WarningsOnUpdate returns warnings for the given update.
func (clusterCIDRStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

// AllowUnconditionalUpdate is the default update policy for ClusterCIDR objects.
func (clusterCIDRStrategy) AllowUnconditionalUpdate() bool {
	return true
}
