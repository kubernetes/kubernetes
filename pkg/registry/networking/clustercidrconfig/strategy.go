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

package clustercidrconfig

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/networking"
	"k8s.io/kubernetes/pkg/apis/networking/validation"
)

// clusterCIDRConfigStrategy implements verification logic for ClusterCIDRConfigs.
type clusterCIDRConfigStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating clusterCIDRConfig objects.
var Strategy = clusterCIDRConfigStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// NamespaceScoped returns false because all clusterCIDRConfigs do not need to be within a namespace.
func (clusterCIDRConfigStrategy) NamespaceScoped() bool {
	return false
}

// PrepareForCreate clears the status of a ClusterCIDRConfig before creation.
func (clusterCIDRConfigStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	clusterCIDRConfig := obj.(*networking.ClusterCIDRConfig)
	clusterCIDRConfig.Generation = 1
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (clusterCIDRConfigStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newClusterCIDRConfig := obj.(*networking.ClusterCIDRConfig)
	oldClusterCIDRConfig := old.(*networking.ClusterCIDRConfig)

	// ClusterCIDRConfig spec is immutable, update is not allowed.
	newClusterCIDRConfig.Spec = oldClusterCIDRConfig.Spec
}

// Validate validates a new ClusterCIDRConfig.
func (clusterCIDRConfigStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	clusterCIDRConfig := obj.(*networking.ClusterCIDRConfig)
	return validation.ValidateClusterCIDRConfig(clusterCIDRConfig)
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (clusterCIDRConfigStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

// Canonicalize normalizes the object after validation.
func (clusterCIDRConfigStrategy) Canonicalize(obj runtime.Object) {}

// AllowCreateOnUpdate is false for ClusterCIDRConfig; this means POST is needed to create one.
func (clusterCIDRConfigStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (clusterCIDRConfigStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	validationErrorList := validation.ValidateClusterCIDRConfig(obj.(*networking.ClusterCIDRConfig))
	updateErrorList := validation.ValidateClusterCIDRConfigUpdate(obj.(*networking.ClusterCIDRConfig), old.(*networking.ClusterCIDRConfig))
	return append(validationErrorList, updateErrorList...)
}

// WarningsOnUpdate returns warnings for the given update.
func (clusterCIDRConfigStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

// AllowUnconditionalUpdate is the default update policy for ClusterCIDRConfig objects.
func (clusterCIDRConfigStrategy) AllowUnconditionalUpdate() bool {
	return true
}
