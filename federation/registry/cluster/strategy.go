/*
Copyright 2016 The Kubernetes Authors.

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

package cluster

import (
	"fmt"

	"k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/federation/apis/federation/validation"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
	apistorage "k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

type clusterStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

var Strategy = clusterStrategy{api.Scheme, api.SimpleNameGenerator}

func (clusterStrategy) NamespaceScoped() bool {
	return false
}

func ClusterToSelectableFields(cluster *federation.Cluster) fields.Set {
	return generic.ObjectMetaFieldsSet(&cluster.ObjectMeta, false)
}

// GetAttrs returns labels and fields of a given object for filtering purposes.
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	cluster, ok := obj.(*federation.Cluster)
	if !ok {
		return nil, nil, fmt.Errorf("given object is not a cluster.")
	}
	return labels.Set(cluster.ObjectMeta.Labels), ClusterToSelectableFields(cluster), nil
}

func MatchCluster(label labels.Selector, field fields.Selector) apistorage.SelectionPredicate {
	return apistorage.SelectionPredicate{
		Label:    label,
		Field:    field,
		GetAttrs: GetAttrs,
	}
}

// PrepareForCreate clears fields that are not allowed to be set by end users on creation.
func (clusterStrategy) PrepareForCreate(ctx api.Context, obj runtime.Object) {
	cluster := obj.(*federation.Cluster)
	cluster.Status = federation.ClusterStatus{}
}

// Validate validates a new cluster.
func (clusterStrategy) Validate(ctx api.Context, obj runtime.Object) field.ErrorList {
	cluster := obj.(*federation.Cluster)
	return validation.ValidateCluster(cluster)
}

// Canonicalize normalizes the object after validation.
func (clusterStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is false for cluster.
func (clusterStrategy) AllowCreateOnUpdate() bool {
	return false
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (clusterStrategy) PrepareForUpdate(ctx api.Context, obj, old runtime.Object) {
	cluster := obj.(*federation.Cluster)
	oldCluster := old.(*federation.Cluster)
	cluster.Status = oldCluster.Status
}

// ValidateUpdate is the default update validation for an end user.
func (clusterStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateClusterUpdate(obj.(*federation.Cluster), old.(*federation.Cluster))
}
func (clusterStrategy) AllowUnconditionalUpdate() bool {
	return true
}

type clusterStatusStrategy struct {
	clusterStrategy
}

var StatusStrategy = clusterStatusStrategy{Strategy}

func (clusterStatusStrategy) PrepareForCreate(ctx api.Context, obj runtime.Object) {
	_ = obj.(*federation.Cluster)
}
func (clusterStatusStrategy) PrepareForUpdate(ctx api.Context, obj, old runtime.Object) {
	cluster := obj.(*federation.Cluster)
	oldCluster := old.(*federation.Cluster)
	cluster.Spec = oldCluster.Spec
}

// ValidateUpdate is the default update validation for an end user.
func (clusterStatusStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateClusterStatusUpdate(obj.(*federation.Cluster), old.(*federation.Cluster))
}
