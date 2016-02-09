/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

	"k8s.io/kubernetes/pkg/api"
	//"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
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

func ClusterToSelectableFields(cluster *api.Cluster) fields.Set {
	return fields.Set{}
}

func MatchCluster(label labels.Selector, field fields.Selector) generic.Matcher {
	return &generic.SelectionPredicate{
		Label: label,
		Field: field,
		GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
			pt, ok := obj.(*api.Cluster)
			if !ok {
				return nil, nil, fmt.Errorf("given object is not a cluster.")
			}
			return labels.Set(pt.ObjectMeta.Labels), ClusterToSelectableFields(pt), nil
		},
	}
}

// PrepareForCreate clears fields that are not allowed to be set by end users on creation.
func (clusterStrategy) PrepareForCreate(obj runtime.Object) {
	cluster := obj.(*api.Cluster)
	// TODO
	cluster.Status = api.ClusterStatus{
		Phase: api.ClusterPending,
	}
}

// Validate validates a new cluster.
func (clusterStrategy) Validate(ctx api.Context, obj runtime.Object) field.ErrorList {
	// TODO
	return field.ErrorList{}
}

// Canonicalize normalizes the object after validation.
func (clusterStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is false for cluster.
func (clusterStrategy) AllowCreateOnUpdate() bool {
	return false
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (clusterStrategy) PrepareForUpdate(obj, old runtime.Object) {
	cluster := obj.(*api.Cluster)
	oldCluster := old.(*api.Cluster)
	cluster.Status = oldCluster.Status
}

// ValidateUpdate is the default update validation for an end user.
func (clusterStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) field.ErrorList {
	// TODO
	return field.ErrorList{}
}
func (clusterStrategy) AllowUnconditionalUpdate() bool {
	return true
}

type clusterStatusStrategy struct {
	clusterStrategy
}

var StatusStrategy = clusterStatusStrategy{Strategy}

func (clusterStatusStrategy) PrepareForCreate(obj runtime.Object) {
	_ = obj.(*api.Cluster)
}
func (clusterStatusStrategy) PrepareForUpdate(obj, old runtime.Object) {
	cluster := obj.(*api.Cluster)
	oldCluster := old.(*api.Cluster)
	cluster.Spec = oldCluster.Spec
}

// ValidateUpdate is the default update validation for an end user.
func (clusterStatusStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) field.ErrorList {
	// TODO
	return field.ErrorList{}
}
