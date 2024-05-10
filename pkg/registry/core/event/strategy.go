/*
Copyright 2014 The Kubernetes Authors.

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

package event

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/validation"
)

type eventStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating
// Event objects via the REST API.
var Strategy = eventStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

func (eventStrategy) DefaultGarbageCollectionPolicy(ctx context.Context) rest.GarbageCollectionPolicy {
	return rest.Unsupported
}

func (eventStrategy) NamespaceScoped() bool {
	return true
}

func (eventStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
}

func (eventStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
}

func (eventStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	groupVersion := requestGroupVersion(ctx)
	event := obj.(*api.Event)
	return validation.ValidateEventCreate(event, groupVersion)
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (eventStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string { return nil }

// Canonicalize normalizes the object after validation.
func (eventStrategy) Canonicalize(obj runtime.Object) {
}

func (eventStrategy) AllowCreateOnUpdate() bool {
	return true
}

func (eventStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	groupVersion := requestGroupVersion(ctx)
	event := obj.(*api.Event)
	oldEvent := old.(*api.Event)
	return validation.ValidateEventUpdate(event, oldEvent, groupVersion)
}

// WarningsOnUpdate returns warnings for the given update.
func (eventStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (eventStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// requestGroupVersion returns the group/version associated with the given context, or a zero-value group/version.
func requestGroupVersion(ctx context.Context) schema.GroupVersion {
	if requestInfo, found := genericapirequest.RequestInfoFrom(ctx); found {
		return schema.GroupVersion{Group: requestInfo.APIGroup, Version: requestInfo.APIVersion}
	}
	return schema.GroupVersion{}
}
