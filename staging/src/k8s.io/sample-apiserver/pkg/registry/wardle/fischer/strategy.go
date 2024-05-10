/*
Copyright 2017 The Kubernetes Authors.

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

package fischer

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
)

// NewStrategy creates and returns a fischerStrategy instance
func NewStrategy(typer runtime.ObjectTyper) fischerStrategy {
	return fischerStrategy{typer, names.SimpleNameGenerator}
}

type fischerStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

func (fischerStrategy) NamespaceScoped() bool {
	return false
}

func (fischerStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
}

func (fischerStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
}

func (fischerStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	return field.ErrorList{}
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (fischerStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string { return nil }

func (fischerStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (fischerStrategy) AllowUnconditionalUpdate() bool {
	return false
}

func (fischerStrategy) Canonicalize(obj runtime.Object) {
}

func (fischerStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return field.ErrorList{}
}

// WarningsOnUpdate returns warnings for the given update.
func (fischerStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}
