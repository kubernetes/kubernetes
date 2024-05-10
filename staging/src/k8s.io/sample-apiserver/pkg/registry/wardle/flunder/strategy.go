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

package flunder

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/sample-apiserver/pkg/apis/wardle/validation"

	"k8s.io/sample-apiserver/pkg/apis/wardle"
)

// NewStrategy creates and returns a flunderStrategy instance
func NewStrategy(typer runtime.ObjectTyper) flunderStrategy {
	return flunderStrategy{typer, names.SimpleNameGenerator}
}

type flunderStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

func (flunderStrategy) NamespaceScoped() bool {
	return true
}

func (flunderStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
}

func (flunderStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
}

func (flunderStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	flunder := obj.(*wardle.Flunder)
	return validation.ValidateFlunder(flunder)
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (flunderStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string { return nil }

func (flunderStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (flunderStrategy) AllowUnconditionalUpdate() bool {
	return false
}

func (flunderStrategy) Canonicalize(obj runtime.Object) {
}

func (flunderStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return field.ErrorList{}
}

// WarningsOnUpdate returns warnings for the given update.
func (flunderStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}
