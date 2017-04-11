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

package miskatonic

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
)

// Resource Validation
func (UniversityStrategy) Validate(ctx request.Context, obj runtime.Object) field.ErrorList {
	university := obj.(*University)
	errors := field.ErrorList{}
	if university.Spec.MaxStudents == nil || *university.Spec.MaxStudents < 1 || *university.Spec.MaxStudents > 150 {
		errors = append(errors, field.Invalid(
			field.NewPath("spec", "MaxStudents"),
			*university.Spec.MaxStudents,
			"Must be between 1 and 150"))
	}
	return errors
}

// Scale Subresource
var _ rest.CreaterUpdater = &ScaleUniversityREST{}
var _ rest.Patcher = &ScaleUniversityREST{}

type ScaleUniversityREST struct {
	Registry UniversityRegistry
}

func (r *ScaleUniversityREST) Create(ctx request.Context, obj runtime.Object) (runtime.Object, error) {
	scale := obj.(*Scale)
	u, err := r.Registry.GetUniversity(ctx, scale.Name, &metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	u.Spec.FacultySize = scale.Faculty
	r.Registry.UpdateUniversity(ctx, u)
	return u, nil
}

// Get retrieves the object from the storage. It is required to support Patch.
func (r *ScaleUniversityREST) Get(ctx request.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	return nil, nil
}

// Update alters the status subset of an object.
func (r *ScaleUniversityREST) Update(ctx request.Context, name string, objInfo rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	return nil, false, nil
}

func (r *ScaleUniversityREST) New() runtime.Object {
	return &Scale{}
}
