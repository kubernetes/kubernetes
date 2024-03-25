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

package resourceslice

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/resource"
)

var slice = &resource.ResourceSlice{
	ObjectMeta: metav1.ObjectMeta{
		Name: "valid-class",
	},
	NodeName:   "valid-node-name",
	DriverName: "testdriver.example.com",
	ResourceModel: resource.ResourceModel{
		NamedResources: &resource.NamedResourcesResources{},
	},
}

func TestClassStrategy(t *testing.T) {
	if Strategy.NamespaceScoped() {
		t.Errorf("ResourceSlice must not be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("ResourceSlice should not allow create on update")
	}
}

func TestClassStrategyCreate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	slice := slice.DeepCopy()

	Strategy.PrepareForCreate(ctx, slice)
	errs := Strategy.Validate(ctx, slice)
	if len(errs) != 0 {
		t.Errorf("unexpected error validating for create %v", errs)
	}
}

func TestClassStrategyUpdate(t *testing.T) {
	t.Run("no-changes-okay", func(t *testing.T) {
		ctx := genericapirequest.NewDefaultContext()
		slice := slice.DeepCopy()
		newClass := slice.DeepCopy()
		newClass.ResourceVersion = "4"

		Strategy.PrepareForUpdate(ctx, newClass, slice)
		errs := Strategy.ValidateUpdate(ctx, newClass, slice)
		if len(errs) != 0 {
			t.Errorf("unexpected validation errors: %v", errs)
		}
	})

	t.Run("name-change-not-allowed", func(t *testing.T) {
		ctx := genericapirequest.NewDefaultContext()
		slice := slice.DeepCopy()
		newClass := slice.DeepCopy()
		newClass.Name = "valid-class-2"
		newClass.ResourceVersion = "4"

		Strategy.PrepareForUpdate(ctx, newClass, slice)
		errs := Strategy.ValidateUpdate(ctx, newClass, slice)
		if len(errs) == 0 {
			t.Errorf("expected a validation error")
		}
	})
}
