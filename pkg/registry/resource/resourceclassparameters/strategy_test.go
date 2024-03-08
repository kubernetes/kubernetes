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

package resourceclassparameters

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/resource"
)

var resourceClassParameters = &resource.ResourceClassParameters{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid",
		Namespace: "ns",
	},
}

func TestClassStrategy(t *testing.T) {
	if !Strategy.NamespaceScoped() {
		t.Errorf("ResourceClassParameters must be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("ResourceClassParameters should not allow create on update")
	}
}

func TestClassStrategyCreate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	resourceClassParameters := resourceClassParameters.DeepCopy()

	Strategy.PrepareForCreate(ctx, resourceClassParameters)
	errs := Strategy.Validate(ctx, resourceClassParameters)
	if len(errs) != 0 {
		t.Errorf("unexpected error validating for create %v", errs)
	}
}

func TestClassStrategyUpdate(t *testing.T) {
	t.Run("no-changes-okay", func(t *testing.T) {
		ctx := genericapirequest.NewDefaultContext()
		resourceClassParameters := resourceClassParameters.DeepCopy()
		newObj := resourceClassParameters.DeepCopy()
		newObj.ResourceVersion = "4"

		Strategy.PrepareForUpdate(ctx, newObj, resourceClassParameters)
		errs := Strategy.ValidateUpdate(ctx, newObj, resourceClassParameters)
		if len(errs) != 0 {
			t.Errorf("unexpected validation errors: %v", errs)
		}
	})

	t.Run("name-change-not-allowed", func(t *testing.T) {
		ctx := genericapirequest.NewDefaultContext()
		resourceClassParameters := resourceClassParameters.DeepCopy()
		newObj := resourceClassParameters.DeepCopy()
		newObj.Name += "-2"
		newObj.ResourceVersion = "4"

		Strategy.PrepareForUpdate(ctx, newObj, resourceClassParameters)
		errs := Strategy.ValidateUpdate(ctx, newObj, resourceClassParameters)
		if len(errs) == 0 {
			t.Errorf("expected a validation error")
		}
	})
}
