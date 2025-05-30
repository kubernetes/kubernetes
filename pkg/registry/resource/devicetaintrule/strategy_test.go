/*
Copyright 2025 The Kubernetes Authors.

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

package devicetaintrule

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/resource"
)

var patch = &resource.DeviceTaintRule{
	ObjectMeta: metav1.ObjectMeta{
		Name: "valid-patch",
	},
	Spec: resource.DeviceTaintRuleSpec{
		Taint: resource.DeviceTaint{
			Key:    "example.com/tainted",
			Effect: resource.DeviceTaintEffectNoExecute,
		},
	},
}

func TestDeviceTaintRuleStrategy(t *testing.T) {
	if Strategy.NamespaceScoped() {
		t.Errorf("DeviceTaintRule must not be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("DeviceTaintRule should not allow create on update")
	}
}

func TestDeviceTaintRuleStrategyCreate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	patch := patch.DeepCopy()

	Strategy.PrepareForCreate(ctx, patch)
	errs := Strategy.Validate(ctx, patch)
	if len(errs) != 0 {
		t.Errorf("unexpected error validating for create %v", errs)
	}
}

func TestDeviceTaintRuleStrategyUpdate(t *testing.T) {
	t.Run("no-changes-okay", func(t *testing.T) {
		ctx := genericapirequest.NewDefaultContext()
		patch := patch.DeepCopy()
		newPatch := patch.DeepCopy()
		newPatch.ResourceVersion = "4"

		Strategy.PrepareForUpdate(ctx, newPatch, patch)
		errs := Strategy.ValidateUpdate(ctx, newPatch, patch)
		if len(errs) != 0 {
			t.Errorf("unexpected validation errors: %v", errs)
		}
	})

	t.Run("name-change-not-allowed", func(t *testing.T) {
		ctx := genericapirequest.NewDefaultContext()
		patch := patch.DeepCopy()
		newPatch := patch.DeepCopy()
		newPatch.Name = "valid-patch-2"
		newPatch.ResourceVersion = "4"

		Strategy.PrepareForUpdate(ctx, newPatch, patch)
		errs := Strategy.ValidateUpdate(ctx, newPatch, patch)
		if len(errs) == 0 {
			t.Errorf("expected a validation error")
		}
	})
}
