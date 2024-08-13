/*
Copyright 2021 The Kubernetes Authors.

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

package podtemplate

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func TestStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if !Strategy.NamespaceScoped() {
		t.Errorf("must be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("should not allow create on update")
	}

	podTemplate := &api.PodTemplate{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "mytemplate",
			Namespace:  metav1.NamespaceDefault,
			Generation: 999,
		},
		Template: api.PodTemplateSpec{
			Spec: podtest.MakePod("",
				podtest.SetRestartPolicy(api.RestartPolicyOnFailure),
			).Spec,
		},
	}

	Strategy.PrepareForCreate(ctx, podTemplate)
	if podTemplate.Generation != 1 {
		t.Errorf("expected Generation=1, got %d", podTemplate.Generation)
	}
	errs := Strategy.Validate(ctx, podTemplate)
	if len(errs) != 0 {
		t.Errorf("Unexpected error validating %v", errs)
	}

	// ensure we do not change generation for non-spec updates
	updatedLabel := podTemplate.DeepCopy()
	updatedLabel.Labels = map[string]string{"a": "true"}
	Strategy.PrepareForUpdate(ctx, updatedLabel, podTemplate)
	if updatedLabel.Generation != 1 {
		t.Errorf("expected Generation=1, got %d", updatedLabel.Generation)
	}

	updatedTemplate := podTemplate.DeepCopy()
	updatedTemplate.ResourceVersion = "10"
	updatedTemplate.Generation = 999
	updatedTemplate.Template.Spec.RestartPolicy = api.RestartPolicyNever

	// ensure generation is updated for spec changes
	Strategy.PrepareForUpdate(ctx, updatedTemplate, podTemplate)
	if updatedTemplate.Generation != 2 {
		t.Errorf("expected Generation=2, got %d", updatedTemplate.Generation)
	}
	errs = Strategy.ValidateUpdate(ctx, updatedTemplate, podTemplate)
	if len(errs) != 0 {
		t.Errorf("Unexpected error validating %v", errs)
	}

	invalidUpdatedTemplate := updatedTemplate.DeepCopy()
	invalidUpdatedTemplate.Name = "changed"
	Strategy.PrepareForUpdate(ctx, invalidUpdatedTemplate, podTemplate)
	errs = Strategy.ValidateUpdate(ctx, invalidUpdatedTemplate, podTemplate)
	if len(errs) == 0 {
		t.Errorf("expected error validating, got none")
	}
}
