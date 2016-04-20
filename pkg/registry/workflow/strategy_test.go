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

package workflow

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
)

func NewWorkflow(name string) *extensions.Workflow {
	return &extensions.Workflow{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: api.NamespaceDefault,
		},
		Spec: extensions.WorkflowSpec{
			Steps: map[string]extensions.WorkflowStep{
				"one": {},
			},
			Selector: &unversioned.LabelSelector{
				MatchLabels: map[string]string{"a": "b"},
			},
		},
		Status: extensions.WorkflowStatus{
			Statuses: map[string]extensions.WorkflowStepStatus{},
		},
	}
}

func TestWorkflowStrategy(t *testing.T) {
	ctx := api.NewDefaultContext()
	if !Strategy.NamespaceScoped() {
		t.Errorf("Workflow must be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("Workflow should not allow create on update")
	}
	workflow := NewWorkflow("my-workflow")
	Strategy.PrepareForCreate(workflow)
	if !reflect.DeepEqual(workflow.Status, extensions.WorkflowStatus{}) {
		t.Errorf("Workflow does not allow setting status on create")
	}
	errs := Strategy.Validate(ctx, workflow)
	if len(errs) != 0 {
		t.Errorf("Unexpected error validating %v", errs)
	}

	updatedWorkflow := NewWorkflow("my-workflow")
	Strategy.PrepareForUpdate(updatedWorkflow, workflow)
	if !api.Semantic.DeepEqual(updatedWorkflow.Status, workflow.Status) {
		t.Errorf("PrepareForUpdate should have preserved prior version status")
	}
	errs = Strategy.ValidateUpdate(ctx, updatedWorkflow, workflow)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}
}

func TestWorkflowStatusStrategy(t *testing.T) {
	ctx := api.NewDefaultContext()
	if !StatusStrategy.NamespaceScoped() {
		t.Errorf("Workflow must be namespace scoped")
	}
	if StatusStrategy.AllowCreateOnUpdate() {
		t.Errorf("Workflow should not allow create on update")
	}
	oldWorkflow := NewWorkflow("my-workflow")
	oldWorkflow.ResourceVersion = "42"

	newWorkflow := NewWorkflow("my-workflow")
	newWorkflow.ResourceVersion = "100"
	newWorkflow.Status.Statuses["step_two"] = extensions.WorkflowStepStatus{}

	StatusStrategy.PrepareForUpdate(newWorkflow, oldWorkflow)
	if len(newWorkflow.Status.Statuses) != 1 {
		t.Errorf("Workflow status updates must allow changes to workflow status")
	}
	errs := StatusStrategy.ValidateUpdate(ctx, newWorkflow, oldWorkflow)
	if len(errs) != 0 {
		t.Errorf("Unexpected error %v", errs)
	}
	if newWorkflow.ResourceVersion != "100" {
		t.Errorf("Incoming resource version on update should not be mutated")
	}

}

func TestSelectableFieldLabelConversions(t *testing.T) {
	apitesting.TestSelectableFieldLabelConversionsOfKind(t,
		testapi.Batch.GroupVersion().String(),
		"Workflow",
		labels.Set(WorkflowToSelectableFields(&extensions.Workflow{})),
		nil,
	)
}

func TestMatchWorkflow(t *testing.T) {
	testcases := []struct {
		w             *extensions.Workflow
		fieldSelector fields.Selector
		expectMatch   bool
	}{
		{
			w:             NewWorkflow("mydag"),
			fieldSelector: fields.ParseSelectorOrDie("metadata.name=mydag"),
			expectMatch:   true,
		},
		{
			w:             NewWorkflow("mydag"),
			fieldSelector: fields.ParseSelectorOrDie("metadata.name=yourdag"),
			expectMatch:   false,
		},
	}
	for _, tc := range testcases {
		result, err := MatchWorkflow(labels.Everything(), tc.fieldSelector).Matches(tc.w)
		if err != nil {
			t.Errorf("Unexpected error %v", err)
		}
		if result != tc.expectMatch {
			t.Errorf("Result %v, Expected %v, Selector: %v, Workflow: %v", result, tc.expectMatch, tc.fieldSelector.String(), tc.w)
		}
	}
}
