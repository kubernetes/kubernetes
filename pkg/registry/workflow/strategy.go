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
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apis/extensions/validation"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

// workflowStrategy implements behavior for Workflow objects.
type workflowStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Workflow object via REST API.
var Strategy = workflowStrategy{api.Scheme, api.SimpleNameGenerator}

// NamespaceScoped returns true because all workflows need to be within a namespace.
func (workflowStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate clears the status of a workflow before creation.
func (workflowStrategy) PrepareForCreate(obj runtime.Object) {
	workflow := obj.(*extensions.Workflow)
	workflow.Status = extensions.WorkflowStatus{}
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (workflowStrategy) PrepareForUpdate(obj, old runtime.Object) {
	newWorkflow := obj.(*extensions.Workflow)
	oldWorkflow := old.(*extensions.Workflow)
	newWorkflow.Status = oldWorkflow.Status
}

// Validate validates a new workflow.
func (workflowStrategy) Validate(ctx api.Context, obj runtime.Object) field.ErrorList {
	workflow := obj.(*extensions.Workflow)
	return validation.ValidateWorkflow(workflow)
}

// Canonicalize normalizes the object after validation.
func (workflowStrategy) Canonicalize(obj runtime.Object) {
}

func (workflowStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// AllowCreateOnUpdate is false for workflows; this means a POST is needed to create one.
func (workflowStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (workflowStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) field.ErrorList {
	validationErrorList := validation.ValidateWorkflow(obj.(*extensions.Workflow))
	updateErrorList := validation.ValidateWorkflowUpdate(obj.(*extensions.Workflow), old.(*extensions.Workflow))
	return append(validationErrorList, updateErrorList...)
}

type workflowStatusStrategy struct {
	workflowStrategy
}

var StatusStrategy = workflowStatusStrategy{Strategy}

func (workflowStatusStrategy) PrepareForUpdate(obj, old runtime.Object) {
	newWorkflow := obj.(*extensions.Workflow)
	oldWorkflow := old.(*extensions.Workflow)
	newWorkflow.Spec = oldWorkflow.Spec
}

func (workflowStatusStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateWorkflowUpdateStatus(obj.(*extensions.Workflow), old.(*extensions.Workflow))
}

// WorkflowSelectableFields returns a field set that represents the object for matching purposes.
func WorkflowToSelectableFields(workflow *extensions.Workflow) fields.Set {
	return generic.ObjectMetaFieldsSet(workflow.ObjectMeta, true)
}

// MatchWorkflow is the filter used by the generic etcd backend to route
// watch events from etcd to clients of the apiserver only interested in specific
// labels/fields.
func MatchWorkflow(label labels.Selector, field fields.Selector) generic.Matcher {
	return &generic.SelectionPredicate{
		Label: label,
		Field: field,
		GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
			workflow, ok := obj.(*extensions.Workflow)
			if !ok {
				return nil, nil, fmt.Errorf("not a workflow.")
			}
			return labels.Set(workflow.ObjectMeta.Labels), WorkflowToSelectableFields(workflow), nil
		},
	}
}
