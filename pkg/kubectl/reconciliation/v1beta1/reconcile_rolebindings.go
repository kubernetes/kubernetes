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

package v1beta1

import (
	"fmt"
	"reflect"

	rbacv1beta1 "k8s.io/api/rbac/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
)

// RoleBindingModifier is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
type RoleBindingModifier interface {
	// Get is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	Get(namespace, name string) (RoleBinding, error)
	// Delete is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	Delete(namespace, name string, uid types.UID) error
	// Create is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	Create(RoleBinding) (RoleBinding, error)
	// Update is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	Update(RoleBinding) (RoleBinding, error)
}

// RoleBinding is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
type RoleBinding interface {
	// GetObject is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	GetObject() runtime.Object
	// GetNamespace is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	GetNamespace() string
	// GetName is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	GetName() string
	// GetUID is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	GetUID() types.UID
	// GetLabels is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	GetLabels() map[string]string
	// SetLabels is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	SetLabels(map[string]string)
	// GetAnnotations is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	GetAnnotations() map[string]string
	// SetAnnotations is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	SetAnnotations(map[string]string)
	// GetRoleRef is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	GetRoleRef() rbacv1beta1.RoleRef
	// GetSubjects is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	GetSubjects() []rbacv1beta1.Subject
	// SetSubjects is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	SetSubjects([]rbacv1beta1.Subject)
	// DeepCopyRoleBinding is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
	DeepCopyRoleBinding() RoleBinding
}

// ReconcileRoleBindingOptions holds options for running a role binding reconciliation
type ReconcileRoleBindingOptions struct {
	// RoleBinding is the expected rolebinding that will be reconciled
	RoleBinding RoleBinding
	// Confirm indicates writes should be performed. When false, results are returned as a dry-run.
	Confirm bool
	// RemoveExtraSubjects indicates reconciliation should remove extra subjects from an existing role binding
	RemoveExtraSubjects bool
	// Client is used to look up existing rolebindings, and create/update the rolebinding when Confirm=true
	Client RoleBindingModifier
}

// ReconcileClusterRoleBindingResult holds the result of a reconciliation operation.
type ReconcileClusterRoleBindingResult struct {
	// RoleBinding is the reconciled rolebinding from the reconciliation operation.
	// If the reconcile was performed as a dry-run, or the existing rolebinding was protected, the reconciled rolebinding is not persisted.
	RoleBinding RoleBinding

	// MissingSubjects contains expected subjects that were missing from the currently persisted rolebinding
	MissingSubjects []rbacv1beta1.Subject
	// ExtraSubjects contains extra subjects the currently persisted rolebinding had
	ExtraSubjects []rbacv1beta1.Subject

	// Operation is the API operation required to reconcile.
	// If no reconciliation was needed, it is set to ReconcileNone.
	// If options.Confirm == false, the reconcile was in dry-run mode, so the operation was not performed.
	// If result.Protected == true, the rolebinding opted out of reconciliation, so the operation was not performed.
	// Otherwise, the operation was performed.
	Operation ReconcileOperation
	// Protected indicates an existing role prevented reconciliation
	Protected bool
}

// Run is a duplication in k8s.io/kubernetes/pkg/registry/rbac/reconciliation
func (o *ReconcileRoleBindingOptions) Run() (*ReconcileClusterRoleBindingResult, error) {
	return o.run(0)
}

func (o *ReconcileRoleBindingOptions) run(attempts int) (*ReconcileClusterRoleBindingResult, error) {
	// This keeps us from retrying forever if a rolebinding keeps appearing and disappearing as we reconcile.
	// Conflict errors on update are handled at a higher level.
	if attempts > 3 {
		return nil, fmt.Errorf("exceeded maximum attempts")
	}

	var result *ReconcileClusterRoleBindingResult

	existingBinding, err := o.Client.Get(o.RoleBinding.GetNamespace(), o.RoleBinding.GetName())
	switch {
	case errors.IsNotFound(err):
		result = &ReconcileClusterRoleBindingResult{
			RoleBinding:     o.RoleBinding,
			MissingSubjects: o.RoleBinding.GetSubjects(),
			Operation:       ReconcileCreate,
		}

	case err != nil:
		return nil, err

	default:
		result, err = computeReconciledRoleBinding(existingBinding, o.RoleBinding, o.RemoveExtraSubjects)
		if err != nil {
			return nil, err
		}
	}

	// If reconcile-protected, short-circuit
	if result.Protected {
		return result, nil
	}
	// If we're in dry-run mode, short-circuit
	if !o.Confirm {
		return result, nil
	}

	switch result.Operation {
	case ReconcileRecreate:
		// Try deleting
		err := o.Client.Delete(existingBinding.GetNamespace(), existingBinding.GetName(), existingBinding.GetUID())
		switch {
		case err == nil, errors.IsNotFound(err):
			// object no longer exists, as desired
		case errors.IsConflict(err):
			// delete failed because our UID precondition conflicted
			// this could mean another object exists with a different UID, re-run
			return o.run(attempts + 1)
		default:
			// return other errors
			return nil, err
		}
		// continue to create
		fallthrough
	case ReconcileCreate:
		created, err := o.Client.Create(result.RoleBinding)
		// If created since we started this reconcile, re-run
		if errors.IsAlreadyExists(err) {
			return o.run(attempts + 1)
		}
		if err != nil {
			return nil, err
		}
		result.RoleBinding = created

	case ReconcileUpdate:
		updated, err := o.Client.Update(result.RoleBinding)
		// If deleted since we started this reconcile, re-run
		if errors.IsNotFound(err) {
			return o.run(attempts + 1)
		}
		if err != nil {
			return nil, err
		}
		result.RoleBinding = updated

	case ReconcileNone:
		// no-op

	default:
		return nil, fmt.Errorf("invalid operation: %v", result.Operation)
	}

	return result, nil
}

// computeReconciledRoleBinding returns the rolebinding that must be created and/or updated to make the
// existing rolebinding's subjects, roleref, labels, and annotations match the expected rolebinding
func computeReconciledRoleBinding(existing, expected RoleBinding, removeExtraSubjects bool) (*ReconcileClusterRoleBindingResult, error) {
	result := &ReconcileClusterRoleBindingResult{Operation: ReconcileNone}

	result.Protected = (existing.GetAnnotations()[rbacv1beta1.AutoUpdateAnnotationKey] == "false")

	// Reset the binding completely if the roleRef is different
	if expected.GetRoleRef() != existing.GetRoleRef() {
		result.RoleBinding = expected
		result.Operation = ReconcileRecreate
		return result, nil
	}

	// Start with a copy of the existing object
	result.RoleBinding = existing.DeepCopyRoleBinding()

	// Merge expected annotations and labels
	result.RoleBinding.SetAnnotations(merge(expected.GetAnnotations(), result.RoleBinding.GetAnnotations()))
	if !reflect.DeepEqual(result.RoleBinding.GetAnnotations(), existing.GetAnnotations()) {
		result.Operation = ReconcileUpdate
	}
	result.RoleBinding.SetLabels(merge(expected.GetLabels(), result.RoleBinding.GetLabels()))
	if !reflect.DeepEqual(result.RoleBinding.GetLabels(), existing.GetLabels()) {
		result.Operation = ReconcileUpdate
	}

	// Compute extra and missing subjects
	result.MissingSubjects, result.ExtraSubjects = diffSubjectLists(expected.GetSubjects(), existing.GetSubjects())

	switch {
	case !removeExtraSubjects && len(result.MissingSubjects) > 0:
		// add missing subjects in the union case
		result.RoleBinding.SetSubjects(append(result.RoleBinding.GetSubjects(), result.MissingSubjects...))
		result.Operation = ReconcileUpdate

	case removeExtraSubjects && (len(result.MissingSubjects) > 0 || len(result.ExtraSubjects) > 0):
		// stomp to expected subjects in the non-union case
		result.RoleBinding.SetSubjects(expected.GetSubjects())
		result.Operation = ReconcileUpdate
	}

	return result, nil
}

func contains(list []rbacv1beta1.Subject, item rbacv1beta1.Subject) bool {
	for _, listItem := range list {
		if listItem == item {
			return true
		}
	}
	return false
}

// diffSubjectLists returns lists containing the items unique to each provided list:
//   list1Only = list1 - list2
//   list2Only = list2 - list1
// if both returned lists are empty, the provided lists are equal
func diffSubjectLists(list1 []rbacv1beta1.Subject, list2 []rbacv1beta1.Subject) (list1Only []rbacv1beta1.Subject, list2Only []rbacv1beta1.Subject) {
	for _, list1Item := range list1 {
		if !contains(list2, list1Item) {
			if !contains(list1Only, list1Item) {
				list1Only = append(list1Only, list1Item)
			}
		}
	}
	for _, list2Item := range list2 {
		if !contains(list1, list2Item) {
			if !contains(list2Only, list2Item) {
				list2Only = append(list2Only, list2Item)
			}
		}
	}
	return
}
