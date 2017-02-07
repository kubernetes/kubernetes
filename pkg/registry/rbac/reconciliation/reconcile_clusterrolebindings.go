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

package reconciliation

import (
	"fmt"
	"reflect"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/rbac/internalversion"
)

type ReconcileClusterRoleBindingOptions struct {
	RoleBinding *rbac.ClusterRoleBinding
	Confirm     bool
	Union       bool
	Client      internalversion.ClusterRoleBindingInterface
}

type ReconcileClusterRoleBindingResult struct {
	// Role is the resulting role binding from the reconciliation operation
	RoleBinding *rbac.ClusterRoleBinding

	// MissingSubjects contains expected rules that were missing from the currently persisted role
	MissingSubjects []rbac.Subject
	// ExtraSubjects contains extra permissions the currently persisted role had
	ExtraSubjects []rbac.Subject

	// Operation is the API operation required to reconcile
	Operation ReconcileOperation
	// Protected indicates an existing role prevented reconciliation
	Protected bool
}

func (o *ReconcileClusterRoleBindingOptions) Run() (*ReconcileClusterRoleBindingResult, error) {
	var result *ReconcileClusterRoleBindingResult

	existingBinding, err := o.Client.Get(o.RoleBinding.Name, metav1.GetOptions{})
	switch {
	case errors.IsNotFound(err):
		result = &ReconcileClusterRoleBindingResult{
			RoleBinding:     o.RoleBinding,
			MissingSubjects: o.RoleBinding.Subjects,
			Operation:       ReconcileCreate,
		}

	case err != nil:
		return nil, err

	default:
		result, err = changedRoleBinding(existingBinding, o.RoleBinding, o.Union)
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
		err := o.Client.Delete(
			existingBinding.Name,
			&metav1.DeleteOptions{Preconditions: &metav1.Preconditions{UID: &existingBinding.UID}},
		)
		switch {
		case err == nil:
			// delete was successful, continue to create
			fallthrough
		case errors.IsNotFound(err):
			// object no longer exists, continue to create
			fallthrough
		case errors.IsConflict(err):
			// delete failed because our UID precondition conflicted
			// this could mean another object exists with a different UID, re-run
			return o.Run()
		default:
			// return other errors
			return nil, err
		}
	case ReconcileCreate:
		created, err := o.Client.Create(result.RoleBinding)
		// If created since we started this reconcile, re-run
		if errors.IsAlreadyExists(err) {
			return o.Run()
		}
		if err != nil {
			return nil, err
		}
		result.RoleBinding = created

	case ReconcileUpdate:
		updated, err := o.Client.Update(result.RoleBinding)
		// If deleted since we started this reconcile, re-run
		if errors.IsNotFound(err) {
			return o.Run()
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

func changedRoleBinding(existing, expected *rbac.ClusterRoleBinding, union bool) (*ReconcileClusterRoleBindingResult, error) {
	result := &ReconcileClusterRoleBindingResult{Operation: ReconcileNone}

	result.Protected = (existing.Annotations[ReconcileProtectAnnotation] == "true")

	// Reset the binding completely if the roleRef is different
	if !equivalentRoleRefs(expected.RoleRef, existing.RoleRef) {
		result.RoleBinding = expected
		result.Operation = ReconcileRecreate
		return result, nil
	}

	// Start with a copy of the existing object
	changedObj, err := api.Scheme.DeepCopy(existing)
	if err != nil {
		return nil, err
	}
	result.RoleBinding = changedObj.(*rbac.ClusterRoleBinding)

	// Merge expected annotations and labels
	result.RoleBinding.Annotations = merge(expected.Annotations, result.RoleBinding.Annotations)
	if !reflect.DeepEqual(result.RoleBinding.Annotations, existing.Annotations) {
		result.Operation = ReconcileUpdate
	}
	result.RoleBinding.Labels = merge(expected.Labels, result.RoleBinding.Labels)
	if !reflect.DeepEqual(result.RoleBinding.Labels, existing.Labels) {
		result.Operation = ReconcileUpdate
	}

	// Compute extra and missing subjects
	result.MissingSubjects, result.ExtraSubjects = diffSubjectLists(expected.Subjects, existing.Subjects)

	switch {
	case union && len(result.MissingSubjects) > 0:
		// add missing subjects in the union case
		result.RoleBinding.Subjects = append(result.RoleBinding.Subjects, result.MissingSubjects...)
		result.Operation = ReconcileUpdate

	case !union && (len(result.MissingSubjects) > 0 || len(result.ExtraSubjects) > 0):
		// stomp to expected subjects in the non-union case
		result.RoleBinding.Subjects = expected.Subjects
		result.Operation = ReconcileUpdate
	}

	return result, nil
}

func equivalentRoleRefs(a, b rbac.RoleRef) bool {
	if a.APIGroup == "" || b.APIGroup == "" {
		// If only one specified an APIGroup, don't compare that field
		a.APIGroup = ""
		b.APIGroup = ""
	}
	return a == b
}

func equivalentSubjects(a, b rbac.Subject) bool {
	if a.APIVersion == "" || b.APIVersion == "" {
		// If either one omitted an APIVersion, don't compare that field
		a.APIVersion = ""
		b.APIVersion = ""
	} else {
		// If both specified an APIVersion, only compare the groups
		gvA, errA := schema.ParseGroupVersion(a.APIVersion)
		gvB, errB := schema.ParseGroupVersion(b.APIVersion)
		if errA == nil && errB == nil && gvA.Group == gvB.Group {
			a.APIVersion = b.APIVersion
		}
	}
	return a == b
}

func contains(list []rbac.Subject, item rbac.Subject) bool {
	for _, listItem := range list {
		if equivalentSubjects(listItem, item) {
			return true
		}
	}
	return false
}

// diffSubjectLists returns lists containing the items unique to each provided list:
//   list1Only = list1 - list2
//   list2Only = list2 - list1
// if both returned lists are empty, the provided lists are equal
func diffSubjectLists(list1 []rbac.Subject, list2 []rbac.Subject) (list1Only []rbac.Subject, list2Only []rbac.Subject) {
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
