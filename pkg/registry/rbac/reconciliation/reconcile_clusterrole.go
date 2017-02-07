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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/rbac/internalversion"
	"k8s.io/kubernetes/pkg/registry/rbac/validation"
)

// ReconcileProtectAnnotation is the name of an annotation which prevents reconciliation if set to "true"
const ReconcileProtectAnnotation = "rbac.authorization.kubernetes.io/reconcile-protect"

type ReconcileOperation string

var (
	ReconcileCreate   ReconcileOperation = "create"
	ReconcileUpdate   ReconcileOperation = "update"
	ReconcileRecreate ReconcileOperation = "recreate"
	ReconcileNone     ReconcileOperation = "none"
)

type ReconcileClusterRoleOptions struct {
	// Role is the expected role that will be reconciled
	Role *rbac.ClusterRole
	// Confirm indicates writes should be performed. When false, results are returned as a dry-run.
	Confirm bool
	// Union indicates reconciliation should be additive (no permissions should be removed)
	Union bool
	// Client is used to look up existing roles, and create/update the role when Confirm=true
	Client internalversion.ClusterRoleInterface
}

type ReconcileClusterRoleResult struct {
	// Role is the resulting role from the reconciliation operation
	Role *rbac.ClusterRole

	// MissingRules contains expected rules that were missing from the currently persisted role
	MissingRules []rbac.PolicyRule
	// ExtraRules contains extra permissions the currently persisted role had
	ExtraRules []rbac.PolicyRule

	// Operation is the API operation required to reconcile
	Operation ReconcileOperation
	// Protected indicates an existing role prevented reconciliation
	Protected bool
}

func (o *ReconcileClusterRoleOptions) Run() (*ReconcileClusterRoleResult, error) {
	var result *ReconcileClusterRoleResult

	existing, err := o.Client.Get(o.Role.Name, metav1.GetOptions{})
	switch {
	case errors.IsNotFound(err):
		result = &ReconcileClusterRoleResult{
			Role:         o.Role,
			MissingRules: o.Role.Rules,
			Operation:    ReconcileCreate,
		}

	case err != nil:
		return nil, err

	default:
		result, err = changedRole(existing, o.Role, o.Union)
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
	case ReconcileCreate:
		created, err := o.Client.Create(result.Role)
		// If created since we started this reconcile, re-run
		if errors.IsAlreadyExists(err) {
			return o.Run()
		}
		if err != nil {
			return nil, err
		}
		result.Role = created

	case ReconcileUpdate:
		updated, err := o.Client.Update(result.Role)
		// If deleted since we started this reconcile, re-run
		if errors.IsNotFound(err) {
			return o.Run()
		}
		if err != nil {
			return nil, err
		}
		result.Role = updated

	case ReconcileNone:
		// no-op

	default:
		return nil, fmt.Errorf("invalid operation: %v", result.Operation)
	}

	return result, nil
}

// ChangedClusterRoles returns the roles that must be created and/or updated to
// match the recommended bootstrap policy
func changedRole(existing, expected *rbac.ClusterRole, union bool) (*ReconcileClusterRoleResult, error) {
	result := &ReconcileClusterRoleResult{Operation: ReconcileNone}

	result.Protected = (existing.Annotations[ReconcileProtectAnnotation] == "true")

	// Start with a copy of the existing object
	changedObj, err := api.Scheme.DeepCopy(existing)
	if err != nil {
		return nil, err
	}
	result.Role = changedObj.(*rbac.ClusterRole)

	// Merge expected annotations and labels
	result.Role.Annotations = merge(expected.Annotations, result.Role.Annotations)
	if !reflect.DeepEqual(result.Role.Annotations, existing.Annotations) {
		result.Operation = ReconcileUpdate
	}
	result.Role.Labels = merge(expected.Labels, result.Role.Labels)
	if !reflect.DeepEqual(result.Role.Labels, existing.Labels) {
		result.Operation = ReconcileUpdate
	}

	// Compute extra and missing rules
	_, result.ExtraRules = validation.Covers(expected.Rules, existing.Rules)
	_, result.MissingRules = validation.Covers(existing.Rules, expected.Rules)

	switch {
	case union && len(result.MissingRules) > 0:
		// add missing rules in the union case
		result.Role.Rules = append(result.Role.Rules, result.MissingRules...)
		result.Operation = ReconcileUpdate

	case !union && (len(result.MissingRules) > 0 || len(result.ExtraRules) > 0):
		// stomp to expected rules in the non-union case
		result.Role.Rules = expected.Rules
		result.Operation = ReconcileUpdate
	}

	return result, nil
}

// merge combines the given maps with the later annotations having higher precedence
func merge(maps ...map[string]string) map[string]string {
	var output map[string]string = nil
	for _, m := range maps {
		if m != nil && output == nil {
			output = map[string]string{}
		}
		for k, v := range m {
			output[k] = v
		}
	}
	return output
}
