/*
Copyright 2016 The Kubernetes Authors.

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

package addrolerequest

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	rbacapi "k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/apis/rbac/validation"
	"k8s.io/kubernetes/pkg/client/retry"
	"k8s.io/kubernetes/pkg/registry/rbac/rolebinding"
	"k8s.io/kubernetes/pkg/runtime"
)

type REST struct {
	// TODO this should be a client, but kube doesn't have a loopback concept yet
	roleBindingClient rolebinding.Registry
}

func NewREST(roleBindingClient rolebinding.Registry) *REST {
	return &REST{roleBindingClient: roleBindingClient}
}

func (r *REST) New() runtime.Object {
	return &rbacapi.AddRoleRequest{}
}

// Create registers a given new ResourceAccessReview instance to r.registry.
func (r *REST) Create(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
	addRole, ok := obj.(*rbacapi.AddRoleRequest)
	if !ok {
		return nil, apierrors.NewBadRequest(fmt.Sprintf("not an AddRoleRequest: %#v", obj))
	}
	namespace := api.NamespaceValue(ctx)
	if len(namespace) == 0 || namespace != addRole.Namespace {
		return nil, apierrors.NewBadRequest("the namespace of the provided object does not match the namespace sent on the request")
	}
	if errs := validation.ValidateAddRoleRequest(addRole); len(errs) > 0 {
		return nil, apierrors.NewInvalid(rbacapi.Kind("AddRoleRequest"), addRole.Name, errs)
	}

	addRole.Status = rbacapi.AddRoleRequestStatus{}

	err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		allRoleBindings, err := r.roleBindingClient.ListRoleBindings(ctx, &api.ListOptions{})
		if err != nil {
			return err
		}

		// find a rolebinding that has a roleref that matches our need
		var targetRoleBinding *rbacapi.RoleBinding
		for _, rolebinding := range allRoleBindings.Items {
			if rolebinding.RoleRef == addRole.Spec.RoleRef {
				targetRoleBinding = &rolebinding
				break
			}
		}

		// if we didn't find an existing rolebinding, create a new one
		if targetRoleBinding == nil {
			targetRoleBinding = &rbacapi.RoleBinding{
				ObjectMeta: api.ObjectMeta{GenerateName: addRole.Spec.RoleRef.Name},
				RoleRef:    addRole.Spec.RoleRef,
			}
		}

		// de-dup the subjects
		subjectsToAdd := []rbacapi.Subject{}
		for _, newSubject := range addRole.Spec.Subjects {
			found := false
			for _, existingSubject := range targetRoleBinding.Subjects {
				if existingSubject == newSubject {
					found = true
					break
				}
			}
			if !found {
				for _, existingSubject := range subjectsToAdd {
					if existingSubject == newSubject {
						found = true
						break
					}
				}
			}

			if !found {
				subjectsToAdd = append(subjectsToAdd, newSubject)
			}
		}
		targetRoleBinding.Subjects = subjectsToAdd

		// create instead of update
		if len(targetRoleBinding.ResourceVersion) == 0 {
			createdRoleBinding, err := r.roleBindingClient.CreateRoleBinding(ctx, targetRoleBinding)
			if err != nil {
				return err
			}
			addRole.Status.RoleBindingRef.APIGroup = rbacapi.GroupName
			addRole.Status.RoleBindingRef.Kind = "RoleBinding"
			addRole.Status.RoleBindingRef.Name = createdRoleBinding.Name
		} else {
			if err := r.roleBindingClient.UpdateRoleBinding(ctx, targetRoleBinding); err != nil {
				return err
			}
			addRole.Status.RoleBindingRef.APIGroup = rbacapi.GroupName
			addRole.Status.RoleBindingRef.Kind = "RoleBinding"
			addRole.Status.RoleBindingRef.Name = targetRoleBinding.Name

		}

		return nil
	})
	if err != nil {
		return nil, err
	}

	return addRole, nil
}
