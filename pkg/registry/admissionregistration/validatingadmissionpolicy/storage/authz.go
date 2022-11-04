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

package storage

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
	rbacregistry "k8s.io/kubernetes/pkg/registry/rbac"
)

func (r *REST) beginCreate(ctx context.Context, obj runtime.Object, options *metav1.CreateOptions) (genericregistry.FinishFunc, error) {
	// for superuser, skip all checks
	if rbacregistry.EscalationAllowed(ctx) {
		return noop, nil
	}

	policy := obj.(*admissionregistration.ValidatingAdmissionPolicy)
	if err := r.authorize(ctx, policy); err != nil {
		return nil, errors.NewForbidden(groupResource, policy.Name, err)
	}
	return noop, nil
}

func (r *REST) beginUpdate(ctx context.Context, obj, old runtime.Object, options *metav1.UpdateOptions) (genericregistry.FinishFunc, error) {
	// for superuser, skip all checks
	if rbacregistry.EscalationAllowed(ctx) {
		return noop, nil
	}

	policy := obj.(*admissionregistration.ValidatingAdmissionPolicy)
	oldPolicy := old.(*admissionregistration.ValidatingAdmissionPolicy)

	// both nil, no change
	if policy.Spec.ParamKind == nil && oldPolicy.Spec.ParamKind == nil {
		return noop, nil
	}

	// both non-nil but equivalent
	if policy.Spec.ParamKind != nil && oldPolicy.Spec.ParamKind != nil &&
		policy.Spec.ParamKind.Kind == oldPolicy.Spec.ParamKind.Kind &&
		policy.Spec.ParamKind.APIVersion == oldPolicy.Spec.ParamKind.APIVersion {
		return noop, nil
	}

	// if the policy has no paramKind, no EXTRA permissions are checked
	if policy.Spec.ParamKind == nil {
		return noop, nil
	}

	// changed, authorize
	if err := r.authorize(ctx, policy); err != nil {
		return nil, errors.NewForbidden(groupResource, policy.Name, err)
	}
	return noop, nil
}

func (r *REST) authorize(ctx context.Context, policy *admissionregistration.ValidatingAdmissionPolicy) error {
	if r.authorizer == nil {
		return nil
	}

	user, ok := genericapirequest.UserFrom(ctx)
	if !ok {
		return fmt.Errorf("cannot identify user")
	}

	var resource string
	var apiGroup string
	var apiVersion string

	paramKind := policy.Spec.ParamKind
	gv, gvr, err := func() (schema.GroupVersion, schema.GroupVersionResource, error) {
		gv, err := schema.ParseGroupVersion(paramKind.APIVersion)
		if err != nil {
			return schema.GroupVersion{}, schema.GroupVersionResource{}, err
		}
		gvk := gv.WithKind(paramKind.Kind)
		gvr, err := r.resourceResolver.Resolve(gvk)
		return gv, gvr, err
	}()

	if err != nil {
		// fail back if resolution fails
		utilruntime.HandleError(fmt.Errorf(
			"error resolving paramKind of %s: %v", policy.Name, paramKind,
		))
		// defaults if resolution fails
		resource = "*"
		apiGroup = gv.Group
		apiVersion = gv.Version
	} else {
		resource = gvr.Resource
		apiGroup = gvr.Group
		apiVersion = gvr.Version
	}

	// require that the user can read (verb "get") the referred kind.
	attrs := authorizer.AttributesRecord{
		User:            user,
		Verb:            "get",
		ResourceRequest: true,
		Name:            "*",
		Namespace:       "*",
		APIGroup:        apiGroup,
		APIVersion:      apiVersion,
		Resource:        resource,
	}

	d, _, err := r.authorizer.Authorize(ctx, attrs)
	if err != nil {
		return err
	}
	if d != authorizer.DecisionAllow {
		return fmt.Errorf("user %v cannot read paramKind of ValidatingAdmissionPolicy %s: %v", user, policy.Name, paramKind)
	}
	return nil
}

func noop(context.Context, bool) {}

var _ genericregistry.FinishFunc = noop
