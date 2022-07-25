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

package rbac

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/rbac"
)

// EscalationAllowed checks if the user associated with the context is a superuser
func EscalationAllowed(ctx context.Context) bool {
	u, ok := genericapirequest.UserFrom(ctx)
	if !ok {
		return false
	}

	// system:masters is special because the API server uses it for privileged loopback connections
	// therefore we know that a member of system:masters can always do anything
	for _, group := range u.GetGroups() {
		if group == user.SystemPrivilegedGroup {
			return true
		}
	}

	return false
}

var roleResources = map[schema.GroupResource]bool{
	rbac.SchemeGroupVersion.WithResource("clusterroles").GroupResource(): true,
	rbac.SchemeGroupVersion.WithResource("roles").GroupResource():        true,
}

// RoleEscalationAuthorized checks if the user associated with the context is explicitly authorized to escalate the role resource associated with the context
func RoleEscalationAuthorized(ctx context.Context, a authorizer.Authorizer) bool {
	if a == nil {
		return false
	}

	user, ok := genericapirequest.UserFrom(ctx)
	if !ok {
		return false
	}

	requestInfo, ok := genericapirequest.RequestInfoFrom(ctx)
	if !ok {
		return false
	}

	if !requestInfo.IsResourceRequest {
		return false
	}

	requestResource := schema.GroupResource{Group: requestInfo.APIGroup, Resource: requestInfo.Resource}
	if !roleResources[requestResource] {
		return false
	}

	attrs := authorizer.AttributesRecord{
		User:            user,
		Verb:            "escalate",
		APIGroup:        requestInfo.APIGroup,
		APIVersion:      "*",
		Resource:        requestInfo.Resource,
		Name:            requestInfo.Name,
		Namespace:       requestInfo.Namespace,
		ResourceRequest: true,
	}

	decision, _, err := a.Authorize(ctx, attrs)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf(
			"error authorizing user %#v to escalate %#v named %q in namespace %q: %v",
			user, requestResource, requestInfo.Name, requestInfo.Namespace, err,
		))
	}
	return decision == authorizer.DecisionAllow
}

// BindingAuthorized returns true if the user associated with the context is explicitly authorized to bind the specified roleRef
func BindingAuthorized(ctx context.Context, roleRef rbac.RoleRef, bindingNamespace string, a authorizer.Authorizer) bool {
	if a == nil {
		return false
	}

	user, ok := genericapirequest.UserFrom(ctx)
	if !ok {
		return false
	}

	attrs := authorizer.AttributesRecord{
		User: user,
		Verb: "bind",
		// check against the namespace where the binding is being created (or the empty namespace for clusterrolebindings).
		// this allows delegation to bind particular clusterroles in rolebindings within particular namespaces,
		// and to authorize binding a clusterrole across all namespaces in a clusterrolebinding.
		Namespace:       bindingNamespace,
		ResourceRequest: true,
	}

	// This occurs after defaulting and conversion, so values pulled from the roleRef won't change
	// Invalid APIGroup or Name values will fail validation
	switch roleRef.Kind {
	case "ClusterRole":
		attrs.APIGroup = roleRef.APIGroup
		attrs.APIVersion = "*"
		attrs.Resource = "clusterroles"
		attrs.Name = roleRef.Name
	case "Role":
		attrs.APIGroup = roleRef.APIGroup
		attrs.APIVersion = "*"
		attrs.Resource = "roles"
		attrs.Name = roleRef.Name
	default:
		return false
	}

	decision, _, err := a.Authorize(ctx, attrs)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf(
			"error authorizing user %#v to bind %#v in namespace %s: %v",
			user, roleRef, bindingNamespace, err,
		))
	}
	return decision == authorizer.DecisionAllow
}
