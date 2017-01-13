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
	"fmt"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/request"
	"k8s.io/kubernetes/pkg/apis/rbac"
)

func EscalationAllowed(ctx genericapirequest.Context) bool {
	u, ok := genericapirequest.UserFrom(ctx)
	if !ok {
		// the only way to be without a user is to either have no authenticators by explicitly saying that's your preference
		// or to be connecting via the insecure port, in which case this logically doesn't apply
		return true
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

// BindingAuthorized returns true if the user associated with the context is explicitly authorized to bind the specified roleRef
func BindingAuthorized(ctx genericapirequest.Context, roleRef rbac.RoleRef, bindingNamespace string, a authorizer.Authorizer) bool {
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
		attrs.Resource = "clusterroles"
		attrs.Name = roleRef.Name
	case "Role":
		attrs.APIGroup = roleRef.APIGroup
		attrs.Resource = "roles"
		attrs.Name = roleRef.Name
	default:
		return false
	}

	ok, _, err := a.Authorize(attrs)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf(
			"error authorizing user %#v to bind %#v in namespace %s: %v",
			roleRef, bindingNamespace, user, err,
		))
	}
	return ok
}
