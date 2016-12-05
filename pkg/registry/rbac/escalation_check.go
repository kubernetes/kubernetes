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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/auth/user"
)

func EscalationAllowed(ctx api.Context) bool {
	u, ok := api.UserFrom(ctx)
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
