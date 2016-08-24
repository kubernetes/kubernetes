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

package unversioned

import (
	rbacapi "k8s.io/kubernetes/pkg/apis/rbac"
)

type AddRoleRequestExpansion interface {
	Create(obj *rbacapi.AddRoleRequest) (result *rbacapi.AddRoleRequest, err error)
}

func (c *addRoleRequests) Create(obj *rbacapi.AddRoleRequest) (result *rbacapi.AddRoleRequest, err error) {
	result = &rbacapi.AddRoleRequest{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("addrolerequests").
		Body(obj).
		Do().
		Into(result)
	return
}
