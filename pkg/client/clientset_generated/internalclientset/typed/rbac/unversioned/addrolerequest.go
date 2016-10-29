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
	restclient "k8s.io/kubernetes/pkg/client/restclient"
)

// AddRoleRequestsGetter has a method to return a AddRoleRequestInterface.
// A group's client should implement this interface.
type AddRoleRequestsGetter interface {
	AddRoleRequests(namespace string) AddRoleRequestInterface
}

// AddRoleRequestInterface has methods to work with AddRoleRequest resources.
type AddRoleRequestInterface interface {
	AddRoleRequestExpansion
}

// addRoleRequests implements AddRoleRequestInterface
type addRoleRequests struct {
	client restclient.Interface
	ns     string
}

// newAddRoleRequests returns a AddRoleRequests
func newAddRoleRequests(c *RbacClient, namespace string) *addRoleRequests {
	return &addRoleRequests{
		client: c.RESTClient(),
		ns:     namespace,
	}
}
