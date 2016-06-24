/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

import "k8s.io/kubernetes/pkg/api"

// The SecurityContextConstraintsExpansion interface allows manually adding extra methods to the
// SecurityContextConstraintsInterface.
type SecurityContextConstraintsExpansion interface {
	// PatchStatus modifies the status of an existing security context constraint. It returns the copy
	// of the node that the server returns, or an error.
	PatchStatus(name string, data []byte) (*api.SecurityContextConstraints, error)
}

// PatchStatus modifies the status of an existing security context constraint. It returns the copy of
// the security context constraint that the server returns, or an error.
func (c *securityContextConstraints) PatchStatus(nodeName string, data []byte) (*api.SecurityContextConstraints, error) {
	result := &api.SecurityContextConstraints{}
	err := c.client.Patch(api.StrategicMergePatchType).
		Resource("securitycontextconstraints").
		Name(nodeName).
		SubResource("status").
		Body(data).
		Do().
		Into(result)
	return result, err
}
