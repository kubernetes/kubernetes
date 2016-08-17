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

import "k8s.io/kubernetes/pkg/apis/extensions"

// The DeploymentExpansion interface allows manually adding extra methods to the DeploymentInterface.
type DeploymentExpansion interface {
	Rollback(*extensions.DeploymentRollback) error
}

// Rollback applied the provided DeploymentRollback to the named deployment in the current namespace.
func (c *deployments) Rollback(deploymentRollback *extensions.DeploymentRollback) error {
	return c.client.Post().Namespace(c.ns).Resource("deployments").Name(deploymentRollback.Name).SubResource("rollback").Body(deploymentRollback).Do().Error()
}
