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
type DaemonSetExpansion interface {
	Rollback(*extensions.DaemonSetRollback) error
}

// Rollback applied the provided DaemonSetRollback to the named daemon set in the current namespace.
func (c *daemonSets) Rollback(daemonSetRollback *extensions.DaemonSetRollback) error {
	return c.client.Post().Namespace(c.ns).Resource("daemonsets").Name(daemonSetRollback.Name).SubResource("rollback").Body(daemonSetRollback).Do().Error()
}
