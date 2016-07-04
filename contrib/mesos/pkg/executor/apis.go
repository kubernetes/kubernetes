/*
Copyright 2015 The Kubernetes Authors.

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

package executor

import (
	"k8s.io/kubernetes/contrib/mesos/pkg/node"
	"k8s.io/kubernetes/pkg/api"
	unversionedcore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/unversioned"
)

type kubeAPI interface {
	killPod(ns, name string) error
}

type nodeAPI interface {
	createOrUpdate(hostname string, slaveAttrLabels, annotations map[string]string) (*api.Node, error)
}

// clientAPIWrapper implements kubeAPI and node API, which serve to isolate external dependencies
// such that they're easier to mock in unit test.
type clientAPIWrapper struct {
	client unversionedcore.CoreInterface
}

func (cw *clientAPIWrapper) killPod(ns, name string) error {
	return cw.client.Pods(ns).Delete(name, api.NewDeleteOptions(0))
}

func (cw *clientAPIWrapper) createOrUpdate(hostname string, slaveAttrLabels, annotations map[string]string) (*api.Node, error) {
	return node.CreateOrUpdate(cw.client, hostname, slaveAttrLabels, annotations)
}
