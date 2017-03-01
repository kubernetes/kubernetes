/*
Copyright 2017 The Kubernetes Authors.

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

package framework

import (
	"testing"

	restclient "k8s.io/client-go/rest"
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	daemonsetcontroller "k8s.io/kubernetes/federation/pkg/federation-controller/daemonset"
	"k8s.io/kubernetes/federation/pkg/typeadapters"
)

type DaemonSetFixture struct {
	adapter  typeadapters.FederatedTypeAdapter
	stopChan chan struct{}
}

func (f *DaemonSetFixture) SetUp(t *testing.T, client federationclientset.Interface, config *restclient.Config) {
	f.adapter = typeadapters.NewDaemonSetAdapter(client)
	f.stopChan = make(chan struct{})
	daemonsetcontroller.StartDaemonSetController(config, f.stopChan, true)
}

func (f *DaemonSetFixture) TearDown(t *testing.T) {
	close(f.stopChan)
}

func (f *DaemonSetFixture) Kind() string {
	adapter := &typeadapters.DaemonSetAdapter{}
	return adapter.Kind()
}

func (f *DaemonSetFixture) Adapter() typeadapters.FederatedTypeAdapter {
	return f.adapter
}
