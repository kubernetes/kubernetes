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
	secretcontroller "k8s.io/kubernetes/federation/pkg/federation-controller/secret"
	"k8s.io/kubernetes/federation/pkg/typeadapters"
)

// ControllerFixture manages a federation controller for testing.
type ControllerFixture struct {
	stopChan chan struct{}
}

// NewControllerFixture initializes a new controller fixture
func NewControllerFixture(t *testing.T, kind string, adapterFactory typeadapters.AdapterFactory, config *restclient.Config) *ControllerFixture {
	f := &ControllerFixture{
		stopChan: make(chan struct{}),
	}
	// TODO the generic controller doesn't belong in the secretcontroller package
	secretcontroller.StartFederationSyncController(kind, adapterFactory, config, f.stopChan, true)
	return f
}

func (f *ControllerFixture) TearDown(t *testing.T) {
	close(f.stopChan)
}
