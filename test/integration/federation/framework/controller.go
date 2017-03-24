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
	"fmt"
	"testing"

	restclient "k8s.io/client-go/rest"
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/federation/pkg/typeadapters"
)

// ControllerFixture defines operations for managing a federation
// controller.  Tests written to this interface can then target any
// controller for which an implementation of this interface exists.
type ControllerFixture interface {
	TestFixture

	SetUp(t *testing.T, testClient federationclientset.Interface, config *restclient.Config)

	Kind() string

	Adapter() typeadapters.FederatedTypeAdapter
}

// SetUpControllerFixture configures the given resource fixture to target the provided api fixture
func SetUpControllerFixture(t *testing.T, apiFixture *FederationAPIFixture, controllerFixture ControllerFixture) {
	client := apiFixture.NewClient(fmt.Sprintf("test-%s", controllerFixture.Kind()))
	config := apiFixture.NewConfig()
	controllerFixture.SetUp(t, client, config)
}
