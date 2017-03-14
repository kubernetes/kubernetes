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
	secretcontroller "k8s.io/kubernetes/federation/pkg/federation-controller/secret"
	"k8s.io/kubernetes/federation/pkg/typeadapters"
)

type SecretFixture struct {
	adapter  *typeadapters.SecretAdapter
	stopChan chan struct{}
}

func (f *SecretFixture) SetUp(t *testing.T, client federationclientset.Interface, config *restclient.Config) {
	f.adapter = typeadapters.NewSecretAdapter(client)
	f.stopChan = make(chan struct{})
	secretcontroller.StartSecretController(config, f.stopChan, true)
}

func (f *SecretFixture) TearDown(t *testing.T) {
	close(f.stopChan)
}
func (f *SecretFixture) Kind() string {
	adapter := &typeadapters.SecretAdapter{}
	return adapter.Kind()
}

func (f *SecretFixture) Adapter() typeadapters.FederatedTypeAdapter {
	return f.adapter
}
