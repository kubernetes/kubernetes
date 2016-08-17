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

package kuberuntime

import (
	"testing"

	"github.com/stretchr/testify/assert"
	apitest "k8s.io/kubernetes/pkg/kubelet/api/testing"
)

func createTestFakeRuntimeManager() (*apitest.FakeRuntimeService, *apitest.FakeImageService, *kubeGenericRuntimeManager, error) {
	fakeRuntimeService := apitest.NewFakeRuntimeService()
	fakeImageService := apitest.NewFakeImageService()
	manager, err := NewFakeKubeRuntimeManager(fakeRuntimeService, fakeImageService)
	return fakeRuntimeService, fakeImageService, manager, err
}

func TestNewKubeRuntimeManager(t *testing.T) {
	_, _, _, err := createTestFakeRuntimeManager()
	assert.NoError(t, err)
}

func TestVersion(t *testing.T) {
	_, _, m, err := createTestFakeRuntimeManager()
	assert.NoError(t, err)

	version, err := m.Version()
	assert.NoError(t, err)
	assert.Equal(t, kubeRuntimeAPIVersion, version.String())
}

func TestContainerRuntimeType(t *testing.T) {
	_, _, m, err := createTestFakeRuntimeManager()
	assert.NoError(t, err)

	runtimeType := m.Type()
	assert.Equal(t, "fakeRuntime", runtimeType)
}
