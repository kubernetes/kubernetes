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

// TODO: This file can potentially be moved to a common place used by both e2e and integration tests.

package framework

import (
	"net/http/httptest"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	// When these values are updated, also update cmd/kubelet/app/options/container_runtime.go
	// A copy of these values exist in test/utils/image/manifest.go
	currentPodInfraContainerImageName    = "k8s.gcr.io/pause"
	currentPodInfraContainerImageVersion = "3.1"
)

// CreateTestingNamespace creates a namespace for testing.
func CreateTestingNamespace(baseName string, apiserver *httptest.Server, t *testing.T) *v1.Namespace {
	// TODO: Create a namespace with a given basename.
	// Currently we neither create the namespace nor delete all of its contents at the end.
	// But as long as tests are not using the same namespaces, this should work fine.
	return &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			// TODO: Once we start creating namespaces, switch to GenerateName.
			Name: baseName,
		},
	}
}

// DeleteTestingNamespace is currently a no-op function.
func DeleteTestingNamespace(ns *v1.Namespace, apiserver *httptest.Server, t *testing.T) {
	// TODO: Remove all resources from a given namespace once we implement CreateTestingNamespace.
}
