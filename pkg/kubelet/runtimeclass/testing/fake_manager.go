/*
Copyright 2018 The Kubernetes Authors.

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

package testing

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	fakedynamic "k8s.io/client-go/dynamic/fake"
	"k8s.io/kubernetes/pkg/kubelet/runtimeclass"
	"k8s.io/utils/pointer"
)

const (
	// SandboxRuntimeClass is a valid RuntimeClass pre-populated in the populated dynamic client.
	SandboxRuntimeClass = "sandbox"
	// SandboxRuntimeHandler is the handler associated with the SandboxRuntimeClass.
	SandboxRuntimeHandler = "kata-containers"

	// EmptyRuntimeClass is a valid RuntimeClass without a handler pre-populated in the populated dynamic client.
	EmptyRuntimeClass = "native"
	// InvalidRuntimeClass is an invalid RuntimeClass pre-populated in the populated dynamic client.
	InvalidRuntimeClass = "foo"
)

// NewPopulatedDynamicClient creates a dynamic client for use with the runtimeclass.Manager,
// and populates it with a few test RuntimeClass objects.
func NewPopulatedDynamicClient() dynamic.Interface {
	invalidRC := NewUnstructuredRuntimeClass(InvalidRuntimeClass, "")
	invalidRC.Object["spec"].(map[string]interface{})["runtimeHandler"] = true

	client := fakedynamic.NewSimpleDynamicClient(runtime.NewScheme(),
		NewUnstructuredRuntimeClass(EmptyRuntimeClass, ""),
		NewUnstructuredRuntimeClass(SandboxRuntimeClass, SandboxRuntimeHandler),
		invalidRC,
	)
	return client
}

// StartManagerSync runs the manager, and waits for startup by polling for the expected "native"
// RuntimeClass to be populated. Returns a function to stop the manager, which should be called with
// a defer:
//     defer StartManagerSync(t, m)()
// Any errors are considered fatal to the test.
func StartManagerSync(t *testing.T, m *runtimeclass.Manager) func() {
	stopCh := make(chan struct{})
	go m.Run(stopCh)

	// Wait for informer to populate.
	err := wait.PollImmediate(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		_, err := m.LookupRuntimeHandler(pointer.StringPtr(EmptyRuntimeClass))
		if err != nil {
			if errors.IsNotFound(err) {
				return false, nil
			}
			return false, err
		}
		return true, nil
	})
	require.NoError(t, err, "Failed to start manager")

	return func() {
		close(stopCh)
	}
}

// NewUnstructuredRuntimeClass is a helper to generate an unstructured RuntimeClass resource with
// the given name & handler.
func NewUnstructuredRuntimeClass(name, handler string) *unstructured.Unstructured {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "node.k8s.io/v1alpha1",
			"kind":       "RuntimeClass",
			"metadata": map[string]interface{}{
				"name": name,
			},
			"spec": map[string]interface{}{
				"runtimeHandler": handler,
			},
		},
	}
}
