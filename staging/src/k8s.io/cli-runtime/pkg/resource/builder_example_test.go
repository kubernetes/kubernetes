/*
Copyright 2020 The Kubernetes Authors.

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

package resource_test

import (
	"bytes"
	"fmt"

	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/kubernetes/scheme"
)

var exampleManifest = `
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingWebhookConfiguration
metadata:
  name: mutating1
---
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingWebhookConfigurationList
items:
- apiVersion: admissionregistration.k8s.io/v1
  kind: MutatingWebhookConfiguration
  metadata:
    name: mutating2
- apiVersion: admissionregistration.k8s.io/v1
  kind: MutatingWebhookConfiguration
  metadata:
    name: mutating3
---
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
metadata:
  name: validating1
---
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfigurationList
items:
- apiVersion: admissionregistration.k8s.io/v1
  kind: ValidatingWebhookConfiguration
  metadata:
    name: validating2
- apiVersion: admissionregistration.k8s.io/v1
  kind: ValidatingWebhookConfiguration
  metadata:
    name: validating3
---
apiVersion: v1
kind: List
items:
- apiVersion: admissionregistration.k8s.io/v1
  kind: MutatingWebhookConfiguration
  metadata:
    name: mutating4
- apiVersion: admissionregistration.k8s.io/v1
  kind: ValidatingWebhookConfiguration
  metadata:
    name: validating4
---
`

// ExampleNewLocalBuilderLoad demonstrates using a local resource builder to read typed resources from a manifest
func ExampleNewLocalBuilder() {
	// Create a local builder...
	builder := resource.NewLocalBuilder().
		// Configure with a scheme to get typed objects in the versions registered with the scheme.
		// As an alternative, could call Unstructured() to get unstructured objects.
		WithScheme(scheme.Scheme, scheme.Scheme.PrioritizedVersionsAllGroups()...).
		// Provide input via a Reader.
		// As an alternative, could call Path(false, "/path/to/file") to read from a file.
		Stream(bytes.NewBufferString(exampleManifest), "input").
		// Flatten items contained in List objects
		Flatten().
		// Accumulate as many items as possible
		ContinueOnError()

	// Run the builder
	result := builder.Do()

	if err := result.Err(); err != nil {
		fmt.Println("builder error:", err)
		return
	}

	items, err := result.Infos()
	if err != nil {
		fmt.Println("infos error:", err)
		return
	}

	for _, item := range items {
		fmt.Printf("%s (%T)\n", item.String(), item.Object)
	}

	// Output:
	// Name: "mutating1", Namespace: "" (*v1.MutatingWebhookConfiguration)
	// Name: "mutating2", Namespace: "" (*v1.MutatingWebhookConfiguration)
	// Name: "mutating3", Namespace: "" (*v1.MutatingWebhookConfiguration)
	// Name: "validating1", Namespace: "" (*v1.ValidatingWebhookConfiguration)
	// Name: "validating2", Namespace: "" (*v1.ValidatingWebhookConfiguration)
	// Name: "validating3", Namespace: "" (*v1.ValidatingWebhookConfiguration)
	// Name: "mutating4", Namespace: "" (*v1.MutatingWebhookConfiguration)
	// Name: "validating4", Namespace: "" (*v1.ValidatingWebhookConfiguration)
}
