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

package manifest

import (
	"testing"

	extensions "k8s.io/api/extensions/v1beta1"
)

func TestIngressToManifest(t *testing.T) {
	ing := &extensions.Ingress{}
	// Write the ingress to a file and ensure that there is no error.
	if err := IngressToManifest(ing, "/tmp/ing.yaml"); err != nil {
		t.Fatalf("Error in creating file: %s", err)
	}
	// Writing it again should not return an error.
	if err := IngressToManifest(ing, "/tmp/ing.yaml"); err != nil {
		t.Fatalf("Error in creating file: %s", err)
	}
}
