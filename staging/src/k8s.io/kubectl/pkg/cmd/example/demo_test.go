/*
Copyright 2025 The Kubernetes Authors.

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

package example

import (
	"fmt"
	"testing"

	yaml "sigs.k8s.io/yaml"
)

// TestDemoDumpAllBuilders is a demo harness that calls every builder function
// and prints the rendered YAML to stdout for manual inspection and
// cross-version Kubernetes compatibility auditing.
//
// Typed-struct builders use marshalExample (JSON round-trip + strip zero fields)
// to match the real output users see.  Unstructured map builders (CRD, Gateway,
// HTTPRoute) use yaml.Marshal directly since they have no zero-value noise.
//
// Run with:  go test -v -run TestDemoDumpAllBuilders ./staging/src/k8s.io/kubectl/pkg/cmd/example/
func TestDemoDumpAllBuilders(t *testing.T) {
	type builderEntry struct {
		kind    string
		builder func() ([]byte, error)
	}

	entries := []builderEntry{
		// Typed-struct builders — use marshalExample to strip zero-value fields
		{"Pod", func() ([]byte, error) { return marshalExample(buildPod("example-pod", "")) }},
		{"Deployment", func() ([]byte, error) { return marshalExample(buildDeployment("example-deployment", "", 3)) }},
		{"Service", func() ([]byte, error) { return marshalExample(buildService("example-service")) }},
		{"PersistentVolumeClaim", func() ([]byte, error) { return marshalExample(buildPVC("example-pvc")) }},
		{"Secret", func() ([]byte, error) { return marshalExample(buildSecret("example-secret")) }},
		{"ConfigMap", func() ([]byte, error) { return marshalExample(buildConfigMap("example-configmap")) }},
		{"Job", func() ([]byte, error) { return marshalExample(buildJob("example-job", "")) }},
		{"CronJob", func() ([]byte, error) { return marshalExample(buildCronJob("example-cronjob", "")) }},
		{"Ingress", func() ([]byte, error) { return marshalExample(buildIngress("example-ingress")) }},
		{"NetworkPolicy", func() ([]byte, error) { return marshalExample(buildNetworkPolicy("example-netpol")) }},
		// Unstructured map builders — yaml.Marshal directly (no zero-value noise)
		{"CustomResourceDefinition", func() ([]byte, error) { return yaml.Marshal(buildCRD("example-crd")) }},
		{"Gateway", func() ([]byte, error) { return yaml.Marshal(buildGateway("example-gateway")) }},
		{"HTTPRoute", func() ([]byte, error) { return yaml.Marshal(buildHTTPRoute("example-httproute")) }},
	}

	for _, e := range entries {
		t.Run(e.kind, func(t *testing.T) {
			data, err := e.builder()
			if err != nil {
				t.Fatalf("failed to marshal %s: %v", e.kind, err)
			}
			fmt.Printf("\n--- %s ---\n%s\n", e.kind, string(data))
		})
	}
}
