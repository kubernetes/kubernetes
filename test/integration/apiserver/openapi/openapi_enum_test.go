/*
Copyright 2021 The Kubernetes Authors.

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

package openapi

import (
	"encoding/json"
	"net/http"
	"testing"

	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/apiserver/pkg/util/openapi"
	restclient "k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/validation/spec"
	"k8s.io/kubernetes/pkg/controlplane"
	generated "k8s.io/kubernetes/pkg/generated/openapi"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestEnablingOpenAPIEnumTypes(t *testing.T) {
	const typeToAddEnum = "k8s.io/api/core/v1.ContainerPort"
	const typeToCheckEnum = "io.k8s.api.core.v1.ContainerPort"

	for _, tc := range []struct {
		name            string
		featureEnabled  bool
		enumShouldExist bool
	}{
		{
			name:            "disabled",
			featureEnabled:  false,
			enumShouldExist: false,
		},
		{
			name:            "enabled",
			featureEnabled:  true,
			enumShouldExist: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.OpenAPIEnums, tc.featureEnabled)

			getDefinitionsFn := openapi.GetOpenAPIDefinitionsWithoutDisabledFeatures(func(ref common.ReferenceCallback) map[string]common.OpenAPIDefinition {
				defs := generated.GetOpenAPIDefinitions(ref)
				def := defs[typeToAddEnum]
				// replace protocol to add the would-be enum field.
				def.Schema.Properties["protocol"] = spec.Schema{
					SchemaProps: spec.SchemaProps{
						Description: "Protocol for port. Must be UDP, TCP, or SCTP. Defaults to \\\"TCP\\\".\\n\\nPossible enum values:\\n - `SCTP`: is the SCTP protocol.\\n - `TCP`: is the TCP protocol.\\n - `UDP`: is the UDP protocol.",
						Default:     "TCP",
						Type:        []string{"string"},
						Format:      "",
						Enum:        []interface{}{"SCTP", "TCP", "UDP"},
					},
				}
				defs[typeToAddEnum] = def
				return defs
			})

			_, kubeConfig, tearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{
				ModifyServerConfig: func(config *controlplane.Config) {
					config.ControlPlane.Generic.OpenAPIConfig = framework.DefaultOpenAPIConfig()
					config.ControlPlane.Generic.OpenAPIConfig.GetDefinitions = getDefinitionsFn
				},
			})
			defer tearDownFn()

			rt, err := restclient.TransportFor(kubeConfig)
			if err != nil {
				t.Fatal(err)
			}

			req, err := http.NewRequest("GET", kubeConfig.Host+"/openapi/v2", nil)
			if err != nil {
				t.Fatal(err)
			}
			resp, err := rt.RoundTrip(req)
			if err != nil {
				t.Fatal(err)
			}
			var body struct {
				Definitions map[string]struct {
					Properties map[string]struct {
						Description string   `json:"description"`
						Type        string   `json:"type"`
						Enum        []string `json:"enum"`
					} `json:"properties"`
				} `json:"definitions"`
			}
			err = json.NewDecoder(resp.Body).Decode(&body)
			if err != nil {
				t.Fatal(err)
			}
			protocol, ok := body.Definitions[typeToCheckEnum].Properties["protocol"]
			if !ok {
				t.Fatalf("protocol not found in properties in %v", body)
			}
			if enumExists := len(protocol.Enum) > 0; enumExists != tc.enumShouldExist {
				t.Errorf("expect enum exists: %v, but got %v", tc.enumShouldExist, enumExists)
			}
		})
	}
}
