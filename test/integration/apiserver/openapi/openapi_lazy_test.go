/*
Copyright The Kubernetes Authors.

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
	"sync/atomic"
	"testing"

	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/validation/spec"
	"k8s.io/kubernetes/pkg/controlplane"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestOpenAPIV2LazyBuild(t *testing.T) {
	for _, tc := range []struct {
		name string
		lazy bool
	}{
		{name: "gate enabled", lazy: true},
		{name: "gate disabled", lazy: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.OpenAPIV2LazyBuild, tc.lazy)

			var v2Builds atomic.Int32
			client, _, tearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{
				ModifyServerConfig: func(config *controlplane.Config) {
					getDefs := config.ControlPlane.Generic.OpenAPIConfig.GetDefinitions
					config.ControlPlane.Generic.OpenAPIConfig.GetDefinitions = func(ref common.ReferenceCallback) map[string]common.OpenAPIDefinition {
						v2Builds.Add(1)
						return getDefs(ref)
					}
				},
			})
			defer tearDownFn()

			if tc.lazy && v2Builds.Load() != 0 {
				t.Fatalf("expected 0 OpenAPI v2 builds at startup with gate enabled, got %d", v2Builds.Load())
			}
			if !tc.lazy && v2Builds.Load() != 1 {
				t.Fatalf("expected 1 OpenAPI v2 build at startup with gate disabled, got %d", v2Builds.Load())
			}

			data, err := client.Discovery().RESTClient().Get().AbsPath("/openapi/v2").Do(tCtx).Raw()
			if err != nil {
				t.Fatal(err)
			}
			var swagger spec.Swagger
			if err := json.Unmarshal(data, &swagger); err != nil {
				t.Fatal(err)
			}
			if _, ok := swagger.Definitions["io.k8s.api.core.v1.Pod"]; !ok {
				t.Error("expected built-in definition io.k8s.api.core.v1.Pod in /openapi/v2")
			}
			if v2Builds.Load() != 1 {
				t.Fatalf("expected 1 OpenAPI v2 build after first request, got %d", v2Builds.Load())
			}
		})
	}
}
