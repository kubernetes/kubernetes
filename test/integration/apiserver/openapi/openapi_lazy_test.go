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
	"context"
	"encoding/json"
	"fmt"
	"sync/atomic"
	"testing"

	kubernetes "k8s.io/client-go/kubernetes"
	openapicommon "k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/validation/spec"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

// TestOpenAPIV2LazyBuild verifies when the three servers in the delegation
// chain (apiextensions-apiserver, kube-apiserver, and kube-aggregator) build
// their /openapi/v2 specs: with the OpenAPIV2LazyBuild gate enabled, none of
// the three build at startup and all three build only once /openapi/v2 is first
// requested; with it disabled, all three build at startup.
func TestOpenAPIV2LazyBuild(t *testing.T) {
	for _, tc := range []struct {
		name              string
		lazy              bool
		wantStartupBuilds int32
	}{
		{name: "gate enabled", lazy: true, wantStartupBuilds: 0},
		{name: "gate disabled", lazy: false, wantStartupBuilds: 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var v2Builds atomic.Int32
			opts := apiservertesting.NewDefaultTestServerOptions()
			opts.OpenAPIGetDefinitionsWrapFunc = func(delegate openapicommon.GetOpenAPIDefinitions) openapicommon.GetOpenAPIDefinitions {
				return func(ref openapicommon.ReferenceCallback) map[string]openapicommon.OpenAPIDefinition {
					v2Builds.Add(1)
					return delegate(ref)
				}
			}

			server, err := apiservertesting.StartTestServer(t, opts,
				[]string{fmt.Sprintf("--feature-gates=OpenAPIV2LazyBuild=%v", tc.lazy)}, framework.SharedEtcd())
			if err != nil {
				t.Fatal(err)
			}
			defer server.TearDownFn()

			if got := v2Builds.Load(); got != tc.wantStartupBuilds {
				t.Fatalf("OpenAPI v2 GetDefinitions calls at startup = %d, want %d", got, tc.wantStartupBuilds)
			}

			client, err := kubernetes.NewForConfig(server.ClientConfig)
			if err != nil {
				t.Fatal(err)
			}
			data, err := client.RESTClient().Get().AbsPath("/openapi/v2").Do(context.TODO()).Raw()
			if err != nil {
				t.Fatal(err)
			}
			var swagger spec.Swagger
			if err := json.Unmarshal(data, &swagger); err != nil {
				t.Fatal(err)
			}
			for _, def := range []string{
				"io.k8s.api.core.v1.Pod",
				"io.k8s.apiextensions-apiserver.pkg.apis.apiextensions.v1.CustomResourceDefinition",
				"io.k8s.kube-aggregator.pkg.apis.apiregistration.v1.APIService",
			} {
				if _, ok := swagger.Definitions[def]; !ok {
					t.Errorf("expected definition %q in /openapi/v2", def)
				}
			}

			if got := v2Builds.Load(); got != 3 {
				t.Fatalf("OpenAPI v2 GetDefinitions calls after first /openapi/v2 request = %d, want 3 (apiextensions-apiserver, kube-apiserver, kube-aggregator)", got)
			}
		})
	}
}
