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
	"runtime"
	"strings"
	"testing"

	kubernetes "k8s.io/client-go/kubernetes"
	"k8s.io/kube-openapi/pkg/validation/spec"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

// openAPIV2BuildFrame is the function that builds a generic apiserver's
// OpenAPI v2 spec. Heap memory allocated under it and still in use is the
// retained spec, so its presence tells whether the spec has been built.
const openAPIV2BuildFrame = "k8s.io/apiserver/pkg/server/routes.OpenAPI.InstallV2"

// TestOpenAPIV2LazyBuild verifies when the servers in the delegation chain
// build their /openapi/v2 specs: with the OpenAPIV2LazyBuild gate enabled, not
// at startup and only once the aggregated endpoint is first requested; with it
// disabled, at startup as before. The test server runs in-process, so the
// heap profile of this process shows the builds.
func TestOpenAPIV2LazyBuild(t *testing.T) {
	for _, tc := range []struct {
		name string
		lazy bool
	}{
		{name: "gate enabled", lazy: true},
		{name: "gate disabled", lazy: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			baseline := openAPIV2SpecBytesInUse()

			server, err := apiservertesting.StartTestServer(t, apiservertesting.NewDefaultTestServerOptions(),
				[]string{fmt.Sprintf("--feature-gates=OpenAPIV2LazyBuild=%v", tc.lazy)}, framework.SharedEtcd())
			if err != nil {
				t.Fatal(err)
			}
			defer server.TearDownFn()

			builtAtStartup := openAPIV2SpecBytesInUse() - baseline
			if tc.lazy && builtAtStartup != 0 {
				t.Fatalf("expected no OpenAPI v2 spec to be built at startup with the gate enabled, found %d bytes of built spec in use", builtAtStartup)
			}
			if !tc.lazy && builtAtStartup == 0 {
				t.Fatal("expected the OpenAPI v2 spec to be built at startup with the gate disabled, found none")
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
			if _, ok := swagger.Definitions["io.k8s.api.core.v1.Pod"]; !ok {
				t.Error("expected built-in definition io.k8s.api.core.v1.Pod in /openapi/v2")
			}
			if _, ok := swagger.Definitions["io.k8s.apiextensions-apiserver.pkg.apis.apiextensions.v1.CustomResourceDefinition"]; !ok {
				t.Error("expected apiextensions definition CustomResourceDefinition in /openapi/v2")
			}

			builtAfterRequest := openAPIV2SpecBytesInUse() - baseline
			if builtAfterRequest == 0 {
				t.Fatal("expected the OpenAPI v2 spec to be built after the first request, found none")
			}
			t.Logf("spec bytes in use: %d at startup, %d after the first request", builtAtStartup, builtAfterRequest)
		})
	}
}

// openAPIV2SpecBytesInUse returns the heap bytes in use that were allocated
// under openAPIV2BuildFrame, according to the runtime's memory profile. The
// profile samples roughly one allocation per runtime.MemProfileRate bytes, so
// the value is a sampled count, not the spec's size: zero means no build
// happened, and a build of several megabytes yields many samples.
func openAPIV2SpecBytesInUse() int64 {
	// The memory profile is updated at GC; the second GC makes the first
	// cycle's results visible.
	runtime.GC()
	runtime.GC()
	var records []runtime.MemProfileRecord
	n, ok := runtime.MemProfile(nil, true)
	for !ok {
		records = make([]runtime.MemProfileRecord, n+64)
		n, ok = runtime.MemProfile(records, true)
	}
	records = records[:n]

	var total int64
	for i := range records {
		inUse := records[i].InUseBytes()
		if inUse <= 0 {
			continue
		}
		frames := runtime.CallersFrames(records[i].Stack())
		for {
			frame, more := frames.Next()
			if strings.HasPrefix(frame.Function, openAPIV2BuildFrame) {
				total += inUse
				break
			}
			if !more {
				break
			}
		}
	}
	return total
}
