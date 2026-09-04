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
	"io"
	"net/http"
	"strconv"
	"strings"
	"testing"
	"time"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	kubernetes "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kube-openapi/pkg/validation/spec"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

// TestOpenAPIV2BytesCache exercises the /openapi/v2 endpoint of the full
// server chain (aggregator -> kube-apiserver -> apiextensions) with the
// OpenAPIV2BytesCache feature gate enabled: built-in definitions, ETag/304
// handling, the protobuf encoding, and CRD schema publication.
func TestOpenAPIV2BytesCache(t *testing.T) {
	server, err := apiservertesting.StartTestServer(t, apiservertesting.NewDefaultTestServerOptions(),
		[]string{"--feature-gates=OpenAPIV2BytesCache=true"}, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()
	config := server.ClientConfig

	client, err := kubernetes.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	// The aggregated spec must contain built-in definitions and paths.
	jsonData, err := client.RESTClient().Get().AbsPath("/openapi/v2").Do(context.TODO()).Raw()
	if err != nil {
		t.Fatal(err)
	}
	openAPISpec := spec.Swagger{}
	if err := json.Unmarshal(jsonData, &openAPISpec); err != nil {
		t.Fatal(err)
	}
	if _, ok := openAPISpec.Definitions["io.k8s.api.core.v1.Pod"]; !ok {
		t.Errorf("expected built-in definition io.k8s.api.core.v1.Pod in /openapi/v2")
	}
	if _, ok := openAPISpec.Paths.Paths["/apis/apps/v1/"]; !ok {
		t.Errorf("expected built-in path /apis/apps/v1/ in /openapi/v2")
	}

	rt, err := restclient.TransportFor(config)
	if err != nil {
		t.Fatal(err)
	}
	fetch := func(acceptHeader, ifNoneMatch string) *http.Response {
		t.Helper()
		req, err := http.NewRequest(http.MethodGet, config.Host+"/openapi/v2", nil)
		if err != nil {
			t.Fatal(err)
		}
		if acceptHeader != "" {
			req.Header.Set("Accept", acceptHeader)
		}
		if ifNoneMatch != "" {
			req.Header.Set("If-None-Match", ifNoneMatch)
		}
		resp, err := rt.RoundTrip(req)
		if err != nil {
			t.Fatal(err)
		}
		return resp
	}
	drain := func(resp *http.Response) {
		t.Helper()
		if _, err := io.Copy(io.Discard, resp.Body); err != nil {
			t.Fatal(err)
		}
		if err := resp.Body.Close(); err != nil {
			t.Fatal(err)
		}
	}

	// The served ETag must be usable for conditional requests.
	resp := fetch("application/json", "")
	drain(resp)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 from /openapi/v2, got %v", resp.StatusCode)
	}
	etag := resp.Header.Get("Etag")
	if etag == "" {
		t.Fatal("expected non-empty Etag header from /openapi/v2")
	}
	if _, err := strconv.Unquote(etag); err != nil {
		t.Errorf("expected quoted Etag, got %q: %v", etag, err)
	}
	resp = fetch("application/json", etag)
	drain(resp)
	if resp.StatusCode != http.StatusNotModified {
		t.Errorf("expected 304 for If-None-Match %q, got %v", etag, resp.StatusCode)
	}

	// The protobuf encoding must be served.
	const protobufContentType = "application/com.github.proto-openapi.spec.v2.v1.0+protobuf"
	resp = fetch(protobufContentType, "")
	protoData, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if err := resp.Body.Close(); err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 for protobuf /openapi/v2, got %v", resp.StatusCode)
	}
	if got := resp.Header.Get("Content-Type"); got != protobufContentType {
		t.Errorf("expected Content-Type %q, got %q", protobufContentType, got)
	}
	if len(protoData) == 0 {
		t.Error("expected non-empty protobuf /openapi/v2 response")
	}

	// A new CRD must be published into the aggregated spec.
	apiExtensionClient, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}
	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}
	crd := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: "bytescaches.cr.bar.com",
		},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: "cr.bar.com",
			Scope: apiextensionsv1.NamespaceScoped,
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Plural: "bytescaches",
				Kind:   "BytesCache",
			},
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
				{
					Name:    "v1",
					Served:  true,
					Storage: true,
					Schema: &apiextensionsv1.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensionsv1.JSONSchemaProps{
								"spec": {
									Type: "object",
									Properties: map[string]apiextensionsv1.JSONSchemaProps{
										"replicas": {
											Type: "integer",
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}
	if _, err := fixtures.CreateNewV1CustomResourceDefinition(crd, apiExtensionClient, dynamicClient); err != nil {
		t.Fatal(err)
	}

	if err := wait.PollUntilContextTimeout(context.Background(), time.Second, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
		jsonData, err := client.RESTClient().Get().AbsPath("/openapi/v2").Do(ctx).Raw()
		if err != nil {
			return false, err
		}
		crdSpec := spec.Swagger{}
		if err := json.Unmarshal(jsonData, &crdSpec); err != nil {
			return false, err
		}
		for schemaName := range crdSpec.Definitions {
			if strings.HasPrefix(schemaName, "com.bar.cr.v1.BytesCache") {
				return true, nil
			}
		}
		return false, nil
	}); err != nil {
		t.Fatalf("timed out waiting for CRD schema to be published to /openapi/v2: %v", err)
	}

	// Publishing the CRD must have changed the served ETag.
	resp = fetch("application/json", etag)
	drain(resp)
	if resp.StatusCode != http.StatusOK {
		t.Errorf("expected 200 with a fresh spec after CRD publication for stale If-None-Match %q, got %v", etag, resp.StatusCode)
	}
}
