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
	"net/url"
	"strconv"
	"strings"
	"testing"
	"time"

	openapi_v3 "github.com/google/gnostic-models/openapiv3"
	"google.golang.org/protobuf/proto"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kube-openapi/pkg/handler3"
	"k8s.io/kube-openapi/pkg/spec3"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

type v3Response struct {
	status int
	header http.Header
	body   []byte
}

type v3Client struct {
	t    *testing.T
	host string
	rt   http.RoundTripper
}

func newV3Client(t *testing.T, config *restclient.Config) *v3Client {
	rt, err := restclient.TransportFor(config)
	if err != nil {
		t.Fatal(err)
	}
	return &v3Client{t: t, host: config.Host, rt: rt}
}

// get performs one request without following redirects.
func (c *v3Client) get(path, accept string, headers ...string) v3Response {
	c.t.Helper()
	req, err := http.NewRequest(http.MethodGet, c.host+path, nil)
	if err != nil {
		c.t.Fatal(err)
	}
	if accept != "" {
		req.Header.Set("Accept", accept)
	}
	for i := 0; i+1 < len(headers); i += 2 {
		req.Header.Set(headers[i], headers[i+1])
	}
	resp, err := c.rt.RoundTrip(req)
	if err != nil {
		c.t.Fatal(err)
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			c.t.Fatal(err)
		}
	}()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		c.t.Fatal(err)
	}
	return v3Response{status: resp.StatusCode, header: resp.Header, body: body}
}

func (c *v3Client) discovery() (handler3.OpenAPIV3Discovery, string) {
	c.t.Helper()
	resp := c.get("/openapi/v3", "application/json")
	if resp.status != http.StatusOK {
		c.t.Fatalf("expected 200 from /openapi/v3, got %d: %s", resp.status, resp.body)
	}
	var d handler3.OpenAPIV3Discovery
	if err := json.Unmarshal(resp.body, &d); err != nil {
		c.t.Fatal(err)
	}
	return d, resp.header.Get("Etag")
}

func (c *v3Client) groupVersionJSON(path string) []byte {
	c.t.Helper()
	resp := c.get(path, "application/json")
	if resp.status != http.StatusOK {
		c.t.Fatalf("expected 200 from %s, got %d: %s", path, resp.status, resp.body)
	}
	return resp.body
}

// TestOpenAPIV3LazyBuild exercises /openapi/v3 of the full server chain
// (aggregator -> kube-apiserver -> apiextensions) with the OpenAPIV3LazyBuild
// feature gate enabled and checks that the served documents are identical to
// the ones served with the gate disabled.
func TestOpenAPIV3LazyBuild(t *testing.T) {
	lazyServer, err := apiservertesting.StartTestServer(t, apiservertesting.NewDefaultTestServerOptions(),
		[]string{"--feature-gates=OpenAPIV3LazyBuild=true"}, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer lazyServer.TearDownFn()
	lazy := newV3Client(t, lazyServer.ClientConfig)

	// Discovery lists the built-in group-versions with hash URLs. (In the
	// full chain the discovery document is assembled by kube-aggregator,
	// which sets no Etag; the per-group-version documents are proxied to the
	// lazy service.)
	discovery, _ := lazy.discovery()
	for _, gv := range []string{"api/v1", "apis/apps/v1", "apis/batch/v1", "apis/apiextensions.k8s.io/v1"} {
		if _, ok := discovery.Paths[gv]; !ok {
			t.Errorf("expected %s in /openapi/v3 discovery, got %v", gv, discovery.Paths)
		}
	}

	// Following a discovery URL serves the document with immutable caching
	// and an ETag equal to the hash, without any redirect.
	appsURL := discovery.Paths["apis/apps/v1"].ServerRelativeURL
	u, err := url.Parse(appsURL)
	if err != nil {
		t.Fatal(err)
	}
	hash := u.Query().Get("hash")
	if hash == "" {
		t.Fatalf("expected hash query parameter in %q", appsURL)
	}
	resp := lazy.get(appsURL, "application/json")
	if resp.status != http.StatusOK {
		t.Fatalf("expected 200 from %s, got %d: %s", appsURL, resp.status, resp.body)
	}
	if got := resp.header.Get("Cache-Control"); got != "public, immutable" {
		t.Errorf("expected immutable Cache-Control, got %q", got)
	}
	if etag, err := strconv.Unquote(resp.header.Get("Etag")); err != nil || etag != hash {
		t.Errorf("expected Etag %q to equal the discovery hash %q (err=%v)", resp.header.Get("Etag"), hash, err)
	}
	var appsSpec spec3.OpenAPI
	if err := json.Unmarshal(resp.body, &appsSpec); err != nil {
		t.Fatal(err)
	}
	if _, ok := appsSpec.Components.Schemas["io.k8s.api.apps.v1.Deployment"]; !ok {
		t.Errorf("expected io.k8s.api.apps.v1.Deployment in apps/v1 schemas")
	}

	// Conditional request, stale hash redirect, protobuf.
	if resp := lazy.get("/openapi/v3/apis/apps/v1", "application/json", "If-None-Match", strconv.Quote(hash)); resp.status != http.StatusNotModified {
		t.Errorf("expected 304 for If-None-Match %q, got %d", hash, resp.status)
	}
	resp = lazy.get("/openapi/v3/apis/apps/v1?hash=stale", "application/json")
	if resp.status != http.StatusMovedPermanently || resp.header.Get("Location") != appsURL {
		t.Errorf("expected 301 to %q for a stale hash, got %d %q", appsURL, resp.status, resp.header.Get("Location"))
	}
	resp = lazy.get("/openapi/v3/apis/apps/v1", "application/com.github.proto-openapi.spec.v3.v1.0+protobuf")
	if resp.status != http.StatusOK {
		t.Fatalf("expected 200 for protobuf, got %d", resp.status)
	}
	var doc openapi_v3.Document
	if err := proto.Unmarshal(resp.body, &doc); err != nil {
		t.Fatalf("failed to unmarshal protobuf document: %v", err)
	}
	if doc.Openapi != "3.0.0" {
		t.Errorf("expected openapi 3.0.0 in protobuf document, got %q", doc.Openapi)
	}

	// CRDs are published and withdrawn through the lazy service.
	apiExtensionClient, err := clientset.NewForConfig(lazyServer.ClientConfig)
	if err != nil {
		t.Fatal(err)
	}
	dynamicClient, err := dynamic.NewForConfig(lazyServer.ClientConfig)
	if err != nil {
		t.Fatal(err)
	}
	crd := fixtures.NewRandomNameV1CustomResourceDefinition(apiextensionsv1.NamespaceScoped)
	crdGV := "apis/" + crd.Spec.Group + "/" + crd.Spec.Versions[0].Name
	if _, err := fixtures.CreateNewV1CustomResourceDefinition(crd, apiExtensionClient, dynamicClient); err != nil {
		t.Fatal(err)
	}
	waitForDiscovery := func(present bool) {
		t.Helper()
		if err := wait.PollUntilContextTimeout(context.Background(), 200*time.Millisecond, wait.ForeverTestTimeout, true, func(context.Context) (bool, error) {
			d, _ := lazy.discovery()
			_, ok := d.Paths[crdGV]
			return ok == present, nil
		}); err != nil {
			t.Fatalf("timed out waiting for %s present=%v in discovery: %v", crdGV, present, err)
		}
	}
	waitForDiscovery(true)
	crdBody := lazy.groupVersionJSON("/openapi/v3/" + crdGV)
	if !strings.Contains(string(crdBody), crd.Spec.Names.Kind) {
		t.Errorf("expected CRD kind %q in %s document", crd.Spec.Names.Kind, crdGV)
	}
	if err := apiExtensionClient.ApiextensionsV1().CustomResourceDefinitions().Delete(context.Background(), crd.Name, metav1.DeleteOptions{}); err != nil {
		t.Fatal(err)
	}
	waitForDiscovery(false)

	// Drift guard against the classic (kube-openapi handler3) serving path:
	// the built-in documents must be byte-identical.
	classicServer, err := apiservertesting.StartTestServer(t, apiservertesting.NewDefaultTestServerOptions(),
		[]string{"--feature-gates=OpenAPIV3LazyBuild=false"}, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer classicServer.TearDownFn()
	classic := newV3Client(t, classicServer.ClientConfig)
	classicDiscovery, _ := classic.discovery()
	for gv := range discovery.Paths {
		if _, ok := classicDiscovery.Paths[gv]; !ok {
			t.Errorf("group-version %s served lazily but not classically", gv)
		}
	}
	for gv := range classicDiscovery.Paths {
		if _, ok := discovery.Paths[gv]; !ok {
			t.Errorf("group-version %s served classically but not lazily", gv)
		}
	}
	for _, gv := range []string{"api/v1", "apis/apps/v1", "apis/batch/v1", "apis/apiextensions.k8s.io/v1", "apis/networking.k8s.io/v1"} {
		want := classic.groupVersionJSON("/openapi/v3/" + gv)
		got := lazy.groupVersionJSON("/openapi/v3/" + gv)
		if string(want) != string(got) {
			t.Errorf("%s: lazy document differs from the classic one (len %d vs %d)", gv, len(got), len(want))
		}
	}
}
