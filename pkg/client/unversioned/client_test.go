/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package unversioned

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"path"
	"reflect"
	"strings"
	"testing"

	"github.com/emicklei/go-restful/swagger"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/version"
)

const nameRequiredError = "resource name may not be empty"

type testRequest struct {
	Method  string
	Path    string
	Header  string
	Query   url.Values
	Body    runtime.Object
	RawBody *string
}

type Response struct {
	StatusCode int
	Body       runtime.Object
	RawBody    *string
}

type testClient struct {
	*Client
	Request  testRequest
	Response Response
	Error    bool
	Created  bool
	Version  string
	server   *httptest.Server
	handler  *util.FakeHandler
	// For query args, an optional function to validate the contents
	// useful when the contents can change but still be correct.
	// Maps from query arg key to validator.
	// If no validator is present, string equality is used.
	QueryValidator map[string]func(string, string) bool
}

func (c *testClient) Setup(t *testing.T) *testClient {
	c.handler = &util.FakeHandler{
		StatusCode: c.Response.StatusCode,
	}
	if responseBody := body(t, c.Response.Body, c.Response.RawBody); responseBody != nil {
		c.handler.ResponseBody = *responseBody
	}
	c.server = httptest.NewServer(c.handler)
	if c.Client == nil {
		version := c.Version
		if len(version) == 0 {
			version = testapi.Default.Version()
		}
		c.Client = NewOrDie(&Config{
			Host:    c.server.URL,
			Version: version,
		})

		// TODO: caesarxuchao: hacky way to specify version of Experimental client.
		// We will fix this by supporting multiple group versions in Config
		version = c.Version
		if len(version) == 0 {
			version = testapi.Experimental.Version()
		}
		c.ExperimentalClient = NewExperimentalOrDie(&Config{
			Host:    c.server.URL,
			Version: version,
		})
	}
	c.QueryValidator = map[string]func(string, string) bool{}
	return c
}

func (c *testClient) Validate(t *testing.T, received runtime.Object, err error) {
	c.ValidateCommon(t, err)

	if c.Response.Body != nil && !api.Semantic.DeepDerivative(c.Response.Body, received) {
		t.Errorf("bad response for request %#v: expected %#v, got %#v", c.Request, c.Response.Body, received)
	}
}

func (c *testClient) ValidateRaw(t *testing.T, received []byte, err error) {
	c.ValidateCommon(t, err)

	if c.Response.Body != nil && !reflect.DeepEqual(c.Response.Body, received) {
		t.Errorf("bad response for request %#v: expected %#v, got %#v", c.Request, c.Response.Body, received)
	}
}

func (c *testClient) ValidateCommon(t *testing.T, err error) {
	defer c.server.Close()

	if c.Error {
		if err == nil {
			t.Errorf("error expected for %#v, got none", c.Request)
		}
		return
	}
	if err != nil {
		t.Errorf("no error expected for %#v, got: %v", c.Request, err)
	}

	if c.handler.RequestReceived == nil {
		t.Errorf("handler had an empty request, %#v", c)
		return
	}

	requestBody := body(t, c.Request.Body, c.Request.RawBody)
	actualQuery := c.handler.RequestReceived.URL.Query()
	t.Logf("got query: %v", actualQuery)
	t.Logf("path: %v", c.Request.Path)
	// We check the query manually, so blank it out so that FakeHandler.ValidateRequest
	// won't check it.
	c.handler.RequestReceived.URL.RawQuery = ""
	c.handler.ValidateRequest(t, path.Join(c.Request.Path), c.Request.Method, requestBody)
	for key, values := range c.Request.Query {
		validator, ok := c.QueryValidator[key]
		if !ok {
			switch key {
			case api.LabelSelectorQueryParam(testapi.Default.Version()):
				validator = validateLabels
			case api.FieldSelectorQueryParam(testapi.Default.Version()):
				validator = validateFields
			default:
				validator = func(a, b string) bool { return a == b }
			}
		}
		observed := actualQuery.Get(key)
		wanted := strings.Join(values, "")
		if !validator(wanted, observed) {
			t.Errorf("Unexpected query arg for key: %s.  Expected %s, Received %s", key, wanted, observed)
		}
	}
	if c.Request.Header != "" {
		if c.handler.RequestReceived.Header.Get(c.Request.Header) == "" {
			t.Errorf("header %q not found in request %#v", c.Request.Header, c.handler.RequestReceived)
		}
	}

	if expected, received := requestBody, c.handler.RequestBody; expected != nil && *expected != received {
		t.Errorf("bad body for request %#v: expected %s, got %s", c.Request, *expected, received)
	}
}

// buildResourcePath is a convenience function for knowing if a namespace should be in a path param or not
func buildResourcePath(namespace, resource string) string {
	if len(namespace) > 0 {
		return path.Join("namespaces", namespace, resource)
	}
	return resource
}

// buildQueryValues is a convenience function for knowing if a namespace should be in a query param or not
func buildQueryValues(query url.Values) url.Values {
	v := url.Values{}
	if query != nil {
		for key, values := range query {
			for _, value := range values {
				v.Add(key, value)
			}
		}
	}
	return v
}

func validateLabels(a, b string) bool {
	sA, eA := labels.Parse(a)
	if eA != nil {
		return false
	}
	sB, eB := labels.Parse(b)
	if eB != nil {
		return false
	}
	return sA.String() == sB.String()
}

func validateFields(a, b string) bool {
	sA, _ := fields.ParseSelector(a)
	sB, _ := fields.ParseSelector(b)
	return sA.String() == sB.String()
}

func body(t *testing.T, obj runtime.Object, raw *string) *string {
	if obj != nil {
		_, kind, err := api.Scheme.ObjectVersionAndKind(obj)
		if err != nil {
			t.Errorf("unexpected encoding error: %v", err)
		}
		// TODO: caesarxuchao: we should detect which group an object belongs to
		// by using the version returned by Schem.ObjectVersionAndKind() once we
		// split the schemes for internal objects.
		// TODO: caesarxuchao: we should add a map from kind to group in Scheme.
		var bs []byte
		if api.Scheme.Recognizes(testapi.Default.GroupAndVersion(), kind) {
			bs, err = testapi.Default.Codec().Encode(obj)
			if err != nil {
				t.Errorf("unexpected encoding error: %v", err)
			}
		} else if api.Scheme.Recognizes(testapi.Experimental.GroupAndVersion(), kind) {
			bs, err = testapi.Experimental.Codec().Encode(obj)
			if err != nil {
				t.Errorf("unexpected encoding error: %v", err)
			}
		} else {
			t.Errorf("unexpected kind: %v", kind)
		}
		body := string(bs)
		return &body
	}
	return raw
}

func TestGetServerVersion(t *testing.T) {
	expect := version.Info{
		Major:     "foo",
		Minor:     "bar",
		GitCommit: "baz",
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		output, err := json.Marshal(expect)
		if err != nil {
			t.Errorf("unexpected encoding error: %v", err)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(output)
	}))
	client := NewOrDie(&Config{Host: server.URL})

	got, err := client.ServerVersion()
	if err != nil {
		t.Fatalf("unexpected encoding error: %v", err)
	}
	if e, a := expect, *got; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, got %v", e, a)
	}
}

func TestGetServerAPIVersions(t *testing.T) {
	versions := []string{"v1", "v2", "v3"}
	expect := api.APIVersions{Versions: versions}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		output, err := json.Marshal(expect)
		if err != nil {
			t.Errorf("unexpected encoding error: %v", err)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(output)
	}))
	client := NewOrDie(&Config{Host: server.URL})
	got, err := client.ServerAPIVersions()
	if err != nil {
		t.Fatalf("unexpected encoding error: %v", err)
	}
	if e, a := expect, *got; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, got %v", e, a)
	}
}

func swaggerSchemaFakeServer() (*httptest.Server, error) {
	request := 1
	var sErr error

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		var resp interface{}
		if request == 1 {
			resp = api.APIVersions{Versions: []string{"v1", "v2", "v3"}}
			request++
		} else {
			resp = swagger.ApiDeclaration{}
		}
		output, err := json.Marshal(resp)
		if err != nil {
			sErr = err
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(output)
	}))
	return server, sErr
}

func TestGetSwaggerSchema(t *testing.T) {
	expect := swagger.ApiDeclaration{}

	server, err := swaggerSchemaFakeServer()
	if err != nil {
		t.Errorf("unexpected encoding error: %v", err)
	}

	client := NewOrDie(&Config{Host: server.URL})
	got, err := client.SwaggerSchema("v1")
	if err != nil {
		t.Fatalf("unexpected encoding error: %v", err)
	}
	if e, a := expect, *got; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, got %v", e, a)
	}
}

func TestGetSwaggerSchemaFail(t *testing.T) {
	expErr := "API version: v4 is not supported by the server. Use one of: [v1 v2 v3]"

	server, err := swaggerSchemaFakeServer()
	if err != nil {
		t.Errorf("unexpected encoding error: %v", err)
	}

	client := NewOrDie(&Config{Host: server.URL})
	got, err := client.SwaggerSchema("v4")
	if got != nil {
		t.Fatalf("unexpected response: %v", got)
	}
	if err.Error() != expErr {
		t.Errorf("expected an error, got %v", err)
	}
}
