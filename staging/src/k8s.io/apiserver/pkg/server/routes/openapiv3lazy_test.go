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

package routes

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"

	restful "github.com/emicklei/go-restful/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/apiserver/pkg/server/mux"
	"k8s.io/kube-openapi/pkg/cached"
	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/common/restfuladapter"
	"k8s.io/kube-openapi/pkg/handler3"
	"k8s.io/kube-openapi/pkg/spec3"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

type testWidget struct {
	Name string `json:"name"`
}

type testWidgetList struct {
	Items []testWidget `json:"items"`
}

const testPkg = "k8s.io/apiserver/pkg/server/routes."

func testV3Definitions(ref common.ReferenceCallback) map[string]common.OpenAPIDefinition {
	return map[string]common.OpenAPIDefinition{
		testPkg + "testWidget": {
			Schema: spec.Schema{
				SchemaProps: spec.SchemaProps{
					Type: []string{"object"},
					Properties: map[string]spec.Schema{
						"name": {SchemaProps: spec.SchemaProps{Type: []string{"string"}}},
					},
				},
			},
		},
		testPkg + "testWidgetList": {
			Schema: spec.Schema{
				SchemaProps: spec.SchemaProps{
					Type: []string{"object"},
					Properties: map[string]spec.Schema{
						"items": {SchemaProps: spec.SchemaProps{
							Type:  []string{"array"},
							Items: &spec.SchemaOrArray{Schema: &spec.Schema{SchemaProps: spec.SchemaProps{Ref: ref(testPkg + "testWidget")}}},
						}},
					},
				},
			},
		},
	}
}

func newTestV3Config() *common.OpenAPIV3Config {
	return &common.OpenAPIV3Config{
		Info:           &spec.Info{InfoProps: spec.InfoProps{Title: "test", Version: "v0.0.0"}},
		GetDefinitions: testV3Definitions,
		GetDefinitionName: func(name string) (string, spec.Extensions) {
			return strings.TrimPrefix(name, "k8s.io/apiserver/pkg/server/"), nil
		},
	}
}

func newTestWebService(root string, extraRoute bool) *restful.WebService {
	ws := new(restful.WebService).Path(root)
	ws.Route(ws.GET("/widgets").To(func(*restful.Request, *restful.Response) {}).
		Operation("listWidgets").Produces("application/json").
		Writes(testWidgetList{}).Returns(http.StatusOK, "OK", testWidgetList{}))
	ws.Route(ws.GET("/widgets/{name}").To(func(*restful.Request, *restful.Response) {}).
		Operation("readWidget").Produces("application/json").
		Param(ws.PathParameter("name", "name of the widget")).
		Writes(testWidget{}).Returns(http.StatusOK, "OK", testWidget{}))
	if extraRoute {
		ws.Route(ws.POST("/widgets").To(func(*restful.Request, *restful.Response) {}).
			Operation("createWidget").Consumes("application/json").Produces("application/json").
			Reads(testWidget{}).Writes(testWidget{}).Returns(http.StatusCreated, "Created", testWidget{}))
	}
	return ws
}

// minimalSpec is the smallest OpenAPI v3 document gnostic accepts for the
// protobuf conversion.
func minimalSpec(title string) *spec3.OpenAPI {
	return &spec3.OpenAPI{
		Version: "3.0.0",
		Info:    &spec.Info{InfoProps: spec.InfoProps{Title: title, Version: "v1"}},
		Paths:   &spec3.Paths{Paths: map[string]*spec3.Path{}},
	}
}

func newTestContainer() *restful.Container {
	c := restful.NewContainer()
	c.Add(newTestWebService("/apis/test.example.com/v1", true))
	c.Add(newTestWebService("/api/v1", false))
	return c
}

// newServers installs the classic (handler3) and lazy services over the same
// container and returns test servers for both.
func newServers(t *testing.T) (classic, lazy *httptest.Server, lazyService *OpenAPIV3LazyService) {
	t.Helper()
	c := newTestContainer()
	oa := OpenAPI{V3Config: newTestV3Config()}

	classicMux := mux.NewPathRecorderMux("classic")
	oa.InstallV3(c, classicMux)
	classic = httptest.NewServer(classicMux)
	t.Cleanup(classic.Close)

	lazyMux := mux.NewPathRecorderMux("lazy")
	lazyService = oa.InstallV3Lazy(c, lazyMux, "test-identity")
	lazy = httptest.NewServer(lazyMux)
	t.Cleanup(lazy.Close)
	return classic, lazy, lazyService
}

type response struct {
	status int
	header http.Header
	body   []byte
}

func get(t *testing.T, base, path, accept string, extra ...string) response {
	t.Helper()
	req, err := http.NewRequest(http.MethodGet, base+path, nil)
	require.NoError(t, err)
	if accept != "" {
		req.Header.Set("Accept", accept)
	}
	for i := 0; i+1 < len(extra); i += 2 {
		req.Header.Set(extra[i], extra[i+1])
	}
	client := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	resp, err := client.Do(req)
	require.NoError(t, err)
	defer func() { require.NoError(t, resp.Body.Close()) }()
	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	return response{status: resp.StatusCode, header: resp.Header, body: body}
}

func discovery(t *testing.T, base string) handler3.OpenAPIV3Discovery {
	t.Helper()
	resp := get(t, base, "/openapi/v3", "application/json")
	require.Equal(t, http.StatusOK, resp.status)
	var d handler3.OpenAPIV3Discovery
	require.NoError(t, json.Unmarshal(resp.body, &d))
	return d
}

var hashRE = regexp.MustCompile(`^/openapi/v3/(.+)\?hash=([0-9A-F]{128})$`)

// TestOpenAPIV3LazyMatchesHandler3 is the drift guard: for every
// group-version the lazy service must serve exactly the bytes and headers
// kube-openapi's handler3 serves, in JSON and protobuf, and honor the hash
// query parameter the same way. Only the value of the built-in etag differs
// by design.
func TestOpenAPIV3LazyMatchesHandler3(t *testing.T) {
	classic, lazy, _ := newServers(t)

	classicDisc := discovery(t, classic.URL)
	lazyDisc := discovery(t, lazy.URL)
	require.Len(t, lazyDisc.Paths, 2)
	require.Equal(t, keys(classicDisc.Paths), keys(lazyDisc.Paths))

	for gv, entry := range lazyDisc.Paths {
		t.Run(gv, func(t *testing.T) {
			m := hashRE.FindStringSubmatch(entry.ServerRelativeURL)
			require.NotNil(t, m, "discovery URL %q is not of the expected shape", entry.ServerRelativeURL)
			require.Equal(t, gv, m[1])
			lazyHash := m[2]

			for _, accept := range []string{
				"",
				"application/json",
				"application/" + v3SubTypeProtobuf,
				"application/" + v3SubTypeProtobufDeprecated,
				"*/*",
				"application/*",
			} {
				want := get(t, classic.URL, "/openapi/v3/"+gv, accept)
				got := get(t, lazy.URL, "/openapi/v3/"+gv, accept)
				assert.Equal(t, want.status, got.status, "accept=%q", accept)
				assert.Equal(t, want.header.Get("Content-Type"), got.header.Get("Content-Type"), "accept=%q", accept)
				assert.Equal(t, want.header.Values("Vary"), got.header.Values("Vary"), "accept=%q", accept)
				assert.Equal(t, want.body, got.body, "accept=%q", accept)
				assert.Empty(t, got.header.Get("Cache-Control"), "accept=%q", accept)
				etag, err := strconv.Unquote(got.header.Get("Etag"))
				require.NoError(t, err, "accept=%q", accept)
				assert.Equal(t, lazyHash, etag, "the served ETag must equal the discovery hash, accept=%q", accept)
			}

			// Unsupported media type.
			want := get(t, classic.URL, "/openapi/v3/"+gv, "text/html")
			got := get(t, lazy.URL, "/openapi/v3/"+gv, "text/html")
			assert.Equal(t, http.StatusNotAcceptable, got.status)
			assert.Equal(t, want.status, got.status)

			// Matching hash: immutable caching headers, same body.
			got = get(t, lazy.URL, "/openapi/v3/"+gv+"?hash="+lazyHash, "application/json")
			assert.Equal(t, http.StatusOK, got.status)
			assert.Equal(t, "public, immutable", got.header.Get("Cache-Control"))
			assert.NotEmpty(t, got.header.Get("Expires"))
			assert.Equal(t, get(t, classic.URL, "/openapi/v3/"+gv, "application/json").body, got.body)

			// Stale hash: permanent redirect to the current one, as handler3.
			want = get(t, classic.URL, "/openapi/v3/"+gv+"?hash=stale", "application/json")
			got = get(t, lazy.URL, "/openapi/v3/"+gv+"?hash=stale", "application/json")
			assert.Equal(t, http.StatusMovedPermanently, want.status)
			assert.Equal(t, http.StatusMovedPermanently, got.status)
			assert.Equal(t, constructV3ServerRelativeURL(gv, lazyHash), got.header.Get("Location"))

			// Conditional request against the served ETag.
			got = get(t, lazy.URL, "/openapi/v3/"+gv, "application/json", "If-None-Match", strconv.Quote(lazyHash))
			assert.Equal(t, http.StatusNotModified, got.status)
		})
	}

	// Unknown group-version: handler3 writes nothing (200, empty body).
	want := get(t, classic.URL, "/openapi/v3/apis/nope/v1", "application/json")
	got := get(t, lazy.URL, "/openapi/v3/apis/nope/v1", "application/json")
	assert.Equal(t, want.status, got.status)
	assert.Equal(t, want.body, got.body)

	// Discovery conditional request.
	resp := get(t, lazy.URL, "/openapi/v3", "application/json")
	resp = get(t, lazy.URL, "/openapi/v3", "application/json", "If-None-Match", resp.header.Get("Etag"))
	assert.Equal(t, http.StatusNotModified, resp.status)
}

func keys(m map[string]handler3.OpenAPIV3DiscoveryGroupVersion) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// TestOpenAPIV3LazyBuildIsDeferred verifies that discovery never triggers a
// build, that a group-version is built exactly once on first request, and
// that the builder is released afterwards.
func TestOpenAPIV3LazyBuildIsDeferred(t *testing.T) {
	c := newTestContainer()
	config := newTestV3Config()
	builds := map[string]int{}
	service := NewOpenAPIV3LazyService()
	for _, ws := range c.RegisteredWebServices() {
		gv := ws.RootPath()[1:]
		rcs := restfuladapter.AdaptWebServices([]*restful.WebService{ws})
		service.AddBuiltinGroupVersion(gv, BuiltinV3ETag("id", gv, rcs, config), func() (*spec3.OpenAPI, error) {
			builds[gv]++
			return minimalSpec(gv), nil
		})
	}
	m := mux.NewPathRecorderMux("lazy")
	require.NoError(t, service.RegisterOpenAPIV3VersionedService("/openapi/v3", m))
	server := httptest.NewServer(m)
	defer server.Close()

	disc := discovery(t, server.URL)
	require.Len(t, disc.Paths, 2)
	assert.Empty(t, builds, "discovery must not build any spec")
	// Discovery is served from cache afterwards.
	discovery(t, server.URL)
	assert.Empty(t, builds)

	resp := get(t, server.URL, "/openapi/v3/api/v1", "application/json")
	require.Equal(t, http.StatusOK, resp.status)
	assert.Contains(t, string(resp.body), `"title":"api/v1"`)
	assert.Equal(t, map[string]int{"api/v1": 1}, builds)

	// Subsequent JSON and protobuf requests reuse the bytes.
	get(t, server.URL, "/openapi/v3/api/v1", "application/json")
	pb := get(t, server.URL, "/openapi/v3/api/v1", "application/"+v3SubTypeProtobuf)
	require.Equal(t, http.StatusOK, pb.status)
	assert.NotEmpty(t, pb.body)
	get(t, server.URL, "/openapi/v3/api/v1", "application/"+v3SubTypeProtobuf)
	assert.Equal(t, map[string]int{"api/v1": 1}, builds)

	service.mu.RLock()
	entry := service.entries["api/v1"]
	other := service.entries["apis/test.example.com/v1"]
	service.mu.RUnlock()
	assert.Nil(t, entry.build, "builder must be released after a successful build")
	assert.NotNil(t, entry.jsonBytes)
	assert.NotNil(t, entry.pbBytes)
	assert.NotNil(t, other.build, "unrequested group-version must stay unbuilt")
	assert.Nil(t, other.jsonBytes)

	// The discovery document is unchanged by builds: the etag was known up front.
	assert.Equal(t, disc, discovery(t, server.URL))
}

// TestOpenAPIV3LazyBuildErrorRetried verifies that a failed build is not
// cached: the request fails the way handler3 fails (nothing written) and the
// next request retries the build.
func TestOpenAPIV3LazyBuildErrorRetried(t *testing.T) {
	service := NewOpenAPIV3LazyService()
	calls := 0
	service.AddBuiltinGroupVersion("api/v1", "ETAG", func() (*spec3.OpenAPI, error) {
		calls++
		if calls == 1 {
			return nil, errors.New("transient")
		}
		return minimalSpec("retry"), nil
	})
	m := mux.NewPathRecorderMux("lazy")
	require.NoError(t, service.RegisterOpenAPIV3VersionedService("/openapi/v3", m))
	server := httptest.NewServer(m)
	defer server.Close()

	resp := get(t, server.URL, "/openapi/v3/api/v1", "application/json")
	assert.Empty(t, resp.body)
	assert.Equal(t, 1, calls)

	resp = get(t, server.URL, "/openapi/v3/api/v1", "application/json")
	assert.Equal(t, http.StatusOK, resp.status)
	assert.Contains(t, string(resp.body), `"openapi":"3.0.0"`)
	assert.Equal(t, 2, calls)

	get(t, server.URL, "/openapi/v3/api/v1", "application/json")
	assert.Equal(t, 2, calls)
}

type fakeSource struct {
	spec *spec3.OpenAPI
	etag string
	err  error
	gets int
}

func (f *fakeSource) Get() (*spec3.OpenAPI, string, error) {
	f.gets++
	return f.spec, f.etag, f.err
}

// TestOpenAPIV3LazyDynamicEntries verifies the OpenAPIV3Updater behaviour
// used by the CRD controller: content-hash etags, rebuild when the source
// etag changes, last-success on source errors, and discovery updates on
// add/delete — compared against handler3 where the outputs are deterministic.
func TestOpenAPIV3LazyDynamicEntries(t *testing.T) {
	lazyService := NewOpenAPIV3LazyService()
	classicService := handler3.NewOpenAPIService()
	newServer := func(register func(string, common.PathHandlerByGroupVersion) error) *httptest.Server {
		m := mux.NewPathRecorderMux("m")
		require.NoError(t, register("/openapi/v3", m))
		s := httptest.NewServer(m)
		t.Cleanup(s.Close)
		return s
	}
	lazy := newServer(lazyService.RegisterOpenAPIV3VersionedService)
	classic := newServer(classicService.RegisterOpenAPIV3VersionedService)

	crdSpec := minimalSpec("crd")
	src := &fakeSource{spec: crdSpec, etag: "1"}
	lazyService.UpdateGroupVersionLazy("apis/crd.example.com/v1", src)
	classicService.UpdateGroupVersionLazy("apis/crd.example.com/v1", cached.Static(crdSpec, "1"))

	lazyDisc := discovery(t, lazy.URL)
	classicDisc := discovery(t, classic.URL)
	assert.Equal(t, classicDisc, lazyDisc, "content-hash discovery URLs must match handler3 for dynamic entries")
	discoveryETag1 := get(t, lazy.URL, "/openapi/v3", "application/json").header.Get("Etag")

	for _, accept := range []string{"application/json", "application/" + v3SubTypeProtobuf} {
		want := get(t, classic.URL, "/openapi/v3/apis/crd.example.com/v1", accept)
		got := get(t, lazy.URL, "/openapi/v3/apis/crd.example.com/v1", accept)
		assert.Equal(t, want.status, got.status)
		assert.Equal(t, want.body, got.body)
		assert.Equal(t, want.header.Get("Etag"), got.header.Get("Etag"))
	}

	// A source error is hidden behind the last success.
	src.err = errors.New("boom")
	got := get(t, lazy.URL, "/openapi/v3/apis/crd.example.com/v1", "application/json")
	assert.Equal(t, http.StatusOK, got.status)
	assert.Contains(t, string(got.body), `"title":"crd"`)
	assert.Equal(t, http.StatusOK, get(t, lazy.URL, "/openapi/v3", "application/json").status)
	src.err = nil

	// A changed source etag rebuilds the bytes and the discovery URL.
	src.spec = minimalSpec("crd2")
	src.etag = "2"
	got = get(t, lazy.URL, "/openapi/v3/apis/crd.example.com/v1", "application/json")
	assert.Contains(t, string(got.body), `"title":"crd2"`)
	lazyDisc2 := discovery(t, lazy.URL)
	assert.NotEqual(t, lazyDisc.Paths["apis/crd.example.com/v1"], lazyDisc2.Paths["apis/crd.example.com/v1"])
	discoveryETag2 := get(t, lazy.URL, "/openapi/v3", "application/json").header.Get("Etag")
	assert.NotEqual(t, discoveryETag1, discoveryETag2)
	pb := get(t, lazy.URL, "/openapi/v3/apis/crd.example.com/v1", "application/"+v3SubTypeProtobuf)
	assert.Equal(t, http.StatusOK, pb.status)
	assert.Equal(t, got.header.Get("Etag"), pb.header.Get("Etag"))

	// UpdateGroupVersion (static) and DeleteGroupVersion.
	lazyService.UpdateGroupVersion("apis/other.example.com/v1", minimalSpec("other"))
	assert.Len(t, discovery(t, lazy.URL).Paths, 2)
	assert.Equal(t, []string{"apis/crd.example.com/v1", "apis/other.example.com/v1"}, lazyService.GroupVersions())
	lazyService.DeleteGroupVersion("apis/crd.example.com/v1")
	assert.Equal(t, []string{"apis/other.example.com/v1"}, keys(discovery(t, lazy.URL).Paths))
	got = get(t, lazy.URL, "/openapi/v3/apis/crd.example.com/v1", "application/json")
	assert.Empty(t, got.body)

	// A source that errors before ever succeeding fails discovery, as handler3.
	lazyService.UpdateGroupVersionLazy("apis/broken.example.com/v1", &fakeSource{err: errors.New("never")})
	assert.Equal(t, http.StatusInternalServerError, get(t, lazy.URL, "/openapi/v3", "application/json").status)
}

func TestBuiltinV3ETag(t *testing.T) {
	config := newTestV3Config()
	ws := newTestWebService("/apis/test.example.com/v1", true)
	rcs := restfuladapter.AdaptWebServices([]*restful.WebService{ws})
	base := BuiltinV3ETag("id", "apis/test.example.com/v1", rcs, config)

	assert.Regexp(t, `^[0-9A-F]{128}$`, base, "must have the shape of handler3's content hash")
	assert.Equal(t, base, BuiltinV3ETag("id", "apis/test.example.com/v1", rcs, config), "deterministic")
	assert.Equal(t, base, BuiltinV3ETag("id", "apis/test.example.com/v1",
		restfuladapter.AdaptWebServices([]*restful.WebService{newTestWebService("/apis/test.example.com/v1", true)}), config),
		"independent of route object identity")

	assert.NotEqual(t, base, BuiltinV3ETag("other", "apis/test.example.com/v1", rcs, config), "identity")
	assert.NotEqual(t, base, BuiltinV3ETag("id", "apis/test.example.com/v2", rcs, config), "group-version")
	assert.NotEqual(t, base, BuiltinV3ETag("id", "apis/test.example.com/v1",
		restfuladapter.AdaptWebServices([]*restful.WebService{newTestWebService("/apis/test.example.com/v1", false)}), config),
		"route set")
	config2 := newTestV3Config()
	config2.Info.Version = "v0.0.1"
	assert.NotEqual(t, base, BuiltinV3ETag("id", "apis/test.example.com/v1", rcs, config2), "config info")
	assert.NotEqual(t, base, BuiltinV3ETag("id", "apis/test.example.com/v1", rcs, nil), "nil config")
}

func TestOpenAPIV3LazyServiceImplementsUpdater(t *testing.T) {
	var updater OpenAPIV3Updater = NewOpenAPIV3LazyService()
	require.NotNil(t, updater)
	// Compile-time assertions in the package cover handler3; make sure the
	// method set is exactly what the CRD controller needs at runtime too.
	updater.UpdateGroupVersion("a/v1", &spec3.OpenAPI{})
	updater.UpdateGroupVersionLazy("b/v1", cached.Static(&spec3.OpenAPI{}, "x"))
	updater.DeleteGroupVersion("a/v1")
	assert.Equal(t, []string{"b/v1"}, updater.(*OpenAPIV3LazyService).GroupVersions())
}
