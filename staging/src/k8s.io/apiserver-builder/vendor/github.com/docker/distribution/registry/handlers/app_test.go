package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"testing"

	"github.com/docker/distribution/configuration"
	"github.com/docker/distribution/context"
	"github.com/docker/distribution/registry/api/errcode"
	"github.com/docker/distribution/registry/api/v2"
	"github.com/docker/distribution/registry/auth"
	_ "github.com/docker/distribution/registry/auth/silly"
	"github.com/docker/distribution/registry/storage"
	memorycache "github.com/docker/distribution/registry/storage/cache/memory"
	"github.com/docker/distribution/registry/storage/driver/inmemory"
)

// TestAppDispatcher builds an application with a test dispatcher and ensures
// that requests are properly dispatched and the handlers are constructed.
// This only tests the dispatch mechanism. The underlying dispatchers must be
// tested individually.
func TestAppDispatcher(t *testing.T) {
	driver := inmemory.New()
	ctx := context.Background()
	registry, err := storage.NewRegistry(ctx, driver, storage.BlobDescriptorCacheProvider(memorycache.NewInMemoryBlobDescriptorCacheProvider()), storage.EnableDelete, storage.EnableRedirect)
	if err != nil {
		t.Fatalf("error creating registry: %v", err)
	}
	app := &App{
		Config:   &configuration.Configuration{},
		Context:  ctx,
		router:   v2.Router(),
		driver:   driver,
		registry: registry,
	}
	server := httptest.NewServer(app)
	router := v2.Router()

	serverURL, err := url.Parse(server.URL)
	if err != nil {
		t.Fatalf("error parsing server url: %v", err)
	}

	varCheckingDispatcher := func(expectedVars map[string]string) dispatchFunc {
		return func(ctx *Context, r *http.Request) http.Handler {
			// Always checks the same name context
			if ctx.Repository.Named().Name() != getName(ctx) {
				t.Fatalf("unexpected name: %q != %q", ctx.Repository.Named().Name(), "foo/bar")
			}

			// Check that we have all that is expected
			for expectedK, expectedV := range expectedVars {
				if ctx.Value(expectedK) != expectedV {
					t.Fatalf("unexpected %s in context vars: %q != %q", expectedK, ctx.Value(expectedK), expectedV)
				}
			}

			// Check that we only have variables that are expected
			for k, v := range ctx.Value("vars").(map[string]string) {
				_, ok := expectedVars[k]

				if !ok { // name is checked on context
					// We have an unexpected key, fail
					t.Fatalf("unexpected key %q in vars with value %q", k, v)
				}
			}

			return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
			})
		}
	}

	// unflatten a list of variables, suitable for gorilla/mux, to a map[string]string
	unflatten := func(vars []string) map[string]string {
		m := make(map[string]string)
		for i := 0; i < len(vars)-1; i = i + 2 {
			m[vars[i]] = vars[i+1]
		}

		return m
	}

	for _, testcase := range []struct {
		endpoint string
		vars     []string
	}{
		{
			endpoint: v2.RouteNameManifest,
			vars: []string{
				"name", "foo/bar",
				"reference", "sometag",
			},
		},
		{
			endpoint: v2.RouteNameTags,
			vars: []string{
				"name", "foo/bar",
			},
		},
		{
			endpoint: v2.RouteNameBlobUpload,
			vars: []string{
				"name", "foo/bar",
			},
		},
		{
			endpoint: v2.RouteNameBlobUploadChunk,
			vars: []string{
				"name", "foo/bar",
				"uuid", "theuuid",
			},
		},
	} {
		app.register(testcase.endpoint, varCheckingDispatcher(unflatten(testcase.vars)))
		route := router.GetRoute(testcase.endpoint).Host(serverURL.Host)
		u, err := route.URL(testcase.vars...)

		if err != nil {
			t.Fatal(err)
		}

		resp, err := http.Get(u.String())

		if err != nil {
			t.Fatal(err)
		}

		if resp.StatusCode != http.StatusOK {
			t.Fatalf("unexpected status code: %v != %v", resp.StatusCode, http.StatusOK)
		}
	}
}

// TestNewApp covers the creation of an application via NewApp with a
// configuration.
func TestNewApp(t *testing.T) {
	ctx := context.Background()
	config := configuration.Configuration{
		Storage: configuration.Storage{
			"inmemory": nil,
		},
		Auth: configuration.Auth{
			// For now, we simply test that new auth results in a viable
			// application.
			"silly": {
				"realm":   "realm-test",
				"service": "service-test",
			},
		},
	}

	// Mostly, with this test, given a sane configuration, we are simply
	// ensuring that NewApp doesn't panic. We might want to tweak this
	// behavior.
	app := NewApp(ctx, &config)

	server := httptest.NewServer(app)
	builder, err := v2.NewURLBuilderFromString(server.URL, false)
	if err != nil {
		t.Fatalf("error creating urlbuilder: %v", err)
	}

	baseURL, err := builder.BuildBaseURL()
	if err != nil {
		t.Fatalf("error creating baseURL: %v", err)
	}

	// TODO(stevvooe): The rest of this test might belong in the API tests.

	// Just hit the app and make sure we get a 401 Unauthorized error.
	req, err := http.Get(baseURL)
	if err != nil {
		t.Fatalf("unexpected error during GET: %v", err)
	}
	defer req.Body.Close()

	if req.StatusCode != http.StatusUnauthorized {
		t.Fatalf("unexpected status code during request: %v", err)
	}

	if req.Header.Get("Content-Type") != "application/json; charset=utf-8" {
		t.Fatalf("unexpected content-type: %v != %v", req.Header.Get("Content-Type"), "application/json; charset=utf-8")
	}

	expectedAuthHeader := "Bearer realm=\"realm-test\",service=\"service-test\""
	if e, a := expectedAuthHeader, req.Header.Get("WWW-Authenticate"); e != a {
		t.Fatalf("unexpected WWW-Authenticate header: %q != %q", e, a)
	}

	var errs errcode.Errors
	dec := json.NewDecoder(req.Body)
	if err := dec.Decode(&errs); err != nil {
		t.Fatalf("error decoding error response: %v", err)
	}

	err2, ok := errs[0].(errcode.ErrorCoder)
	if !ok {
		t.Fatalf("not an ErrorCoder: %#v", errs[0])
	}
	if err2.ErrorCode() != errcode.ErrorCodeUnauthorized {
		t.Fatalf("unexpected error code: %v != %v", err2.ErrorCode(), errcode.ErrorCodeUnauthorized)
	}
}

// Test the access record accumulator
func TestAppendAccessRecords(t *testing.T) {
	repo := "testRepo"

	expectedResource := auth.Resource{
		Type: "repository",
		Name: repo,
	}

	expectedPullRecord := auth.Access{
		Resource: expectedResource,
		Action:   "pull",
	}
	expectedPushRecord := auth.Access{
		Resource: expectedResource,
		Action:   "push",
	}
	expectedAllRecord := auth.Access{
		Resource: expectedResource,
		Action:   "*",
	}

	records := []auth.Access{}
	result := appendAccessRecords(records, "GET", repo)
	expectedResult := []auth.Access{expectedPullRecord}
	if ok := reflect.DeepEqual(result, expectedResult); !ok {
		t.Fatalf("Actual access record differs from expected")
	}

	records = []auth.Access{}
	result = appendAccessRecords(records, "HEAD", repo)
	expectedResult = []auth.Access{expectedPullRecord}
	if ok := reflect.DeepEqual(result, expectedResult); !ok {
		t.Fatalf("Actual access record differs from expected")
	}

	records = []auth.Access{}
	result = appendAccessRecords(records, "POST", repo)
	expectedResult = []auth.Access{expectedPullRecord, expectedPushRecord}
	if ok := reflect.DeepEqual(result, expectedResult); !ok {
		t.Fatalf("Actual access record differs from expected")
	}

	records = []auth.Access{}
	result = appendAccessRecords(records, "PUT", repo)
	expectedResult = []auth.Access{expectedPullRecord, expectedPushRecord}
	if ok := reflect.DeepEqual(result, expectedResult); !ok {
		t.Fatalf("Actual access record differs from expected")
	}

	records = []auth.Access{}
	result = appendAccessRecords(records, "PATCH", repo)
	expectedResult = []auth.Access{expectedPullRecord, expectedPushRecord}
	if ok := reflect.DeepEqual(result, expectedResult); !ok {
		t.Fatalf("Actual access record differs from expected")
	}

	records = []auth.Access{}
	result = appendAccessRecords(records, "DELETE", repo)
	expectedResult = []auth.Access{expectedAllRecord}
	if ok := reflect.DeepEqual(result, expectedResult); !ok {
		t.Fatalf("Actual access record differs from expected")
	}

}
