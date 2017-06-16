/*
Copyright 2015 The Kubernetes Authors.

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

package server

import (
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	goruntime "runtime"
	"testing"
	"time"

	// "github.com/go-openapi/spec"
	"github.com/stretchr/testify/assert"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apimachinery"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/openapi"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/version"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/discovery"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	etcdtesting "k8s.io/apiserver/pkg/storage/etcd/testing"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	restclient "k8s.io/client-go/rest"
)

const (
	extensionsGroupName = "extensions"
)

var (
	v1GroupVersion = schema.GroupVersion{Group: "", Version: "v1"}

	scheme         = runtime.NewScheme()
	codecs         = serializer.NewCodecFactory(scheme)
	parameterCodec = runtime.NewParameterCodec(scheme)
)

func init() {
	metav1.AddToGroupVersion(scheme, metav1.SchemeGroupVersion)
	scheme.AddUnversionedTypes(v1GroupVersion,
		&metav1.Status{},
		&metav1.APIVersions{},
		&metav1.APIGroupList{},
		&metav1.APIGroup{},
		&metav1.APIResourceList{},
	)
	example.AddToScheme(scheme)
	examplev1.AddToScheme(scheme)
}

func testGetOpenAPIDefinitions(ref openapi.ReferenceCallback) map[string]openapi.OpenAPIDefinition {
	return map[string]openapi.OpenAPIDefinition{}
}

// setUp is a convience function for setting up for (most) tests.
func setUp(t *testing.T) (*etcdtesting.EtcdTestServer, Config, *assert.Assertions) {
	etcdServer, _ := etcdtesting.NewUnsecuredEtcd3TestClientServer(t, scheme)

	config := NewConfig(codecs)
	config.PublicAddress = net.ParseIP("192.168.10.4")
	config.RequestContextMapper = genericapirequest.NewRequestContextMapper()
	config.LegacyAPIGroupPrefixes = sets.NewString("/api")
	config.LoopbackClientConfig = &restclient.Config{}

	clientset := fake.NewSimpleClientset()
	if clientset == nil {
		t.Fatal("unable to create fake client set")
	}
	config.SharedInformerFactory = informers.NewSharedInformerFactory(clientset, config.LoopbackClientConfig.Timeout)

	// TODO restore this test, but right now, eliminate our cycle
	// config.OpenAPIConfig = DefaultOpenAPIConfig(testGetOpenAPIDefinitions, runtime.NewScheme())
	// config.OpenAPIConfig.Info = &spec.Info{
	// 	InfoProps: spec.InfoProps{
	// 		Title:   "Kubernetes",
	// 		Version: "unversioned",
	// 	},
	// }
	config.SwaggerConfig = DefaultSwaggerConfig()

	return etcdServer, *config, assert.New(t)
}

func newMaster(t *testing.T) (*GenericAPIServer, *etcdtesting.EtcdTestServer, Config, *assert.Assertions) {
	etcdserver, config, assert := setUp(t)

	s, err := config.Complete().New("test", EmptyDelegate)
	if err != nil {
		t.Fatalf("Error in bringing up the server: %v", err)
	}
	return s, etcdserver, config, assert
}

// TestNew verifies that the New function returns a GenericAPIServer
// using the configuration properly.
func TestNew(t *testing.T) {
	s, etcdserver, config, assert := newMaster(t)
	defer etcdserver.Terminate(t)

	// Verify many of the variables match their config counterparts
	assert.Equal(s.legacyAPIGroupPrefixes, config.LegacyAPIGroupPrefixes)
	assert.Equal(s.admissionControl, config.AdmissionControl)
	assert.Equal(s.RequestContextMapper(), config.RequestContextMapper)

	// these values get defaulted
	assert.Equal(net.JoinHostPort(config.PublicAddress.String(), "443"), s.ExternalAddress)
	assert.NotNil(s.swaggerConfig)
	assert.Equal("http://"+s.ExternalAddress, s.swaggerConfig.WebServicesUrl)
}

// Verifies that AddGroupVersions works as expected.
func TestInstallAPIGroups(t *testing.T) {
	etcdserver, config, assert := setUp(t)
	defer etcdserver.Terminate(t)

	config.LegacyAPIGroupPrefixes = sets.NewString("/apiPrefix")
	config.DiscoveryAddresses = discovery.DefaultAddresses{DefaultAddress: "ExternalAddress"}

	s, err := config.SkipComplete().New("test", EmptyDelegate)
	if err != nil {
		t.Fatalf("Error in bringing up the server: %v", err)
	}

	testAPI := func(gv schema.GroupVersion) APIGroupInfo {
		getter, noVerbs := testGetterStorage{}, testNoVerbsStorage{}

		scheme := runtime.NewScheme()
		scheme.AddKnownTypeWithName(gv.WithKind("Getter"), getter.New())
		scheme.AddKnownTypeWithName(gv.WithKind("NoVerb"), noVerbs.New())
		scheme.AddKnownTypes(v1GroupVersion, &metav1.Status{})
		metav1.AddToGroupVersion(scheme, v1GroupVersion)

		interfacesFor := func(version schema.GroupVersion) (*meta.VersionInterfaces, error) {
			return &meta.VersionInterfaces{
				ObjectConvertor:  scheme,
				MetadataAccessor: meta.NewAccessor(),
			}, nil
		}

		mapper := meta.NewDefaultRESTMapperFromScheme([]schema.GroupVersion{gv}, interfacesFor, "", sets.NewString(), sets.NewString(), scheme)
		groupMeta := apimachinery.GroupMeta{
			GroupVersion:  gv,
			GroupVersions: []schema.GroupVersion{gv},
			RESTMapper:    mapper,
			InterfacesFor: interfacesFor,
		}

		return APIGroupInfo{
			GroupMeta: groupMeta,
			VersionedResourcesStorageMap: map[string]map[string]rest.Storage{
				gv.Version: {
					"getter":  &testGetterStorage{Version: gv.Version},
					"noverbs": &testNoVerbsStorage{Version: gv.Version},
				},
			},
			OptionsExternalVersion: &schema.GroupVersion{Version: "v1"},
			ParameterCodec:         parameterCodec,
			NegotiatedSerializer:   codecs,
			Scheme:                 scheme,
		}
	}

	apis := []APIGroupInfo{
		testAPI(schema.GroupVersion{Group: "", Version: "v1"}),
		testAPI(schema.GroupVersion{Group: extensionsGroupName, Version: "v1"}),
		testAPI(schema.GroupVersion{Group: "batch", Version: "v1"}),
	}

	err = s.InstallLegacyAPIGroup("/apiPrefix", &apis[0])
	assert.NoError(err)
	groupPaths := []string{
		config.LegacyAPIGroupPrefixes.List()[0], // /apiPrefix
	}
	for _, api := range apis[1:] {
		err = s.InstallAPIGroup(&api)
		assert.NoError(err)
		groupPaths = append(groupPaths, APIGroupPrefix+"/"+api.GroupMeta.GroupVersion.Group) // /apis/<group>
	}

	server := httptest.NewServer(s.Handler)
	defer server.Close()

	for i := range apis {
		// should serve APIGroup at group path
		info := &apis[i]
		path := groupPaths[i]
		resp, err := http.Get(server.URL + path)
		if err != nil {
			t.Errorf("[%d] unexpected error getting path %q path: %v", i, path, err)
			continue
		}

		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			t.Errorf("[%d] unexpected error reading body at path %q: %v", i, path, err)
			continue
		}

		t.Logf("[%d] json at %s: %s", i, path, string(body))

		if i == 0 {
			// legacy API returns APIVersions
			group := metav1.APIVersions{}
			err = json.Unmarshal(body, &group)
			if err != nil {
				t.Errorf("[%d] unexpected error parsing json body at path %q: %v", i, path, err)
				continue
			}
		} else {
			// API groups return APIGroup
			group := metav1.APIGroup{}
			err = json.Unmarshal(body, &group)
			if err != nil {
				t.Errorf("[%d] unexpected error parsing json body at path %q: %v", i, path, err)
				continue
			}

			if got, expected := group.Name, info.GroupMeta.GroupVersion.Group; got != expected {
				t.Errorf("[%d] unexpected group name at path %q: got=%q expected=%q", i, path, got, expected)
				continue
			}

			if got, expected := group.PreferredVersion.Version, info.GroupMeta.GroupVersion.Version; got != expected {
				t.Errorf("[%d] unexpected group version at path %q: got=%q expected=%q", i, path, got, expected)
				continue
			}
		}

		// should serve APIResourceList at group path + /<group-version>
		path = path + "/" + info.GroupMeta.GroupVersion.Version
		resp, err = http.Get(server.URL + path)
		if err != nil {
			t.Errorf("[%d] unexpected error getting path %q path: %v", i, path, err)
			continue
		}

		body, err = ioutil.ReadAll(resp.Body)
		if err != nil {
			t.Errorf("[%d] unexpected error reading body at path %q: %v", i, path, err)
			continue
		}

		t.Logf("[%d] json at %s: %s", i, path, string(body))

		resources := metav1.APIResourceList{}
		err = json.Unmarshal(body, &resources)
		if err != nil {
			t.Errorf("[%d] unexpected error parsing json body at path %q: %v", i, path, err)
			continue
		}

		if got, expected := resources.GroupVersion, info.GroupMeta.GroupVersion.String(); got != expected {
			t.Errorf("[%d] unexpected groupVersion at path %q: got=%q expected=%q", i, path, got, expected)
			continue
		}

		// the verbs should match the features of resources
		for _, r := range resources.APIResources {
			switch r.Name {
			case "getter":
				if got, expected := sets.NewString([]string(r.Verbs)...), sets.NewString("get"); !got.Equal(expected) {
					t.Errorf("[%d] unexpected verbs for resource %s/%s: got=%v expected=%v", i, resources.GroupVersion, r.Name, got, expected)
				}
			case "noverbs":
				if r.Verbs == nil {
					t.Errorf("[%d] unexpected nil verbs slice. Expected: []string{}", i)
				}
				if got, expected := sets.NewString([]string(r.Verbs)...), sets.NewString(); !got.Equal(expected) {
					t.Errorf("[%d] unexpected verbs for resource %s/%s: got=%v expected=%v", i, resources.GroupVersion, r.Name, got, expected)
				}
			}
		}
	}
}

func TestPrepareRun(t *testing.T) {
	s, etcdserver, config, assert := newMaster(t)
	defer etcdserver.Terminate(t)

	assert.NotNil(config.SwaggerConfig)

	server := httptest.NewServer(s.Handler.Director)
	defer server.Close()
	done := make(chan struct{})

	s.PrepareRun()
	s.RunPostStartHooks(done)

	// swagger is installed in PrepareRun
	resp, err := http.Get(server.URL + "/swaggerapi/")
	assert.NoError(err)
	assert.Equal(http.StatusOK, resp.StatusCode)

	// healthz checks are installed in PrepareRun
	resp, err = http.Get(server.URL + "/healthz")
	assert.NoError(err)
	assert.Equal(http.StatusOK, resp.StatusCode)
	resp, err = http.Get(server.URL + "/healthz/ping")
	assert.NoError(err)
	assert.Equal(http.StatusOK, resp.StatusCode)
}

// TestCustomHandlerChain verifies the handler chain with custom handler chain builder functions.
func TestCustomHandlerChain(t *testing.T) {
	etcdserver, config, _ := setUp(t)
	defer etcdserver.Terminate(t)

	var protected, called bool

	config.BuildHandlerChainFunc = func(apiHandler http.Handler, c *Config) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			protected = true
			apiHandler.ServeHTTP(w, req)
		})
	}
	handler := http.HandlerFunc(func(r http.ResponseWriter, req *http.Request) {
		called = true
	})

	s, err := config.SkipComplete().New("test", EmptyDelegate)
	if err != nil {
		t.Fatalf("Error in bringing up the server: %v", err)
	}

	s.Handler.NonGoRestfulMux.Handle("/nonswagger", handler)
	s.Handler.NonGoRestfulMux.Handle("/secret", handler)

	type Test struct {
		handler   http.Handler
		path      string
		protected bool
	}
	for i, test := range []Test{
		{s.Handler, "/nonswagger", true},
		{s.Handler, "/secret", true},
	} {
		protected, called = false, false

		var w io.Reader
		req, err := http.NewRequest("GET", test.path, w)
		if err != nil {
			t.Errorf("%d: Unexpected http error: %v", i, err)
			continue
		}

		test.handler.ServeHTTP(httptest.NewRecorder(), req)

		if !called {
			t.Errorf("%d: Expected handler to be called.", i)
		}
		if test.protected != protected {
			t.Errorf("%d: Expected protected=%v, got protected=%v.", i, test.protected, protected)
		}
	}
}

// TestNotRestRoutesHaveAuth checks that special non-routes are behind authz/authn.
func TestNotRestRoutesHaveAuth(t *testing.T) {
	etcdserver, config, _ := setUp(t)
	defer etcdserver.Terminate(t)

	authz := mockAuthorizer{}

	config.LegacyAPIGroupPrefixes = sets.NewString("/apiPrefix")
	config.Authorizer = &authz

	config.EnableSwaggerUI = true
	config.EnableIndex = true
	config.EnableProfiling = true
	config.SwaggerConfig = DefaultSwaggerConfig()

	kubeVersion := fakeVersion()
	config.Version = &kubeVersion

	s, err := config.SkipComplete().New("test", EmptyDelegate)
	if err != nil {
		t.Fatalf("Error in bringing up the server: %v", err)
	}

	for _, test := range []struct {
		route string
	}{
		{"/"},
		{"/swagger-ui/"},
		{"/debug/pprof/"},
		{"/version"},
	} {
		resp := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", test.route, nil)
		s.Handler.ServeHTTP(resp, req)
		if resp.Code != 200 {
			t.Errorf("route %q expected to work: code %d", test.route, resp.Code)
			continue
		}

		if authz.lastURI != test.route {
			t.Errorf("route %q expected to go through authorization, last route did: %q", test.route, authz.lastURI)
		}
	}
}

type mockAuthorizer struct {
	lastURI string
}

func (authz *mockAuthorizer) Authorize(a authorizer.Attributes) (authorized bool, reason string, err error) {
	authz.lastURI = a.GetPath()
	return true, "", nil
}

type mockAuthenticator struct {
	lastURI string
}

func (authn *mockAuthenticator) AuthenticateRequest(req *http.Request) (user.Info, bool, error) {
	authn.lastURI = req.RequestURI
	return &user.DefaultInfo{
		Name: "foo",
	}, true, nil
}

func decodeResponse(resp *http.Response, obj interface{}) error {
	defer resp.Body.Close()

	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, obj); err != nil {
		return err
	}
	return nil
}

type testGetterStorage struct {
	Version string
}

func (p *testGetterStorage) New() runtime.Object {
	return &metav1.APIGroup{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Getter",
			APIVersion: p.Version,
		},
	}
}

func (p *testGetterStorage) Get(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	return nil, nil
}

type testNoVerbsStorage struct {
	Version string
}

func (p *testNoVerbsStorage) New() runtime.Object {
	return &metav1.APIGroup{
		TypeMeta: metav1.TypeMeta{
			Kind:       "NoVerbs",
			APIVersion: p.Version,
		},
	}
}

func fakeVersion() version.Info {
	return version.Info{
		Major:        "42",
		Minor:        "42",
		GitVersion:   "42",
		GitCommit:    "34973274ccef6ab4dfaaf86599792fa9c3fe4689",
		GitTreeState: "Dirty",
		BuildDate:    time.Now().String(),
		GoVersion:    goruntime.Version(),
		Compiler:     goruntime.Compiler,
		Platform:     fmt.Sprintf("%s/%s", goruntime.GOOS, goruntime.GOARCH),
	}
}
