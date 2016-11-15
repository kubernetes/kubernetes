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

package genericapiserver

import (
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/auth/authorizer"
	"k8s.io/kubernetes/pkg/auth/user"
	openapigen "k8s.io/kubernetes/pkg/generated/openapi"
	etcdtesting "k8s.io/kubernetes/pkg/storage/etcd/testing"
	utilnet "k8s.io/kubernetes/pkg/util/net"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/version"

	"github.com/go-openapi/spec"
	"github.com/stretchr/testify/assert"
)

// setUp is a convience function for setting up for (most) tests.
func setUp(t *testing.T) (*etcdtesting.EtcdTestServer, Config, *assert.Assertions) {
	etcdServer, _ := etcdtesting.NewUnsecuredEtcd3TestClientServer(t)

	config := NewConfig()
	config.PublicAddress = net.ParseIP("192.168.10.4")
	config.RequestContextMapper = api.NewRequestContextMapper()
	config.LegacyAPIGroupPrefixes = sets.NewString("/api")

	config.EnableOpenAPISupport = true
	config.EnableSwaggerSupport = true
	config.OpenAPIConfig.Definitions = openapigen.OpenAPIDefinitions
	config.OpenAPIConfig.Info = &spec.Info{
		InfoProps: spec.InfoProps{
			Title:   "Kubernetes",
			Version: "unversioned",
		},
	}

	return etcdServer, *config, assert.New(t)
}

func newMaster(t *testing.T) (*GenericAPIServer, *etcdtesting.EtcdTestServer, Config, *assert.Assertions) {
	etcdserver, config, assert := setUp(t)

	s, err := config.Complete().New()
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
	assert.Equal(s.enableSwaggerSupport, config.EnableSwaggerSupport)
	assert.Equal(s.legacyAPIGroupPrefixes, config.LegacyAPIGroupPrefixes)
	assert.Equal(s.admissionControl, config.AdmissionControl)
	assert.Equal(s.RequestContextMapper(), config.RequestContextMapper)

	// these values get defaulted
	assert.Equal(s.ExternalAddress, net.JoinHostPort(config.PublicAddress.String(), "6443"))
}

// Verifies that AddGroupVersions works as expected.
func TestInstallAPIGroups(t *testing.T) {
	etcdserver, config, assert := setUp(t)
	defer etcdserver.Terminate(t)

	config.LegacyAPIGroupPrefixes = sets.NewString("/apiPrefix")

	s, err := config.SkipComplete().New()
	if err != nil {
		t.Fatalf("Error in bringing up the server: %v", err)
	}

	apiGroupMeta := registered.GroupOrDie(api.GroupName)
	extensionsGroupMeta := registered.GroupOrDie(extensions.GroupName)
	s.InstallLegacyAPIGroup("/apiPrefix", &APIGroupInfo{
		// legacy group version
		GroupMeta:                    *apiGroupMeta,
		VersionedResourcesStorageMap: map[string]map[string]rest.Storage{},
		ParameterCodec:               api.ParameterCodec,
		NegotiatedSerializer:         api.Codecs,
	})

	apiGroupsInfo := []APIGroupInfo{
		{
			// extensions group version
			GroupMeta:                    *extensionsGroupMeta,
			VersionedResourcesStorageMap: map[string]map[string]rest.Storage{},
			OptionsExternalVersion:       &apiGroupMeta.GroupVersion,
			ParameterCodec:               api.ParameterCodec,
			NegotiatedSerializer:         api.Codecs,
		},
	}
	for i := range apiGroupsInfo {
		s.InstallAPIGroup(&apiGroupsInfo[i])
	}

	server := httptest.NewServer(s.InsecureHandler)
	defer server.Close()
	validPaths := []string{
		// "/api"
		config.LegacyAPIGroupPrefixes.List()[0],
		// "/api/v1"
		config.LegacyAPIGroupPrefixes.List()[0] + "/" + apiGroupMeta.GroupVersion.Version,
		// "/apis/extensions"
		APIGroupPrefix + "/" + extensionsGroupMeta.GroupVersion.Group,
		// "/apis/extensions/v1beta1"
		APIGroupPrefix + "/" + extensionsGroupMeta.GroupVersion.String(),
	}
	for _, path := range validPaths {
		_, err := http.Get(server.URL + path)
		if !assert.NoError(err) {
			t.Errorf("unexpected error: %v, for path: %s", err, path)
		}
	}
}

func TestPrepareRun(t *testing.T) {
	s, etcdserver, config, assert := newMaster(t)
	defer etcdserver.Terminate(t)

	assert.True(config.EnableSwaggerSupport)
	assert.True(config.EnableOpenAPISupport)

	server := httptest.NewServer(s.HandlerContainer.ServeMux)
	defer server.Close()

	s.PrepareRun()

	// openapi is installed in PrepareRun
	resp, err := http.Get(server.URL + "/swagger.json")
	assert.NoError(err)
	assert.Equal(http.StatusOK, resp.StatusCode)

	// swagger is installed in PrepareRun
	resp, err = http.Get(server.URL + "/swaggerapi/")
	assert.NoError(err)
	assert.Equal(http.StatusOK, resp.StatusCode)
}

// TestCustomHandlerChain verifies the handler chain with custom handler chain builder functions.
func TestCustomHandlerChain(t *testing.T) {
	etcdserver, config, _ := setUp(t)
	defer etcdserver.Terminate(t)

	var protected, called bool

	config.BuildHandlerChainsFunc = func(apiHandler http.Handler, c *Config) (secure, insecure http.Handler) {
		return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				protected = true
				apiHandler.ServeHTTP(w, req)
			}), http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				protected = false
				apiHandler.ServeHTTP(w, req)
			})
	}
	handler := http.HandlerFunc(func(r http.ResponseWriter, req *http.Request) {
		called = true
	})

	s, err := config.SkipComplete().New()
	if err != nil {
		t.Fatalf("Error in bringing up the server: %v", err)
	}

	s.HandlerContainer.NonSwaggerRoutes.Handle("/nonswagger", handler)
	s.HandlerContainer.SecretRoutes.Handle("/secret", handler)

	type Test struct {
		handler   http.Handler
		path      string
		protected bool
	}
	for i, test := range []Test{
		{s.Handler, "/nonswagger", true},
		{s.Handler, "/secret", true},
		{s.InsecureHandler, "/nonswagger", false},
		{s.InsecureHandler, "/secret", false},
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
	config.EnableSwaggerSupport = true

	kubeVersion := version.Get()
	config.Version = &kubeVersion

	s, err := config.SkipComplete().New()
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

func getGroupList(server *httptest.Server) (*unversioned.APIGroupList, error) {
	resp, err := http.Get(server.URL + "/apis")
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected server response, expected %d, actual: %d", http.StatusOK, resp.StatusCode)
	}

	groupList := unversioned.APIGroupList{}
	err = decodeResponse(resp, &groupList)
	return &groupList, err
}

func TestDiscoveryAtAPIS(t *testing.T) {
	master, etcdserver, _, assert := newMaster(t)
	defer etcdserver.Terminate(t)

	server := httptest.NewServer(master.InsecureHandler)
	groupList, err := getGroupList(server)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assert.Equal(0, len(groupList.Groups))

	// Add a Group.
	extensionsVersions := []unversioned.GroupVersionForDiscovery{
		{
			GroupVersion: testapi.Extensions.GroupVersion().String(),
			Version:      testapi.Extensions.GroupVersion().Version,
		},
	}
	extensionsPreferredVersion := unversioned.GroupVersionForDiscovery{
		GroupVersion: extensions.GroupName + "/preferred",
		Version:      "preferred",
	}
	master.AddAPIGroupForDiscovery(unversioned.APIGroup{
		Name:             extensions.GroupName,
		Versions:         extensionsVersions,
		PreferredVersion: extensionsPreferredVersion,
	})

	groupList, err = getGroupList(server)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	assert.Equal(1, len(groupList.Groups))
	groupListGroup := groupList.Groups[0]
	assert.Equal(extensions.GroupName, groupListGroup.Name)
	assert.Equal(extensionsVersions, groupListGroup.Versions)
	assert.Equal(extensionsPreferredVersion, groupListGroup.PreferredVersion)
	assert.Equal(master.discoveryAddresses.ServerAddressByClientCIDRs(utilnet.GetClientIP(&http.Request{})), groupListGroup.ServerAddressByClientCIDRs)

	// Remove the group.
	master.RemoveAPIGroupForDiscovery(extensions.GroupName)
	groupList, err = getGroupList(server)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	assert.Equal(0, len(groupList.Groups))
}

func TestGetServerAddressByClientCIDRs(t *testing.T) {
	publicAddressCIDRMap := []unversioned.ServerAddressByClientCIDR{
		{
			ClientCIDR:    "0.0.0.0/0",
			ServerAddress: "ExternalAddress",
		},
	}
	internalAddressCIDRMap := []unversioned.ServerAddressByClientCIDR{
		publicAddressCIDRMap[0],
		{
			ClientCIDR:    "10.0.0.0/24",
			ServerAddress: "serviceIP",
		},
	}
	internalIP := "10.0.0.1"
	publicIP := "1.1.1.1"
	testCases := []struct {
		Request     http.Request
		ExpectedMap []unversioned.ServerAddressByClientCIDR
	}{
		{
			Request:     http.Request{},
			ExpectedMap: publicAddressCIDRMap,
		},
		{
			Request: http.Request{
				Header: map[string][]string{
					"X-Real-Ip": {internalIP},
				},
			},
			ExpectedMap: internalAddressCIDRMap,
		},
		{
			Request: http.Request{
				Header: map[string][]string{
					"X-Real-Ip": {publicIP},
				},
			},
			ExpectedMap: publicAddressCIDRMap,
		},
		{
			Request: http.Request{
				Header: map[string][]string{
					"X-Forwarded-For": {internalIP},
				},
			},
			ExpectedMap: internalAddressCIDRMap,
		},
		{
			Request: http.Request{
				Header: map[string][]string{
					"X-Forwarded-For": {publicIP},
				},
			},
			ExpectedMap: publicAddressCIDRMap,
		},

		{
			Request: http.Request{
				RemoteAddr: internalIP,
			},
			ExpectedMap: internalAddressCIDRMap,
		},
		{
			Request: http.Request{
				RemoteAddr: publicIP,
			},
			ExpectedMap: publicAddressCIDRMap,
		},
		{
			Request: http.Request{
				RemoteAddr: "invalidIP",
			},
			ExpectedMap: publicAddressCIDRMap,
		},
	}

	_, ipRange, _ := net.ParseCIDR("10.0.0.0/24")
	discoveryAddresses := DefaultDiscoveryAddresses{DefaultAddress: "ExternalAddress"}
	discoveryAddresses.DiscoveryCIDRRules = append(discoveryAddresses.DiscoveryCIDRRules,
		DiscoveryCIDRRule{IPRange: *ipRange, Address: "serviceIP"})

	for i, test := range testCases {
		if a, e := discoveryAddresses.ServerAddressByClientCIDRs(utilnet.GetClientIP(&test.Request)), test.ExpectedMap; reflect.DeepEqual(e, a) != true {
			t.Fatalf("test case %d failed. expected: %v, actual: %v", i+1, e, a)
		}
	}
}
