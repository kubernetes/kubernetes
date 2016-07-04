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
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strconv"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"

	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apiserver"
	etcdtesting "k8s.io/kubernetes/pkg/storage/etcd/testing"
	utilnet "k8s.io/kubernetes/pkg/util/net"

	"github.com/stretchr/testify/assert"
)

// setUp is a convience function for setting up for (most) tests.
func setUp(t *testing.T) (GenericAPIServer, *etcdtesting.EtcdTestServer, Config, *assert.Assertions) {
	etcdServer := etcdtesting.NewEtcdTestClientServer(t)

	genericapiserver := GenericAPIServer{}
	config := Config{}
	config.PublicAddress = net.ParseIP("192.168.10.4")

	return genericapiserver, etcdServer, config, assert.New(t)
}

func newMaster(t *testing.T) (*GenericAPIServer, *etcdtesting.EtcdTestServer, Config, *assert.Assertions) {
	_, etcdserver, config, assert := setUp(t)

	config.ProxyDialer = func(network, addr string) (net.Conn, error) { return nil, nil }
	config.ProxyTLSClientConfig = &tls.Config{}
	config.Serializer = api.Codecs
	config.APIPrefix = "/api"
	config.APIGroupPrefix = "/apis"

	s, err := New(&config)
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
	assert.Equal(s.enableLogsSupport, config.EnableLogsSupport)
	assert.Equal(s.enableUISupport, config.EnableUISupport)
	assert.Equal(s.enableSwaggerSupport, config.EnableSwaggerSupport)
	assert.Equal(s.enableSwaggerUI, config.EnableSwaggerUI)
	assert.Equal(s.enableProfiling, config.EnableProfiling)
	assert.Equal(s.APIPrefix, config.APIPrefix)
	assert.Equal(s.APIGroupPrefix, config.APIGroupPrefix)
	assert.Equal(s.corsAllowedOriginList, config.CorsAllowedOriginList)
	assert.Equal(s.authenticator, config.Authenticator)
	assert.Equal(s.authorizer, config.Authorizer)
	assert.Equal(s.AdmissionControl, config.AdmissionControl)
	assert.Equal(s.RequestContextMapper, config.RequestContextMapper)
	assert.Equal(s.cacheTimeout, config.CacheTimeout)
	assert.Equal(s.ExternalAddress, config.ExternalHost)
	assert.Equal(s.ClusterIP, config.PublicAddress)
	assert.Equal(s.PublicReadWritePort, config.ReadWritePort)
	assert.Equal(s.ServiceReadWriteIP, config.ServiceReadWriteIP)

	// These functions should point to the same memory location
	serverDialer, _ := utilnet.Dialer(s.ProxyTransport)
	serverDialerFunc := fmt.Sprintf("%p", serverDialer)
	configDialerFunc := fmt.Sprintf("%p", config.ProxyDialer)
	assert.Equal(serverDialerFunc, configDialerFunc)

	assert.Equal(s.ProxyTransport.(*http.Transport).TLSClientConfig, config.ProxyTLSClientConfig)
}

// Verifies that AddGroupVersions works as expected.
func TestInstallAPIGroups(t *testing.T) {
	_, etcdserver, config, assert := setUp(t)
	defer etcdserver.Terminate(t)

	config.ProxyDialer = func(network, addr string) (net.Conn, error) { return nil, nil }
	config.ProxyTLSClientConfig = &tls.Config{}
	config.APIPrefix = "/apiPrefix"
	config.APIGroupPrefix = "/apiGroupPrefix"
	config.Serializer = api.Codecs

	s, err := New(&config)
	if err != nil {
		t.Fatalf("Error in bringing up the server: %v", err)
	}

	apiGroupMeta := registered.GroupOrDie(api.GroupName)
	extensionsGroupMeta := registered.GroupOrDie(extensions.GroupName)
	apiGroupsInfo := []APIGroupInfo{
		{
			// legacy group version
			GroupMeta:                    *apiGroupMeta,
			VersionedResourcesStorageMap: map[string]map[string]rest.Storage{},
			IsLegacyGroup:                true,
			ParameterCodec:               api.ParameterCodec,
			NegotiatedSerializer:         api.Codecs,
		},
		{
			// extensions group version
			GroupMeta:                    *extensionsGroupMeta,
			VersionedResourcesStorageMap: map[string]map[string]rest.Storage{},
			OptionsExternalVersion:       &apiGroupMeta.GroupVersion,
			ParameterCodec:               api.ParameterCodec,
			NegotiatedSerializer:         api.Codecs,
		},
	}
	s.InstallAPIGroups(apiGroupsInfo)

	server := httptest.NewServer(s.HandlerContainer.ServeMux)
	defer server.Close()
	validPaths := []string{
		// "/api"
		config.APIPrefix,
		// "/api/v1"
		config.APIPrefix + "/" + apiGroupMeta.GroupVersion.Version,
		// "/apis/extensions"
		config.APIGroupPrefix + "/" + extensionsGroupMeta.GroupVersion.Group,
		// "/apis/extensions/v1beta1"
		config.APIGroupPrefix + "/" + extensionsGroupMeta.GroupVersion.String(),
	}
	for _, path := range validPaths {
		_, err := http.Get(server.URL + path)
		if !assert.NoError(err) {
			t.Errorf("unexpected error: %v, for path: %s", err, path)
		}
	}
}

// TestNewHandlerContainer verifies that NewHandlerContainer uses the
// mux provided
func TestNewHandlerContainer(t *testing.T) {
	assert := assert.New(t)
	mux := http.NewServeMux()
	container := NewHandlerContainer(mux, nil)
	assert.Equal(mux, container.ServeMux, "ServerMux's do not match")
}

// TestHandleWithAuth verifies HandleWithAuth adds the path
// to the MuxHelper.RegisteredPaths.
func TestHandleWithAuth(t *testing.T) {
	server, etcdserver, _, assert := setUp(t)
	defer etcdserver.Terminate(t)

	mh := apiserver.MuxHelper{Mux: http.NewServeMux()}
	server.MuxHelper = &mh
	handler := func(r http.ResponseWriter, w *http.Request) { w.Write(nil) }
	server.HandleWithAuth("/test", http.HandlerFunc(handler))

	assert.Contains(server.MuxHelper.RegisteredPaths, "/test", "Path not found in MuxHelper")
}

// TestHandleFuncWithAuth verifies HandleFuncWithAuth adds the path
// to the MuxHelper.RegisteredPaths.
func TestHandleFuncWithAuth(t *testing.T) {
	server, etcdserver, _, assert := setUp(t)
	defer etcdserver.Terminate(t)

	mh := apiserver.MuxHelper{Mux: http.NewServeMux()}
	server.MuxHelper = &mh
	handler := func(r http.ResponseWriter, w *http.Request) { w.Write(nil) }
	server.HandleFuncWithAuth("/test", handler)

	assert.Contains(server.MuxHelper.RegisteredPaths, "/test", "Path not found in MuxHelper")
}

// TestInstallSwaggerAPI verifies that the swagger api is added
// at the proper endpoint.
func TestInstallSwaggerAPI(t *testing.T) {
	server, etcdserver, _, assert := setUp(t)
	defer etcdserver.Terminate(t)

	mux := http.NewServeMux()
	server.HandlerContainer = NewHandlerContainer(mux, nil)

	// Ensure swagger isn't installed without the call
	ws := server.HandlerContainer.RegisteredWebServices()
	if !assert.Equal(len(ws), 0) {
		for x := range ws {
			assert.NotEqual("/swaggerapi", ws[x].RootPath(), "SwaggerAPI was installed without a call to InstallSwaggerAPI()")
		}
	}

	// Install swagger and test
	server.InstallSwaggerAPI()
	ws = server.HandlerContainer.RegisteredWebServices()
	if assert.NotEqual(0, len(ws), "SwaggerAPI not installed.") {
		assert.Equal("/swaggerapi/", ws[0].RootPath(), "SwaggerAPI did not install to the proper path. %s != /swaggerapi", ws[0].RootPath())
	}

	// Empty externalHost verification
	mux = http.NewServeMux()
	server.HandlerContainer = NewHandlerContainer(mux, nil)
	server.ExternalAddress = ""
	server.ClusterIP = net.IPv4(10, 10, 10, 10)
	server.PublicReadWritePort = 1010
	server.InstallSwaggerAPI()
	if assert.NotEqual(0, len(ws), "SwaggerAPI not installed.") {
		assert.Equal("/swaggerapi/", ws[0].RootPath(), "SwaggerAPI did not install to the proper path. %s != /swaggerapi", ws[0].RootPath())
	}
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

	server := httptest.NewServer(master.HandlerContainer.ServeMux)
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
	assert.Equal(master.getServerAddressByClientCIDRs(&http.Request{}), groupListGroup.ServerAddressByClientCIDRs)

	// Remove the group.
	master.RemoveAPIGroupForDiscovery(extensions.GroupName)
	groupList, err = getGroupList(server)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	assert.Equal(0, len(groupList.Groups))
}

func TestGetServerAddressByClientCIDRs(t *testing.T) {
	s, etcdserver, _, _ := newMaster(t)
	defer etcdserver.Terminate(t)

	publicAddressCIDRMap := []unversioned.ServerAddressByClientCIDR{
		{
			ClientCIDR: "0.0.0.0/0",

			ServerAddress: s.ExternalAddress,
		},
	}
	internalAddressCIDRMap := []unversioned.ServerAddressByClientCIDR{
		publicAddressCIDRMap[0],
		{
			ClientCIDR:    s.ServiceClusterIPRange.String(),
			ServerAddress: net.JoinHostPort(s.ServiceReadWriteIP.String(), strconv.Itoa(s.ServiceReadWritePort)),
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

	for i, test := range testCases {
		if a, e := s.getServerAddressByClientCIDRs(&test.Request), test.ExpectedMap; reflect.DeepEqual(e, a) != true {
			t.Fatalf("test case %d failed. expected: %v, actual: %v", i+1, e, a)
		}
	}
}
