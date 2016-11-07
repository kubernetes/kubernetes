/*
Copyright 2014 The Kubernetes Authors.

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

package master

import (
	"crypto/tls"
	"encoding/json"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apis/apps"
	appsapiv1beta1 "k8s.io/kubernetes/pkg/apis/apps/v1beta1"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	autoscalingapiv1 "k8s.io/kubernetes/pkg/apis/autoscaling/v1"
	"k8s.io/kubernetes/pkg/apis/batch"
	batchapiv1 "k8s.io/kubernetes/pkg/apis/batch/v1"
	batchapiv2alpha1 "k8s.io/kubernetes/pkg/apis/batch/v2alpha1"
	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/apis/extensions"
	extensionsapiv1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/restclient"
	openapigen "k8s.io/kubernetes/pkg/generated/openapi"
	"k8s.io/kubernetes/pkg/genericapiserver"
	kubeletclient "k8s.io/kubernetes/pkg/kubelet/client"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/runtime"
	etcdtesting "k8s.io/kubernetes/pkg/storage/etcd/testing"
	utilnet "k8s.io/kubernetes/pkg/util/net"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/version"

	"github.com/go-openapi/loads"
	"github.com/go-openapi/spec"
	"github.com/go-openapi/strfmt"
	"github.com/go-openapi/validate"
	"github.com/stretchr/testify/assert"
)

// setUp is a convience function for setting up for (most) tests.
func setUp(t *testing.T) (*Master, *etcdtesting.EtcdTestServer, Config, *assert.Assertions) {
	server, storageConfig := etcdtesting.NewUnsecuredEtcd3TestClientServer(t)

	config := &Config{
		GenericConfig:        genericapiserver.NewConfig(),
		APIServerServicePort: 443,
		MasterCount:          1,
	}

	resourceEncoding := genericapiserver.NewDefaultResourceEncodingConfig()
	resourceEncoding.SetVersionEncoding(api.GroupName, registered.GroupOrDie(api.GroupName).GroupVersion, unversioned.GroupVersion{Group: api.GroupName, Version: runtime.APIVersionInternal})
	resourceEncoding.SetVersionEncoding(autoscaling.GroupName, *testapi.Autoscaling.GroupVersion(), unversioned.GroupVersion{Group: autoscaling.GroupName, Version: runtime.APIVersionInternal})
	resourceEncoding.SetVersionEncoding(batch.GroupName, *testapi.Batch.GroupVersion(), unversioned.GroupVersion{Group: batch.GroupName, Version: runtime.APIVersionInternal})
	resourceEncoding.SetVersionEncoding(apps.GroupName, *testapi.Apps.GroupVersion(), unversioned.GroupVersion{Group: apps.GroupName, Version: runtime.APIVersionInternal})
	resourceEncoding.SetVersionEncoding(extensions.GroupName, *testapi.Extensions.GroupVersion(), unversioned.GroupVersion{Group: extensions.GroupName, Version: runtime.APIVersionInternal})
	resourceEncoding.SetVersionEncoding(rbac.GroupName, *testapi.Rbac.GroupVersion(), unversioned.GroupVersion{Group: rbac.GroupName, Version: runtime.APIVersionInternal})
	resourceEncoding.SetVersionEncoding(certificates.GroupName, *testapi.Certificates.GroupVersion(), unversioned.GroupVersion{Group: certificates.GroupName, Version: runtime.APIVersionInternal})
	storageFactory := genericapiserver.NewDefaultStorageFactory(*storageConfig, testapi.StorageMediaType(), api.Codecs, resourceEncoding, DefaultAPIResourceConfigSource())

	kubeVersion := version.Get()
	config.GenericConfig.Version = &kubeVersion
	config.StorageFactory = storageFactory
	config.GenericConfig.LoopbackClientConfig = &restclient.Config{APIPath: "/api", ContentConfig: restclient.ContentConfig{NegotiatedSerializer: api.Codecs}}
	config.GenericConfig.APIResourceConfigSource = DefaultAPIResourceConfigSource()
	config.GenericConfig.PublicAddress = net.ParseIP("192.168.10.4")
	config.GenericConfig.LegacyAPIGroupPrefixes = sets.NewString("/api")
	config.GenericConfig.APIResourceConfigSource = DefaultAPIResourceConfigSource()
	config.GenericConfig.RequestContextMapper = api.NewRequestContextMapper()
	config.GenericConfig.LoopbackClientConfig = &restclient.Config{APIPath: "/api", ContentConfig: restclient.ContentConfig{NegotiatedSerializer: api.Codecs}}
	config.GenericConfig.EnableMetrics = true
	config.EnableCoreControllers = false
	config.KubeletClientConfig = kubeletclient.KubeletClientConfig{Port: 10250}
	config.ProxyTransport = utilnet.SetTransportDefaults(&http.Transport{
		Dial:            func(network, addr string) (net.Conn, error) { return nil, nil },
		TLSClientConfig: &tls.Config{},
	})

	master, err := config.Complete().New()
	if err != nil {
		t.Fatal(err)
	}

	return master, server, *config, assert.New(t)
}

func newMaster(t *testing.T) (*Master, *etcdtesting.EtcdTestServer, Config, *assert.Assertions) {
	_, etcdserver, config, assert := setUp(t)

	master, err := config.Complete().New()
	if err != nil {
		t.Fatalf("Error in bringing up the master: %v", err)
	}

	return master, etcdserver, config, assert
}

// limitedAPIResourceConfigSource only enables the core group, the extensions group, the batch group, and the autoscaling group.
func limitedAPIResourceConfigSource() *genericapiserver.ResourceConfig {
	ret := genericapiserver.NewResourceConfig()
	ret.EnableVersions(
		apiv1.SchemeGroupVersion,
		extensionsapiv1beta1.SchemeGroupVersion,
		batchapiv1.SchemeGroupVersion,
		batchapiv2alpha1.SchemeGroupVersion,
		appsapiv1beta1.SchemeGroupVersion,
		autoscalingapiv1.SchemeGroupVersion,
	)
	return ret
}

// newLimitedMaster only enables the core group, the extensions group, the batch group, and the autoscaling group.
func newLimitedMaster(t *testing.T) (*Master, *etcdtesting.EtcdTestServer, Config, *assert.Assertions) {
	_, etcdserver, config, assert := setUp(t)
	config.GenericConfig.APIResourceConfigSource = limitedAPIResourceConfigSource()
	master, err := config.Complete().New()
	if err != nil {
		t.Fatalf("Error in bringing up the master: %v", err)
	}

	return master, etcdserver, config, assert
}

// TestVersion tests /version
func TestVersion(t *testing.T) {
	s, etcdserver, _, _ := newMaster(t)
	defer etcdserver.Terminate(t)

	req, _ := http.NewRequest("GET", "/version", nil)
	resp := httptest.NewRecorder()
	s.GenericAPIServer.InsecureHandler.ServeHTTP(resp, req)
	if resp.Code != 200 {
		t.Fatalf("expected http 200, got: %d", resp.Code)
	}

	var info version.Info
	err := json.NewDecoder(resp.Body).Decode(&info)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if !reflect.DeepEqual(version.Get(), info) {
		t.Errorf("Expected %#v, Got %#v", version.Get(), info)
	}
}

type fakeEndpointReconciler struct{}

func (*fakeEndpointReconciler) ReconcileEndpoints(serviceName string, ip net.IP, endpointPorts []api.EndpointPort, reconcilePorts bool) error {
	return nil
}

// TestGetNodeAddresses verifies that proper results are returned
// when requesting node addresses.
func TestGetNodeAddresses(t *testing.T) {
	assert := assert.New(t)

	fakeNodeClient := fake.NewSimpleClientset(registrytest.MakeNodeList([]string{"node1", "node2"}, api.NodeResources{})).Core().Nodes()
	addressProvider := nodeAddressProvider{fakeNodeClient}

	// Fail case (no addresses associated with nodes)
	nodes, _ := fakeNodeClient.List(api.ListOptions{})
	addrs, err := addressProvider.externalAddresses()

	assert.Error(err, "addresses should have caused an error as there are no addresses.")
	assert.Equal([]string(nil), addrs)

	// Pass case with External type IP
	nodes, _ = fakeNodeClient.List(api.ListOptions{})
	for index := range nodes.Items {
		nodes.Items[index].Status.Addresses = []api.NodeAddress{{Type: api.NodeExternalIP, Address: "127.0.0.1"}}
		fakeNodeClient.Update(&nodes.Items[index])
	}
	addrs, err = addressProvider.externalAddresses()
	assert.NoError(err, "addresses should not have returned an error.")
	assert.Equal([]string{"127.0.0.1", "127.0.0.1"}, addrs)

	// Pass case with LegacyHost type IP
	nodes, _ = fakeNodeClient.List(api.ListOptions{})
	for index := range nodes.Items {
		nodes.Items[index].Status.Addresses = []api.NodeAddress{{Type: api.NodeLegacyHostIP, Address: "127.0.0.2"}}
		fakeNodeClient.Update(&nodes.Items[index])
	}
	addrs, err = addressProvider.externalAddresses()
	assert.NoError(err, "addresses failback should not have returned an error.")
	assert.Equal([]string{"127.0.0.2", "127.0.0.2"}, addrs)
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

// Because we need to be backwards compatible with release 1.1, at endpoints
// that exist in release 1.1, the responses should have empty APIVersion.
func TestAPIVersionOfDiscoveryEndpoints(t *testing.T) {
	master, etcdserver, _, assert := newMaster(t)
	defer etcdserver.Terminate(t)

	server := httptest.NewServer(master.GenericAPIServer.HandlerContainer.ServeMux)

	// /api exists in release-1.1
	resp, err := http.Get(server.URL + "/api")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	apiVersions := unversioned.APIVersions{}
	assert.NoError(decodeResponse(resp, &apiVersions))
	assert.Equal(apiVersions.APIVersion, "")

	// /api/v1 exists in release-1.1
	resp, err = http.Get(server.URL + "/api/v1")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	resourceList := unversioned.APIResourceList{}
	assert.NoError(decodeResponse(resp, &resourceList))
	assert.Equal(resourceList.APIVersion, "")

	// /apis exists in release-1.1
	resp, err = http.Get(server.URL + "/apis")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	groupList := unversioned.APIGroupList{}
	assert.NoError(decodeResponse(resp, &groupList))
	assert.Equal(groupList.APIVersion, "")

	// /apis/extensions exists in release-1.1
	resp, err = http.Get(server.URL + "/apis/extensions")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	group := unversioned.APIGroup{}
	assert.NoError(decodeResponse(resp, &group))
	assert.Equal(group.APIVersion, "")

	// /apis/extensions/v1beta1 exists in release-1.1
	resp, err = http.Get(server.URL + "/apis/extensions/v1beta1")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	resourceList = unversioned.APIResourceList{}
	assert.NoError(decodeResponse(resp, &resourceList))
	assert.Equal(resourceList.APIVersion, "")

	// /apis/autoscaling doesn't exist in release-1.1, so the APIVersion field
	// should be non-empty in the results returned by the server.
	resp, err = http.Get(server.URL + "/apis/autoscaling")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	group = unversioned.APIGroup{}
	assert.NoError(decodeResponse(resp, &group))
	assert.Equal(group.APIVersion, "v1")

	// apis/autoscaling/v1 doesn't exist in release-1.1, so the APIVersion field
	// should be non-empty in the results returned by the server.

	resp, err = http.Get(server.URL + "/apis/autoscaling/v1")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	resourceList = unversioned.APIResourceList{}
	assert.NoError(decodeResponse(resp, &resourceList))
	assert.Equal(resourceList.APIVersion, "v1")

}

// TestValidOpenAPISpec verifies that the open api is added
// at the proper endpoint and the spec is valid.
func TestValidOpenAPISpec(t *testing.T) {
	_, etcdserver, config, assert := setUp(t)
	defer etcdserver.Terminate(t)

	config.GenericConfig.OpenAPIConfig.Definitions = openapigen.OpenAPIDefinitions
	config.GenericConfig.EnableOpenAPISupport = true
	config.GenericConfig.EnableIndex = true
	config.GenericConfig.OpenAPIConfig.Info = &spec.Info{
		InfoProps: spec.InfoProps{
			Title:   "Kubernetes",
			Version: "unversioned",
		},
	}
	master, err := config.Complete().New()
	if err != nil {
		t.Fatalf("Error in bringing up the master: %v", err)
	}

	// make sure swagger.json is not registered before calling PrepareRun.
	server := httptest.NewServer(master.GenericAPIServer.HandlerContainer.ServeMux)
	defer server.Close()
	resp, err := http.Get(server.URL + "/swagger.json")
	if !assert.NoError(err) {
		t.Errorf("unexpected error: %v", err)
	}
	assert.Equal(http.StatusNotFound, resp.StatusCode)

	master.GenericAPIServer.PrepareRun()

	resp, err = http.Get(server.URL + "/swagger.json")
	if !assert.NoError(err) {
		t.Errorf("unexpected error: %v", err)
	}
	assert.Equal(http.StatusOK, resp.StatusCode)

	// as json schema
	var sch spec.Schema
	if assert.NoError(decodeResponse(resp, &sch)) {
		validator := validate.NewSchemaValidator(spec.MustLoadSwagger20Schema(), nil, "", strfmt.Default)
		res := validator.Validate(&sch)
		assert.NoError(res.AsError())
	}

	// TODO(mehdy): The actual validation part of these tests are timing out on jerkin but passing locally. Enable it after debugging timeout issue.
	disableValidation := true

	// Validate OpenApi spec
	doc, err := loads.Spec(server.URL + "/swagger.json")
	if assert.NoError(err) {
		validator := validate.NewSpecValidator(doc.Schema(), strfmt.Default)
		if !disableValidation {
			res, warns := validator.Validate(doc)
			assert.NoError(res.AsError())
			if !warns.IsValid() {
				t.Logf("Open API spec on root has some warnings : %v", warns)
			}
		} else {
			t.Logf("Validation is disabled because it is timing out on jenkins put passing locally.")
		}
	}
}
