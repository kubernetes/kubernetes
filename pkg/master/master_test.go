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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/sets"
	genericapirequest "k8s.io/apiserver/pkg/request"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
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
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/genericapiserver"
	kubeletclient "k8s.io/kubernetes/pkg/kubelet/client"
	etcdtesting "k8s.io/kubernetes/pkg/storage/etcd/testing"
	"k8s.io/kubernetes/pkg/version"

	"github.com/stretchr/testify/assert"
)

// setUp is a convience function for setting up for (most) tests.
func setUp(t *testing.T) (*Master, *etcdtesting.EtcdTestServer, Config, *assert.Assertions) {
	server, storageConfig := etcdtesting.NewUnsecuredEtcd3TestClientServer(t)

	config := &Config{
		GenericConfig:           genericapiserver.NewConfig(),
		APIResourceConfigSource: DefaultAPIResourceConfigSource(),
		APIServerServicePort:    443,
		MasterCount:             1,
	}

	resourceEncoding := genericapiserver.NewDefaultResourceEncodingConfig()
	resourceEncoding.SetVersionEncoding(api.GroupName, api.Registry.GroupOrDie(api.GroupName).GroupVersion, schema.GroupVersion{Group: api.GroupName, Version: runtime.APIVersionInternal})
	resourceEncoding.SetVersionEncoding(autoscaling.GroupName, *testapi.Autoscaling.GroupVersion(), schema.GroupVersion{Group: autoscaling.GroupName, Version: runtime.APIVersionInternal})
	resourceEncoding.SetVersionEncoding(batch.GroupName, *testapi.Batch.GroupVersion(), schema.GroupVersion{Group: batch.GroupName, Version: runtime.APIVersionInternal})
	resourceEncoding.SetVersionEncoding(apps.GroupName, *testapi.Apps.GroupVersion(), schema.GroupVersion{Group: apps.GroupName, Version: runtime.APIVersionInternal})
	resourceEncoding.SetVersionEncoding(extensions.GroupName, *testapi.Extensions.GroupVersion(), schema.GroupVersion{Group: extensions.GroupName, Version: runtime.APIVersionInternal})
	resourceEncoding.SetVersionEncoding(rbac.GroupName, *testapi.Rbac.GroupVersion(), schema.GroupVersion{Group: rbac.GroupName, Version: runtime.APIVersionInternal})
	resourceEncoding.SetVersionEncoding(certificates.GroupName, *testapi.Certificates.GroupVersion(), schema.GroupVersion{Group: certificates.GroupName, Version: runtime.APIVersionInternal})
	storageFactory := genericapiserver.NewDefaultStorageFactory(*storageConfig, testapi.StorageMediaType(), api.Codecs, resourceEncoding, DefaultAPIResourceConfigSource())

	kubeVersion := version.Get()
	config.GenericConfig.Version = &kubeVersion
	config.StorageFactory = storageFactory
	config.GenericConfig.LoopbackClientConfig = &restclient.Config{APIPath: "/api", ContentConfig: restclient.ContentConfig{NegotiatedSerializer: api.Codecs}}
	config.GenericConfig.PublicAddress = net.ParseIP("192.168.10.4")
	config.GenericConfig.LegacyAPIGroupPrefixes = sets.NewString("/api")
	config.GenericConfig.RequestContextMapper = genericapirequest.NewRequestContextMapper()
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
	config.APIResourceConfigSource = limitedAPIResourceConfigSource()
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

func makeNodeList(nodes []string, nodeResources apiv1.NodeResources) *apiv1.NodeList {
	list := apiv1.NodeList{
		Items: make([]apiv1.Node, len(nodes)),
	}
	for i := range nodes {
		list.Items[i].Name = nodes[i]
		list.Items[i].Status.Capacity = nodeResources.Capacity
	}
	return &list
}

// TestGetNodeAddresses verifies that proper results are returned
// when requesting node addresses.
func TestGetNodeAddresses(t *testing.T) {
	assert := assert.New(t)

	fakeNodeClient := fake.NewSimpleClientset(makeNodeList([]string{"node1", "node2"}, apiv1.NodeResources{})).Core().Nodes()
	addressProvider := nodeAddressProvider{fakeNodeClient}

	// Fail case (no addresses associated with nodes)
	nodes, _ := fakeNodeClient.List(apiv1.ListOptions{})
	addrs, err := addressProvider.externalAddresses()

	assert.Error(err, "addresses should have caused an error as there are no addresses.")
	assert.Equal([]string(nil), addrs)

	// Pass case with External type IP
	nodes, _ = fakeNodeClient.List(apiv1.ListOptions{})
	for index := range nodes.Items {
		nodes.Items[index].Status.Addresses = []apiv1.NodeAddress{{Type: apiv1.NodeExternalIP, Address: "127.0.0.1"}}
		fakeNodeClient.Update(&nodes.Items[index])
	}
	addrs, err = addressProvider.externalAddresses()
	assert.NoError(err, "addresses should not have returned an error.")
	assert.Equal([]string{"127.0.0.1", "127.0.0.1"}, addrs)

	// Pass case with LegacyHost type IP
	nodes, _ = fakeNodeClient.List(apiv1.ListOptions{})
	for index := range nodes.Items {
		nodes.Items[index].Status.Addresses = []apiv1.NodeAddress{{Type: apiv1.NodeLegacyHostIP, Address: "127.0.0.2"}}
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
	apiVersions := metav1.APIVersions{}
	assert.NoError(decodeResponse(resp, &apiVersions))
	assert.Equal(apiVersions.APIVersion, "")

	// /api/v1 exists in release-1.1
	resp, err = http.Get(server.URL + "/api/v1")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	resourceList := metav1.APIResourceList{}
	assert.NoError(decodeResponse(resp, &resourceList))
	assert.Equal(resourceList.APIVersion, "")

	// /apis exists in release-1.1
	resp, err = http.Get(server.URL + "/apis")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	groupList := metav1.APIGroupList{}
	assert.NoError(decodeResponse(resp, &groupList))
	assert.Equal(groupList.APIVersion, "")

	// /apis/extensions exists in release-1.1
	resp, err = http.Get(server.URL + "/apis/extensions")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	group := metav1.APIGroup{}
	assert.NoError(decodeResponse(resp, &group))
	assert.Equal(group.APIVersion, "")

	// /apis/extensions/v1beta1 exists in release-1.1
	resp, err = http.Get(server.URL + "/apis/extensions/v1beta1")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	resourceList = metav1.APIResourceList{}
	assert.NoError(decodeResponse(resp, &resourceList))
	assert.Equal(resourceList.APIVersion, "")

	// /apis/autoscaling doesn't exist in release-1.1, so the APIVersion field
	// should be non-empty in the results returned by the server.
	resp, err = http.Get(server.URL + "/apis/autoscaling")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	group = metav1.APIGroup{}
	assert.NoError(decodeResponse(resp, &group))
	assert.Equal(group.APIVersion, "v1")

	// apis/autoscaling/v1 doesn't exist in release-1.1, so the APIVersion field
	// should be non-empty in the results returned by the server.

	resp, err = http.Get(server.URL + "/apis/autoscaling/v1")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	resourceList = metav1.APIResourceList{}
	assert.NoError(decodeResponse(resp, &resourceList))
	assert.Equal(resourceList.APIVersion, "v1")

}
