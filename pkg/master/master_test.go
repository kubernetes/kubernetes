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
	"strings"
	"testing"

	appsapiv1beta1 "k8s.io/api/apps/v1beta1"
	autoscalingapiv1 "k8s.io/api/autoscaling/v1"
	batchapiv1 "k8s.io/api/batch/v1"
	batchapiv1beta1 "k8s.io/api/batch/v1beta1"
	certificatesapiv1beta1 "k8s.io/api/certificates/v1beta1"
	apiv1 "k8s.io/api/core/v1"
	extensionsapiv1beta1 "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/version"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/options"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	etcdtesting "k8s.io/apiserver/pkg/storage/etcd/testing"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apis/rbac"
	kubeletclient "k8s.io/kubernetes/pkg/kubelet/client"
	"k8s.io/kubernetes/pkg/master/reconcilers"
	certificatesrest "k8s.io/kubernetes/pkg/registry/certificates/rest"
	corerest "k8s.io/kubernetes/pkg/registry/core/rest"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	kubeversion "k8s.io/kubernetes/pkg/version"

	"github.com/stretchr/testify/assert"
)

// setUp is a convience function for setting up for (most) tests.
func setUp(t *testing.T) (*etcdtesting.EtcdTestServer, Config, informers.SharedInformerFactory, *assert.Assertions) {
	server, storageConfig := etcdtesting.NewUnsecuredEtcd3TestClientServer(t, legacyscheme.Scheme)

	config := &Config{
		GenericConfig: genericapiserver.NewConfig(legacyscheme.Codecs),
		ExtraConfig: ExtraConfig{
			APIResourceConfigSource: DefaultAPIResourceConfigSource(),
			APIServerServicePort:    443,
			MasterCount:             1,
			EndpointReconcilerType:  reconcilers.MasterCountReconcilerType,
		},
	}

	resourceEncoding := serverstorage.NewDefaultResourceEncodingConfig(legacyscheme.Registry)
	resourceEncoding.SetVersionEncoding(api.GroupName, legacyscheme.Registry.GroupOrDie(api.GroupName).GroupVersion, schema.GroupVersion{Group: api.GroupName, Version: runtime.APIVersionInternal})
	resourceEncoding.SetVersionEncoding(autoscaling.GroupName, *testapi.Autoscaling.GroupVersion(), schema.GroupVersion{Group: autoscaling.GroupName, Version: runtime.APIVersionInternal})
	resourceEncoding.SetVersionEncoding(batch.GroupName, *testapi.Batch.GroupVersion(), schema.GroupVersion{Group: batch.GroupName, Version: runtime.APIVersionInternal})
	// FIXME (soltysh): this GroupVersionResource override should be configurable
	resourceEncoding.SetResourceEncoding(schema.GroupResource{Group: "batch", Resource: "cronjobs"}, schema.GroupVersion{Group: batch.GroupName, Version: "v1beta1"}, schema.GroupVersion{Group: batch.GroupName, Version: runtime.APIVersionInternal})
	resourceEncoding.SetVersionEncoding(apps.GroupName, *testapi.Apps.GroupVersion(), schema.GroupVersion{Group: apps.GroupName, Version: runtime.APIVersionInternal})
	resourceEncoding.SetVersionEncoding(extensions.GroupName, *testapi.Extensions.GroupVersion(), schema.GroupVersion{Group: extensions.GroupName, Version: runtime.APIVersionInternal})
	resourceEncoding.SetVersionEncoding(rbac.GroupName, *testapi.Rbac.GroupVersion(), schema.GroupVersion{Group: rbac.GroupName, Version: runtime.APIVersionInternal})
	resourceEncoding.SetVersionEncoding(certificates.GroupName, *testapi.Certificates.GroupVersion(), schema.GroupVersion{Group: certificates.GroupName, Version: runtime.APIVersionInternal})
	storageFactory := serverstorage.NewDefaultStorageFactory(*storageConfig, testapi.StorageMediaType(), legacyscheme.Codecs, resourceEncoding, DefaultAPIResourceConfigSource(), nil)

	err := options.NewEtcdOptions(storageConfig).ApplyWithStorageFactoryTo(storageFactory, config.GenericConfig)
	if err != nil {
		t.Fatal(err)
	}

	kubeVersion := kubeversion.Get()
	config.GenericConfig.Version = &kubeVersion
	config.ExtraConfig.StorageFactory = storageFactory
	config.GenericConfig.LoopbackClientConfig = &restclient.Config{APIPath: "/api", ContentConfig: restclient.ContentConfig{NegotiatedSerializer: legacyscheme.Codecs}}
	config.GenericConfig.PublicAddress = net.ParseIP("192.168.10.4")
	config.GenericConfig.LegacyAPIGroupPrefixes = sets.NewString("/api")
	config.GenericConfig.RequestContextMapper = genericapirequest.NewRequestContextMapper()
	config.GenericConfig.LoopbackClientConfig = &restclient.Config{APIPath: "/api", ContentConfig: restclient.ContentConfig{NegotiatedSerializer: legacyscheme.Codecs}}
	config.GenericConfig.EnableMetrics = true
	config.ExtraConfig.EnableCoreControllers = false
	config.ExtraConfig.KubeletClientConfig = kubeletclient.KubeletClientConfig{Port: 10250}
	config.ExtraConfig.ProxyTransport = utilnet.SetTransportDefaults(&http.Transport{
		Dial:            func(network, addr string) (net.Conn, error) { return nil, nil },
		TLSClientConfig: &tls.Config{},
	})

	clientset, err := kubernetes.NewForConfig(config.GenericConfig.LoopbackClientConfig)
	if err != nil {
		t.Fatalf("unable to create client set due to %v", err)
	}
	sharedInformers := informers.NewSharedInformerFactory(clientset, config.GenericConfig.LoopbackClientConfig.Timeout)

	return server, *config, sharedInformers, assert.New(t)
}

// TestLegacyRestStorageStrategies ensures that all Storage objects which are using the generic registry Store have
// their various strategies properly wired up. This surfaced as a bug where strategies defined Export functions, but
// they were never used outside of unit tests because the export strategies were not assigned inside the Store.
func TestLegacyRestStorageStrategies(t *testing.T) {
	_, etcdserver, masterCfg, _ := newMaster(t)
	defer etcdserver.Terminate(t)

	storageProvider := corerest.LegacyRESTStorageProvider{
		StorageFactory:       masterCfg.ExtraConfig.StorageFactory,
		ProxyTransport:       masterCfg.ExtraConfig.ProxyTransport,
		KubeletClientConfig:  masterCfg.ExtraConfig.KubeletClientConfig,
		EventTTL:             masterCfg.ExtraConfig.EventTTL,
		ServiceIPRange:       masterCfg.ExtraConfig.ServiceIPRange,
		ServiceNodePortRange: masterCfg.ExtraConfig.ServiceNodePortRange,
		LoopbackClientConfig: masterCfg.GenericConfig.LoopbackClientConfig,
	}

	_, apiGroupInfo, err := storageProvider.NewLegacyRESTStorage(masterCfg.GenericConfig.RESTOptionsGetter)
	if err != nil {
		t.Errorf("failed to create legacy REST storage: %v", err)
	}

	// Any new stores with export logic will need to be added here:
	exceptions := registrytest.StrategyExceptions{
		// Only these stores should have an export strategy defined:
		HasExportStrategy: []string{
			"secrets",
			"limitRanges",
			"nodes",
			"podTemplates",
		},
	}

	strategyErrors := registrytest.ValidateStorageStrategies(apiGroupInfo.VersionedResourcesStorageMap["v1"], exceptions)
	for _, err := range strategyErrors {
		t.Error(err)
	}
}

func TestCertificatesRestStorageStrategies(t *testing.T) {
	_, etcdserver, masterCfg, _ := newMaster(t)
	defer etcdserver.Terminate(t)

	certStorageProvider := certificatesrest.RESTStorageProvider{}
	apiGroupInfo, _ := certStorageProvider.NewRESTStorage(masterCfg.ExtraConfig.APIResourceConfigSource, masterCfg.GenericConfig.RESTOptionsGetter)

	exceptions := registrytest.StrategyExceptions{
		HasExportStrategy: []string{
			"certificatesigningrequests",
		},
	}

	strategyErrors := registrytest.ValidateStorageStrategies(
		apiGroupInfo.VersionedResourcesStorageMap[certificatesapiv1beta1.SchemeGroupVersion.Version], exceptions)
	for _, err := range strategyErrors {
		t.Error(err)
	}
}

func newMaster(t *testing.T) (*Master, *etcdtesting.EtcdTestServer, Config, *assert.Assertions) {
	etcdserver, config, sharedInformers, assert := setUp(t)

	master, err := config.Complete(sharedInformers).New(genericapiserver.EmptyDelegate)
	if err != nil {
		t.Fatalf("Error in bringing up the master: %v", err)
	}

	return master, etcdserver, config, assert
}

// limitedAPIResourceConfigSource only enables the core group, the extensions group, the batch group, and the autoscaling group.
func limitedAPIResourceConfigSource() *serverstorage.ResourceConfig {
	ret := serverstorage.NewResourceConfig()
	ret.EnableVersions(
		apiv1.SchemeGroupVersion,
		extensionsapiv1beta1.SchemeGroupVersion,
		batchapiv1.SchemeGroupVersion,
		batchapiv1beta1.SchemeGroupVersion,
		appsapiv1beta1.SchemeGroupVersion,
		autoscalingapiv1.SchemeGroupVersion,
	)
	return ret
}

// newLimitedMaster only enables the core group, the extensions group, the batch group, and the autoscaling group.
func newLimitedMaster(t *testing.T) (*Master, *etcdtesting.EtcdTestServer, Config, *assert.Assertions) {
	etcdserver, config, sharedInformers, assert := setUp(t)
	config.ExtraConfig.APIResourceConfigSource = limitedAPIResourceConfigSource()
	master, err := config.Complete(sharedInformers).New(genericapiserver.EmptyDelegate)
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
	s.GenericAPIServer.Handler.ServeHTTP(resp, req)
	if resp.Code != 200 {
		t.Fatalf("expected http 200, got: %d", resp.Code)
	}

	var info version.Info
	err := json.NewDecoder(resp.Body).Decode(&info)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if !reflect.DeepEqual(kubeversion.Get(), info) {
		t.Errorf("Expected %#v, Got %#v", kubeversion.Get(), info)
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
	nodes, _ := fakeNodeClient.List(metav1.ListOptions{})
	addrs, err := addressProvider.externalAddresses()

	assert.Error(err, "addresses should have caused an error as there are no addresses.")
	assert.Equal([]string(nil), addrs)

	// Pass case with External type IP
	nodes, _ = fakeNodeClient.List(metav1.ListOptions{})
	for index := range nodes.Items {
		nodes.Items[index].Status.Addresses = []apiv1.NodeAddress{{Type: apiv1.NodeExternalIP, Address: "127.0.0.1"}}
		fakeNodeClient.Update(&nodes.Items[index])
	}
	addrs, err = addressProvider.externalAddresses()
	assert.NoError(err, "addresses should not have returned an error.")
	assert.Equal([]string{"127.0.0.1", "127.0.0.1"}, addrs)
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

	server := httptest.NewServer(genericapirequest.WithRequestContext(master.GenericAPIServer.Handler.GoRestfulContainer.ServeMux, master.GenericAPIServer.RequestContextMapper()))

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

func TestNoAlphaVersionsEnabledByDefault(t *testing.T) {
	config := DefaultAPIResourceConfigSource()
	for gv, gvConfig := range config.GroupVersionResourceConfigs {
		if gvConfig.Enable && strings.Contains(gv.Version, "alpha") {
			t.Errorf("Alpha API version %s enabled by default", gv.String())
		}
	}
}
