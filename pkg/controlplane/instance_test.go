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

package controlplane

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	certificatesapiv1beta1 "k8s.io/api/certificates/v1beta1"
	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/version"
	"k8s.io/apiserver/pkg/authorization/authorizerfactory"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/options"
	"k8s.io/apiserver/pkg/server/resourceconfig"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	restclient "k8s.io/client-go/rest"
	kubeversion "k8s.io/component-base/version"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/networking"
	apisstorage "k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/controlplane/reconcilers"
	"k8s.io/kubernetes/pkg/controlplane/storageversionhashdata"
	kubeletclient "k8s.io/kubernetes/pkg/kubelet/client"
	certificatesrest "k8s.io/kubernetes/pkg/registry/certificates/rest"
	corerest "k8s.io/kubernetes/pkg/registry/core/rest"
	"k8s.io/kubernetes/pkg/registry/registrytest"

	"github.com/stretchr/testify/assert"
)

// setUp is a convenience function for setting up for (most) tests.
func setUp(t *testing.T) (*etcd3testing.EtcdTestServer, Config, *assert.Assertions) {
	server, storageConfig := etcd3testing.NewUnsecuredEtcd3TestClientServer(t)

	config := &Config{
		GenericConfig: genericapiserver.NewConfig(legacyscheme.Codecs),
		ExtraConfig: ExtraConfig{
			APIResourceConfigSource: DefaultAPIResourceConfigSource(),
			APIServerServicePort:    443,
			MasterCount:             1,
			EndpointReconcilerType:  reconcilers.MasterCountReconcilerType,
		},
	}

	resourceEncoding := serverstorage.NewDefaultResourceEncodingConfig(legacyscheme.Scheme)
	// This configures the testing apiserver the same way the real apiserver is
	// configured. The storage versions of these resources are different
	// from the storage versions of other resources in their group.
	resourceEncodingOverrides := []schema.GroupVersionResource{
		batch.Resource("cronjobs").WithVersion("v1beta1"),
		apisstorage.Resource("volumeattachments").WithVersion("v1beta1"),
		networking.Resource("ingresses").WithVersion("v1beta1"),
	}
	resourceEncoding = resourceconfig.MergeResourceEncodingConfigs(resourceEncoding, resourceEncodingOverrides)
	storageFactory := serverstorage.NewDefaultStorageFactory(*storageConfig, "application/vnd.kubernetes.protobuf", legacyscheme.Codecs, resourceEncoding, DefaultAPIResourceConfigSource(), nil)

	etcdOptions := options.NewEtcdOptions(storageConfig)
	// unit tests don't need watch cache and it leaks lots of goroutines with etcd testing functions during unit tests
	etcdOptions.EnableWatchCache = false
	err := etcdOptions.ApplyWithStorageFactoryTo(storageFactory, config.GenericConfig)
	if err != nil {
		t.Fatal(err)
	}

	kubeVersion := kubeversion.Get()
	config.GenericConfig.Authorization.Authorizer = authorizerfactory.NewAlwaysAllowAuthorizer()
	config.GenericConfig.Version = &kubeVersion
	config.ExtraConfig.StorageFactory = storageFactory
	config.GenericConfig.LoopbackClientConfig = &restclient.Config{APIPath: "/api", ContentConfig: restclient.ContentConfig{NegotiatedSerializer: legacyscheme.Codecs}}
	config.GenericConfig.PublicAddress = net.ParseIP("192.168.10.4")
	config.GenericConfig.LegacyAPIGroupPrefixes = sets.NewString("/api")
	config.ExtraConfig.KubeletClientConfig = kubeletclient.KubeletClientConfig{Port: 10250}
	config.ExtraConfig.ProxyTransport = utilnet.SetTransportDefaults(&http.Transport{
		DialContext:     func(ctx context.Context, network, addr string) (net.Conn, error) { return nil, nil },
		TLSClientConfig: &tls.Config{},
	})

	// set fake SecureServingInfo because the listener port is needed for the kubernetes service
	config.GenericConfig.SecureServing = &genericapiserver.SecureServingInfo{Listener: fakeLocalhost443Listener{}}

	clientset, err := kubernetes.NewForConfig(config.GenericConfig.LoopbackClientConfig)
	if err != nil {
		t.Fatalf("unable to create client set due to %v", err)
	}
	config.ExtraConfig.VersionedInformers = informers.NewSharedInformerFactory(clientset, config.GenericConfig.LoopbackClientConfig.Timeout)

	return server, *config, assert.New(t)
}

type fakeLocalhost443Listener struct{}

func (fakeLocalhost443Listener) Accept() (net.Conn, error) {
	return nil, nil
}

func (fakeLocalhost443Listener) Close() error {
	return nil
}

func (fakeLocalhost443Listener) Addr() net.Addr {
	return &net.TCPAddr{
		IP:   net.IPv4(127, 0, 0, 1),
		Port: 443,
	}
}

// TestLegacyRestStorageStrategies ensures that all Storage objects which are using the generic registry Store have
// their various strategies properly wired up. This surfaced as a bug where strategies defined Export functions, but
// they were never used outside of unit tests because the export strategies were not assigned inside the Store.
func TestLegacyRestStorageStrategies(t *testing.T) {
	_, etcdserver, apiserverCfg, _ := newInstance(t)
	defer etcdserver.Terminate(t)

	storageProvider := corerest.LegacyRESTStorageProvider{
		StorageFactory:       apiserverCfg.ExtraConfig.StorageFactory,
		ProxyTransport:       apiserverCfg.ExtraConfig.ProxyTransport,
		KubeletClientConfig:  apiserverCfg.ExtraConfig.KubeletClientConfig,
		EventTTL:             apiserverCfg.ExtraConfig.EventTTL,
		ServiceIPRange:       apiserverCfg.ExtraConfig.ServiceIPRange,
		ServiceNodePortRange: apiserverCfg.ExtraConfig.ServiceNodePortRange,
		LoopbackClientConfig: apiserverCfg.GenericConfig.LoopbackClientConfig,
	}

	_, apiGroupInfo, err := storageProvider.NewLegacyRESTStorage(apiserverCfg.GenericConfig.RESTOptionsGetter)
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
	_, etcdserver, apiserverCfg, _ := newInstance(t)
	defer etcdserver.Terminate(t)

	certStorageProvider := certificatesrest.RESTStorageProvider{}
	apiGroupInfo, _, err := certStorageProvider.NewRESTStorage(apiserverCfg.ExtraConfig.APIResourceConfigSource, apiserverCfg.GenericConfig.RESTOptionsGetter)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}

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

func newInstance(t *testing.T) (*Instance, *etcd3testing.EtcdTestServer, Config, *assert.Assertions) {
	etcdserver, config, assert := setUp(t)

	apiserver, err := config.Complete().New(genericapiserver.NewEmptyDelegate())
	if err != nil {
		t.Fatalf("Error in bringing up the master: %v", err)
	}

	return apiserver, etcdserver, config, assert
}

// TestVersion tests /version
func TestVersion(t *testing.T) {
	s, etcdserver, _, _ := newInstance(t)
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

	fakeNodeClient := fake.NewSimpleClientset(makeNodeList([]string{"node1", "node2"}, apiv1.NodeResources{})).CoreV1().Nodes()
	addressProvider := nodeAddressProvider{fakeNodeClient}

	// Fail case (no addresses associated with nodes)
	addrs, err := addressProvider.externalAddresses()

	assert.Error(err, "addresses should have caused an error as there are no addresses.")
	assert.Equal([]string(nil), addrs)

	// Pass case with External type IP
	nodes, _ := fakeNodeClient.List(context.TODO(), metav1.ListOptions{})
	for index := range nodes.Items {
		nodes.Items[index].Status.Addresses = []apiv1.NodeAddress{{Type: apiv1.NodeExternalIP, Address: "127.0.0.1"}}
		fakeNodeClient.Update(context.TODO(), &nodes.Items[index], metav1.UpdateOptions{})
	}
	addrs, err = addressProvider.externalAddresses()
	assert.NoError(err, "addresses should not have returned an error.")
	assert.Equal([]string{"127.0.0.1", "127.0.0.1"}, addrs)
}

func TestGetNodeAddressesWithOnlySomeExternalIP(t *testing.T) {
	assert := assert.New(t)

	fakeNodeClient := fake.NewSimpleClientset(makeNodeList([]string{"node1", "node2", "node3"}, apiv1.NodeResources{})).CoreV1().Nodes()
	addressProvider := nodeAddressProvider{fakeNodeClient}

	// Pass case with 1 External type IP (index == 1) and nodes (indexes 0 & 2) have no External IP.
	nodes, _ := fakeNodeClient.List(context.TODO(), metav1.ListOptions{})
	nodes.Items[1].Status.Addresses = []apiv1.NodeAddress{{Type: apiv1.NodeExternalIP, Address: "127.0.0.1"}}
	fakeNodeClient.Update(context.TODO(), &nodes.Items[1], metav1.UpdateOptions{})

	addrs, err := addressProvider.externalAddresses()
	assert.NoError(err, "addresses should not have returned an error.")
	assert.Equal([]string{"127.0.0.1"}, addrs)
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
	apiserver, etcdserver, _, assert := newInstance(t)
	defer etcdserver.Terminate(t)

	server := httptest.NewServer(apiserver.GenericAPIServer.Handler.GoRestfulContainer.ServeMux)

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

// This test doesn't cover the apiregistration and apiextensions group, as they are installed by other apiservers.
func TestStorageVersionHashes(t *testing.T) {
	apiserver, etcdserver, _, _ := newInstance(t)
	defer etcdserver.Terminate(t)

	server := httptest.NewServer(apiserver.GenericAPIServer.Handler.GoRestfulContainer.ServeMux)

	c := &restclient.Config{
		Host:          server.URL,
		APIPath:       "/api",
		ContentConfig: restclient.ContentConfig{NegotiatedSerializer: legacyscheme.Codecs},
	}
	discover := discovery.NewDiscoveryClientForConfigOrDie(c)
	all, err := discover.ServerResources()
	if err != nil {
		t.Error(err)
	}
	var count int
	for _, g := range all {
		for _, r := range g.APIResources {
			if strings.Contains(r.Name, "/") ||
				storageversionhashdata.NoStorageVersionHash.Has(g.GroupVersion+"/"+r.Name) {
				if r.StorageVersionHash != "" {
					t.Errorf("expect resource %s/%s to have empty storageVersionHash, got hash %q", g.GroupVersion, r.Name, r.StorageVersionHash)
				}
				continue
			}
			if r.StorageVersionHash == "" {
				t.Errorf("expect the storageVersionHash of %s/%s to exist", g.GroupVersion, r.Name)
				continue
			}
			// Uncomment the following line if you want to update storageversionhash/data.go
			// fmt.Printf("\"%s/%s\": \"%s\",\n", g.GroupVersion, r.Name, r.StorageVersionHash)
			expected := storageversionhashdata.GVRToStorageVersionHash[g.GroupVersion+"/"+r.Name]
			if r.StorageVersionHash != expected {
				t.Errorf("expect the storageVersionHash of %s/%s to be %q, got %q", g.GroupVersion, r.Name, expected, r.StorageVersionHash)
			}
			count++
		}
	}
	if count != len(storageversionhashdata.GVRToStorageVersionHash) {
		t.Errorf("please remove the redundant entries from GVRToStorageVersionHash")
	}
}

func TestStorageVersionHashEqualities(t *testing.T) {
	apiserver, etcdserver, _, assert := newInstance(t)
	defer etcdserver.Terminate(t)

	server := httptest.NewServer(apiserver.GenericAPIServer.Handler.GoRestfulContainer.ServeMux)

	// Test 1: extensions/v1beta1/ingresses and apps/v1/ingresses have
	// the same storage version hash.
	resp, err := http.Get(server.URL + "/apis/extensions/v1beta1")
	assert.Empty(err)
	extList := metav1.APIResourceList{}
	assert.NoError(decodeResponse(resp, &extList))
	var extIngressHash, appsIngressHash string
	for _, r := range extList.APIResources {
		if r.Name == "ingresses" {
			extIngressHash = r.StorageVersionHash
			assert.NotEmpty(extIngressHash)
		}
	}

	resp, err = http.Get(server.URL + "/apis/networking.k8s.io/v1beta1")
	assert.Empty(err)
	appsList := metav1.APIResourceList{}
	assert.NoError(decodeResponse(resp, &appsList))
	for _, r := range appsList.APIResources {
		if r.Name == "ingresses" {
			appsIngressHash = r.StorageVersionHash
			assert.NotEmpty(appsIngressHash)
		}
	}
	if len(extIngressHash) > 0 && len(appsIngressHash) > 0 {
		assert.Equal(extIngressHash, appsIngressHash)
	}

	// Test 2: batch/v1/jobs and batch/v1beta1/cronjobs have different
	// storage version hashes.
	resp, err = http.Get(server.URL + "/apis/batch/v1")
	assert.Empty(err)
	batchv1 := metav1.APIResourceList{}
	assert.NoError(decodeResponse(resp, &batchv1))
	var jobsHash string
	for _, r := range batchv1.APIResources {
		if r.Name == "jobs" {
			jobsHash = r.StorageVersionHash
		}
	}
	assert.NotEmpty(jobsHash)

	resp, err = http.Get(server.URL + "/apis/batch/v1beta1")
	assert.Empty(err)
	batchv1beta1 := metav1.APIResourceList{}
	assert.NoError(decodeResponse(resp, &batchv1beta1))
	var cronjobsHash string
	for _, r := range batchv1beta1.APIResources {
		if r.Name == "cronjobs" {
			cronjobsHash = r.StorageVersionHash
		}
	}
	assert.NotEmpty(cronjobsHash)
	assert.NotEqual(jobsHash, cronjobsHash)
}

func TestNoAlphaVersionsEnabledByDefault(t *testing.T) {
	config := DefaultAPIResourceConfigSource()
	for gv, enable := range config.GroupVersionConfigs {
		if enable && strings.Contains(gv.Version, "alpha") {
			t.Errorf("Alpha API version %s enabled by default", gv.String())
		}
	}
}
