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
	"bytes"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apis/apps"
	appsapi "k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	autoscalingapiv1 "k8s.io/kubernetes/pkg/apis/autoscaling/v1"
	"k8s.io/kubernetes/pkg/apis/batch"
	batchapiv1 "k8s.io/kubernetes/pkg/apis/batch/v1"
	batchapiv2alpha1 "k8s.io/kubernetes/pkg/apis/batch/v2alpha1"
	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/apis/extensions"
	extensionsapiv1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/apiserver"
	"k8s.io/kubernetes/pkg/genericapiserver"
	"k8s.io/kubernetes/pkg/kubelet/client"
	"k8s.io/kubernetes/pkg/registry/endpoint"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/namespace"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/registry/thirdpartyresourcedata"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/storage/etcd/etcdtest"
	etcdtesting "k8s.io/kubernetes/pkg/storage/etcd/testing"
	"k8s.io/kubernetes/pkg/storage/storagebackend"
	"k8s.io/kubernetes/pkg/util/intstr"
	utilnet "k8s.io/kubernetes/pkg/util/net"
	"k8s.io/kubernetes/pkg/util/sets"

	"github.com/go-openapi/loads"
	"github.com/go-openapi/spec"
	"github.com/go-openapi/strfmt"
	"github.com/go-openapi/validate"
	"github.com/stretchr/testify/assert"
	"golang.org/x/net/context"
)

// setUp is a convience function for setting up for (most) tests.
func setUp(t *testing.T) (*Master, *etcdtesting.EtcdTestServer, Config, *assert.Assertions) {
	server := etcdtesting.NewUnsecuredEtcdTestClientServer(t)

	master := &Master{
		GenericAPIServer: &genericapiserver.GenericAPIServer{},
	}
	config := Config{
		Config: &genericapiserver.Config{},
	}

	storageConfig := storagebackend.Config{
		Prefix:   etcdtest.PathPrefix(),
		CAFile:   server.CAFile,
		KeyFile:  server.KeyFile,
		CertFile: server.CertFile,
	}
	for _, url := range server.ClientURLs {
		storageConfig.ServerList = append(storageConfig.ServerList, url.String())
	}

	resourceEncoding := genericapiserver.NewDefaultResourceEncodingConfig()
	resourceEncoding.SetVersionEncoding(api.GroupName, *testapi.Default.GroupVersion(), unversioned.GroupVersion{Group: api.GroupName, Version: runtime.APIVersionInternal})
	resourceEncoding.SetVersionEncoding(autoscaling.GroupName, *testapi.Autoscaling.GroupVersion(), unversioned.GroupVersion{Group: autoscaling.GroupName, Version: runtime.APIVersionInternal})
	resourceEncoding.SetVersionEncoding(batch.GroupName, *testapi.Batch.GroupVersion(), unversioned.GroupVersion{Group: batch.GroupName, Version: runtime.APIVersionInternal})
	resourceEncoding.SetVersionEncoding(apps.GroupName, *testapi.Apps.GroupVersion(), unversioned.GroupVersion{Group: apps.GroupName, Version: runtime.APIVersionInternal})
	resourceEncoding.SetVersionEncoding(extensions.GroupName, *testapi.Extensions.GroupVersion(), unversioned.GroupVersion{Group: extensions.GroupName, Version: runtime.APIVersionInternal})
	resourceEncoding.SetVersionEncoding(rbac.GroupName, *testapi.Rbac.GroupVersion(), unversioned.GroupVersion{Group: rbac.GroupName, Version: runtime.APIVersionInternal})
	resourceEncoding.SetVersionEncoding(certificates.GroupName, *testapi.Certificates.GroupVersion(), unversioned.GroupVersion{Group: certificates.GroupName, Version: runtime.APIVersionInternal})
	storageFactory := genericapiserver.NewDefaultStorageFactory(storageConfig, testapi.StorageMediaType(), api.Codecs, resourceEncoding, DefaultAPIResourceConfigSource())

	config.StorageFactory = storageFactory
	config.APIResourceConfigSource = DefaultAPIResourceConfigSource()
	config.PublicAddress = net.ParseIP("192.168.10.4")
	config.Serializer = api.Codecs
	config.KubeletClient = client.FakeKubeletClient{}
	config.APIPrefix = "/api"
	config.APIGroupPrefix = "/apis"
	config.APIResourceConfigSource = DefaultAPIResourceConfigSource()
	config.ProxyDialer = func(network, addr string) (net.Conn, error) { return nil, nil }
	config.ProxyTLSClientConfig = &tls.Config{}

	// TODO: this is kind of hacky.  The trouble is that the sync loop
	// runs in a go-routine and there is no way to validate in the test
	// that the sync routine has actually run.  The right answer here
	// is probably to add some sort of callback that we can register
	// to validate that it's actually been run, but for now we don't
	// run the sync routine and register types manually.
	config.disableThirdPartyControllerForTesting = true

	master.nodeRegistry = registrytest.NewNodeRegistry([]string{"node1", "node2"}, api.NodeResources{})

	return master, server, config, assert.New(t)
}

func newMaster(t *testing.T) (*Master, *etcdtesting.EtcdTestServer, Config, *assert.Assertions) {
	_, etcdserver, config, assert := setUp(t)

	master, err := New(&config)
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
		appsapi.SchemeGroupVersion,
		autoscalingapiv1.SchemeGroupVersion,
	)
	return ret
}

// newLimitedMaster only enables the core group, the extensions group, the batch group, and the autoscaling group.
func newLimitedMaster(t *testing.T) (*Master, *etcdtesting.EtcdTestServer, Config, *assert.Assertions) {
	_, etcdserver, config, assert := setUp(t)
	config.APIResourceConfigSource = limitedAPIResourceConfigSource()
	master, err := New(&config)
	if err != nil {
		t.Fatalf("Error in bringing up the master: %v", err)
	}

	return master, etcdserver, config, assert
}

// TestNew verifies that the New function returns a Master
// using the configuration properly.
func TestNew(t *testing.T) {
	master, etcdserver, config, assert := newMaster(t)
	defer etcdserver.Terminate(t)

	// Verify many of the variables match their config counterparts
	assert.Equal(master.enableCoreControllers, config.EnableCoreControllers)
	assert.Equal(master.tunneler, config.Tunneler)
	assert.Equal(master.RequestContextMapper(), config.RequestContextMapper)
	assert.Equal(master.MasterCount, config.MasterCount)
	assert.Equal(master.ClusterIP, config.PublicAddress)
	assert.Equal(master.PublicReadWritePort, config.ReadWritePort)
	assert.Equal(master.ServiceReadWriteIP, config.ServiceReadWriteIP)

	// These functions should point to the same memory location
	masterDialer, _ := utilnet.Dialer(master.ProxyTransport)
	masterDialerFunc := fmt.Sprintf("%p", masterDialer)
	configDialerFunc := fmt.Sprintf("%p", config.ProxyDialer)
	assert.Equal(masterDialerFunc, configDialerFunc)

	assert.Equal(master.ProxyTransport.(*http.Transport).TLSClientConfig, config.ProxyTLSClientConfig)
}

// TestNamespaceSubresources ensures the namespace subresource parsing in apiserver/handlers.go doesn't drift
func TestNamespaceSubresources(t *testing.T) {
	master, etcdserver, _, _ := newMaster(t)
	defer etcdserver.Terminate(t)

	expectedSubresources := apiserver.NamespaceSubResourcesForTest
	foundSubresources := sets.NewString()

	for k := range master.v1ResourcesStorage {
		parts := strings.Split(k, "/")
		if len(parts) == 2 && parts[0] == "namespaces" {
			foundSubresources.Insert(parts[1])
		}
	}

	if !reflect.DeepEqual(expectedSubresources.List(), foundSubresources.List()) {
		t.Errorf("Expected namespace subresources %#v, got %#v. Update apiserver/handlers.go#namespaceSubresources", expectedSubresources.List(), foundSubresources.List())
	}
}

// TestGetServersToValidate verifies the unexported getServersToValidate function
func TestGetServersToValidate(t *testing.T) {
	master, etcdserver, config, assert := setUp(t)
	defer etcdserver.Terminate(t)

	servers := master.getServersToValidate(&config)

	// Expected servers to validate: scheduler, controller-manager and etcd.
	assert.Equal(3, len(servers), "unexpected server list: %#v", servers)

	for _, server := range []string{"scheduler", "controller-manager", "etcd-0"} {
		if _, ok := servers[server]; !ok {
			t.Errorf("server list missing: %s", server)
		}
	}
}

// TestFindExternalAddress verifies both pass and fail cases for the unexported
// findExternalAddress function
func TestFindExternalAddress(t *testing.T) {
	assert := assert.New(t)
	expectedIP := "172.0.0.1"

	nodes := []*api.Node{new(api.Node), new(api.Node), new(api.Node)}
	nodes[0].Status.Addresses = []api.NodeAddress{{Type: "ExternalIP", Address: expectedIP}}
	nodes[1].Status.Addresses = []api.NodeAddress{{Type: "LegacyHostIP", Address: expectedIP}}
	nodes[2].Status.Addresses = []api.NodeAddress{{Type: "ExternalIP", Address: expectedIP}, {Type: "LegacyHostIP", Address: "172.0.0.2"}}

	// Pass Case
	for _, node := range nodes {
		ip, err := findExternalAddress(node)
		assert.NoError(err, "error getting node external address")
		assert.Equal(expectedIP, ip, "expected ip to be %s, but was %s", expectedIP, ip)
	}

	// Fail case
	_, err := findExternalAddress(new(api.Node))
	assert.Error(err, "expected findExternalAddress to fail on a node with missing ip information")
}

type fakeEndpointReconciler struct{}

func (*fakeEndpointReconciler) ReconcileEndpoints(serviceName string, ip net.IP, endpointPorts []api.EndpointPort, reconcilePorts bool) error {
	return nil
}

// TestNewBootstrapController verifies master fields are properly copied into controller
func TestNewBootstrapController(t *testing.T) {
	// Tests a subset of inputs to ensure they are set properly in the controller
	master, etcdserver, _, assert := setUp(t)
	defer etcdserver.Terminate(t)

	portRange := utilnet.PortRange{Base: 10, Size: 10}

	master.namespaceRegistry = namespace.NewRegistry(nil)
	master.serviceRegistry = registrytest.NewServiceRegistry()
	master.endpointRegistry = endpoint.NewRegistry(nil)

	master.ServiceNodePortRange = portRange
	master.MasterCount = 1
	master.ServiceReadWritePort = 1000
	master.PublicReadWritePort = 1010

	// test with an empty EndpointReconcilerConfig to ensure the defaults are applied
	controller := master.NewBootstrapController(EndpointReconcilerConfig{})

	assert.Equal(controller.NamespaceRegistry, master.namespaceRegistry)
	assert.Equal(controller.EndpointReconciler, NewMasterCountEndpointReconciler(master.MasterCount, master.endpointRegistry))
	assert.Equal(controller.EndpointInterval, DefaultEndpointReconcilerInterval)
	assert.Equal(controller.ServiceRegistry, master.serviceRegistry)
	assert.Equal(controller.ServiceNodePortRange, portRange)
	assert.Equal(controller.ServicePort, master.ServiceReadWritePort)
	assert.Equal(controller.PublicServicePort, master.PublicReadWritePort)

	// test with a filled-in EndpointReconcilerConfig to make sure its values are used
	controller = master.NewBootstrapController(EndpointReconcilerConfig{
		Reconciler: &fakeEndpointReconciler{},
		Interval:   5 * time.Second,
	})
	assert.Equal(controller.EndpointReconciler, &fakeEndpointReconciler{})
	assert.Equal(controller.EndpointInterval, 5*time.Second)
}

// TestControllerServicePorts verifies master extraServicePorts are
// correctly copied into controller
func TestControllerServicePorts(t *testing.T) {
	master, etcdserver, _, assert := setUp(t)
	defer etcdserver.Terminate(t)

	master.namespaceRegistry = namespace.NewRegistry(nil)
	master.serviceRegistry = registrytest.NewServiceRegistry()
	master.endpointRegistry = endpoint.NewRegistry(nil)

	master.ExtraServicePorts = []api.ServicePort{
		{
			Name:       "additional-port-1",
			Port:       1000,
			Protocol:   api.ProtocolTCP,
			TargetPort: intstr.FromInt(1000),
		},
		{
			Name:       "additional-port-2",
			Port:       1010,
			Protocol:   api.ProtocolTCP,
			TargetPort: intstr.FromInt(1010),
		},
	}

	controller := master.NewBootstrapController(EndpointReconcilerConfig{})

	assert.Equal(int32(1000), controller.ExtraServicePorts[0].Port)
	assert.Equal(int32(1010), controller.ExtraServicePorts[1].Port)
}

// TestGetNodeAddresses verifies that proper results are returned
// when requesting node addresses.
func TestGetNodeAddresses(t *testing.T) {
	master, etcdserver, _, assert := setUp(t)
	defer etcdserver.Terminate(t)

	// Fail case (no addresses associated with nodes)
	nodes, _ := master.nodeRegistry.ListNodes(api.NewDefaultContext(), nil)
	addrs, err := master.getNodeAddresses()

	assert.Error(err, "getNodeAddresses should have caused an error as there are no addresses.")
	assert.Equal([]string(nil), addrs)

	// Pass case with External type IP
	nodes, _ = master.nodeRegistry.ListNodes(api.NewDefaultContext(), nil)
	for index := range nodes.Items {
		nodes.Items[index].Status.Addresses = []api.NodeAddress{{Type: api.NodeExternalIP, Address: "127.0.0.1"}}
	}
	addrs, err = master.getNodeAddresses()
	assert.NoError(err, "getNodeAddresses should not have returned an error.")
	assert.Equal([]string{"127.0.0.1", "127.0.0.1"}, addrs)

	// Pass case with LegacyHost type IP
	nodes, _ = master.nodeRegistry.ListNodes(api.NewDefaultContext(), nil)
	for index := range nodes.Items {
		nodes.Items[index].Status.Addresses = []api.NodeAddress{{Type: api.NodeLegacyHostIP, Address: "127.0.0.2"}}
	}
	addrs, err = master.getNodeAddresses()
	assert.NoError(err, "getNodeAddresses failback should not have returned an error.")
	assert.Equal([]string{"127.0.0.2", "127.0.0.2"}, addrs)
}

// Because we need to be backwards compatible with release 1.1, at endpoints
// that exist in release 1.1, the responses should have empty APIVersion.
func TestAPIVersionOfDiscoveryEndpoints(t *testing.T) {
	master, etcdserver, _, assert := newMaster(t)
	defer etcdserver.Terminate(t)

	server := httptest.NewServer(master.HandlerContainer.ServeMux)

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

func TestDiscoveryAtAPIS(t *testing.T) {
	master, etcdserver, _, assert := newLimitedMaster(t)
	defer etcdserver.Terminate(t)

	server := httptest.NewServer(master.HandlerContainer.ServeMux)
	resp, err := http.Get(server.URL + "/apis")
	if !assert.NoError(err) {
		t.Errorf("unexpected error: %v", err)
	}

	assert.Equal(http.StatusOK, resp.StatusCode)

	groupList := unversioned.APIGroupList{}
	assert.NoError(decodeResponse(resp, &groupList))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expectGroupNames := sets.NewString(autoscaling.GroupName, batch.GroupName, apps.GroupName, extensions.GroupName)
	expectVersions := map[string][]unversioned.GroupVersionForDiscovery{
		autoscaling.GroupName: {
			{
				GroupVersion: testapi.Autoscaling.GroupVersion().String(),
				Version:      testapi.Autoscaling.GroupVersion().Version,
			},
		},
		// batch is using its pkg/apis/batch/ types here since during installation
		// both versions get installed and testapi.go currently does not support
		// multi-versioned clients
		batch.GroupName: {
			{
				GroupVersion: batchapiv1.SchemeGroupVersion.String(),
				Version:      batchapiv1.SchemeGroupVersion.Version,
			},
			{
				GroupVersion: batchapiv2alpha1.SchemeGroupVersion.String(),
				Version:      batchapiv2alpha1.SchemeGroupVersion.Version,
			},
		},
		apps.GroupName: {
			{
				GroupVersion: testapi.Apps.GroupVersion().String(),
				Version:      testapi.Apps.GroupVersion().Version,
			},
		},
		extensions.GroupName: {
			{
				GroupVersion: testapi.Extensions.GroupVersion().String(),
				Version:      testapi.Extensions.GroupVersion().Version,
			},
		},
	}
	expectPreferredVersion := map[string]unversioned.GroupVersionForDiscovery{
		autoscaling.GroupName: {
			GroupVersion: registered.GroupOrDie(autoscaling.GroupName).GroupVersion.String(),
			Version:      registered.GroupOrDie(autoscaling.GroupName).GroupVersion.Version,
		},
		batch.GroupName: {
			GroupVersion: registered.GroupOrDie(batch.GroupName).GroupVersion.String(),
			Version:      registered.GroupOrDie(batch.GroupName).GroupVersion.Version,
		},
		apps.GroupName: {
			GroupVersion: registered.GroupOrDie(apps.GroupName).GroupVersion.String(),
			Version:      registered.GroupOrDie(apps.GroupName).GroupVersion.Version,
		},
		extensions.GroupName: {
			GroupVersion: registered.GroupOrDie(extensions.GroupName).GroupVersion.String(),
			Version:      registered.GroupOrDie(extensions.GroupName).GroupVersion.Version,
		},
	}

	assert.Equal(4, len(groupList.Groups))
	for _, group := range groupList.Groups {
		if !expectGroupNames.Has(group.Name) {
			t.Errorf("got unexpected group %s", group.Name)
		}
		assert.Equal(expectVersions[group.Name], group.Versions)
		assert.Equal(expectPreferredVersion[group.Name], group.PreferredVersion)
	}

	thirdPartyGV := unversioned.GroupVersionForDiscovery{GroupVersion: "company.com/v1", Version: "v1"}
	master.addThirdPartyResourceStorage("/apis/company.com/v1", "foos", nil,
		unversioned.APIGroup{
			Name:             "company.com",
			Versions:         []unversioned.GroupVersionForDiscovery{thirdPartyGV},
			PreferredVersion: thirdPartyGV,
		})

	resp, err = http.Get(server.URL + "/apis")
	if !assert.NoError(err) {
		t.Errorf("unexpected error: %v", err)
	}
	assert.Equal(http.StatusOK, resp.StatusCode)
	assert.NoError(decodeResponse(resp, &groupList))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	assert.Equal(5, len(groupList.Groups))

	expectGroupNames.Insert("company.com")
	expectVersions["company.com"] = []unversioned.GroupVersionForDiscovery{thirdPartyGV}
	expectPreferredVersion["company.com"] = thirdPartyGV
	for _, group := range groupList.Groups {
		if !expectGroupNames.Has(group.Name) {
			t.Errorf("got unexpected group %s", group.Name)
		}
		assert.Equal(expectVersions[group.Name], group.Versions)
		assert.Equal(expectPreferredVersion[group.Name], group.PreferredVersion)
	}
}

var versionsToTest = []string{"v1", "v3"}

type Foo struct {
	unversioned.TypeMeta `json:",inline"`
	api.ObjectMeta       `json:"metadata,omitempty" description:"standard object metadata"`

	SomeField  string `json:"someField"`
	OtherField int    `json:"otherField"`
}

type FooList struct {
	unversioned.TypeMeta `json:",inline"`
	unversioned.ListMeta `json:"metadata,omitempty" description:"standard list metadata; see http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata"`

	Items []Foo `json:"items"`
}

func initThirdParty(t *testing.T, version, name string) (*Master, *etcdtesting.EtcdTestServer, *httptest.Server, *assert.Assertions) {
	return initThirdPartyMultiple(t, []string{version}, []string{name})
}

func initThirdPartyMultiple(t *testing.T, versions, names []string) (*Master, *etcdtesting.EtcdTestServer, *httptest.Server, *assert.Assertions) {
	master, etcdserver, _, assert := newMaster(t)
	_, master.ServiceClusterIPRange, _ = net.ParseCIDR("10.0.0.0/24")

	for ix := range names {
		api := &extensions.ThirdPartyResource{
			ObjectMeta: api.ObjectMeta{
				Name: names[ix],
			},
			Versions: []extensions.APIVersion{
				{
					Name: versions[ix],
				},
			},
		}
		hasRsrc, err := master.HasThirdPartyResource(api)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if !hasRsrc {
			err := master.InstallThirdPartyResource(api)
			if !assert.NoError(err) {
				t.Errorf("Failed to install API: %v", err)
				t.FailNow()
			}
		} else {
			t.Errorf("Expected %s: %v not to be present!", names[ix], api)
		}
	}

	server := httptest.NewServer(master.HandlerContainer.ServeMux)
	return master, etcdserver, server, assert
}

func TestInstallMultipleAPIs(t *testing.T) {
	names := []string{"foo.company.com", "bar.company.com"}
	versions := []string{"v1", "v1"}

	_, etcdserver, server, assert := initThirdPartyMultiple(t, versions, names)
	defer server.Close()
	defer etcdserver.Terminate(t)
	for ix := range names {
		kind, group, err := thirdpartyresourcedata.ExtractApiGroupAndKind(
			&extensions.ThirdPartyResource{ObjectMeta: api.ObjectMeta{Name: names[ix]}})
		assert.NoError(err, "Failed to extract group & kind")

		plural, _ := meta.KindToResource(unversioned.GroupVersionKind{
			Group:   group,
			Version: versions[ix],
			Kind:    kind,
		})

		resp, err := http.Get(
			fmt.Sprintf("%s/apis/%s/%s/namespaces/default/%s", server.URL, group, versions[ix], plural.Resource))
		if !assert.NoError(err, "Failed to do HTTP GET") {
			return
		}
		defer resp.Body.Close()

		assert.Equal(http.StatusOK, resp.StatusCode)

		data, err := ioutil.ReadAll(resp.Body)
		assert.NoError(err)

		obj := map[string]interface{}{}
		if err = json.Unmarshal(data, &obj); err != nil {
			assert.NoError(err, fmt.Sprintf("unexpected error: %v", err))
		}
		kindOut, found := obj["kind"]
		if !found {
			t.Errorf("Missing 'kind' in %v", obj)
		}
		assert.Equal(kindOut, kind+"List")
	}
}

func TestInstallThirdPartyAPIList(t *testing.T) {
	for _, version := range versionsToTest {
		testInstallThirdPartyAPIListVersion(t, version)
	}
}

func testInstallThirdPartyAPIListVersion(t *testing.T, version string) {
	tests := []struct {
		items []Foo
		name  string
		test  string
	}{
		{
			name: "foo.company.com",
			test: "null",
		},
		{
			items: []Foo{},
			name:  "foo.company.com",
			test:  "empty",
		},
		{
			items: []Foo{},
			name:  "policy.company.com",
			test:  "plurals",
		},
		{
			items: []Foo{
				{
					ObjectMeta: api.ObjectMeta{
						Name: "test",
					},
					TypeMeta: unversioned.TypeMeta{
						Kind:       "Foo",
						APIVersion: version,
					},
					SomeField:  "test field",
					OtherField: 10,
				},
				{
					ObjectMeta: api.ObjectMeta{
						Name: "bar",
					},
					TypeMeta: unversioned.TypeMeta{
						Kind:       "Foo",
						APIVersion: version,
					},
					SomeField:  "test field another",
					OtherField: 20,
				},
			},
			name: "foo.company.com",
			test: "real list",
		},
	}
	for _, test := range tests {
		func() {
			master, etcdserver, server, assert := initThirdParty(t, version, test.name)
			defer server.Close()
			defer etcdserver.Terminate(t)

			kind, group, err := thirdpartyresourcedata.ExtractApiGroupAndKind(
				&extensions.ThirdPartyResource{ObjectMeta: api.ObjectMeta{Name: test.name}})
			assert.NoError(err, test.test)

			plural, _ := meta.KindToResource(unversioned.GroupVersionKind{
				Group:   group,
				Version: version,
				Kind:    kind,
			})

			if test.items != nil {
				s, destroyFunc := generic.NewRawStorage(master.thirdPartyStorageConfig)
				defer destroyFunc()
				err := createThirdPartyList(
					s,
					fmt.Sprintf("/ThirdPartyResourceData/%s/%s/default", group, plural.Resource),
					test.items)
				if !assert.NoError(err, test.test) {
					return
				}
			}

			resp, err := http.Get(
				fmt.Sprintf("%s/apis/%s/%s/namespaces/default/%s", server.URL, group, version, plural.Resource))
			if !assert.NoError(err, test.test) {
				return
			}
			defer resp.Body.Close()

			assert.Equal(http.StatusOK, resp.StatusCode, test.test)

			data, err := ioutil.ReadAll(resp.Body)
			assert.NoError(err, test.test)

			list := FooList{}
			if err = json.Unmarshal(data, &list); err != nil {
				assert.NoError(err, "unexpected error: %v %s", err, test.test)
			}

			if test.items == nil {
				if len(list.Items) != 0 {
					assert.NoError(err, "expected no items, saw: %v %s", err, list.Items, test.test)
				}
				return
			}

			if len(list.Items) != len(test.items) {
				t.Fatalf("(%s) unexpected length: %d vs %d", test.name, len(list.Items), len(test.items))
			}
			// The order of elements in LIST is not guaranteed.
			mapping := make(map[string]int)
			for ix := range test.items {
				mapping[test.items[ix].Name] = ix
			}
			for ix := range list.Items {
				// Copy things that are set dynamically on the server
				expectedObj := test.items[mapping[list.Items[ix].Name]]
				expectedObj.SelfLink = list.Items[ix].SelfLink
				expectedObj.ResourceVersion = list.Items[ix].ResourceVersion
				expectedObj.Namespace = list.Items[ix].Namespace
				expectedObj.UID = list.Items[ix].UID
				expectedObj.CreationTimestamp = list.Items[ix].CreationTimestamp

				// We endure the order of items by sorting them (using 'mapping')
				// so that this function passes.
				if !reflect.DeepEqual(list.Items[ix], expectedObj) {
					t.Errorf("(%s) expected:\n%#v\nsaw:\n%#v\n", test.name, expectedObj, list.Items[ix])
				}
			}
		}()
	}
}

func encodeToThirdParty(name string, obj interface{}) (runtime.Object, error) {
	serial, err := json.Marshal(obj)
	if err != nil {
		return nil, err
	}
	thirdPartyData := extensions.ThirdPartyResourceData{
		ObjectMeta: api.ObjectMeta{Name: name},
		Data:       serial,
	}
	return &thirdPartyData, nil
}

func createThirdPartyObject(s storage.Interface, path, name string, obj interface{}) error {
	data, err := encodeToThirdParty(name, obj)
	if err != nil {
		return err
	}
	return s.Create(context.TODO(), etcdtest.AddPrefix(path), data, nil, 0)
}

func createThirdPartyList(s storage.Interface, path string, list []Foo) error {
	for _, obj := range list {
		if err := createThirdPartyObject(s, path+"/"+obj.Name, obj.Name, obj); err != nil {
			return err
		}
	}
	return nil
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

func TestInstallThirdPartyAPIGet(t *testing.T) {
	for _, version := range versionsToTest {
		testInstallThirdPartyAPIGetVersion(t, version)
	}
}

func testInstallThirdPartyAPIGetVersion(t *testing.T, version string) {
	master, etcdserver, server, assert := initThirdParty(t, version, "foo.company.com")
	defer server.Close()
	defer etcdserver.Terminate(t)

	expectedObj := Foo{
		ObjectMeta: api.ObjectMeta{
			Name: "test",
		},
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Foo",
			APIVersion: version,
		},
		SomeField:  "test field",
		OtherField: 10,
	}
	s, destroyFunc := generic.NewRawStorage(master.thirdPartyStorageConfig)
	defer destroyFunc()
	if !assert.NoError(createThirdPartyObject(s, "/ThirdPartyResourceData/company.com/foos/default/test", "test", expectedObj)) {
		t.FailNow()
		return
	}

	resp, err := http.Get(server.URL + "/apis/company.com/" + version + "/namespaces/default/foos/test")
	if !assert.NoError(err) {
		return
	}

	assert.Equal(http.StatusOK, resp.StatusCode)

	item := Foo{}
	assert.NoError(decodeResponse(resp, &item))
	if !assert.False(reflect.DeepEqual(item, expectedObj)) {
		t.Errorf("expected objects to not be equal:\n%v\nsaw:\n%v\n", expectedObj, item)
	}
	// Fill in data that the apiserver injects
	expectedObj.SelfLink = item.SelfLink
	expectedObj.ResourceVersion = item.ResourceVersion
	if !assert.True(reflect.DeepEqual(item, expectedObj)) {
		t.Errorf("expected:\n%#v\nsaw:\n%#v\n", expectedObj, item)
	}
}

func TestInstallThirdPartyAPIPost(t *testing.T) {
	registered.AddThirdPartyAPIGroupVersions(unversioned.GroupVersion{Group: "company.com", Version: "v1"}, unversioned.GroupVersion{Group: "company.com", Version: "v3"})

	for _, version := range versionsToTest {
		testInstallThirdPartyAPIPostForVersion(t, version)
	}
}

func testInstallThirdPartyAPIPostForVersion(t *testing.T, version string) {
	master, etcdserver, server, assert := initThirdParty(t, version, "foo.company.com")
	defer server.Close()
	defer etcdserver.Terminate(t)

	inputObj := Foo{
		ObjectMeta: api.ObjectMeta{
			Name: "test",
		},
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Foo",
			APIVersion: "company.com/" + version,
		},
		SomeField:  "test field",
		OtherField: 10,
	}
	data, err := json.Marshal(inputObj)
	if !assert.NoError(err) {
		return
	}

	resp, err := http.Post(server.URL+"/apis/company.com/"+version+"/namespaces/default/foos", "application/json", bytes.NewBuffer(data))
	if !assert.NoError(err) {
		t.Fatalf("unexpected error: %v", err)
	}

	assert.Equal(http.StatusCreated, resp.StatusCode)

	item := Foo{}
	assert.NoError(decodeResponse(resp, &item))

	// fill in fields set by the apiserver
	expectedObj := inputObj
	expectedObj.SelfLink = item.SelfLink
	expectedObj.ResourceVersion = item.ResourceVersion
	expectedObj.Namespace = item.Namespace
	expectedObj.UID = item.UID
	expectedObj.CreationTimestamp = item.CreationTimestamp
	if !assert.True(reflect.DeepEqual(item, expectedObj)) {
		t.Errorf("expected:\n%v\nsaw:\n%v\n", expectedObj, item)
	}

	thirdPartyObj := extensions.ThirdPartyResourceData{}
	s, destroyFunc := generic.NewRawStorage(master.thirdPartyStorageConfig)
	defer destroyFunc()
	err = s.Get(context.TODO(), etcdtest.AddPrefix("/ThirdPartyResourceData/company.com/foos/default/test"), &thirdPartyObj, false)
	if !assert.NoError(err) {
		t.FailNow()
	}

	item = Foo{}
	assert.NoError(json.Unmarshal(thirdPartyObj.Data, &item))

	if !assert.True(reflect.DeepEqual(item, inputObj)) {
		t.Errorf("expected:\n%v\nsaw:\n%v\n", inputObj, item)
	}
}

func TestInstallThirdPartyAPIDelete(t *testing.T) {
	for _, version := range versionsToTest {
		testInstallThirdPartyAPIDeleteVersion(t, version)
	}
}

func testInstallThirdPartyAPIDeleteVersion(t *testing.T, version string) {
	master, etcdserver, server, assert := initThirdParty(t, version, "foo.company.com")
	defer server.Close()
	defer etcdserver.Terminate(t)

	expectedObj := Foo{
		ObjectMeta: api.ObjectMeta{
			Name:      "test",
			Namespace: "default",
		},
		TypeMeta: unversioned.TypeMeta{
			Kind: "Foo",
		},
		SomeField:  "test field",
		OtherField: 10,
	}
	s, destroyFunc := generic.NewRawStorage(master.thirdPartyStorageConfig)
	defer destroyFunc()
	if !assert.NoError(createThirdPartyObject(s, "/ThirdPartyResourceData/company.com/foos/default/test", "test", expectedObj)) {
		t.FailNow()
		return
	}

	resp, err := http.Get(server.URL + "/apis/company.com/" + version + "/namespaces/default/foos/test")
	if !assert.NoError(err) {
		return
	}

	assert.Equal(http.StatusOK, resp.StatusCode)

	item := Foo{}
	assert.NoError(decodeResponse(resp, &item))

	// Fill in fields set by the apiserver
	expectedObj.SelfLink = item.SelfLink
	expectedObj.ResourceVersion = item.ResourceVersion
	expectedObj.Namespace = item.Namespace
	if !assert.True(reflect.DeepEqual(item, expectedObj)) {
		t.Errorf("expected:\n%v\nsaw:\n%v\n", expectedObj, item)
	}

	resp, err = httpDelete(server.URL + "/apis/company.com/" + version + "/namespaces/default/foos/test")
	if !assert.NoError(err) {
		return
	}

	assert.Equal(http.StatusOK, resp.StatusCode)

	resp, err = http.Get(server.URL + "/apis/company.com/" + version + "/namespaces/default/foos/test")
	if !assert.NoError(err) {
		return
	}

	assert.Equal(http.StatusNotFound, resp.StatusCode)

	expectedDeletedKey := etcdtest.AddPrefix("ThirdPartyResourceData/company.com/foos/default/test")
	thirdPartyObj := extensions.ThirdPartyResourceData{}
	err = s.Get(context.TODO(), expectedDeletedKey, &thirdPartyObj, false)
	if !storage.IsNotFound(err) {
		t.Errorf("expected deletion didn't happen: %v", err)
	}
}

func httpDelete(url string) (*http.Response, error) {
	req, err := http.NewRequest("DELETE", url, nil)
	if err != nil {
		return nil, err
	}
	client := &http.Client{}
	return client.Do(req)
}

func TestInstallThirdPartyAPIListOptions(t *testing.T) {
	for _, version := range versionsToTest {
		testInstallThirdPartyAPIListOptionsForVersion(t, version)
	}
}

func testInstallThirdPartyAPIListOptionsForVersion(t *testing.T, version string) {
	_, etcdserver, server, assert := initThirdParty(t, version, "foo.company.com")
	defer server.Close()
	defer etcdserver.Terminate(t)

	// send a GET request with query parameter
	resp, err := httpGetWithRV(server.URL + "/apis/company.com/" + version + "/namespaces/default/foos")
	if !assert.NoError(err) {
		t.Fatalf("unexpected error: %v", err)
	}
	assert.Equal(http.StatusOK, resp.StatusCode)
}

func httpGetWithRV(url string) (*http.Response, error) {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	q := req.URL.Query()
	// resourceversion is part of a ListOptions
	q.Add("resourceversion", "0")
	req.URL.RawQuery = q.Encode()
	client := &http.Client{}
	return client.Do(req)
}

func TestInstallThirdPartyResourceRemove(t *testing.T) {
	for _, version := range versionsToTest {
		testInstallThirdPartyResourceRemove(t, version)
	}
}

func testInstallThirdPartyResourceRemove(t *testing.T, version string) {
	master, etcdserver, server, assert := initThirdParty(t, version, "foo.company.com")
	defer server.Close()
	defer etcdserver.Terminate(t)

	expectedObj := Foo{
		ObjectMeta: api.ObjectMeta{
			Name: "test",
		},
		TypeMeta: unversioned.TypeMeta{
			Kind: "Foo",
		},
		SomeField:  "test field",
		OtherField: 10,
	}
	s, destroyFunc := generic.NewRawStorage(master.thirdPartyStorageConfig)
	defer destroyFunc()
	if !assert.NoError(createThirdPartyObject(s, "/ThirdPartyResourceData/company.com/foos/default/test", "test", expectedObj)) {
		t.FailNow()
		return
	}
	secondObj := expectedObj
	secondObj.Name = "bar"
	if !assert.NoError(createThirdPartyObject(s, "/ThirdPartyResourceData/company.com/foos/default/bar", "bar", secondObj)) {
		t.FailNow()
		return
	}

	resp, err := http.Get(server.URL + "/apis/company.com/" + version + "/namespaces/default/foos/test")
	if !assert.NoError(err) {
		t.FailNow()
		return
	}

	if resp.StatusCode != http.StatusOK {
		t.Errorf("unexpected status: %v", resp)
	}

	item := Foo{}
	if err := decodeResponse(resp, &item); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// TODO: validate etcd set things here
	item.ObjectMeta = expectedObj.ObjectMeta

	if !assert.True(reflect.DeepEqual(item, expectedObj)) {
		t.Errorf("expected:\n%v\nsaw:\n%v\n", expectedObj, item)
	}

	path := makeThirdPartyPath("company.com")
	master.RemoveThirdPartyResource(path + "/foos")

	resp, err = http.Get(server.URL + "/apis/company.com/" + version + "/namespaces/default/foos/test")
	if !assert.NoError(err) {
		return
	}

	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("unexpected status: %v", resp)
	}

	expectedDeletedKeys := []string{
		etcdtest.AddPrefix("/ThirdPartyResourceData/company.com/foos/default/test"),
		etcdtest.AddPrefix("/ThirdPartyResourceData/company.com/foos/default/bar"),
	}
	for _, key := range expectedDeletedKeys {
		thirdPartyObj := extensions.ThirdPartyResourceData{}
		s, destroyFunc := generic.NewRawStorage(master.thirdPartyStorageConfig)
		err := s.Get(context.TODO(), key, &thirdPartyObj, false)
		if !storage.IsNotFound(err) {
			t.Errorf("expected deletion didn't happen: %v", err)
		}
		destroyFunc()
	}
	installed := master.ListThirdPartyResources()
	if len(installed) != 0 {
		t.Errorf("Resource(s) still installed: %v", installed)
	}
	services := master.HandlerContainer.RegisteredWebServices()
	for ix := range services {
		if strings.HasPrefix(services[ix].RootPath(), "/apis/company.com") {
			t.Errorf("Web service still installed at %s: %#v", services[ix].RootPath(), services[ix])
		}
	}
}

func TestThirdPartyDiscovery(t *testing.T) {
	for _, version := range versionsToTest {
		testThirdPartyDiscovery(t, version)
	}
}

type FakeTunneler struct {
	SecondsSinceSyncValue       int64
	SecondsSinceSSHKeySyncValue int64
}

func (t *FakeTunneler) Run(genericapiserver.AddressFunc)        {}
func (t *FakeTunneler) Stop()                                   {}
func (t *FakeTunneler) Dial(net, addr string) (net.Conn, error) { return nil, nil }
func (t *FakeTunneler) SecondsSinceSync() int64                 { return t.SecondsSinceSyncValue }
func (t *FakeTunneler) SecondsSinceSSHKeySync() int64           { return t.SecondsSinceSSHKeySyncValue }

// TestIsTunnelSyncHealthy verifies that the 600 second lag test
// is honored.
func TestIsTunnelSyncHealthy(t *testing.T) {
	assert := assert.New(t)
	tunneler := &FakeTunneler{}
	master := &Master{
		GenericAPIServer: &genericapiserver.GenericAPIServer{},
		tunneler:         tunneler,
	}

	// Pass case: 540 second lag
	tunneler.SecondsSinceSyncValue = 540
	err := master.IsTunnelSyncHealthy(nil)
	assert.NoError(err, "IsTunnelSyncHealthy() should not have returned an error.")

	// Fail case: 720 second lag
	tunneler.SecondsSinceSyncValue = 720
	err = master.IsTunnelSyncHealthy(nil)
	assert.Error(err, "IsTunnelSyncHealthy() should have returned an error.")
}

func testThirdPartyDiscovery(t *testing.T, version string) {
	_, etcdserver, server, assert := initThirdParty(t, version, "foo.company.com")
	defer server.Close()
	defer etcdserver.Terminate(t)

	resp, err := http.Get(server.URL + "/apis/company.com/")
	if !assert.NoError(err) {
		return
	}
	assert.Equal(http.StatusOK, resp.StatusCode)

	group := unversioned.APIGroup{}
	assert.NoError(decodeResponse(resp, &group))
	assert.Equal(group.APIVersion, "v1")
	assert.Equal(group.Kind, "APIGroup")
	assert.Equal(group.Name, "company.com")
	expectedVersion := unversioned.GroupVersionForDiscovery{
		GroupVersion: "company.com/" + version,
		Version:      version,
	}

	assert.Equal(group.Versions, []unversioned.GroupVersionForDiscovery{expectedVersion})
	assert.Equal(group.PreferredVersion, expectedVersion)

	resp, err = http.Get(server.URL + "/apis/company.com/" + version)
	if !assert.NoError(err) {
		return
	}
	assert.Equal(http.StatusOK, resp.StatusCode)

	resourceList := unversioned.APIResourceList{}
	assert.NoError(decodeResponse(resp, &resourceList))
	assert.Equal(resourceList.APIVersion, "v1")
	assert.Equal(resourceList.Kind, "APIResourceList")
	assert.Equal(resourceList.GroupVersion, "company.com/"+version)
	assert.Equal(resourceList.APIResources, []unversioned.APIResource{
		{
			Name:       "foos",
			Namespaced: true,
			Kind:       "Foo",
		},
	})
}

// TestValidOpenAPISpec verifies that the open api is added
// at the proper endpoint and the spec is valid.
func TestValidOpenAPISpec(t *testing.T) {
	_, etcdserver, config, assert := setUp(t)
	defer etcdserver.Terminate(t)

	config.EnableOpenAPISupport = true
	config.OpenAPIInfo = spec.Info{
		InfoProps: spec.InfoProps{
			Title:   "Kubernetes",
			Version: "unversioned",
		},
	}
	master, err := New(&config)
	if err != nil {
		t.Fatalf("Error in bringing up the master: %v", err)
	}

	// make sure swagger.json is not registered before calling install api.
	server := httptest.NewServer(master.HandlerContainer.ServeMux)
	resp, err := http.Get(server.URL + "/swagger.json")
	if !assert.NoError(err) {
		t.Errorf("unexpected error: %v", err)
	}
	assert.Equal(http.StatusNotFound, resp.StatusCode)

	master.InstallOpenAPI()
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

	// Validate OpenApi spec
	doc, err := loads.Spec(server.URL + "/swagger.json")
	if assert.NoError(err) {
		// TODO(mehdy): This test is timing out on jerkin but passing locally. Enable it after debugging timeout issue.
		_ = validate.NewSpecValidator(doc.Schema(), strfmt.Default)
		// res, _ := validator.Validate(doc)
		// assert.NoError(res.AsError())
	}
}
