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
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	autoscalingapiv2beta1 "k8s.io/api/autoscaling/v2beta1"
	autoscalingapiv2beta2 "k8s.io/api/autoscaling/v2beta2"
	batchapiv1beta1 "k8s.io/api/batch/v1beta1"
	certificatesapiv1beta1 "k8s.io/api/certificates/v1beta1"
	discoveryv1beta1 "k8s.io/api/discovery/v1beta1"
	eventsv1beta1 "k8s.io/api/events/v1beta1"
	nodev1beta1 "k8s.io/api/node/v1beta1"
	policyapiv1beta1 "k8s.io/api/policy/v1beta1"
	storageapiv1beta1 "k8s.io/api/storage/v1beta1"
	extensionsapiserver "k8s.io/apiextensions-apiserver/pkg/apiserver"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/version"
	"k8s.io/apiserver/pkg/authorization/authorizerfactory"
	openapinamer "k8s.io/apiserver/pkg/endpoints/openapi"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/options"
	"k8s.io/apiserver/pkg/server/resourceconfig"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/apiserver/pkg/util/openapi"
	utilversion "k8s.io/apiserver/pkg/util/version"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	kubeversion "k8s.io/component-base/version"
	aggregatorscheme "k8s.io/kube-aggregator/pkg/apiserver/scheme"
	netutils "k8s.io/utils/net"

	"k8s.io/kubernetes/pkg/api/legacyscheme"
	flowcontrolv1bet3 "k8s.io/kubernetes/pkg/apis/flowcontrol/v1beta3"
	controlplaneapiserver "k8s.io/kubernetes/pkg/controlplane/apiserver"
	"k8s.io/kubernetes/pkg/controlplane/reconcilers"
	"k8s.io/kubernetes/pkg/controlplane/storageversionhashdata"
	generatedopenapi "k8s.io/kubernetes/pkg/generated/openapi"
	"k8s.io/kubernetes/pkg/kubeapiserver"
	kubeletclient "k8s.io/kubernetes/pkg/kubelet/client"
	certificatesrest "k8s.io/kubernetes/pkg/registry/certificates/rest"
	corerest "k8s.io/kubernetes/pkg/registry/core/rest"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

// setUp is a convenience function for setting up for (most) tests.
func setUp(t *testing.T) (*etcd3testing.EtcdTestServer, Config, *assert.Assertions) {
	server, storageConfig := etcd3testing.NewUnsecuredEtcd3TestClientServer(t)

	config := &Config{
		ControlPlane: controlplaneapiserver.Config{
			Generic: genericapiserver.NewConfig(legacyscheme.Codecs),
			Extra: controlplaneapiserver.Extra{
				APIResourceConfigSource: DefaultAPIResourceConfigSource(),
			},
		},
		Extra: Extra{
			APIServerServicePort:   443,
			MasterCount:            1,
			EndpointReconcilerType: reconcilers.MasterCountReconcilerType,
			ServiceIPRange:         net.IPNet{IP: netutils.ParseIPSloppy("10.0.0.0"), Mask: net.CIDRMask(24, 32)},
		},
	}

	config.ControlPlane.Generic.EffectiveVersion = utilversion.DefaultKubeEffectiveVersion()
	storageFactoryConfig := kubeapiserver.NewStorageFactoryConfig()
	storageFactoryConfig.DefaultResourceEncoding.SetEffectiveVersion(config.ControlPlane.Generic.EffectiveVersion)
	storageConfig.StorageObjectCountTracker = config.ControlPlane.Generic.StorageObjectCountTracker
	resourceEncoding := resourceconfig.MergeResourceEncodingConfigs(storageFactoryConfig.DefaultResourceEncoding, storageFactoryConfig.ResourceEncodingOverrides)
	storageFactory := serverstorage.NewDefaultStorageFactory(*storageConfig, "application/vnd.kubernetes.protobuf", storageFactoryConfig.Serializer, resourceEncoding, DefaultAPIResourceConfigSource(), nil)
	etcdOptions := options.NewEtcdOptions(storageConfig)
	// unit tests don't need watch cache and it leaks lots of goroutines with etcd testing functions during unit tests
	etcdOptions.EnableWatchCache = false
	err := etcdOptions.ApplyWithStorageFactoryTo(storageFactory, config.ControlPlane.Generic)
	if err != nil {
		t.Fatal(err)
	}

	kubeVersion := kubeversion.Get()
	config.ControlPlane.Generic.Authorization.Authorizer = authorizerfactory.NewAlwaysAllowAuthorizer()
	config.ControlPlane.Generic.Version = &kubeVersion
	config.ControlPlane.StorageFactory = storageFactory
	config.ControlPlane.Generic.LoopbackClientConfig = &restclient.Config{APIPath: "/api", ContentConfig: restclient.ContentConfig{NegotiatedSerializer: legacyscheme.Codecs}}
	config.ControlPlane.Generic.PublicAddress = netutils.ParseIPSloppy("192.168.10.4")
	config.ControlPlane.Generic.LegacyAPIGroupPrefixes = sets.NewString("/api")
	config.Extra.KubeletClientConfig = kubeletclient.KubeletClientConfig{Port: 10250}
	config.ControlPlane.ProxyTransport = utilnet.SetTransportDefaults(&http.Transport{
		DialContext:     func(ctx context.Context, network, addr string) (net.Conn, error) { return nil, nil },
		TLSClientConfig: &tls.Config{},
	})

	// set fake SecureServingInfo because the listener port is needed for the kubernetes service
	config.ControlPlane.Generic.SecureServing = &genericapiserver.SecureServingInfo{Listener: fakeLocalhost443Listener{}}

	getOpenAPIDefinitions := openapi.GetOpenAPIDefinitionsWithoutDisabledFeatures(generatedopenapi.GetOpenAPIDefinitions)
	namer := openapinamer.NewDefinitionNamer(legacyscheme.Scheme, extensionsapiserver.Scheme, aggregatorscheme.Scheme)
	config.ControlPlane.Generic.OpenAPIV3Config = genericapiserver.DefaultOpenAPIV3Config(getOpenAPIDefinitions, namer)

	clientset, err := kubernetes.NewForConfig(config.ControlPlane.Generic.LoopbackClientConfig)
	if err != nil {
		t.Fatalf("unable to create client set due to %v", err)
	}
	config.ControlPlane.VersionedInformers = informers.NewSharedInformerFactory(clientset, config.ControlPlane.Generic.LoopbackClientConfig.Timeout)

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

	storageProvider, err := corerest.New(corerest.Config{
		GenericConfig: corerest.GenericConfig{
			StorageFactory:       apiserverCfg.ControlPlane.Extra.StorageFactory,
			EventTTL:             apiserverCfg.ControlPlane.Extra.EventTTL,
			LoopbackClientConfig: apiserverCfg.ControlPlane.Generic.LoopbackClientConfig,
			Informers:            apiserverCfg.ControlPlane.Extra.VersionedInformers,
		},
		Proxy: corerest.ProxyConfig{
			Transport:           apiserverCfg.ControlPlane.Extra.ProxyTransport,
			KubeletClientConfig: apiserverCfg.Extra.KubeletClientConfig,
		},
		Services: corerest.ServicesConfig{
			ClusterIPRange: apiserverCfg.Extra.ServiceIPRange,
			NodePortRange:  apiserverCfg.Extra.ServiceNodePortRange,
		},
	})
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}

	apiGroupInfo, err := storageProvider.NewRESTStorage(serverstorage.NewResourceConfig(), apiserverCfg.ControlPlane.Generic.RESTOptionsGetter)
	if err != nil {
		t.Errorf("failed to create legacy REST storage: %v", err)
	}

	strategyErrors := registrytest.ValidateStorageStrategies(apiGroupInfo.VersionedResourcesStorageMap["v1"])
	for _, err := range strategyErrors {
		t.Error(err)
	}
}

func TestCertificatesRestStorageStrategies(t *testing.T) {
	_, etcdserver, apiserverCfg, _ := newInstance(t)
	defer etcdserver.Terminate(t)

	certStorageProvider := certificatesrest.RESTStorageProvider{}
	apiGroupInfo, err := certStorageProvider.NewRESTStorage(apiserverCfg.ControlPlane.APIResourceConfigSource, apiserverCfg.ControlPlane.Generic.RESTOptionsGetter)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}

	strategyErrors := registrytest.ValidateStorageStrategies(
		apiGroupInfo.VersionedResourcesStorageMap[certificatesapiv1beta1.SchemeGroupVersion.Version])
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
	s.ControlPlane.GenericAPIServer.Handler.ServeHTTP(resp, req)
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

func decodeResponse(resp *http.Response, obj interface{}) error {
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
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

	server := httptest.NewServer(apiserver.ControlPlane.GenericAPIServer.Handler.GoRestfulContainer.ServeMux)

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

	// /apis/autoscaling doesn't exist in release-1.1, so the APIVersion field
	// should be non-empty in the results returned by the server.
	resp, err = http.Get(server.URL + "/apis/autoscaling")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	group := metav1.APIGroup{}
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

	server := httptest.NewServer(apiserver.ControlPlane.GenericAPIServer.Handler.GoRestfulContainer.ServeMux)

	c := &restclient.Config{
		Host:          server.URL,
		APIPath:       "/api",
		ContentConfig: restclient.ContentConfig{NegotiatedSerializer: legacyscheme.Codecs},
	}
	discover := discovery.NewDiscoveryClientForConfigOrDie(c).WithLegacy()
	_, all, err := discover.ServerGroupsAndResources()
	if err != nil {
		t.Error(err)
	}
	var count int
	apiResources := sets.NewString()
	for _, g := range all {
		for _, r := range g.APIResources {
			apiResources.Insert(g.GroupVersion + "/" + r.Name)
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
		knownResources := sets.StringKeySet(storageversionhashdata.GVRToStorageVersionHash)
		t.Errorf("please remove the redundant entries from GVRToStorageVersionHash: %v", knownResources.Difference(apiResources).List())
	}
}

func TestNoAlphaVersionsEnabledByDefault(t *testing.T) {
	config := DefaultAPIResourceConfigSource()
	for gv, enable := range config.GroupVersionConfigs {
		if enable && strings.Contains(gv.Version, "alpha") {
			t.Errorf("Alpha API version %s enabled by default", gv.String())
		}
	}

	for gvr, enabled := range config.ResourceConfigs {
		if !strings.Contains(gvr.Version, "alpha") || !enabled {
			continue
		}

		// we have enabled an alpha api by resource {g,v,r}, we also expect the
		// alpha api by version {g,v} to be disabled. This is so a programmer
		// remembers to add the new alpha version to alphaAPIGroupVersionsDisabledByDefault.
		gr := gvr.GroupVersion()
		if enabled, found := config.GroupVersionConfigs[gr]; !found || enabled {
			t.Errorf("Alpha API version %q should be disabled by default", gr.String())
		}
	}
}

func TestNoBetaVersionsEnabledByDefault(t *testing.T) {
	config := DefaultAPIResourceConfigSource()
	for gv, enable := range config.GroupVersionConfigs {
		if enable && strings.Contains(gv.Version, "beta") {
			t.Errorf("Beta API version %s enabled by default", gv.String())
		}
	}

	for gvr, enabled := range config.ResourceConfigs {
		if !strings.Contains(gvr.Version, "beta") || !enabled {
			continue
		}

		// we have enabled a beta api by resource {g,v,r}, we also expect the
		// beta api by version {g,v} to be disabled. This is so a programmer
		// remembers to add the new beta version to betaAPIGroupVersionsDisabledByDefault.
		gr := gvr.GroupVersion()
		if enabled, found := config.GroupVersionConfigs[gr]; !found || enabled {
			t.Errorf("Beta API version %q should be disabled by default", gr.String())
		}
	}
}

func TestDefaultVars(t *testing.T) {
	// stableAPIGroupVersionsEnabledByDefault should not contain beta or alpha
	for i := range stableAPIGroupVersionsEnabledByDefault {
		gv := stableAPIGroupVersionsEnabledByDefault[i]
		if strings.Contains(gv.Version, "beta") || strings.Contains(gv.Version, "alpha") {
			t.Errorf("stableAPIGroupVersionsEnabledByDefault should contain stable version, but found: %q", gv.String())
		}
	}

	// legacyBetaEnabledByDefaultResources should contain only beta version
	for i := range legacyBetaEnabledByDefaultResources {
		gv := legacyBetaEnabledByDefaultResources[i]
		if !strings.Contains(gv.Version, "beta") {
			t.Errorf("legacyBetaEnabledByDefaultResources should contain beta version, but found: %q", gv.String())
		}
	}

	// betaAPIGroupVersionsDisabledByDefault should contain only beta version
	for i := range betaAPIGroupVersionsDisabledByDefault {
		gv := betaAPIGroupVersionsDisabledByDefault[i]
		if !strings.Contains(gv.Version, "beta") {
			t.Errorf("betaAPIGroupVersionsDisabledByDefault should contain beta version, but found: %q", gv.String())
		}
	}

	// alphaAPIGroupVersionsDisabledByDefault should contain only alpha version
	for i := range alphaAPIGroupVersionsDisabledByDefault {
		gv := alphaAPIGroupVersionsDisabledByDefault[i]
		if !strings.Contains(gv.Version, "alpha") {
			t.Errorf("alphaAPIGroupVersionsDisabledByDefault should contain alpha version, but found: %q", gv.String())
		}
	}
}

func TestNewBetaResourcesEnabledByDefault(t *testing.T) {
	// legacyEnabledBetaResources is nearly a duplication from elsewhere.  This is intentional.  These types already have
	// GA equivalents available and should therefore never have a beta enabled by default again.
	legacyEnabledBetaResources := map[schema.GroupVersionResource]bool{
		autoscalingapiv2beta1.SchemeGroupVersion.WithResource("horizontalpodautoscalers"): true,
		autoscalingapiv2beta2.SchemeGroupVersion.WithResource("horizontalpodautoscalers"): true,
		batchapiv1beta1.SchemeGroupVersion.WithResource("cronjobs"):                       true,
		discoveryv1beta1.SchemeGroupVersion.WithResource("endpointslices"):                true,
		eventsv1beta1.SchemeGroupVersion.WithResource("events"):                           true,
		nodev1beta1.SchemeGroupVersion.WithResource("runtimeclasses"):                     true,
		policyapiv1beta1.SchemeGroupVersion.WithResource("poddisruptionbudgets"):          true,
		policyapiv1beta1.SchemeGroupVersion.WithResource("podsecuritypolicies"):           true,
		storageapiv1beta1.SchemeGroupVersion.WithResource("csinodes"):                     true,
	}

	// legacyBetaResourcesWithoutStableEquivalents contains those groupresources that were enabled by default as beta
	// before we changed that policy and do not have stable versions. These resources are allowed to have additional
	// beta versions enabled by default.  Nothing new should be added here.  There are no future exceptions because there
	// are no more beta resources enabled by default.
	legacyBetaResourcesWithoutStableEquivalents := map[schema.GroupResource]bool{
		flowcontrolv1bet3.SchemeGroupVersion.WithResource("flowschemas").GroupResource():                 true,
		flowcontrolv1bet3.SchemeGroupVersion.WithResource("prioritylevelconfigurations").GroupResource(): true,
	}

	config := DefaultAPIResourceConfigSource()
	for gvr, enable := range config.ResourceConfigs {
		if !strings.Contains(gvr.Version, "beta") {
			continue // only check beta things
		}
		if !enable {
			continue // only check things that are enabled
		}
		if legacyEnabledBetaResources[gvr] {
			continue // this is a legacy beta resource
		}
		if legacyBetaResourcesWithoutStableEquivalents[gvr.GroupResource()] {
			continue // this is another beta of a legacy beta resource with no stable equivalent
		}
		t.Errorf("no new beta resources can be enabled by default, see https://github.com/kubernetes/enhancements/blob/0ad0fc8269165ca300d05ca51c7ce190a79976a5/keps/sig-architecture/3136-beta-apis-off-by-default/README.md: %v", gvr)
	}
}
