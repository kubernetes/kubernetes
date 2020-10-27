/*
Copyright 2016 The Kubernetes Authors.

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

package rest

import (
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"k8s.io/klog/v2"

	"k8s.io/apimachinery/pkg/runtime/schema"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/apiserver/pkg/storage/etcd3"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	policyclient "k8s.io/client-go/kubernetes/typed/policy/v1beta1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/cluster/ports"
	"k8s.io/kubernetes/pkg/features"
	kubeletclient "k8s.io/kubernetes/pkg/kubelet/client"
	"k8s.io/kubernetes/pkg/registry/core/componentstatus"
	configmapstore "k8s.io/kubernetes/pkg/registry/core/configmap/storage"
	endpointsstore "k8s.io/kubernetes/pkg/registry/core/endpoint/storage"
	eventstore "k8s.io/kubernetes/pkg/registry/core/event/storage"
	limitrangestore "k8s.io/kubernetes/pkg/registry/core/limitrange/storage"
	namespacestore "k8s.io/kubernetes/pkg/registry/core/namespace/storage"
	nodestore "k8s.io/kubernetes/pkg/registry/core/node/storage"
	pvstore "k8s.io/kubernetes/pkg/registry/core/persistentvolume/storage"
	pvcstore "k8s.io/kubernetes/pkg/registry/core/persistentvolumeclaim/storage"
	podstore "k8s.io/kubernetes/pkg/registry/core/pod/storage"
	podtemplatestore "k8s.io/kubernetes/pkg/registry/core/podtemplate/storage"
	"k8s.io/kubernetes/pkg/registry/core/rangeallocation"
	controllerstore "k8s.io/kubernetes/pkg/registry/core/replicationcontroller/storage"
	resourcequotastore "k8s.io/kubernetes/pkg/registry/core/resourcequota/storage"
	secretstore "k8s.io/kubernetes/pkg/registry/core/secret/storage"
	"k8s.io/kubernetes/pkg/registry/core/service/allocator"
	serviceallocator "k8s.io/kubernetes/pkg/registry/core/service/allocator/storage"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
	"k8s.io/kubernetes/pkg/registry/core/service/portallocator"
	servicestore "k8s.io/kubernetes/pkg/registry/core/service/storage"
	serviceaccountstore "k8s.io/kubernetes/pkg/registry/core/serviceaccount/storage"
	kubeschedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/serviceaccount"
	utilsnet "k8s.io/utils/net"
)

// LegacyRESTStorageProvider provides information needed to build RESTStorage for core, but
// does NOT implement the "normal" RESTStorageProvider (yet!)
type LegacyRESTStorageProvider struct {
	StorageFactory serverstorage.StorageFactory
	// Used for custom proxy dialing, and proxy TLS options
	ProxyTransport      http.RoundTripper
	KubeletClientConfig kubeletclient.KubeletClientConfig
	EventTTL            time.Duration

	// ServiceIPRange is used to build cluster IPs for discovery.
	ServiceIPRange net.IPNet
	// allocates ips for secondary service cidr in dual  stack clusters
	SecondaryServiceIPRange net.IPNet
	ServiceNodePortRange    utilnet.PortRange

	ServiceAccountIssuer        serviceaccount.TokenGenerator
	ServiceAccountMaxExpiration time.Duration
	ExtendExpiration            bool

	APIAudiences authenticator.Audiences

	LoopbackClientConfig *restclient.Config
}

// LegacyRESTStorage returns stateful information about particular instances of REST storage to
// master.go for wiring controllers.
// TODO remove this by running the controller as a poststarthook
type LegacyRESTStorage struct {
	ServiceClusterIPAllocator          rangeallocation.RangeRegistry
	SecondaryServiceClusterIPAllocator rangeallocation.RangeRegistry
	ServiceNodePortAllocator           rangeallocation.RangeRegistry
}

func (c LegacyRESTStorageProvider) NewLegacyRESTStorage(restOptionsGetter generic.RESTOptionsGetter) (LegacyRESTStorage, genericapiserver.APIGroupInfo, error) {
	apiGroupInfo := genericapiserver.APIGroupInfo{
		PrioritizedVersions:          legacyscheme.Scheme.PrioritizedVersionsForGroup(""),
		VersionedResourcesStorageMap: map[string]map[string]rest.Storage{},
		Scheme:                       legacyscheme.Scheme,
		ParameterCodec:               legacyscheme.ParameterCodec,
		NegotiatedSerializer:         legacyscheme.Codecs,
	}

	var podDisruptionClient policyclient.PodDisruptionBudgetsGetter
	if policyGroupVersion := (schema.GroupVersion{Group: "policy", Version: "v1beta1"}); legacyscheme.Scheme.IsVersionRegistered(policyGroupVersion) {
		var err error
		podDisruptionClient, err = policyclient.NewForConfig(c.LoopbackClientConfig)
		if err != nil {
			return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, err
		}
	}
	restStorage := LegacyRESTStorage{}

	podTemplateStorage, err := podtemplatestore.NewREST(restOptionsGetter)
	if err != nil {
		return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, err
	}

	eventStorage, err := eventstore.NewREST(restOptionsGetter, uint64(c.EventTTL.Seconds()))
	if err != nil {
		return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, err
	}
	limitRangeStorage, err := limitrangestore.NewREST(restOptionsGetter)
	if err != nil {
		return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, err
	}

	resourceQuotaStorage, resourceQuotaStatusStorage, err := resourcequotastore.NewREST(restOptionsGetter)
	if err != nil {
		return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, err
	}
	secretStorage, err := secretstore.NewREST(restOptionsGetter)
	if err != nil {
		return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, err
	}
	persistentVolumeStorage, persistentVolumeStatusStorage, err := pvstore.NewREST(restOptionsGetter)
	if err != nil {
		return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, err
	}
	persistentVolumeClaimStorage, persistentVolumeClaimStatusStorage, err := pvcstore.NewREST(restOptionsGetter)
	if err != nil {
		return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, err
	}
	configMapStorage, err := configmapstore.NewREST(restOptionsGetter)
	if err != nil {
		return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, err
	}

	namespaceStorage, namespaceStatusStorage, namespaceFinalizeStorage, err := namespacestore.NewREST(restOptionsGetter)
	if err != nil {
		return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, err
	}

	endpointsStorage, err := endpointsstore.NewREST(restOptionsGetter)
	if err != nil {
		return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, err
	}

	nodeStorage, err := nodestore.NewStorage(restOptionsGetter, c.KubeletClientConfig, c.ProxyTransport)
	if err != nil {
		return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, err
	}

	podStorage, err := podstore.NewStorage(
		restOptionsGetter,
		nodeStorage.KubeletConnectionInfo,
		c.ProxyTransport,
		podDisruptionClient,
	)
	if err != nil {
		return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, err
	}

	var serviceAccountStorage *serviceaccountstore.REST
	if c.ServiceAccountIssuer != nil && utilfeature.DefaultFeatureGate.Enabled(features.TokenRequest) {
		serviceAccountStorage, err = serviceaccountstore.NewREST(restOptionsGetter, c.ServiceAccountIssuer, c.APIAudiences, c.ServiceAccountMaxExpiration, podStorage.Pod.Store, secretStorage.Store, c.ExtendExpiration)
	} else {
		serviceAccountStorage, err = serviceaccountstore.NewREST(restOptionsGetter, nil, nil, 0, nil, nil, false)
	}
	if err != nil {
		return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, err
	}

	var serviceClusterIPRegistry rangeallocation.RangeRegistry
	serviceClusterIPRange := c.ServiceIPRange
	if serviceClusterIPRange.IP == nil {
		return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, fmt.Errorf("service clusterIPRange is missing")
	}

	serviceStorageConfig, err := c.StorageFactory.NewConfig(api.Resource("services"))
	if err != nil {
		return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, err
	}

	serviceClusterIPAllocator, err := ipallocator.NewAllocatorCIDRRange(&serviceClusterIPRange, func(max int, rangeSpec string) (allocator.Interface, error) {
		mem := allocator.NewAllocationMap(max, rangeSpec)
		// TODO etcdallocator package to return a storage interface via the storageFactory
		etcd, err := serviceallocator.NewEtcd(mem, "/ranges/serviceips", api.Resource("serviceipallocations"), serviceStorageConfig)
		if err != nil {
			return nil, err
		}
		serviceClusterIPRegistry = etcd
		return etcd, nil
	})
	if err != nil {
		return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, fmt.Errorf("cannot create cluster IP allocator: %v", err)
	}
	restStorage.ServiceClusterIPAllocator = serviceClusterIPRegistry

	// allocator for secondary service ip range
	var secondaryServiceClusterIPAllocator ipallocator.Interface
	if utilfeature.DefaultFeatureGate.Enabled(features.IPv6DualStack) && c.SecondaryServiceIPRange.IP != nil {
		var secondaryServiceClusterIPRegistry rangeallocation.RangeRegistry
		secondaryServiceClusterIPAllocator, err = ipallocator.NewAllocatorCIDRRange(&c.SecondaryServiceIPRange, func(max int, rangeSpec string) (allocator.Interface, error) {
			mem := allocator.NewAllocationMap(max, rangeSpec)
			// TODO etcdallocator package to return a storage interface via the storageFactory
			etcd, err := serviceallocator.NewEtcd(mem, "/ranges/secondaryserviceips", api.Resource("serviceipallocations"), serviceStorageConfig)
			if err != nil {
				return nil, err
			}
			secondaryServiceClusterIPRegistry = etcd
			return etcd, nil
		})
		if err != nil {
			return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, fmt.Errorf("cannot create cluster secondary IP allocator: %v", err)
		}
		restStorage.SecondaryServiceClusterIPAllocator = secondaryServiceClusterIPRegistry
	}

	var serviceNodePortRegistry rangeallocation.RangeRegistry
	serviceNodePortAllocator, err := portallocator.NewPortAllocatorCustom(c.ServiceNodePortRange, func(max int, rangeSpec string) (allocator.Interface, error) {
		mem := allocator.NewAllocationMap(max, rangeSpec)
		// TODO etcdallocator package to return a storage interface via the storageFactory
		etcd, err := serviceallocator.NewEtcd(mem, "/ranges/servicenodeports", api.Resource("servicenodeportallocations"), serviceStorageConfig)
		if err != nil {
			return nil, err
		}
		serviceNodePortRegistry = etcd
		return etcd, nil
	})
	if err != nil {
		return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, fmt.Errorf("cannot create cluster port allocator: %v", err)
	}
	restStorage.ServiceNodePortAllocator = serviceNodePortRegistry

	controllerStorage, err := controllerstore.NewStorage(restOptionsGetter)
	if err != nil {
		return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, err
	}

	serviceRESTStorage, serviceStatusStorage, err := servicestore.NewGenericREST(restOptionsGetter, serviceClusterIPRange, secondaryServiceClusterIPAllocator != nil)
	if err != nil {
		return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, err
	}

	serviceRest, serviceRestProxy := servicestore.NewREST(serviceRESTStorage,
		endpointsStorage,
		podStorage.Pod,
		serviceClusterIPAllocator,
		secondaryServiceClusterIPAllocator,
		serviceNodePortAllocator,
		c.ProxyTransport)

	restStorageMap := map[string]rest.Storage{
		"pods":             podStorage.Pod,
		"pods/attach":      podStorage.Attach,
		"pods/status":      podStorage.Status,
		"pods/log":         podStorage.Log,
		"pods/exec":        podStorage.Exec,
		"pods/portforward": podStorage.PortForward,
		"pods/proxy":       podStorage.Proxy,
		"pods/binding":     podStorage.Binding,
		"bindings":         podStorage.LegacyBinding,

		"podTemplates": podTemplateStorage,

		"replicationControllers":        controllerStorage.Controller,
		"replicationControllers/status": controllerStorage.Status,

		"services":        serviceRest,
		"services/proxy":  serviceRestProxy,
		"services/status": serviceStatusStorage,

		"endpoints": endpointsStorage,

		"nodes":        nodeStorage.Node,
		"nodes/status": nodeStorage.Status,
		"nodes/proxy":  nodeStorage.Proxy,

		"events": eventStorage,

		"limitRanges":                   limitRangeStorage,
		"resourceQuotas":                resourceQuotaStorage,
		"resourceQuotas/status":         resourceQuotaStatusStorage,
		"namespaces":                    namespaceStorage,
		"namespaces/status":             namespaceStatusStorage,
		"namespaces/finalize":           namespaceFinalizeStorage,
		"secrets":                       secretStorage,
		"serviceAccounts":               serviceAccountStorage,
		"persistentVolumes":             persistentVolumeStorage,
		"persistentVolumes/status":      persistentVolumeStatusStorage,
		"persistentVolumeClaims":        persistentVolumeClaimStorage,
		"persistentVolumeClaims/status": persistentVolumeClaimStatusStorage,
		"configMaps":                    configMapStorage,

		"componentStatuses": componentstatus.NewStorage(componentStatusStorage{c.StorageFactory}.serversToValidate),
	}
	if legacyscheme.Scheme.IsVersionRegistered(schema.GroupVersion{Group: "autoscaling", Version: "v1"}) {
		restStorageMap["replicationControllers/scale"] = controllerStorage.Scale
	}
	if legacyscheme.Scheme.IsVersionRegistered(schema.GroupVersion{Group: "policy", Version: "v1beta1"}) {
		restStorageMap["pods/eviction"] = podStorage.Eviction
	}
	if serviceAccountStorage.Token != nil {
		restStorageMap["serviceaccounts/token"] = serviceAccountStorage.Token
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.EphemeralContainers) {
		restStorageMap["pods/ephemeralcontainers"] = podStorage.EphemeralContainers
	}
	apiGroupInfo.VersionedResourcesStorageMap["v1"] = restStorageMap

	return restStorage, apiGroupInfo, nil
}

func (p LegacyRESTStorageProvider) GroupName() string {
	return api.GroupName
}

type componentStatusStorage struct {
	storageFactory serverstorage.StorageFactory
}

func (s componentStatusStorage) serversToValidate() map[string]*componentstatus.Server {
	// this is fragile, which assumes that the default port is being used
	serversToValidate := map[string]*componentstatus.Server{
		"controller-manager": {Addr: "127.0.0.1", Port: ports.InsecureKubeControllerManagerPort, Path: "/healthz"},
		"scheduler":          {Addr: "127.0.0.1", Port: kubeschedulerconfig.DefaultInsecureSchedulerPort, Path: "/healthz"},
	}

	for ix, machine := range s.storageFactory.Backends() {
		etcdUrl, err := url.Parse(machine.Server)
		if err != nil {
			klog.Errorf("Failed to parse etcd url for validation: %v", err)
			continue
		}
		var port int
		var addr string
		if strings.Contains(etcdUrl.Host, ":") {
			var portString string
			addr, portString, err = net.SplitHostPort(etcdUrl.Host)
			if err != nil {
				klog.Errorf("Failed to split host/port: %s (%v)", etcdUrl.Host, err)
				continue
			}
			port, _ = utilsnet.ParsePort(portString, true)
		} else {
			addr = etcdUrl.Host
			port = 2379
		}
		// TODO: etcd health checking should be abstracted in the storage tier
		serversToValidate[fmt.Sprintf("etcd-%d", ix)] = &componentstatus.Server{
			Addr:        addr,
			EnableHTTPS: etcdUrl.Scheme == "https",
			TLSConfig:   machine.TLSConfig,
			Port:        port,
			Path:        "/health",
			Validate:    etcd3.EtcdHealthCheck,
		}
	}
	return serversToValidate
}
