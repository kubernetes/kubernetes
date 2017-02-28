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
	"strconv"
	"strings"
	"time"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/runtime/schema"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	etcdutil "k8s.io/apiserver/pkg/storage/etcd/util"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/api"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	policyclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/policy/internalversion"
	kubeletclient "k8s.io/kubernetes/pkg/kubelet/client"
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/pkg/registry/core/componentstatus"
	configmapstore "k8s.io/kubernetes/pkg/registry/core/configmap/storage"
	"k8s.io/kubernetes/pkg/registry/core/endpoint"
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
	"k8s.io/kubernetes/pkg/registry/core/service"
	"k8s.io/kubernetes/pkg/registry/core/service/allocator"
	serviceallocator "k8s.io/kubernetes/pkg/registry/core/service/allocator/storage"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
	"k8s.io/kubernetes/pkg/registry/core/service/portallocator"
	servicestore "k8s.io/kubernetes/pkg/registry/core/service/storage"
	serviceaccountstore "k8s.io/kubernetes/pkg/registry/core/serviceaccount/storage"
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
	ServiceIPRange       net.IPNet
	ServiceNodePortRange utilnet.PortRange

	LoopbackClientConfig *restclient.Config
}

// LegacyRESTStorage returns stateful information about particular instances of REST storage to
// master.go for wiring controllers.
// TODO remove this by running the controller as a poststarthook
type LegacyRESTStorage struct {
	ServiceClusterIPAllocator rangeallocation.RangeRegistry
	ServiceNodePortAllocator  rangeallocation.RangeRegistry
}

func (c LegacyRESTStorageProvider) NewLegacyRESTStorage(restOptionsGetter generic.RESTOptionsGetter, apiResourceConfigSource serverstorage.APIResourceConfigSource) (LegacyRESTStorage, genericapiserver.APIGroupInfo, error) {
	apiGroupInfo := genericapiserver.APIGroupInfo{
		GroupMeta:                    *api.Registry.GroupOrDie(api.GroupName),
		VersionedResourcesStorageMap: map[string]map[string]rest.Storage{},
		Scheme:                      api.Scheme,
		ParameterCodec:              api.ParameterCodec,
		NegotiatedSerializer:        api.Codecs,
		SubresourceGroupVersionKind: map[string]schema.GroupVersionKind{},
	}
	if autoscalingGroupVersion := (schema.GroupVersion{Group: "autoscaling", Version: "v1"}); api.Registry.IsEnabledVersion(autoscalingGroupVersion) {
		apiGroupInfo.SubresourceGroupVersionKind["replicationcontrollers/scale"] = autoscalingGroupVersion.WithKind("Scale")
	}

	var podDisruptionClient policyclient.PodDisruptionBudgetsGetter
	if policyGroupVersion := (schema.GroupVersion{Group: "policy", Version: "v1beta1"}); api.Registry.IsEnabledVersion(policyGroupVersion) {
		apiGroupInfo.SubresourceGroupVersionKind["pods/eviction"] = policyGroupVersion.WithKind("Eviction")

		var err error
		podDisruptionClient, err = policyclient.NewForConfig(c.LoopbackClientConfig)
		if err != nil {
			return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, err
		}
	}
	restStorage := LegacyRESTStorage{}

	podTemplateStorage := podtemplatestore.NewREST(restOptionsGetter)

	limitRangeStorage := limitrangestore.NewREST(restOptionsGetter)

	resourceQuotaStorage, resourceQuotaStatusStorage := resourcequotastore.NewREST(restOptionsGetter)
	serviceAccountStorage := serviceaccountstore.NewREST(restOptionsGetter)
	persistentVolumeStorage, persistentVolumeStatusStorage := pvstore.NewREST(restOptionsGetter)
	persistentVolumeClaimStorage, persistentVolumeClaimStatusStorage := pvcstore.NewREST(restOptionsGetter)

	endpointsStorage := endpointsstore.NewREST(restOptionsGetter)
	endpointRegistry := endpoint.NewRegistry(endpointsStorage)

	nodeStorage, err := nodestore.NewStorage(restOptionsGetter, c.KubeletClientConfig, c.ProxyTransport)
	if err != nil {
		return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, err
	}

	podStorage := podstore.NewStorage(
		restOptionsGetter,
		nodeStorage.KubeletConnectionInfo,
		c.ProxyTransport,
		podDisruptionClient,
	)

	serviceRESTStorage, serviceStatusStorage := servicestore.NewREST(restOptionsGetter)
	serviceRegistry := service.NewRegistry(serviceRESTStorage)

	var serviceClusterIPRegistry rangeallocation.RangeRegistry
	serviceClusterIPRange := c.ServiceIPRange
	if serviceClusterIPRange.IP == nil {
		return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, fmt.Errorf("service clusterIPRange is missing")
	}

	serviceStorageConfig, err := c.StorageFactory.NewConfig(api.Resource("services"))
	if err != nil {
		return LegacyRESTStorage{}, genericapiserver.APIGroupInfo{}, err
	}

	ServiceClusterIPAllocator := ipallocator.NewAllocatorCIDRRange(&serviceClusterIPRange, func(max int, rangeSpec string) allocator.Interface {
		mem := allocator.NewAllocationMap(max, rangeSpec)
		// TODO etcdallocator package to return a storage interface via the storageFactory
		etcd := serviceallocator.NewEtcd(mem, "/ranges/serviceips", api.Resource("serviceipallocations"), serviceStorageConfig)
		serviceClusterIPRegistry = etcd
		return etcd
	})
	restStorage.ServiceClusterIPAllocator = serviceClusterIPRegistry

	var serviceNodePortRegistry rangeallocation.RangeRegistry
	ServiceNodePortAllocator := portallocator.NewPortAllocatorCustom(c.ServiceNodePortRange, func(max int, rangeSpec string) allocator.Interface {
		mem := allocator.NewAllocationMap(max, rangeSpec)
		// TODO etcdallocator package to return a storage interface via the storageFactory
		etcd := serviceallocator.NewEtcd(mem, "/ranges/servicenodeports", api.Resource("servicenodeportallocations"), serviceStorageConfig)
		serviceNodePortRegistry = etcd
		return etcd
	})
	restStorage.ServiceNodePortAllocator = serviceNodePortRegistry

	controllerStorage := controllerstore.NewStorage(restOptionsGetter)

	serviceRest := service.NewStorage(serviceRegistry, endpointRegistry, ServiceClusterIPAllocator, ServiceNodePortAllocator, c.ProxyTransport)

	restStorageMap := map[string]rest.Storage{}
	version := apiv1.SchemeGroupVersion
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("pods")) {
		restStorageMap["pods"] = podStorage.Pod
		restStorageMap["pods/attach"] = podStorage.Attach
		restStorageMap["pods/status"] = podStorage.Status
		restStorageMap["pods/log"] = podStorage.Log
		restStorageMap["pods/exec"] = podStorage.Exec
		restStorageMap["pods/portforward"] = podStorage.PortForward
		restStorageMap["pods/proxy"] = podStorage.Proxy
		restStorageMap["pods/binding"] = podStorage.Binding
	} else {
		glog.V(1).Infof("Skipping disabled resource %s/pods", version.String())
	}
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("bindings")) {
		restStorageMap["bindings"] = podStorage.Binding
	} else {
		glog.V(1).Infof("Skipping disabled resource %s/bindings", version.String())
	}
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("podtemplates")) {
		restStorageMap["podTemplates"] = podTemplateStorage
	} else {
		glog.V(1).Infof("Skipping disabled resource %s/podtemplates", version.String())
	}
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("replicationcontrollers")) {
		restStorageMap["replicationControllers"] = controllerStorage.Controller
		restStorageMap["replicationControllers/status"] = controllerStorage.Status
	} else {
		glog.V(1).Infof("Skipping disabled resource %s/replicationcontrollers", version.String())
	}
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("services")) {
		restStorageMap["services/proxy"] = serviceRest.Proxy
		restStorageMap["services/status"] = serviceStatusStorage
		restStorageMap["services"] = serviceRest.Service
	} else {
		glog.V(1).Infof("Skipping disabled resource %s/services", version.String())
	}
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("endpoints")) {
		restStorageMap["endpoints"] = endpointsStorage
	} else {
		glog.V(1).Infof("Skipping disabled resource %s/endpoints", version.String())
	}
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("nodes")) {
		restStorageMap["nodes/status"] = nodeStorage.Status
		restStorageMap["nodes/proxy"] = nodeStorage.Proxy

		restStorageMap["nodes"] = nodeStorage.Node
	} else {
		glog.V(1).Infof("Skipping disabled resource %s/nodes", version.String())
	}
	c.AddEventsStorage(apiResourceConfigSource, restOptionsGetter, uint64(c.EventTTL.Seconds()), restStorageMap)
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("limitranges")) {
		restStorageMap["limitRanges"] = limitRangeStorage
	} else {
		glog.V(1).Infof("Skipping disabled resource %s/limitranges", version.String())
	}
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("resourcequotas")) {
		restStorageMap["resourceQuotas"] = resourceQuotaStorage
		restStorageMap["resourceQuotas/status"] = resourceQuotaStatusStorage
	} else {
		glog.V(1).Infof("Skipping disabled resource %s/resourcequotas", version.String())
	}
	c.AddNamespacesStorage(apiResourceConfigSource, restOptionsGetter, restStorageMap)
	c.AddSecretsStorage(apiResourceConfigSource, restOptionsGetter, restStorageMap)
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("serviceaccounts")) {
		restStorageMap["serviceAccounts"] = serviceAccountStorage
	} else {
		glog.V(1).Infof("Skipping disabled resource %s/serviceaccounts", version.String())
	}
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("persistentvolumes")) {
		restStorageMap["persistentVolumes"] = persistentVolumeStorage
		restStorageMap["persistentVolumes/status"] = persistentVolumeStatusStorage
		restStorageMap["persistentVolumeClaims"] = persistentVolumeClaimStorage
		restStorageMap["persistentVolumeClaims/status"] = persistentVolumeClaimStatusStorage
	} else {
		glog.V(1).Infof("Skipping disabled resource %s/persistentvolumes", version.String())
	}
	c.AddConfigMapsStorage(apiResourceConfigSource, restOptionsGetter, restStorageMap)
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("componentstatuses")) {
		restStorageMap["componentStatuses"] = componentstatus.NewStorage(componentStatusStorage{c.StorageFactory}.serversToValidate)
	} else {
		glog.V(1).Infof("Skipping disabled resource %s/componentstatuses", version.String())
	}

	if api.Registry.IsEnabledVersion(schema.GroupVersion{Group: "autoscaling", Version: "v1"}) {
		restStorageMap["replicationControllers/scale"] = controllerStorage.Scale
	}
	if api.Registry.IsEnabledVersion(schema.GroupVersion{Group: "policy", Version: "v1beta1"}) {
		restStorageMap["pods/eviction"] = podStorage.Eviction
	}
	apiGroupInfo.VersionedResourcesStorageMap["v1"] = restStorageMap

	return restStorage, apiGroupInfo, nil
}

// Adds rest storage interfaces for /events resource to the given storage map.
func (c LegacyRESTStorageProvider) AddEventsStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter, ttl uint64, restStorageMap map[string]rest.Storage) {
	version := apiv1.SchemeGroupVersion
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("events")) {
		eventStorage := eventstore.NewREST(restOptionsGetter, ttl)
		restStorageMap["events"] = eventStorage
	} else {
		glog.V(1).Infof("Skipping disabled resource %s/events", version.String())
	}
}

// Adds rest storage interfaces for /namespaces resource to the given storage map.
func (c LegacyRESTStorageProvider) AddNamespacesStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter, restStorageMap map[string]rest.Storage) {
	version := apiv1.SchemeGroupVersion
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("namespaces")) {
		namespaceStorage, namespaceStatusStorage, namespaceFinalizeStorage := namespacestore.NewREST(restOptionsGetter)
		restStorageMap["namespaces"] = namespaceStorage
		restStorageMap["namespaces/status"] = namespaceStatusStorage
		restStorageMap["namespaces/finalize"] = namespaceFinalizeStorage
	} else {
		glog.V(1).Infof("Skipping disabled resource %s/namespaces", version.String())
	}
}

// Adds rest storage interfaces for /secrets resource to the given storage map.
func (c LegacyRESTStorageProvider) AddSecretsStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter, restStorageMap map[string]rest.Storage) {
	version := apiv1.SchemeGroupVersion
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("secrets")) {
		secretStorage := secretstore.NewREST(restOptionsGetter)
		restStorageMap["secrets"] = secretStorage
	} else {
		glog.V(1).Infof("Skipping disabled resource %s/secrets", version.String())
	}
}

// Adds rest storage interfaces for /configmaps resource to the given storage map.
func (c LegacyRESTStorageProvider) AddConfigMapsStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter, restStorageMap map[string]rest.Storage) {
	version := apiv1.SchemeGroupVersion
	if apiResourceConfigSource.ResourceEnabled(version.WithResource("configmaps")) {
		configMapStorage := configmapstore.NewREST(restOptionsGetter)
		restStorageMap["configMaps"] = configMapStorage
	} else {
		glog.V(1).Infof("Skipping disabled resource %s/configmaps", version.String())
	}
}

func (p LegacyRESTStorageProvider) GroupName() string {
	return api.GroupName
}

type componentStatusStorage struct {
	storageFactory serverstorage.StorageFactory
}

func (s componentStatusStorage) serversToValidate() map[string]componentstatus.Server {
	serversToValidate := map[string]componentstatus.Server{
		"controller-manager": {Addr: "127.0.0.1", Port: ports.ControllerManagerPort, Path: "/healthz"},
		"scheduler":          {Addr: "127.0.0.1", Port: ports.SchedulerPort, Path: "/healthz"},
	}

	for ix, machine := range s.storageFactory.Backends() {
		etcdUrl, err := url.Parse(machine)
		if err != nil {
			glog.Errorf("Failed to parse etcd url for validation: %v", err)
			continue
		}
		var port int
		var addr string
		if strings.Contains(etcdUrl.Host, ":") {
			var portString string
			addr, portString, err = net.SplitHostPort(etcdUrl.Host)
			if err != nil {
				glog.Errorf("Failed to split host/port: %s (%v)", etcdUrl.Host, err)
				continue
			}
			port, _ = strconv.Atoi(portString)
		} else {
			addr = etcdUrl.Host
			port = 2379
		}
		// TODO: etcd health checking should be abstracted in the storage tier
		serversToValidate[fmt.Sprintf("etcd-%d", ix)] = componentstatus.Server{
			Addr:        addr,
			EnableHTTPS: etcdUrl.Scheme == "https",
			Port:        port,
			Path:        "/health",
			Validate:    etcdutil.EtcdHealthCheck,
		}
	}
	return serversToValidate
}
