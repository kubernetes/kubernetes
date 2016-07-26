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
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/api/unversioned"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apis/apps"
	appsapi "k8s.io/kubernetes/pkg/apis/apps/v1alpha1"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	autoscalingapiv1 "k8s.io/kubernetes/pkg/apis/autoscaling/v1"
	"k8s.io/kubernetes/pkg/apis/batch"
	batchapiv1 "k8s.io/kubernetes/pkg/apis/batch/v1"
	batchapiv2alpha1 "k8s.io/kubernetes/pkg/apis/batch/v2alpha1"
	"k8s.io/kubernetes/pkg/apis/certificates"
	certificatesapiv1alpha1 "k8s.io/kubernetes/pkg/apis/certificates/v1alpha1"
	"k8s.io/kubernetes/pkg/apis/extensions"
	extensionsapiv1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/apis/policy"
	policyapiv1alpha1 "k8s.io/kubernetes/pkg/apis/policy/v1alpha1"
	"k8s.io/kubernetes/pkg/apis/rbac"
	rbacapi "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1"
	rbacvalidation "k8s.io/kubernetes/pkg/apis/rbac/validation"
	"k8s.io/kubernetes/pkg/apiserver"
	apiservermetrics "k8s.io/kubernetes/pkg/apiserver/metrics"
	"k8s.io/kubernetes/pkg/genericapiserver"
	"k8s.io/kubernetes/pkg/healthz"
	kubeletclient "k8s.io/kubernetes/pkg/kubelet/client"
	"k8s.io/kubernetes/pkg/master/ports"
	certificateetcd "k8s.io/kubernetes/pkg/registry/certificates/etcd"
	"k8s.io/kubernetes/pkg/registry/clusterrole"
	clusterroleetcd "k8s.io/kubernetes/pkg/registry/clusterrole/etcd"
	clusterrolepolicybased "k8s.io/kubernetes/pkg/registry/clusterrole/policybased"
	"k8s.io/kubernetes/pkg/registry/clusterrolebinding"
	clusterrolebindingetcd "k8s.io/kubernetes/pkg/registry/clusterrolebinding/etcd"
	clusterrolebindingpolicybased "k8s.io/kubernetes/pkg/registry/clusterrolebinding/policybased"
	"k8s.io/kubernetes/pkg/registry/componentstatus"
	configmapetcd "k8s.io/kubernetes/pkg/registry/configmap/etcd"
	controlleretcd "k8s.io/kubernetes/pkg/registry/controller/etcd"
	deploymentetcd "k8s.io/kubernetes/pkg/registry/deployment/etcd"
	"k8s.io/kubernetes/pkg/registry/endpoint"
	endpointsetcd "k8s.io/kubernetes/pkg/registry/endpoint/etcd"
	eventetcd "k8s.io/kubernetes/pkg/registry/event/etcd"
	expcontrolleretcd "k8s.io/kubernetes/pkg/registry/experimental/controller/etcd"
	"k8s.io/kubernetes/pkg/registry/generic"
	ingressetcd "k8s.io/kubernetes/pkg/registry/ingress/etcd"
	jobetcd "k8s.io/kubernetes/pkg/registry/job/etcd"
	limitrangeetcd "k8s.io/kubernetes/pkg/registry/limitrange/etcd"
	"k8s.io/kubernetes/pkg/registry/namespace"
	namespaceetcd "k8s.io/kubernetes/pkg/registry/namespace/etcd"
	networkpolicyetcd "k8s.io/kubernetes/pkg/registry/networkpolicy/etcd"
	"k8s.io/kubernetes/pkg/registry/node"
	nodeetcd "k8s.io/kubernetes/pkg/registry/node/etcd"
	pvetcd "k8s.io/kubernetes/pkg/registry/persistentvolume/etcd"
	pvcetcd "k8s.io/kubernetes/pkg/registry/persistentvolumeclaim/etcd"
	petsetetcd "k8s.io/kubernetes/pkg/registry/petset/etcd"
	podetcd "k8s.io/kubernetes/pkg/registry/pod/etcd"
	poddisruptionbudgetetcd "k8s.io/kubernetes/pkg/registry/poddisruptionbudget/etcd"
	pspetcd "k8s.io/kubernetes/pkg/registry/podsecuritypolicy/etcd"
	podtemplateetcd "k8s.io/kubernetes/pkg/registry/podtemplate/etcd"
	replicasetetcd "k8s.io/kubernetes/pkg/registry/replicaset/etcd"
	resourcequotaetcd "k8s.io/kubernetes/pkg/registry/resourcequota/etcd"
	"k8s.io/kubernetes/pkg/registry/role"
	roleetcd "k8s.io/kubernetes/pkg/registry/role/etcd"
	rolepolicybased "k8s.io/kubernetes/pkg/registry/role/policybased"
	"k8s.io/kubernetes/pkg/registry/rolebinding"
	rolebindingetcd "k8s.io/kubernetes/pkg/registry/rolebinding/etcd"
	rolebindingpolicybased "k8s.io/kubernetes/pkg/registry/rolebinding/policybased"
	secretetcd "k8s.io/kubernetes/pkg/registry/secret/etcd"
	"k8s.io/kubernetes/pkg/registry/service"
	etcdallocator "k8s.io/kubernetes/pkg/registry/service/allocator/etcd"
	serviceetcd "k8s.io/kubernetes/pkg/registry/service/etcd"
	ipallocator "k8s.io/kubernetes/pkg/registry/service/ipallocator"
	serviceaccountetcd "k8s.io/kubernetes/pkg/registry/serviceaccount/etcd"
	thirdpartyresourceetcd "k8s.io/kubernetes/pkg/registry/thirdpartyresource/etcd"
	"k8s.io/kubernetes/pkg/registry/thirdpartyresourcedata"
	thirdpartyresourcedataetcd "k8s.io/kubernetes/pkg/registry/thirdpartyresourcedata/etcd"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	etcdmetrics "k8s.io/kubernetes/pkg/storage/etcd/metrics"
	etcdutil "k8s.io/kubernetes/pkg/storage/etcd/util"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wait"

	daemonetcd "k8s.io/kubernetes/pkg/registry/daemonset/etcd"
	horizontalpodautoscaleretcd "k8s.io/kubernetes/pkg/registry/horizontalpodautoscaler/etcd"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
	"k8s.io/kubernetes/pkg/registry/service/allocator"
	"k8s.io/kubernetes/pkg/registry/service/portallocator"
)

const (
	// DefaultEndpointReconcilerInterval is the default amount of time for how often the endpoints for
	// the kubernetes Service are reconciled.
	DefaultEndpointReconcilerInterval = 10 * time.Second
)

type Config struct {
	*genericapiserver.Config

	EnableCoreControllers    bool
	EndpointReconcilerConfig EndpointReconcilerConfig
	DeleteCollectionWorkers  int
	EventTTL                 time.Duration
	KubeletClient            kubeletclient.KubeletClient
	// RESTStorageProviders provides RESTStorage building methods keyed by groupName
	RESTStorageProviders map[string]RESTStorageProvider
	// Used to start and monitor tunneling
	Tunneler genericapiserver.Tunneler

	disableThirdPartyControllerForTesting bool
}

// EndpointReconcilerConfig holds the endpoint reconciler and endpoint reconciliation interval to be
// used by the master.
type EndpointReconcilerConfig struct {
	Reconciler EndpointReconciler
	Interval   time.Duration
}

// Master contains state for a Kubernetes cluster master/api server.
type Master struct {
	*genericapiserver.GenericAPIServer

	// Map of v1 resources to their REST storages.
	v1ResourcesStorage map[string]rest.Storage

	enableCoreControllers   bool
	deleteCollectionWorkers int
	// registries are internal client APIs for accessing the storage layer
	// TODO: define the internal typed interface in a way that clients can
	// also be replaced
	nodeRegistry              node.Registry
	namespaceRegistry         namespace.Registry
	serviceRegistry           service.Registry
	endpointRegistry          endpoint.Registry
	serviceClusterIPAllocator service.RangeRegistry
	serviceNodePortAllocator  service.RangeRegistry

	// storage for third party objects
	thirdPartyStorage storage.Interface
	// map from api path to a tuple of (storage for the objects, APIGroup)
	thirdPartyResources map[string]thirdPartyEntry
	// protects the map
	thirdPartyResourcesLock sync.RWMutex
	// Useful for reliable testing.  Shouldn't be used otherwise.
	disableThirdPartyControllerForTesting bool

	// Used to start and monitor tunneling
	tunneler genericapiserver.Tunneler
}

// thirdPartyEntry combines objects storage and API group into one struct
// for easy lookup.
type thirdPartyEntry struct {
	storage *thirdpartyresourcedataetcd.REST
	group   unversioned.APIGroup
}

type RESTOptionsGetter func(resource unversioned.GroupResource) generic.RESTOptions
type RESTStorageProvider func(apiResourceConfigSource genericapiserver.APIResourceConfigSource, restOptionsGetter RESTOptionsGetter) (genericapiserver.APIGroupInfo, bool)

// New returns a new instance of Master from the given config.
// Certain config fields will be set to a default value if unset.
// Certain config fields must be specified, including:
//   KubeletClient
func New(c *Config) (*Master, error) {
	if c.KubeletClient == nil {
		return nil, fmt.Errorf("Master.New() called with config.KubeletClient == nil")
	}

	s, err := genericapiserver.New(c.Config)
	if err != nil {
		return nil, err
	}

	m := &Master{
		GenericAPIServer:        s,
		enableCoreControllers:   c.EnableCoreControllers,
		deleteCollectionWorkers: c.DeleteCollectionWorkers,
		tunneler:                c.Tunneler,

		disableThirdPartyControllerForTesting: c.disableThirdPartyControllerForTesting,
	}

	// Add some hardcoded storage for now.  Append to the map.
	if c.RESTStorageProviders == nil {
		c.RESTStorageProviders = map[string]RESTStorageProvider{}
	}
	c.RESTStorageProviders[autoscaling.GroupName] = buildAutoscalingResources
	c.RESTStorageProviders[batch.GroupName] = buildBatchResources
	c.RESTStorageProviders[policy.GroupName] = buildPolicyResources
	c.RESTStorageProviders[appsapi.GroupName] = buildAppsResources
	c.RESTStorageProviders[rbac.GroupName] = buildRBACResources(c.AuthorizerRBACSuperUser)
	c.RESTStorageProviders[certificates.GroupName] = buildCertificateResources
	m.InstallAPIs(c)

	// TODO: Attempt clean shutdown?
	if m.enableCoreControllers {
		m.NewBootstrapController(c.EndpointReconcilerConfig).Start()
	}

	return m, nil
}

var defaultMetricsHandler = prometheus.Handler().ServeHTTP

// MetricsWithReset is a handler that resets metrics when DELETE is passed to the endpoint.
func MetricsWithReset(w http.ResponseWriter, req *http.Request) {
	if req.Method == "DELETE" {
		apiservermetrics.Reset()
		etcdmetrics.Reset()
		io.WriteString(w, "metrics reset\n")
		return
	}
	defaultMetricsHandler(w, req)
}

func (m *Master) InstallAPIs(c *Config) {
	apiGroupsInfo := []genericapiserver.APIGroupInfo{}

	// Install v1 unless disabled.
	if c.APIResourceConfigSource.AnyResourcesForVersionEnabled(apiv1.SchemeGroupVersion) {
		// Install v1 API.
		m.initV1ResourcesStorage(c)
		apiGroupInfo := genericapiserver.APIGroupInfo{
			GroupMeta: *registered.GroupOrDie(api.GroupName),
			VersionedResourcesStorageMap: map[string]map[string]rest.Storage{
				"v1": m.v1ResourcesStorage,
			},
			IsLegacyGroup:        true,
			Scheme:               api.Scheme,
			ParameterCodec:       api.ParameterCodec,
			NegotiatedSerializer: api.Codecs,
		}
		if autoscalingGroupVersion := (unversioned.GroupVersion{Group: "autoscaling", Version: "v1"}); registered.IsEnabledVersion(autoscalingGroupVersion) {
			apiGroupInfo.SubresourceGroupVersionKind = map[string]unversioned.GroupVersionKind{
				"replicationcontrollers/scale": autoscalingGroupVersion.WithKind("Scale"),
			}
		}
		apiGroupsInfo = append(apiGroupsInfo, apiGroupInfo)
	}

	// Run the tunneler.
	healthzChecks := []healthz.HealthzChecker{}
	if m.tunneler != nil {
		m.tunneler.Run(m.getNodeAddresses)
		healthzChecks = append(healthzChecks, healthz.NamedCheck("SSH Tunnel Check", m.IsTunnelSyncHealthy))
		prometheus.NewGaugeFunc(prometheus.GaugeOpts{
			Name: "apiserver_proxy_tunnel_sync_latency_secs",
			Help: "The time since the last successful synchronization of the SSH tunnels for proxy requests.",
		}, func() float64 { return float64(m.tunneler.SecondsSinceSync()) })
	}
	healthz.InstallHandler(m.MuxHelper, healthzChecks...)

	if c.EnableProfiling {
		m.MuxHelper.HandleFunc("/metrics", MetricsWithReset)
	} else {
		m.MuxHelper.HandleFunc("/metrics", defaultMetricsHandler)
	}

	// Install extensions unless disabled.
	if c.APIResourceConfigSource.AnyResourcesForVersionEnabled(extensionsapiv1beta1.SchemeGroupVersion) {
		var err error
		m.thirdPartyStorage, err = c.StorageFactory.New(extensions.Resource("thirdpartyresources"))
		if err != nil {
			glog.Fatalf("Error getting third party storage: %v", err)
		}
		m.thirdPartyResources = map[string]thirdPartyEntry{}

		extensionResources := m.getExtensionResources(c)
		extensionsGroupMeta := registered.GroupOrDie(extensions.GroupName)

		apiGroupInfo := genericapiserver.APIGroupInfo{
			GroupMeta: *extensionsGroupMeta,
			VersionedResourcesStorageMap: map[string]map[string]rest.Storage{
				"v1beta1": extensionResources,
			},
			OptionsExternalVersion: &registered.GroupOrDie(api.GroupName).GroupVersion,
			Scheme:                 api.Scheme,
			ParameterCodec:         api.ParameterCodec,
			NegotiatedSerializer:   api.Codecs,
		}
		apiGroupsInfo = append(apiGroupsInfo, apiGroupInfo)
	}

	restOptionsGetter := func(resource unversioned.GroupResource) generic.RESTOptions {
		return m.GetRESTOptionsOrDie(c, resource)
	}

	// stabilize order.
	// TODO find a better way to configure priority of groups
	for _, group := range sets.StringKeySet(c.RESTStorageProviders).List() {
		if !c.APIResourceConfigSource.AnyResourcesForGroupEnabled(group) {
			continue
		}
		restStorageBuilder := c.RESTStorageProviders[group]
		apiGroupInfo, enabled := restStorageBuilder(c.APIResourceConfigSource, restOptionsGetter)
		if !enabled {
			continue
		}

		apiGroupsInfo = append(apiGroupsInfo, apiGroupInfo)
	}

	if err := m.InstallAPIGroups(apiGroupsInfo); err != nil {
		glog.Fatalf("Error in registering group versions: %v", err)
	}
}

func (m *Master) initV1ResourcesStorage(c *Config) {
	restOptions := func(resource string) generic.RESTOptions {
		return m.GetRESTOptionsOrDie(c, api.Resource(resource))
	}

	podTemplateStorage := podtemplateetcd.NewREST(restOptions("podTemplates"))

	eventStorage := eventetcd.NewREST(restOptions("events"), uint64(c.EventTTL.Seconds()))
	limitRangeStorage := limitrangeetcd.NewREST(restOptions("limitRanges"))

	resourceQuotaStorage, resourceQuotaStatusStorage := resourcequotaetcd.NewREST(restOptions("resourceQuotas"))
	secretStorage := secretetcd.NewREST(restOptions("secrets"))
	serviceAccountStorage := serviceaccountetcd.NewREST(restOptions("serviceAccounts"))
	persistentVolumeStorage, persistentVolumeStatusStorage := pvetcd.NewREST(restOptions("persistentVolumes"))
	persistentVolumeClaimStorage, persistentVolumeClaimStatusStorage := pvcetcd.NewREST(restOptions("persistentVolumeClaims"))
	configMapStorage := configmapetcd.NewREST(restOptions("configMaps"))

	namespaceStorage, namespaceStatusStorage, namespaceFinalizeStorage := namespaceetcd.NewREST(restOptions("namespaces"))
	m.namespaceRegistry = namespace.NewRegistry(namespaceStorage)

	endpointsStorage := endpointsetcd.NewREST(restOptions("endpoints"))
	m.endpointRegistry = endpoint.NewRegistry(endpointsStorage)

	nodeStorage := nodeetcd.NewStorage(restOptions("nodes"), c.KubeletClient, m.ProxyTransport)
	m.nodeRegistry = node.NewRegistry(nodeStorage.Node)

	podStorage := podetcd.NewStorage(
		restOptions("pods"),
		kubeletclient.ConnectionInfoGetter(nodeStorage.Node),
		m.ProxyTransport,
	)

	serviceRESTStorage, serviceStatusStorage := serviceetcd.NewREST(restOptions("services"))
	m.serviceRegistry = service.NewRegistry(serviceRESTStorage)

	var serviceClusterIPRegistry service.RangeRegistry
	serviceClusterIPRange := m.ServiceClusterIPRange
	if serviceClusterIPRange == nil {
		glog.Fatalf("service clusterIPRange is nil")
		return
	}

	serviceStorage, err := c.StorageFactory.New(api.Resource("services"))
	if err != nil {
		glog.Fatal(err.Error())
	}

	serviceClusterIPAllocator := ipallocator.NewAllocatorCIDRRange(serviceClusterIPRange, func(max int, rangeSpec string) allocator.Interface {
		mem := allocator.NewAllocationMap(max, rangeSpec)
		// TODO etcdallocator package to return a storage interface via the storageFactory
		etcd := etcdallocator.NewEtcd(mem, "/ranges/serviceips", api.Resource("serviceipallocations"), serviceStorage)
		serviceClusterIPRegistry = etcd
		return etcd
	})
	m.serviceClusterIPAllocator = serviceClusterIPRegistry

	var serviceNodePortRegistry service.RangeRegistry
	serviceNodePortAllocator := portallocator.NewPortAllocatorCustom(m.ServiceNodePortRange, func(max int, rangeSpec string) allocator.Interface {
		mem := allocator.NewAllocationMap(max, rangeSpec)
		// TODO etcdallocator package to return a storage interface via the storageFactory
		etcd := etcdallocator.NewEtcd(mem, "/ranges/servicenodeports", api.Resource("servicenodeportallocations"), serviceStorage)
		serviceNodePortRegistry = etcd
		return etcd
	})
	m.serviceNodePortAllocator = serviceNodePortRegistry

	controllerStorage := controlleretcd.NewStorage(restOptions("replicationControllers"))

	serviceRest := service.NewStorage(m.serviceRegistry, m.endpointRegistry, serviceClusterIPAllocator, serviceNodePortAllocator, m.ProxyTransport)

	// TODO: Factor out the core API registration
	m.v1ResourcesStorage = map[string]rest.Storage{
		"pods":             podStorage.Pod,
		"pods/attach":      podStorage.Attach,
		"pods/status":      podStorage.Status,
		"pods/log":         podStorage.Log,
		"pods/exec":        podStorage.Exec,
		"pods/portforward": podStorage.PortForward,
		"pods/proxy":       podStorage.Proxy,
		"pods/binding":     podStorage.Binding,
		"bindings":         podStorage.Binding,

		"podTemplates": podTemplateStorage,

		"replicationControllers":        controllerStorage.Controller,
		"replicationControllers/status": controllerStorage.Status,

		"services":        serviceRest.Service,
		"services/proxy":  serviceRest.Proxy,
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

		"componentStatuses": componentstatus.NewStorage(func() map[string]apiserver.Server { return m.getServersToValidate(c) }),
	}
	if registered.IsEnabledVersion(unversioned.GroupVersion{Group: "autoscaling", Version: "v1"}) {
		m.v1ResourcesStorage["replicationControllers/scale"] = controllerStorage.Scale
	}
}

// NewBootstrapController returns a controller for watching the core capabilities of the master.  If
// endpointReconcilerConfig.Interval is 0, the default value of DefaultEndpointReconcilerInterval
// will be used instead.  If endpointReconcilerConfig.Reconciler is nil, the default
// MasterCountEndpointReconciler will be used.
func (m *Master) NewBootstrapController(endpointReconcilerConfig EndpointReconcilerConfig) *Controller {
	if endpointReconcilerConfig.Interval == 0 {
		endpointReconcilerConfig.Interval = DefaultEndpointReconcilerInterval
	}

	if endpointReconcilerConfig.Reconciler == nil {
		// use a default endpoint	reconciler if nothing is set
		// m.endpointRegistry is set via m.InstallAPIs -> m.initV1ResourcesStorage
		endpointReconcilerConfig.Reconciler = NewMasterCountEndpointReconciler(m.MasterCount, m.endpointRegistry)
	}

	return &Controller{
		NamespaceRegistry: m.namespaceRegistry,
		ServiceRegistry:   m.serviceRegistry,

		EndpointReconciler: endpointReconcilerConfig.Reconciler,
		EndpointInterval:   endpointReconcilerConfig.Interval,

		SystemNamespaces:         []string{api.NamespaceSystem},
		SystemNamespacesInterval: 1 * time.Minute,

		ServiceClusterIPRegistry: m.serviceClusterIPAllocator,
		ServiceClusterIPRange:    m.ServiceClusterIPRange,
		ServiceClusterIPInterval: 3 * time.Minute,

		ServiceNodePortRegistry: m.serviceNodePortAllocator,
		ServiceNodePortRange:    m.ServiceNodePortRange,
		ServiceNodePortInterval: 3 * time.Minute,

		PublicIP: m.ClusterIP,

		ServiceIP:                 m.ServiceReadWriteIP,
		ServicePort:               m.ServiceReadWritePort,
		ExtraServicePorts:         m.ExtraServicePorts,
		ExtraEndpointPorts:        m.ExtraEndpointPorts,
		PublicServicePort:         m.PublicReadWritePort,
		KubernetesServiceNodePort: m.KubernetesServiceNodePort,
	}
}

func (m *Master) getServersToValidate(c *Config) map[string]apiserver.Server {
	serversToValidate := map[string]apiserver.Server{
		"controller-manager": {Addr: "127.0.0.1", Port: ports.ControllerManagerPort, Path: "/healthz"},
		"scheduler":          {Addr: "127.0.0.1", Port: ports.SchedulerPort, Path: "/healthz"},
	}

	for ix, machine := range c.StorageFactory.Backends() {
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
			port = 4001
		}
		// TODO: etcd health checking should be abstracted in the storage tier
		serversToValidate[fmt.Sprintf("etcd-%d", ix)] = apiserver.Server{
			Addr:        addr,
			EnableHTTPS: etcdUrl.Scheme == "https",
			Port:        port,
			Path:        "/health",
			Validate:    etcdutil.EtcdHealthCheck,
		}
	}
	return serversToValidate
}

// HasThirdPartyResource returns true if a particular third party resource currently installed.
func (m *Master) HasThirdPartyResource(rsrc *extensions.ThirdPartyResource) (bool, error) {
	_, group, err := thirdpartyresourcedata.ExtractApiGroupAndKind(rsrc)
	if err != nil {
		return false, err
	}
	path := makeThirdPartyPath(group)
	services := m.HandlerContainer.RegisteredWebServices()
	for ix := range services {
		if services[ix].RootPath() == path {
			return true, nil
		}
	}
	return false, nil
}

func (m *Master) removeThirdPartyStorage(path string) error {
	m.thirdPartyResourcesLock.Lock()
	defer m.thirdPartyResourcesLock.Unlock()
	storage, found := m.thirdPartyResources[path]
	if found {
		if err := m.removeAllThirdPartyResources(storage.storage); err != nil {
			return err
		}
		delete(m.thirdPartyResources, path)
		m.RemoveAPIGroupForDiscovery(getThirdPartyGroupName(path))
	}
	return nil
}

// RemoveThirdPartyResource removes all resources matching `path`.  Also deletes any stored data
func (m *Master) RemoveThirdPartyResource(path string) error {
	if err := m.removeThirdPartyStorage(path); err != nil {
		return err
	}

	services := m.HandlerContainer.RegisteredWebServices()
	for ix := range services {
		root := services[ix].RootPath()
		if root == path || strings.HasPrefix(root, path+"/") {
			m.HandlerContainer.Remove(services[ix])
		}
	}
	return nil
}

func (m *Master) removeAllThirdPartyResources(registry *thirdpartyresourcedataetcd.REST) error {
	ctx := api.NewDefaultContext()
	existingData, err := registry.List(ctx, nil)
	if err != nil {
		return err
	}
	list, ok := existingData.(*extensions.ThirdPartyResourceDataList)
	if !ok {
		return fmt.Errorf("expected a *ThirdPartyResourceDataList, got %#v", list)
	}
	for ix := range list.Items {
		item := &list.Items[ix]
		if _, err := registry.Delete(ctx, item.Name, nil); err != nil {
			return err
		}
	}
	return nil
}

// ListThirdPartyResources lists all currently installed third party resources
func (m *Master) ListThirdPartyResources() []string {
	m.thirdPartyResourcesLock.RLock()
	defer m.thirdPartyResourcesLock.RUnlock()
	result := []string{}
	for key := range m.thirdPartyResources {
		result = append(result, key)
	}
	return result
}

func (m *Master) hasThirdPartyResourceStorage(path string) bool {
	m.thirdPartyResourcesLock.Lock()
	defer m.thirdPartyResourcesLock.Unlock()
	_, found := m.thirdPartyResources[path]
	return found
}

func (m *Master) addThirdPartyResourceStorage(path string, storage *thirdpartyresourcedataetcd.REST, apiGroup unversioned.APIGroup) {
	m.thirdPartyResourcesLock.Lock()
	defer m.thirdPartyResourcesLock.Unlock()
	m.thirdPartyResources[path] = thirdPartyEntry{storage, apiGroup}
	m.AddAPIGroupForDiscovery(apiGroup)
}

// InstallThirdPartyResource installs a third party resource specified by 'rsrc'.  When a resource is
// installed a corresponding RESTful resource is added as a valid path in the web service provided by
// the master.
//
// For example, if you install a resource ThirdPartyResource{ Name: "foo.company.com", Versions: {"v1"} }
// then the following RESTful resource is created on the server:
//   http://<host>/apis/company.com/v1/foos/...
func (m *Master) InstallThirdPartyResource(rsrc *extensions.ThirdPartyResource) error {
	kind, group, err := thirdpartyresourcedata.ExtractApiGroupAndKind(rsrc)
	if err != nil {
		return err
	}
	plural, _ := meta.KindToResource(unversioned.GroupVersionKind{
		Group:   group,
		Version: rsrc.Versions[0].Name,
		Kind:    kind,
	})
	path := makeThirdPartyPath(group)

	thirdparty := m.thirdpartyapi(group, kind, rsrc.Versions[0].Name, plural.Resource)

	// If storage exists, this group has already been added, just update
	// the group with the new API
	if m.hasThirdPartyResourceStorage(path) {
		return thirdparty.UpdateREST(m.HandlerContainer)
	}

	if err := thirdparty.InstallREST(m.HandlerContainer); err != nil {
		glog.Errorf("Unable to setup thirdparty api: %v", err)
	}
	groupVersion := unversioned.GroupVersionForDiscovery{
		GroupVersion: group + "/" + rsrc.Versions[0].Name,
		Version:      rsrc.Versions[0].Name,
	}
	apiGroup := unversioned.APIGroup{
		Name:             group,
		Versions:         []unversioned.GroupVersionForDiscovery{groupVersion},
		PreferredVersion: groupVersion,
	}
	apiserver.AddGroupWebService(api.Codecs, m.HandlerContainer, path, apiGroup)

	m.addThirdPartyResourceStorage(path, thirdparty.Storage[plural.Resource].(*thirdpartyresourcedataetcd.REST), apiGroup)
	apiserver.InstallServiceErrorHandler(api.Codecs, m.HandlerContainer, m.NewRequestInfoResolver(), []string{thirdparty.GroupVersion.String()})
	return nil
}

func (m *Master) thirdpartyapi(group, kind, version, pluralResource string) *apiserver.APIGroupVersion {
	resourceStorage := thirdpartyresourcedataetcd.NewREST(
		generic.RESTOptions{
			Storage:                 m.thirdPartyStorage,
			Decorator:               generic.UndecoratedStorage,
			DeleteCollectionWorkers: m.deleteCollectionWorkers,
		},
		group,
		kind,
	)

	apiRoot := makeThirdPartyPath("")

	storage := map[string]rest.Storage{
		pluralResource: resourceStorage,
	}

	optionsExternalVersion := registered.GroupOrDie(api.GroupName).GroupVersion
	internalVersion := unversioned.GroupVersion{Group: group, Version: runtime.APIVersionInternal}
	externalVersion := unversioned.GroupVersion{Group: group, Version: version}

	return &apiserver.APIGroupVersion{
		Root:                apiRoot,
		GroupVersion:        externalVersion,
		RequestInfoResolver: m.NewRequestInfoResolver(),

		Creater:   thirdpartyresourcedata.NewObjectCreator(group, version, api.Scheme),
		Convertor: api.Scheme,
		Copier:    api.Scheme,
		Typer:     api.Scheme,

		Mapper:                 thirdpartyresourcedata.NewMapper(registered.GroupOrDie(extensions.GroupName).RESTMapper, kind, version, group),
		Linker:                 registered.GroupOrDie(extensions.GroupName).SelfLinker,
		Storage:                storage,
		OptionsExternalVersion: &optionsExternalVersion,

		Serializer:     thirdpartyresourcedata.NewNegotiatedSerializer(api.Codecs, kind, externalVersion, internalVersion),
		ParameterCodec: thirdpartyresourcedata.NewThirdPartyParameterCodec(api.ParameterCodec),

		Context: m.RequestContextMapper,

		MinRequestTimeout: m.MinRequestTimeout,
	}
}

func (m *Master) GetRESTOptionsOrDie(c *Config, resource unversioned.GroupResource) generic.RESTOptions {
	storage, err := c.StorageFactory.New(resource)
	if err != nil {
		glog.Fatalf("Unable to find storage destination for %v, due to %v", resource, err.Error())
	}

	return generic.RESTOptions{
		Storage:                 storage,
		Decorator:               m.StorageDecorator(),
		DeleteCollectionWorkers: m.deleteCollectionWorkers,
	}
}

// getExperimentalResources returns the resources for extensions api
func (m *Master) getExtensionResources(c *Config) map[string]rest.Storage {
	restOptions := func(resource string) generic.RESTOptions {
		return m.GetRESTOptionsOrDie(c, extensions.Resource(resource))
	}

	// TODO update when we support more than one version of this group
	version := extensionsapiv1beta1.SchemeGroupVersion

	storage := map[string]rest.Storage{}

	if c.APIResourceConfigSource.ResourceEnabled(version.WithResource("horizontalpodautoscalers")) {
		hpaStorage, hpaStatusStorage := horizontalpodautoscaleretcd.NewREST(restOptions("horizontalpodautoscalers"))
		storage["horizontalpodautoscalers"] = hpaStorage
		storage["horizontalpodautoscalers/status"] = hpaStatusStorage

		controllerStorage := expcontrolleretcd.NewStorage(m.GetRESTOptionsOrDie(c, api.Resource("replicationControllers")))
		storage["replicationcontrollers"] = controllerStorage.ReplicationController
		storage["replicationcontrollers/scale"] = controllerStorage.Scale
	}
	thirdPartyResourceStorage := thirdpartyresourceetcd.NewREST(restOptions("thirdpartyresources"))
	if c.APIResourceConfigSource.ResourceEnabled(version.WithResource("thirdpartyresources")) {
		thirdPartyControl := ThirdPartyController{
			master: m,
			thirdPartyResourceRegistry: thirdPartyResourceStorage,
		}
		if !m.disableThirdPartyControllerForTesting {
			go wait.Forever(func() {
				if err := thirdPartyControl.SyncResources(); err != nil {
					glog.Warningf("third party resource sync failed: %v", err)
				}
			}, 10*time.Second)
		}
		storage["thirdpartyresources"] = thirdPartyResourceStorage
	}

	daemonSetStorage, daemonSetStatusStorage := daemonetcd.NewREST(restOptions("daemonsets"))
	if c.APIResourceConfigSource.ResourceEnabled(version.WithResource("daemonsets")) {
		storage["daemonsets"] = daemonSetStorage
		storage["daemonsets/status"] = daemonSetStatusStorage
	}
	deploymentStorage := deploymentetcd.NewStorage(restOptions("deployments"))
	if c.APIResourceConfigSource.ResourceEnabled(version.WithResource("deployments")) {
		storage["deployments"] = deploymentStorage.Deployment
		storage["deployments/status"] = deploymentStorage.Status
		storage["deployments/rollback"] = deploymentStorage.Rollback
		storage["deployments/scale"] = deploymentStorage.Scale
	}
	if c.APIResourceConfigSource.ResourceEnabled(version.WithResource("jobs")) {
		jobsStorage, jobsStatusStorage := jobetcd.NewREST(restOptions("jobs"))
		storage["jobs"] = jobsStorage
		storage["jobs/status"] = jobsStatusStorage
	}
	ingressStorage, ingressStatusStorage := ingressetcd.NewREST(restOptions("ingresses"))
	if c.APIResourceConfigSource.ResourceEnabled(version.WithResource("ingresses")) {
		storage["ingresses"] = ingressStorage
		storage["ingresses/status"] = ingressStatusStorage
	}
	podSecurityPolicyStorage := pspetcd.NewREST(restOptions("podsecuritypolicy"))
	if c.APIResourceConfigSource.ResourceEnabled(version.WithResource("podsecuritypolicy")) {
		storage["podSecurityPolicies"] = podSecurityPolicyStorage
	}
	replicaSetStorage := replicasetetcd.NewStorage(restOptions("replicasets"))
	if c.APIResourceConfigSource.ResourceEnabled(version.WithResource("replicasets")) {
		storage["replicasets"] = replicaSetStorage.ReplicaSet
		storage["replicasets/status"] = replicaSetStorage.Status
		storage["replicasets/scale"] = replicaSetStorage.Scale
	}
	networkPolicyStorage := networkpolicyetcd.NewREST(restOptions("networkpolicies"))
	if c.APIResourceConfigSource.ResourceEnabled(version.WithResource("networkpolicies")) {
		storage["networkpolicies"] = networkPolicyStorage
	}

	return storage
}

// NewDefaultAPIGroupInfo returns a complete APIGroupInfo stubbed with "normal" values
// exposed for easier composition from other packages
func NewDefaultAPIGroupInfo(group string) genericapiserver.APIGroupInfo {
	groupMeta := registered.GroupOrDie(group)

	return genericapiserver.APIGroupInfo{
		GroupMeta:                    *groupMeta,
		VersionedResourcesStorageMap: map[string]map[string]rest.Storage{},
		OptionsExternalVersion:       &registered.GroupOrDie(api.GroupName).GroupVersion,
		Scheme:                       api.Scheme,
		ParameterCodec:               api.ParameterCodec,
		NegotiatedSerializer:         api.Codecs,
	}
}

func buildCertificateResources(apiResourceConfigSource genericapiserver.APIResourceConfigSource, restOptionsGetter RESTOptionsGetter) (genericapiserver.APIGroupInfo, bool) {
	apiGroupInfo := NewDefaultAPIGroupInfo(certificates.GroupName)

	storageForVersion := func(version unversioned.GroupVersion) map[string]rest.Storage {
		storage := map[string]rest.Storage{}
		if apiResourceConfigSource.ResourceEnabled(version.WithResource("certificatesigningrequests")) {
			csrStorage, csrStatusStorage, csrApprovalStorage := certificateetcd.NewREST(restOptionsGetter(certificates.Resource("certificatesigningrequests")))
			storage["certificatesigningrequests"] = csrStorage
			storage["certificatesigningrequests/status"] = csrStatusStorage
			storage["certificatesigningrequests/approval"] = csrApprovalStorage
		}
		return storage
	}

	if apiResourceConfigSource.AnyResourcesForVersionEnabled(certificatesapiv1alpha1.SchemeGroupVersion) {
		apiGroupInfo.VersionedResourcesStorageMap[certificatesapiv1alpha1.SchemeGroupVersion.Version] = storageForVersion(certificatesapiv1alpha1.SchemeGroupVersion)
		apiGroupInfo.GroupMeta.GroupVersion = certificatesapiv1alpha1.SchemeGroupVersion
	}

	return apiGroupInfo, true
}

func buildAutoscalingResources(apiResourceConfigSource genericapiserver.APIResourceConfigSource, restOptionsGetter RESTOptionsGetter) (genericapiserver.APIGroupInfo, bool) {
	apiGroupInfo := NewDefaultAPIGroupInfo(autoscaling.GroupName)

	storageForVersion := func(version unversioned.GroupVersion) map[string]rest.Storage {
		storage := map[string]rest.Storage{}
		if apiResourceConfigSource.ResourceEnabled(version.WithResource("horizontalpodautoscalers")) {
			hpaStorage, hpaStatusStorage := horizontalpodautoscaleretcd.NewREST(restOptionsGetter(autoscaling.Resource("horizontalpodautoscalers")))
			storage["horizontalpodautoscalers"] = hpaStorage
			storage["horizontalpodautoscalers/status"] = hpaStatusStorage
		}
		return storage
	}

	if apiResourceConfigSource.AnyResourcesForVersionEnabled(autoscalingapiv1.SchemeGroupVersion) {
		apiGroupInfo.VersionedResourcesStorageMap[autoscalingapiv1.SchemeGroupVersion.Version] = storageForVersion(autoscalingapiv1.SchemeGroupVersion)
		apiGroupInfo.GroupMeta.GroupVersion = autoscalingapiv1.SchemeGroupVersion
	}

	return apiGroupInfo, true
}

func buildBatchResources(apiResourceConfigSource genericapiserver.APIResourceConfigSource, restOptionsGetter RESTOptionsGetter) (genericapiserver.APIGroupInfo, bool) {
	apiGroupInfo := NewDefaultAPIGroupInfo(batch.GroupName)

	storageForVersion := func(version unversioned.GroupVersion) map[string]rest.Storage {
		storage := map[string]rest.Storage{}
		if apiResourceConfigSource.ResourceEnabled(version.WithResource("jobs")) {
			jobsStorage, jobsStatusStorage := jobetcd.NewREST(restOptionsGetter(batch.Resource("jobs")))
			storage["jobs"] = jobsStorage
			storage["jobs/status"] = jobsStatusStorage
		}
		return storage
	}

	if apiResourceConfigSource.AnyResourcesForVersionEnabled(batchapiv2alpha1.SchemeGroupVersion) {
		apiGroupInfo.VersionedResourcesStorageMap[batchapiv2alpha1.SchemeGroupVersion.Version] = storageForVersion(batchapiv2alpha1.SchemeGroupVersion)
		apiGroupInfo.GroupMeta.GroupVersion = batchapiv2alpha1.SchemeGroupVersion
	}
	if apiResourceConfigSource.AnyResourcesForVersionEnabled(batchapiv1.SchemeGroupVersion) {
		apiGroupInfo.VersionedResourcesStorageMap[batchapiv1.SchemeGroupVersion.Version] = storageForVersion(batchapiv1.SchemeGroupVersion)
		apiGroupInfo.GroupMeta.GroupVersion = batchapiv1.SchemeGroupVersion
	}

	return apiGroupInfo, true
}

func buildPolicyResources(apiResourceConfigSource genericapiserver.APIResourceConfigSource, restOptionsGetter RESTOptionsGetter) (genericapiserver.APIGroupInfo, bool) {
	apiGroupInfo := NewDefaultAPIGroupInfo(policy.GroupName)

	storageForVersion := func(version unversioned.GroupVersion) map[string]rest.Storage {
		storage := map[string]rest.Storage{}
		if apiResourceConfigSource.ResourceEnabled(version.WithResource("poddisruptionbudgets")) {
			poddisruptionbudgetStorage, poddisruptionbudgetStatusStorage := poddisruptionbudgetetcd.NewREST(restOptionsGetter(policy.Resource("poddisruptionbudgets")))
			storage["poddisruptionbudgets"] = poddisruptionbudgetStorage
			storage["poddisruptionbudgets/status"] = poddisruptionbudgetStatusStorage
		}
		return storage
	}

	if apiResourceConfigSource.AnyResourcesForVersionEnabled(policyapiv1alpha1.SchemeGroupVersion) {
		apiGroupInfo.VersionedResourcesStorageMap[policyapiv1alpha1.SchemeGroupVersion.Version] = storageForVersion(policyapiv1alpha1.SchemeGroupVersion)
		apiGroupInfo.GroupMeta.GroupVersion = policyapiv1alpha1.SchemeGroupVersion
	}

	return apiGroupInfo, true
}

func buildAppsResources(apiResourceConfigSource genericapiserver.APIResourceConfigSource, restOptionsGetter RESTOptionsGetter) (genericapiserver.APIGroupInfo, bool) {
	apiGroupInfo := NewDefaultAPIGroupInfo(appsapi.GroupName)

	storageForVersion := func(version unversioned.GroupVersion) map[string]rest.Storage {
		storage := map[string]rest.Storage{}
		if apiResourceConfigSource.ResourceEnabled(version.WithResource("petsets")) {
			petsetStorage, petsetStatusStorage := petsetetcd.NewREST(restOptionsGetter(apps.Resource("petsets")))
			storage["petsets"] = petsetStorage
			storage["petsets/status"] = petsetStatusStorage
		}
		return storage
	}

	if apiResourceConfigSource.AnyResourcesForVersionEnabled(appsapi.SchemeGroupVersion) {
		apiGroupInfo.VersionedResourcesStorageMap[appsapi.SchemeGroupVersion.Version] = storageForVersion(appsapi.SchemeGroupVersion)
		apiGroupInfo.GroupMeta.GroupVersion = appsapi.SchemeGroupVersion
	}

	return apiGroupInfo, true
}

func buildRBACResources(authorizerRBACSuperUser string) RESTStorageProvider {
	return func(apiResourceConfigSource genericapiserver.APIResourceConfigSource, restOptionsGetter RESTOptionsGetter) (genericapiserver.APIGroupInfo, bool) {
		apiGroupInfo := NewDefaultAPIGroupInfo(rbac.GroupName)

		storageForVersion := func(version unversioned.GroupVersion) map[string]rest.Storage {
			once := new(sync.Once)
			var authorizationRuleResolver rbacvalidation.AuthorizationRuleResolver
			newRuleValidator := func() rbacvalidation.AuthorizationRuleResolver {
				once.Do(func() {
					authorizationRuleResolver = rbacvalidation.NewDefaultRuleResolver(
						role.NewRegistry(roleetcd.NewREST(restOptionsGetter(rbac.Resource("roles")))),
						rolebinding.NewRegistry(rolebindingetcd.NewREST(restOptionsGetter(rbac.Resource("rolebindings")))),
						clusterrole.NewRegistry(clusterroleetcd.NewREST(restOptionsGetter(rbac.Resource("clusterroles")))),
						clusterrolebinding.NewRegistry(clusterrolebindingetcd.NewREST(restOptionsGetter(rbac.Resource("clusterrolebindings")))),
					)
				})
				return authorizationRuleResolver
			}

			storage := map[string]rest.Storage{}
			if apiResourceConfigSource.ResourceEnabled(version.WithResource("roles")) {
				rolesStorage := roleetcd.NewREST(restOptionsGetter(rbac.Resource("roles")))
				storage["roles"] = rolepolicybased.NewStorage(rolesStorage, newRuleValidator(), authorizerRBACSuperUser)
			}
			if apiResourceConfigSource.ResourceEnabled(version.WithResource("rolebindings")) {
				roleBindingsStorage := rolebindingetcd.NewREST(restOptionsGetter(rbac.Resource("rolebindings")))
				storage["rolebindings"] = rolebindingpolicybased.NewStorage(roleBindingsStorage, newRuleValidator(), authorizerRBACSuperUser)
			}
			if apiResourceConfigSource.ResourceEnabled(version.WithResource("clusterroles")) {
				clusterRolesStorage := clusterroleetcd.NewREST(restOptionsGetter(rbac.Resource("clusterroles")))
				storage["clusterroles"] = clusterrolepolicybased.NewStorage(clusterRolesStorage, newRuleValidator(), authorizerRBACSuperUser)
			}
			if apiResourceConfigSource.ResourceEnabled(version.WithResource("clusterrolebindings")) {
				clusterRoleBindingsStorage := clusterrolebindingetcd.NewREST(restOptionsGetter(rbac.Resource("clusterrolebindings")))
				storage["clusterrolebindings"] = clusterrolebindingpolicybased.NewStorage(clusterRoleBindingsStorage, newRuleValidator(), authorizerRBACSuperUser)
			}
			return storage
		}

		if apiResourceConfigSource.AnyResourcesForVersionEnabled(rbacapi.SchemeGroupVersion) {
			apiGroupInfo.VersionedResourcesStorageMap[rbacapi.SchemeGroupVersion.Version] = storageForVersion(rbacapi.SchemeGroupVersion)
			apiGroupInfo.GroupMeta.GroupVersion = rbacapi.SchemeGroupVersion
		}

		return apiGroupInfo, true
	}
}

// findExternalAddress returns ExternalIP of provided node with fallback to LegacyHostIP.
func findExternalAddress(node *api.Node) (string, error) {
	var fallback string
	for ix := range node.Status.Addresses {
		addr := &node.Status.Addresses[ix]
		if addr.Type == api.NodeExternalIP {
			return addr.Address, nil
		}
		if fallback == "" && addr.Type == api.NodeLegacyHostIP {
			fallback = addr.Address
		}
	}
	if fallback != "" {
		return fallback, nil
	}
	return "", fmt.Errorf("Couldn't find external address: %v", node)
}

func (m *Master) getNodeAddresses() ([]string, error) {
	nodes, err := m.nodeRegistry.ListNodes(api.NewDefaultContext(), nil)
	if err != nil {
		return nil, err
	}
	addrs := []string{}
	for ix := range nodes.Items {
		node := &nodes.Items[ix]
		addr, err := findExternalAddress(node)
		if err != nil {
			return nil, err
		}
		addrs = append(addrs, addr)
	}
	return addrs, nil
}

func (m *Master) IsTunnelSyncHealthy(req *http.Request) error {
	if m.tunneler == nil {
		return nil
	}
	lag := m.tunneler.SecondsSinceSync()
	if lag > 600 {
		return fmt.Errorf("Tunnel sync is taking to long: %d", lag)
	}
	sshKeyLag := m.tunneler.SecondsSinceSSHKeySync()
	if sshKeyLag > 600 {
		return fmt.Errorf("SSHKey sync is taking to long: %d", sshKeyLag)
	}
	return nil
}

func DefaultAPIResourceConfigSource() *genericapiserver.ResourceConfig {
	ret := genericapiserver.NewResourceConfig()
	ret.EnableVersions(
		apiv1.SchemeGroupVersion,
		extensionsapiv1beta1.SchemeGroupVersion,
		batchapiv1.SchemeGroupVersion,
		autoscalingapiv1.SchemeGroupVersion,
		appsapi.SchemeGroupVersion,
		policyapiv1alpha1.SchemeGroupVersion,
		rbacapi.SchemeGroupVersion,
		certificatesapiv1alpha1.SchemeGroupVersion,
	)

	// all extensions resources except these are disabled by default
	ret.EnableResources(
		extensionsapiv1beta1.SchemeGroupVersion.WithResource("daemonsets"),
		extensionsapiv1beta1.SchemeGroupVersion.WithResource("deployments"),
		extensionsapiv1beta1.SchemeGroupVersion.WithResource("horizontalpodautoscalers"),
		extensionsapiv1beta1.SchemeGroupVersion.WithResource("ingresses"),
		extensionsapiv1beta1.SchemeGroupVersion.WithResource("jobs"),
		extensionsapiv1beta1.SchemeGroupVersion.WithResource("networkpolicies"),
		extensionsapiv1beta1.SchemeGroupVersion.WithResource("replicasets"),
		extensionsapiv1beta1.SchemeGroupVersion.WithResource("thirdpartyresources"),
	)

	return ret
}
