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
	appsapi "k8s.io/kubernetes/pkg/apis/apps/v1alpha1"
	authenticationv1beta1 "k8s.io/kubernetes/pkg/apis/authentication/v1beta1"
	"k8s.io/kubernetes/pkg/apis/authorization"
	authorizationapiv1beta1 "k8s.io/kubernetes/pkg/apis/authorization/v1beta1"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	autoscalingapiv1 "k8s.io/kubernetes/pkg/apis/autoscaling/v1"
	"k8s.io/kubernetes/pkg/apis/batch"
	batchapiv1 "k8s.io/kubernetes/pkg/apis/batch/v1"
	"k8s.io/kubernetes/pkg/apis/certificates"
	certificatesapiv1alpha1 "k8s.io/kubernetes/pkg/apis/certificates/v1alpha1"
	"k8s.io/kubernetes/pkg/apis/extensions"
	extensionsapiv1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/apis/policy"
	policyapiv1alpha1 "k8s.io/kubernetes/pkg/apis/policy/v1alpha1"
	"k8s.io/kubernetes/pkg/apis/rbac"
	rbacapi "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1"
	"k8s.io/kubernetes/pkg/apis/storage"
	storageapiv1beta1 "k8s.io/kubernetes/pkg/apis/storage/v1beta1"
	"k8s.io/kubernetes/pkg/apiserver"
	"k8s.io/kubernetes/pkg/genericapiserver"
	"k8s.io/kubernetes/pkg/healthz"
	kubeletclient "k8s.io/kubernetes/pkg/kubelet/client"
	"k8s.io/kubernetes/pkg/master/ports"

	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/routes"
	"k8s.io/kubernetes/pkg/runtime"
	etcdutil "k8s.io/kubernetes/pkg/storage/etcd/util"
	"k8s.io/kubernetes/pkg/storage/storagebackend"
	"k8s.io/kubernetes/pkg/util/sets"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"

	// RESTStorage installers
	appsrest "k8s.io/kubernetes/pkg/registry/apps/rest"
	authenticationrest "k8s.io/kubernetes/pkg/registry/authentication/rest"
	authorizationrest "k8s.io/kubernetes/pkg/registry/authorization/rest"
	autoscalingrest "k8s.io/kubernetes/pkg/registry/autoscaling/rest"
	batchrest "k8s.io/kubernetes/pkg/registry/batch/rest"
	certificatesrest "k8s.io/kubernetes/pkg/registry/certificates/rest"
	corerest "k8s.io/kubernetes/pkg/registry/core/rest"
	extensionsrest "k8s.io/kubernetes/pkg/registry/extensions/rest"
	policyrest "k8s.io/kubernetes/pkg/registry/policy/rest"
	rbacrest "k8s.io/kubernetes/pkg/registry/rbac/rest"
	storagerest "k8s.io/kubernetes/pkg/registry/storage/rest"

	// direct etcd registry dependencies
	podetcd "k8s.io/kubernetes/pkg/registry/core/pod/etcd"
	"k8s.io/kubernetes/pkg/registry/extensions/thirdpartyresourcedata"
	thirdpartyresourcedataetcd "k8s.io/kubernetes/pkg/registry/extensions/thirdpartyresourcedata/etcd"
)

const (
	// DefaultEndpointReconcilerInterval is the default amount of time for how often the endpoints for
	// the kubernetes Service are reconciled.
	DefaultEndpointReconcilerInterval = 10 * time.Second
)

type Config struct {
	GenericConfig *genericapiserver.Config

	StorageFactory           genericapiserver.StorageFactory
	EnableWatchCache         bool
	EnableCoreControllers    bool
	EndpointReconcilerConfig EndpointReconcilerConfig
	DeleteCollectionWorkers  int
	EventTTL                 time.Duration
	KubeletClient            kubeletclient.KubeletClient
	// genericapiserver.RESTStorageProviders provides RESTStorage building methods keyed by groupName
	RESTStorageProviders map[string]genericapiserver.RESTStorageProvider
	// Used to start and monitor tunneling
	Tunneler          genericapiserver.Tunneler
	EnableUISupport   bool
	EnableLogsSupport bool

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

	legacyRESTStorageProvider corerest.LegacyRESTStorageProvider
	legacyRESTStorage         corerest.LegacyRESTStorage

	enableCoreControllers   bool
	deleteCollectionWorkers int

	// storage for third party objects
	thirdPartyStorageConfig *storagebackend.Config
	// map from api path to a tuple of (storage for the objects, APIGroup)
	thirdPartyResources map[string]*thirdPartyEntry
	// protects the map
	thirdPartyResourcesLock sync.RWMutex
	// Useful for reliable testing.  Shouldn't be used otherwise.
	disableThirdPartyControllerForTesting bool

	// Used to start and monitor tunneling
	tunneler genericapiserver.Tunneler

	restOptionsFactory restOptionsFactory
}

// thirdPartyEntry combines objects storage and API group into one struct
// for easy lookup.
type thirdPartyEntry struct {
	// Map from plural resource name to entry
	storage map[string]*thirdpartyresourcedataetcd.REST
	group   unversioned.APIGroup
}

type RESTOptionsGetter func(resource unversioned.GroupResource) generic.RESTOptions

type RESTStorageProvider interface {
	NewRESTStorage(apiResourceConfigSource genericapiserver.APIResourceConfigSource, restOptionsGetter RESTOptionsGetter) (groupInfo genericapiserver.APIGroupInfo, enabled bool)
}

type completedConfig struct {
	*Config
}

// Complete fills in any fields not set that are required to have valid data. It's mutating the receiver.
func (c *Config) Complete() completedConfig {
	c.GenericConfig.Complete()

	// enable swagger UI only if general UI support is on
	c.GenericConfig.EnableSwaggerUI = c.GenericConfig.EnableSwaggerUI && c.EnableUISupport

	return completedConfig{c}
}

// SkipComplete provides a way to construct a server instance without config completion.
func (c *Config) SkipComplete() completedConfig {
	return completedConfig{c}
}

// New returns a new instance of Master from the given config.
// Certain config fields will be set to a default value if unset.
// Certain config fields must be specified, including:
//   KubeletClient
func (c completedConfig) New() (*Master, error) {
	if c.KubeletClient == nil {
		return nil, fmt.Errorf("Master.New() called with config.KubeletClient == nil")
	}

	s, err := c.Config.GenericConfig.SkipComplete().New() // completion is done in Complete, no need for a second time
	if err != nil {
		return nil, err
	}

	if c.EnableUISupport {
		routes.UIRedirect{}.Install(s.HandlerContainer)
	}
	if c.EnableLogsSupport {
		routes.Logs{}.Install(s.HandlerContainer)
	}

	m := &Master{
		GenericAPIServer:        s,
		enableCoreControllers:   c.EnableCoreControllers,
		deleteCollectionWorkers: c.DeleteCollectionWorkers,
		tunneler:                c.Tunneler,

		disableThirdPartyControllerForTesting: c.disableThirdPartyControllerForTesting,

		restOptionsFactory: restOptionsFactory{
			deleteCollectionWorkers: c.DeleteCollectionWorkers,
			enableGarbageCollection: c.GenericConfig.EnableGarbageCollection,
			storageFactory:          c.StorageFactory,
		},

		legacyRESTStorageProvider: corerest.LegacyRESTStorageProvider{
			StorageFactory:            c.StorageFactory,
			ProxyTransport:            s.ProxyTransport,
			KubeletClient:             c.KubeletClient,
			EventTTL:                  c.EventTTL,
			ServiceClusterIPRange:     c.GenericConfig.ServiceClusterIPRange,
			ServiceNodePortRange:      c.GenericConfig.ServiceNodePortRange,
			ComponentStatusServerFunc: func() map[string]apiserver.Server { return getServersToValidate(c.StorageFactory) },
		},
	}

	if c.EnableWatchCache {
		m.restOptionsFactory.storageDecorator = registry.StorageWithCacher
	} else {
		m.restOptionsFactory.storageDecorator = generic.UndecoratedStorage
	}

	// Add some hardcoded storage for now.  Append to the map.
	if c.RESTStorageProviders == nil {
		c.RESTStorageProviders = map[string]genericapiserver.RESTStorageProvider{}
	}
	c.RESTStorageProviders[appsapi.GroupName] = appsrest.RESTStorageProvider{}
	c.RESTStorageProviders[authenticationv1beta1.GroupName] = authenticationrest.RESTStorageProvider{Authenticator: c.GenericConfig.Authenticator}
	c.RESTStorageProviders[authorization.GroupName] = authorizationrest.RESTStorageProvider{Authorizer: c.GenericConfig.Authorizer}
	c.RESTStorageProviders[autoscaling.GroupName] = autoscalingrest.RESTStorageProvider{}
	c.RESTStorageProviders[batch.GroupName] = batchrest.RESTStorageProvider{}
	c.RESTStorageProviders[certificates.GroupName] = certificatesrest.RESTStorageProvider{}
	c.RESTStorageProviders[extensions.GroupName] = extensionsrest.RESTStorageProvider{
		ResourceInterface:                     m,
		DisableThirdPartyControllerForTesting: m.disableThirdPartyControllerForTesting,
	}
	c.RESTStorageProviders[policy.GroupName] = policyrest.RESTStorageProvider{}
	c.RESTStorageProviders[rbac.GroupName] = &rbacrest.RESTStorageProvider{AuthorizerRBACSuperUser: c.GenericConfig.AuthorizerRBACSuperUser}
	c.RESTStorageProviders[storage.GroupName] = storagerest.RESTStorageProvider{}
	m.InstallAPIs(c.Config)

	// TODO: Attempt clean shutdown?
	if m.enableCoreControllers {
		m.NewBootstrapController(c.EndpointReconcilerConfig).Start()
	}

	return m, nil
}

func (m *Master) InstallAPIs(c *Config) {
	restOptionsGetter := func(resource unversioned.GroupResource) generic.RESTOptions {
		return m.restOptionsFactory.NewFor(resource)
	}

	apiGroupsInfo := []genericapiserver.APIGroupInfo{}

	// Install v1 unless disabled.
	if c.GenericConfig.APIResourceConfigSource.AnyResourcesForVersionEnabled(apiv1.SchemeGroupVersion) {
		legacyRESTStorage, apiGroupInfo, err := m.legacyRESTStorageProvider.NewLegacyRESTStorage(restOptionsGetter)
		if err != nil {
			glog.Fatalf("Error building core storage: %v", err)
		}
		m.legacyRESTStorage = legacyRESTStorage

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
	healthz.InstallHandler(&m.HandlerContainer.NonSwaggerRoutes, healthzChecks...)

	if c.GenericConfig.EnableProfiling {
		routes.MetricsWithReset{}.Install(m.HandlerContainer)
	} else {
		routes.DefaultMetrics{}.Install(m.HandlerContainer)
	}

	// Install third party resource support if requested
	// TODO seems like this bit ought to be unconditional and the REST API is controlled by the config
	if c.GenericConfig.APIResourceConfigSource.ResourceEnabled(extensionsapiv1beta1.SchemeGroupVersion.WithResource("thirdpartyresources")) {
		var err error
		m.thirdPartyStorageConfig, err = c.StorageFactory.NewConfig(extensions.Resource("thirdpartyresources"))
		if err != nil {
			glog.Fatalf("Error getting third party storage: %v", err)
		}
		m.thirdPartyResources = map[string]*thirdPartyEntry{}
	}

	// stabilize order.
	// TODO find a better way to configure priority of groups
	for _, group := range sets.StringKeySet(c.RESTStorageProviders).List() {
		if !c.GenericConfig.APIResourceConfigSource.AnyResourcesForGroupEnabled(group) {
			glog.V(1).Infof("Skipping disabled API group %q.", group)
			continue
		}
		restStorageBuilder := c.RESTStorageProviders[group]
		apiGroupInfo, enabled := restStorageBuilder.NewRESTStorage(c.GenericConfig.APIResourceConfigSource, restOptionsGetter)
		if !enabled {
			glog.Warningf("Problem initializing API group %q, skipping.", group)
			continue
		}
		glog.V(1).Infof("Enabling API group %q.", group)

		if postHookProvider, ok := restStorageBuilder.(genericapiserver.PostStartHookProvider); ok {
			name, hook, err := postHookProvider.PostStartHook()
			if err != nil {
				glog.Fatalf("Error building PostStartHook: %v", err)
			}
			if err := m.GenericAPIServer.AddPostStartHook(name, hook); err != nil {
				glog.Fatalf("Error registering PostStartHook %q: %v", name, err)
			}
		}

		// This is here so that, if the policy group is present, the eviction
		// subresource handler wil be able to find poddisruptionbudgets
		// TODO(lavalamp) find a better way for groups to discover and interact
		// with each other
		if group == "policy" {
			storage := apiGroupsInfo[0].VersionedResourcesStorageMap["v1"]["pods/eviction"]
			evictionStorage := storage.(*podetcd.EvictionREST)

			storage = apiGroupInfo.VersionedResourcesStorageMap["v1alpha1"]["poddisruptionbudgets"]
			evictionStorage.PodDisruptionBudgetLister = storage.(rest.Lister)
			evictionStorage.PodDisruptionBudgetUpdater = storage.(rest.Updater)
		}

		apiGroupsInfo = append(apiGroupsInfo, apiGroupInfo)
	}

	for i := range apiGroupsInfo {
		if err := m.InstallAPIGroup(&apiGroupsInfo[i]); err != nil {
			glog.Fatalf("Error in registering group versions: %v", err)
		}
	}
}

// NewBootstrapController returns a controller for watching the core capabilities of the master.  If
// endpointReconcilerConfig.Interval is 0, the default value of DefaultEndpointReconcilerInterval
// will be used instead.  If endpointReconcilerConfig.Reconciler is nil, the default
// MasterCountEndpointReconciler will be used.
// TODO this should be kicked off as a server PostHook
func (m *Master) NewBootstrapController(endpointReconcilerConfig EndpointReconcilerConfig) *Controller {
	if endpointReconcilerConfig.Interval == 0 {
		endpointReconcilerConfig.Interval = DefaultEndpointReconcilerInterval
	}

	if endpointReconcilerConfig.Reconciler == nil {
		// use a default endpoint	reconciler if nothing is set
		// m.endpointRegistry is set via m.InstallAPIs -> m.initV1ResourcesStorage
		endpointReconcilerConfig.Reconciler = NewMasterCountEndpointReconciler(m.MasterCount, m.legacyRESTStorage.EndpointRegistry)
	}

	return &Controller{
		NamespaceRegistry: m.legacyRESTStorage.NamespaceRegistry,
		ServiceRegistry:   m.legacyRESTStorage.ServiceRegistry,

		EndpointReconciler: endpointReconcilerConfig.Reconciler,
		EndpointInterval:   endpointReconcilerConfig.Interval,

		SystemNamespaces:         []string{api.NamespaceSystem},
		SystemNamespacesInterval: 1 * time.Minute,

		ServiceClusterIPRegistry: m.legacyRESTStorage.ServiceClusterIPAllocator,
		ServiceClusterIPRange:    m.legacyRESTStorageProvider.ServiceClusterIPRange,
		ServiceClusterIPInterval: 3 * time.Minute,

		ServiceNodePortRegistry: m.legacyRESTStorage.ServiceNodePortAllocator,
		ServiceNodePortRange:    m.legacyRESTStorageProvider.ServiceNodePortRange,
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

func getServersToValidate(storageFactory genericapiserver.StorageFactory) map[string]apiserver.Server {
	serversToValidate := map[string]apiserver.Server{
		"controller-manager": {Addr: "127.0.0.1", Port: ports.ControllerManagerPort, Path: "/healthz"},
		"scheduler":          {Addr: "127.0.0.1", Port: ports.SchedulerPort, Path: "/healthz"},
	}

	for ix, machine := range storageFactory.Backends() {
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
	kind, group, err := thirdpartyresourcedata.ExtractApiGroupAndKind(rsrc)
	if err != nil {
		return false, err
	}
	path := extensionsrest.MakeThirdPartyPath(group)
	m.thirdPartyResourcesLock.Lock()
	defer m.thirdPartyResourcesLock.Unlock()
	entry := m.thirdPartyResources[path]
	if entry == nil {
		return false, nil
	}
	plural, _ := meta.KindToResource(unversioned.GroupVersionKind{
		Group:   group,
		Version: rsrc.Versions[0].Name,
		Kind:    kind,
	})
	_, found := entry.storage[plural.Resource]
	return found, nil
}

func (m *Master) removeThirdPartyStorage(path, resource string) error {
	m.thirdPartyResourcesLock.Lock()
	defer m.thirdPartyResourcesLock.Unlock()
	entry, found := m.thirdPartyResources[path]
	if !found {
		return nil
	}
	storage, found := entry.storage[resource]
	if !found {
		return nil
	}
	if err := m.removeAllThirdPartyResources(storage); err != nil {
		return err
	}
	delete(entry.storage, resource)
	if len(entry.storage) == 0 {
		delete(m.thirdPartyResources, path)
		m.RemoveAPIGroupForDiscovery(extensionsrest.GetThirdPartyGroupName(path))
	} else {
		m.thirdPartyResources[path] = entry
	}
	return nil
}

// RemoveThirdPartyResource removes all resources matching `path`.  Also deletes any stored data
func (m *Master) RemoveThirdPartyResource(path string) error {
	ix := strings.LastIndex(path, "/")
	if ix == -1 {
		return fmt.Errorf("expected <api-group>/<resource-plural-name>, saw: %s", path)
	}
	resource := path[ix+1:]
	path = path[0:ix]

	if err := m.removeThirdPartyStorage(path, resource); err != nil {
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
// The format is <path>/<resource-plural-name>
func (m *Master) ListThirdPartyResources() []string {
	m.thirdPartyResourcesLock.RLock()
	defer m.thirdPartyResourcesLock.RUnlock()
	result := []string{}
	for key := range m.thirdPartyResources {
		for rsrc := range m.thirdPartyResources[key].storage {
			result = append(result, key+"/"+rsrc)
		}
	}
	return result
}

func (m *Master) getExistingThirdPartyResources(path string) []unversioned.APIResource {
	result := []unversioned.APIResource{}
	m.thirdPartyResourcesLock.Lock()
	defer m.thirdPartyResourcesLock.Unlock()
	entry := m.thirdPartyResources[path]
	if entry != nil {
		for key, obj := range entry.storage {
			result = append(result, unversioned.APIResource{
				Name:       key,
				Namespaced: true,
				Kind:       obj.Kind(),
			})
		}
	}
	return result
}

func (m *Master) hasThirdPartyGroupStorage(path string) bool {
	m.thirdPartyResourcesLock.Lock()
	defer m.thirdPartyResourcesLock.Unlock()
	_, found := m.thirdPartyResources[path]
	return found
}

func (m *Master) addThirdPartyResourceStorage(path, resource string, storage *thirdpartyresourcedataetcd.REST, apiGroup unversioned.APIGroup) {
	m.thirdPartyResourcesLock.Lock()
	defer m.thirdPartyResourcesLock.Unlock()
	entry, found := m.thirdPartyResources[path]
	if entry == nil {
		entry = &thirdPartyEntry{
			group:   apiGroup,
			storage: map[string]*thirdpartyresourcedataetcd.REST{},
		}
		m.thirdPartyResources[path] = entry
	}
	entry.storage[resource] = storage
	if !found {
		m.AddAPIGroupForDiscovery(apiGroup)
	}
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
	path := extensionsrest.MakeThirdPartyPath(group)

	groupVersion := unversioned.GroupVersionForDiscovery{
		GroupVersion: group + "/" + rsrc.Versions[0].Name,
		Version:      rsrc.Versions[0].Name,
	}
	apiGroup := unversioned.APIGroup{
		Name:             group,
		Versions:         []unversioned.GroupVersionForDiscovery{groupVersion},
		PreferredVersion: groupVersion,
	}

	thirdparty := m.thirdpartyapi(group, kind, rsrc.Versions[0].Name, plural.Resource)

	// If storage exists, this group has already been added, just update
	// the group with the new API
	if m.hasThirdPartyGroupStorage(path) {
		m.addThirdPartyResourceStorage(path, plural.Resource, thirdparty.Storage[plural.Resource].(*thirdpartyresourcedataetcd.REST), apiGroup)
		return thirdparty.UpdateREST(m.HandlerContainer.Container)
	}

	if err := thirdparty.InstallREST(m.HandlerContainer.Container); err != nil {
		glog.Errorf("Unable to setup thirdparty api: %v", err)
	}
	m.HandlerContainer.Add(apiserver.NewGroupWebService(api.Codecs, path, apiGroup))

	m.addThirdPartyResourceStorage(path, plural.Resource, thirdparty.Storage[plural.Resource].(*thirdpartyresourcedataetcd.REST), apiGroup)
	return nil
}

func (m *Master) thirdpartyapi(group, kind, version, pluralResource string) *apiserver.APIGroupVersion {
	resourceStorage := thirdpartyresourcedataetcd.NewREST(
		generic.RESTOptions{
			StorageConfig:           m.thirdPartyStorageConfig,
			Decorator:               generic.UndecoratedStorage,
			DeleteCollectionWorkers: m.deleteCollectionWorkers,
		},
		group,
		kind,
	)

	storage := map[string]rest.Storage{
		pluralResource: resourceStorage,
	}

	optionsExternalVersion := registered.GroupOrDie(api.GroupName).GroupVersion
	internalVersion := unversioned.GroupVersion{Group: group, Version: runtime.APIVersionInternal}
	externalVersion := unversioned.GroupVersion{Group: group, Version: version}

	apiRoot := extensionsrest.MakeThirdPartyPath("")
	return &apiserver.APIGroupVersion{
		Root:         apiRoot,
		GroupVersion: externalVersion,

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

		Context: m.RequestContextMapper(),

		MinRequestTimeout: m.MinRequestTimeout(),

		ResourceLister: dynamicLister{m, extensionsrest.MakeThirdPartyPath(group)},
	}
}

type restOptionsFactory struct {
	deleteCollectionWorkers int
	enableGarbageCollection bool
	storageFactory          genericapiserver.StorageFactory
	storageDecorator        generic.StorageDecorator
}

func (f restOptionsFactory) NewFor(resource unversioned.GroupResource) generic.RESTOptions {
	storageConfig, err := f.storageFactory.NewConfig(resource)
	if err != nil {
		glog.Fatalf("Unable to find storage destination for %v, due to %v", resource, err.Error())
	}

	return generic.RESTOptions{
		StorageConfig:           storageConfig,
		Decorator:               f.storageDecorator,
		DeleteCollectionWorkers: f.deleteCollectionWorkers,
		EnableGarbageCollection: f.enableGarbageCollection,
		ResourcePrefix:          f.storageFactory.ResourcePrefix(resource),
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
	nodes, err := m.legacyRESTStorage.NodeRegistry.ListNodes(api.NewDefaultContext(), nil)
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
		authenticationv1beta1.SchemeGroupVersion,
		autoscalingapiv1.SchemeGroupVersion,
		appsapi.SchemeGroupVersion,
		policyapiv1alpha1.SchemeGroupVersion,
		rbacapi.SchemeGroupVersion,
		storageapiv1beta1.SchemeGroupVersion,
		certificatesapiv1alpha1.SchemeGroupVersion,
		authorizationapiv1beta1.SchemeGroupVersion,
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
