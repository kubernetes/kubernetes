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
	"reflect"
	"strconv"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	appsapi "k8s.io/kubernetes/pkg/apis/apps/v1beta1"
	authenticationv1beta1 "k8s.io/kubernetes/pkg/apis/authentication/v1beta1"
	authorizationapiv1beta1 "k8s.io/kubernetes/pkg/apis/authorization/v1beta1"
	autoscalingapiv1 "k8s.io/kubernetes/pkg/apis/autoscaling/v1"
	batchapiv1 "k8s.io/kubernetes/pkg/apis/batch/v1"
	certificatesapiv1alpha1 "k8s.io/kubernetes/pkg/apis/certificates/v1alpha1"
	extensionsapiv1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	policyapiv1beta1 "k8s.io/kubernetes/pkg/apis/policy/v1beta1"
	rbacapi "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1"
	storageapiv1beta1 "k8s.io/kubernetes/pkg/apis/storage/v1beta1"
	coreclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	"k8s.io/kubernetes/pkg/genericapiserver"
	"k8s.io/kubernetes/pkg/genericapiserver/options"
	"k8s.io/kubernetes/pkg/healthz"
	kubeletclient "k8s.io/kubernetes/pkg/kubelet/client"
	"k8s.io/kubernetes/pkg/master/thirdparty"
	utilnet "k8s.io/kubernetes/pkg/util/net"
	nodeutil "k8s.io/kubernetes/pkg/util/node"

	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/routes"

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
	KubeletClientConfig      kubeletclient.KubeletClientConfig

	// Used to start and monitor tunneling
	Tunneler          genericapiserver.Tunneler
	EnableUISupport   bool
	EnableLogsSupport bool
	ProxyTransport    http.RoundTripper

	// Values to build the IP addresses used by discovery
	// The range of IPs to be assigned to services with type=ClusterIP or greater
	ServiceIPRange net.IPNet
	// The IP address for the GenericAPIServer service (must be inside ServiceIPRange)
	APIServerServiceIP net.IP
	// Port for the apiserver service.
	APIServerServicePort int

	// TODO, we can probably group service related items into a substruct to make it easier to configure
	// the API server items and `Extra*` fields likely fit nicely together.

	// The range of ports to be assigned to services with type=NodePort or greater
	ServiceNodePortRange utilnet.PortRange
	// Additional ports to be exposed on the GenericAPIServer service
	// extraServicePorts is injectable in the event that more ports
	// (other than the default 443/tcp) are exposed on the GenericAPIServer
	// and those ports need to be load balanced by the GenericAPIServer
	// service because this pkg is linked by out-of-tree projects
	// like openshift which want to use the GenericAPIServer but also do
	// more stuff.
	ExtraServicePorts []api.ServicePort
	// Additional ports to be exposed on the GenericAPIServer endpoints
	// Port names should align with ports defined in ExtraServicePorts
	ExtraEndpointPorts []api.EndpointPort
	// If non-zero, the "kubernetes" services uses this port as NodePort.
	// TODO(sttts): move into master
	KubernetesServiceNodePort int

	// Number of masters running; all masters must be started with the
	// same value for this field. (Numbers > 1 currently untested.)
	MasterCount int
}

// EndpointReconcilerConfig holds the endpoint reconciler and endpoint reconciliation interval to be
// used by the master.
type EndpointReconcilerConfig struct {
	Reconciler EndpointReconciler
	Interval   time.Duration
}

// Master contains state for a Kubernetes cluster master/api server.
type Master struct {
	GenericAPIServer *genericapiserver.GenericAPIServer
}

type completedConfig struct {
	*Config
}

// Complete fills in any fields not set that are required to have valid data. It's mutating the receiver.
func (c *Config) Complete() completedConfig {
	c.GenericConfig.Complete()

	serviceIPRange, apiServerServiceIP, err := genericapiserver.DefaultServiceIPRange(c.ServiceIPRange)
	if err != nil {
		glog.Fatalf("Error determining service IP ranges: %v", err)
	}
	if c.ServiceIPRange.IP == nil {
		c.ServiceIPRange = serviceIPRange
	}
	if c.APIServerServiceIP == nil {
		c.APIServerServiceIP = apiServerServiceIP
	}

	discoveryAddresses := genericapiserver.DefaultDiscoveryAddresses{DefaultAddress: c.GenericConfig.ExternalAddress}
	discoveryAddresses.DiscoveryCIDRRules = append(discoveryAddresses.DiscoveryCIDRRules,
		genericapiserver.DiscoveryCIDRRule{IPRange: c.ServiceIPRange, Address: net.JoinHostPort(c.APIServerServiceIP.String(), strconv.Itoa(c.APIServerServicePort))})
	c.GenericConfig.DiscoveryAddresses = discoveryAddresses

	if c.ServiceNodePortRange.Size == 0 {
		// TODO: Currently no way to specify an empty range (do we need to allow this?)
		// We should probably allow this for clouds that don't require NodePort to do load-balancing (GCE)
		// but then that breaks the strict nestedness of ServiceType.
		// Review post-v1
		c.ServiceNodePortRange = options.DefaultServiceNodePortRange
		glog.Infof("Node port range unspecified. Defaulting to %v.", c.ServiceNodePortRange)
	}

	// enable swagger UI only if general UI support is on
	c.GenericConfig.EnableSwaggerUI = c.GenericConfig.EnableSwaggerUI && c.EnableUISupport

	if c.EndpointReconcilerConfig.Interval == 0 {
		c.EndpointReconcilerConfig.Interval = DefaultEndpointReconcilerInterval
	}

	if c.EndpointReconcilerConfig.Reconciler == nil {
		// use a default endpoint reconciler if nothing is set
		endpointClient := coreclient.NewForConfigOrDie(c.GenericConfig.LoopbackClientConfig)
		c.EndpointReconcilerConfig.Reconciler = NewMasterCountEndpointReconciler(c.MasterCount, endpointClient)
	}

	// this has always been hardcoded true in the past
	c.GenericConfig.EnableMetrics = true

	return completedConfig{c}
}

// SkipComplete provides a way to construct a server instance without config completion.
func (c *Config) SkipComplete() completedConfig {
	return completedConfig{c}
}

// New returns a new instance of Master from the given config.
// Certain config fields will be set to a default value if unset.
// Certain config fields must be specified, including:
//   KubeletClientConfig
func (c completedConfig) New() (*Master, error) {
	if reflect.DeepEqual(c.KubeletClientConfig, kubeletclient.KubeletClientConfig{}) {
		return nil, fmt.Errorf("Master.New() called with empty config.KubeletClientConfig")
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
		GenericAPIServer: s,
	}

	restOptionsFactory := restOptionsFactory{
		deleteCollectionWorkers: c.DeleteCollectionWorkers,
		enableGarbageCollection: c.GenericConfig.EnableGarbageCollection,
		storageFactory:          c.StorageFactory,
	}

	if c.EnableWatchCache {
		restOptionsFactory.storageDecorator = registry.StorageWithCacher
	} else {
		restOptionsFactory.storageDecorator = generic.UndecoratedStorage
	}

	// install legacy rest storage
	if c.GenericConfig.APIResourceConfigSource.AnyResourcesForVersionEnabled(apiv1.SchemeGroupVersion) {
		legacyRESTStorageProvider := corerest.LegacyRESTStorageProvider{
			StorageFactory:       c.StorageFactory,
			ProxyTransport:       c.ProxyTransport,
			KubeletClientConfig:  c.KubeletClientConfig,
			EventTTL:             c.EventTTL,
			ServiceIPRange:       c.ServiceIPRange,
			ServiceNodePortRange: c.ServiceNodePortRange,
			LoopbackClientConfig: c.GenericConfig.LoopbackClientConfig,
		}
		m.InstallLegacyAPI(c.Config, restOptionsFactory.NewFor, legacyRESTStorageProvider)
	}

	restStorageProviders := []genericapiserver.RESTStorageProvider{
		appsrest.RESTStorageProvider{},
		authenticationrest.RESTStorageProvider{Authenticator: c.GenericConfig.Authenticator},
		authorizationrest.RESTStorageProvider{Authorizer: c.GenericConfig.Authorizer},
		autoscalingrest.RESTStorageProvider{},
		batchrest.RESTStorageProvider{},
		certificatesrest.RESTStorageProvider{},
		extensionsrest.RESTStorageProvider{ResourceInterface: thirdparty.NewThirdPartyResourceServer(s, c.StorageFactory)},
		policyrest.RESTStorageProvider{},
		rbacrest.RESTStorageProvider{AuthorizerRBACSuperUser: c.GenericConfig.AuthorizerRBACSuperUser},
		storagerest.RESTStorageProvider{},
	}
	m.InstallAPIs(c.Config.GenericConfig.APIResourceConfigSource, restOptionsFactory.NewFor, restStorageProviders...)

	if c.Tunneler != nil {
		m.installTunneler(c.Tunneler, coreclient.NewForConfigOrDie(c.GenericConfig.LoopbackClientConfig).Nodes())
	}

	return m, nil
}

func (m *Master) InstallLegacyAPI(c *Config, restOptionsGetter genericapiserver.RESTOptionsGetter, legacyRESTStorageProvider corerest.LegacyRESTStorageProvider) {
	legacyRESTStorage, apiGroupInfo, err := legacyRESTStorageProvider.NewLegacyRESTStorage(restOptionsGetter)
	if err != nil {
		glog.Fatalf("Error building core storage: %v", err)
	}

	if c.EnableCoreControllers {
		serviceClient := coreclient.NewForConfigOrDie(c.GenericConfig.LoopbackClientConfig)
		bootstrapController := c.NewBootstrapController(legacyRESTStorage, serviceClient)
		if err := m.GenericAPIServer.AddPostStartHook("bootstrap-controller", bootstrapController.PostStartHook); err != nil {
			glog.Fatalf("Error registering PostStartHook %q: %v", "bootstrap-controller", err)
		}
	}

	if err := m.GenericAPIServer.InstallLegacyAPIGroup(genericapiserver.DefaultLegacyAPIPrefix, &apiGroupInfo); err != nil {
		glog.Fatalf("Error in registering group versions: %v", err)
	}
}

func (m *Master) installTunneler(tunneler genericapiserver.Tunneler, nodeClient coreclient.NodeInterface) {
	tunneler.Run(nodeAddressProvider{nodeClient}.externalAddresses)
	m.GenericAPIServer.AddHealthzChecks(healthz.NamedCheck("SSH Tunnel Check", genericapiserver.TunnelSyncHealthChecker(tunneler)))
	prometheus.NewGaugeFunc(prometheus.GaugeOpts{
		Name: "apiserver_proxy_tunnel_sync_latency_secs",
		Help: "The time since the last successful synchronization of the SSH tunnels for proxy requests.",
	}, func() float64 { return float64(tunneler.SecondsSinceSync()) })
}

// InstallAPIs will install the APIs for the restStorageProviders if they are enabled.
func (m *Master) InstallAPIs(apiResourceConfigSource genericapiserver.APIResourceConfigSource, restOptionsGetter genericapiserver.RESTOptionsGetter, restStorageProviders ...genericapiserver.RESTStorageProvider) {
	apiGroupsInfo := []genericapiserver.APIGroupInfo{}

	for _, restStorageBuilder := range restStorageProviders {
		groupName := restStorageBuilder.GroupName()
		if !apiResourceConfigSource.AnyResourcesForGroupEnabled(groupName) {
			glog.V(1).Infof("Skipping disabled API group %q.", groupName)
			continue
		}
		apiGroupInfo, enabled := restStorageBuilder.NewRESTStorage(apiResourceConfigSource, restOptionsGetter)
		if !enabled {
			glog.Warningf("Problem initializing API group %q, skipping.", groupName)
			continue
		}
		glog.V(1).Infof("Enabling API group %q.", groupName)

		if postHookProvider, ok := restStorageBuilder.(genericapiserver.PostStartHookProvider); ok {
			name, hook, err := postHookProvider.PostStartHook()
			if err != nil {
				glog.Fatalf("Error building PostStartHook: %v", err)
			}
			if err := m.GenericAPIServer.AddPostStartHook(name, hook); err != nil {
				glog.Fatalf("Error registering PostStartHook %q: %v", name, err)
			}
		}

		apiGroupsInfo = append(apiGroupsInfo, apiGroupInfo)
	}

	for i := range apiGroupsInfo {
		if err := m.GenericAPIServer.InstallAPIGroup(&apiGroupsInfo[i]); err != nil {
			glog.Fatalf("Error in registering group versions: %v", err)
		}
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

type nodeAddressProvider struct {
	nodeClient coreclient.NodeInterface
}

func (n nodeAddressProvider) externalAddresses() (addresses []string, err error) {
	preferredAddressTypes := []api.NodeAddressType{
		api.NodeExternalIP,
		api.NodeLegacyHostIP,
	}
	nodes, err := n.nodeClient.List(api.ListOptions{})
	if err != nil {
		return nil, err
	}
	addrs := []string{}
	for ix := range nodes.Items {
		node := &nodes.Items[ix]
		addr, err := nodeutil.GetPreferredNodeAddress(node, preferredAddressTypes)
		if err != nil {
			return nil, err
		}
		addrs = append(addrs, addr)
	}
	return addrs, nil
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
		policyapiv1beta1.SchemeGroupVersion,
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
