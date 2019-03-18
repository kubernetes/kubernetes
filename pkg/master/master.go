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

	admissionregistrationv1beta1 "k8s.io/api/admissionregistration/v1beta1"
	appsv1 "k8s.io/api/apps/v1"
	appsv1beta1 "k8s.io/api/apps/v1beta1"
	appsv1beta2 "k8s.io/api/apps/v1beta2"
	auditregistrationv1alpha1 "k8s.io/api/auditregistration/v1alpha1"
	authenticationv1 "k8s.io/api/authentication/v1"
	authenticationv1beta1 "k8s.io/api/authentication/v1beta1"
	authorizationapiv1 "k8s.io/api/authorization/v1"
	authorizationapiv1beta1 "k8s.io/api/authorization/v1beta1"
	autoscalingapiv1 "k8s.io/api/autoscaling/v1"
	autoscalingapiv2beta1 "k8s.io/api/autoscaling/v2beta1"
	autoscalingapiv2beta2 "k8s.io/api/autoscaling/v2beta2"
	batchapiv1 "k8s.io/api/batch/v1"
	batchapiv1beta1 "k8s.io/api/batch/v1beta1"
	batchapiv2alpha1 "k8s.io/api/batch/v2alpha1"
	certificatesapiv1beta1 "k8s.io/api/certificates/v1beta1"
	coordinationapiv1 "k8s.io/api/coordination/v1"
	coordinationapiv1beta1 "k8s.io/api/coordination/v1beta1"
	apiv1 "k8s.io/api/core/v1"
	eventsv1beta1 "k8s.io/api/events/v1beta1"
	extensionsapiv1beta1 "k8s.io/api/extensions/v1beta1"
	networkingapiv1 "k8s.io/api/networking/v1"
	networkingapiv1beta1 "k8s.io/api/networking/v1beta1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	nodev1beta1 "k8s.io/api/node/v1beta1"
	policyapiv1beta1 "k8s.io/api/policy/v1beta1"
	rbacv1 "k8s.io/api/rbac/v1"
	rbacv1alpha1 "k8s.io/api/rbac/v1alpha1"
	rbacv1beta1 "k8s.io/api/rbac/v1beta1"
	schedulingapiv1 "k8s.io/api/scheduling/v1"
	schedulingv1alpha1 "k8s.io/api/scheduling/v1alpha1"
	schedulingapiv1beta1 "k8s.io/api/scheduling/v1beta1"
	settingsv1alpha1 "k8s.io/api/settings/v1alpha1"
	storageapiv1 "k8s.io/api/storage/v1"
	storageapiv1alpha1 "k8s.io/api/storage/v1alpha1"
	storageapiv1beta1 "k8s.io/api/storage/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apiserver/pkg/endpoints/discovery"
	"k8s.io/apiserver/pkg/registry/generic"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/healthz"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	storagefactory "k8s.io/apiserver/pkg/storage/storagebackend/factory"
	"k8s.io/client-go/informers"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	api "k8s.io/kubernetes/pkg/apis/core"
	kubeoptions "k8s.io/kubernetes/pkg/kubeapiserver/options"
	kubeletclient "k8s.io/kubernetes/pkg/kubelet/client"
	"k8s.io/kubernetes/pkg/master/reconcilers"
	"k8s.io/kubernetes/pkg/master/tunneler"
	"k8s.io/kubernetes/pkg/routes"
	"k8s.io/kubernetes/pkg/serviceaccount"
	nodeutil "k8s.io/kubernetes/pkg/util/node"

	"github.com/prometheus/client_golang/prometheus"
	"k8s.io/klog"

	// RESTStorage installers
	admissionregistrationrest "k8s.io/kubernetes/pkg/registry/admissionregistration/rest"
	appsrest "k8s.io/kubernetes/pkg/registry/apps/rest"
	auditregistrationrest "k8s.io/kubernetes/pkg/registry/auditregistration/rest"
	authenticationrest "k8s.io/kubernetes/pkg/registry/authentication/rest"
	authorizationrest "k8s.io/kubernetes/pkg/registry/authorization/rest"
	autoscalingrest "k8s.io/kubernetes/pkg/registry/autoscaling/rest"
	batchrest "k8s.io/kubernetes/pkg/registry/batch/rest"
	certificatesrest "k8s.io/kubernetes/pkg/registry/certificates/rest"
	coordinationrest "k8s.io/kubernetes/pkg/registry/coordination/rest"
	corerest "k8s.io/kubernetes/pkg/registry/core/rest"
	eventsrest "k8s.io/kubernetes/pkg/registry/events/rest"
	extensionsrest "k8s.io/kubernetes/pkg/registry/extensions/rest"
	networkingrest "k8s.io/kubernetes/pkg/registry/networking/rest"
	noderest "k8s.io/kubernetes/pkg/registry/node/rest"
	policyrest "k8s.io/kubernetes/pkg/registry/policy/rest"
	rbacrest "k8s.io/kubernetes/pkg/registry/rbac/rest"
	schedulingrest "k8s.io/kubernetes/pkg/registry/scheduling/rest"
	settingsrest "k8s.io/kubernetes/pkg/registry/settings/rest"
	storagerest "k8s.io/kubernetes/pkg/registry/storage/rest"
)

const (
	// DefaultEndpointReconcilerInterval is the default amount of time for how often the endpoints for
	// the kubernetes Service are reconciled.
	DefaultEndpointReconcilerInterval = 10 * time.Second
	// DefaultEndpointReconcilerTTL is the default TTL timeout for the storage layer
	DefaultEndpointReconcilerTTL = 15 * time.Second
)

type ExtraConfig struct {
	ClientCARegistrationHook ClientCARegistrationHook

	APIResourceConfigSource  serverstorage.APIResourceConfigSource
	StorageFactory           serverstorage.StorageFactory
	EndpointReconcilerConfig EndpointReconcilerConfig
	EventTTL                 time.Duration
	KubeletClientConfig      kubeletclient.KubeletClientConfig

	// Used to start and monitor tunneling
	Tunneler          tunneler.Tunneler
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
	ExtraServicePorts []apiv1.ServicePort
	// Additional ports to be exposed on the GenericAPIServer endpoints
	// Port names should align with ports defined in ExtraServicePorts
	ExtraEndpointPorts []apiv1.EndpointPort
	// If non-zero, the "kubernetes" services uses this port as NodePort.
	KubernetesServiceNodePort int

	// Number of masters running; all masters must be started with the
	// same value for this field. (Numbers > 1 currently untested.)
	MasterCount int

	// MasterEndpointReconcileTTL sets the time to live in seconds of an
	// endpoint record recorded by each master. The endpoints are checked at an
	// interval that is 2/3 of this value and this value defaults to 15s if
	// unset. In very large clusters, this value may be increased to reduce the
	// possibility that the master endpoint record expires (due to other load
	// on the etcd server) and causes masters to drop in and out of the
	// kubernetes service record. It is not recommended to set this value below
	// 15s.
	MasterEndpointReconcileTTL time.Duration

	// Selects which reconciler to use
	EndpointReconcilerType reconcilers.Type

	ServiceAccountIssuer        serviceaccount.TokenGenerator
	ServiceAccountMaxExpiration time.Duration

	VersionedInformers informers.SharedInformerFactory
}

type Config struct {
	GenericConfig *genericapiserver.Config
	ExtraConfig   ExtraConfig
}

type completedConfig struct {
	GenericConfig genericapiserver.CompletedConfig
	ExtraConfig   *ExtraConfig
}

type CompletedConfig struct {
	// Embed a private pointer that cannot be instantiated outside of this package.
	*completedConfig
}

// EndpointReconcilerConfig holds the endpoint reconciler and endpoint reconciliation interval to be
// used by the master.
type EndpointReconcilerConfig struct {
	Reconciler reconcilers.EndpointReconciler
	Interval   time.Duration
}

// Master contains state for a Kubernetes cluster master/api server.
type Master struct {
	GenericAPIServer *genericapiserver.GenericAPIServer

	ClientCARegistrationHook ClientCARegistrationHook
}

func (c *Config) createMasterCountReconciler() reconcilers.EndpointReconciler {
	endpointClient := corev1client.NewForConfigOrDie(c.GenericConfig.LoopbackClientConfig)
	return reconcilers.NewMasterCountEndpointReconciler(c.ExtraConfig.MasterCount, endpointClient)
}

func (c *Config) createNoneReconciler() reconcilers.EndpointReconciler {
	return reconcilers.NewNoneEndpointReconciler()
}

func (c *Config) createLeaseReconciler() reconcilers.EndpointReconciler {
	endpointClient := corev1client.NewForConfigOrDie(c.GenericConfig.LoopbackClientConfig)
	ttl := c.ExtraConfig.MasterEndpointReconcileTTL
	config, err := c.ExtraConfig.StorageFactory.NewConfig(api.Resource("apiServerIPInfo"))
	if err != nil {
		klog.Fatalf("Error determining service IP ranges: %v", err)
	}
	leaseStorage, _, err := storagefactory.Create(*config)
	if err != nil {
		klog.Fatalf("Error creating storage factory: %v", err)
	}
	masterLeases := reconcilers.NewLeases(leaseStorage, "/masterleases/", ttl)
	return reconcilers.NewLeaseEndpointReconciler(endpointClient, masterLeases)
}

func (c *Config) createEndpointReconciler() reconcilers.EndpointReconciler {
	klog.Infof("Using reconciler: %v", c.ExtraConfig.EndpointReconcilerType)
	switch c.ExtraConfig.EndpointReconcilerType {
	// there are numerous test dependencies that depend on a default controller
	case "", reconcilers.MasterCountReconcilerType:
		return c.createMasterCountReconciler()
	case reconcilers.LeaseEndpointReconcilerType:
		return c.createLeaseReconciler()
	case reconcilers.NoneEndpointReconcilerType:
		return c.createNoneReconciler()
	default:
		klog.Fatalf("Reconciler not implemented: %v", c.ExtraConfig.EndpointReconcilerType)
	}
	return nil
}

// Complete fills in any fields not set that are required to have valid data. It's mutating the receiver.
func (cfg *Config) Complete() CompletedConfig {
	c := completedConfig{
		cfg.GenericConfig.Complete(cfg.ExtraConfig.VersionedInformers),
		&cfg.ExtraConfig,
	}

	serviceIPRange, apiServerServiceIP, err := DefaultServiceIPRange(c.ExtraConfig.ServiceIPRange)
	if err != nil {
		klog.Fatalf("Error determining service IP ranges: %v", err)
	}
	if c.ExtraConfig.ServiceIPRange.IP == nil {
		c.ExtraConfig.ServiceIPRange = serviceIPRange
	}
	if c.ExtraConfig.APIServerServiceIP == nil {
		c.ExtraConfig.APIServerServiceIP = apiServerServiceIP
	}

	discoveryAddresses := discovery.DefaultAddresses{DefaultAddress: c.GenericConfig.ExternalAddress}
	discoveryAddresses.CIDRRules = append(discoveryAddresses.CIDRRules,
		discovery.CIDRRule{IPRange: c.ExtraConfig.ServiceIPRange, Address: net.JoinHostPort(c.ExtraConfig.APIServerServiceIP.String(), strconv.Itoa(c.ExtraConfig.APIServerServicePort))})
	c.GenericConfig.DiscoveryAddresses = discoveryAddresses

	if c.ExtraConfig.ServiceNodePortRange.Size == 0 {
		// TODO: Currently no way to specify an empty range (do we need to allow this?)
		// We should probably allow this for clouds that don't require NodePort to do load-balancing (GCE)
		// but then that breaks the strict nestedness of ServiceType.
		// Review post-v1
		c.ExtraConfig.ServiceNodePortRange = kubeoptions.DefaultServiceNodePortRange
		klog.Infof("Node port range unspecified. Defaulting to %v.", c.ExtraConfig.ServiceNodePortRange)
	}

	if c.ExtraConfig.EndpointReconcilerConfig.Interval == 0 {
		c.ExtraConfig.EndpointReconcilerConfig.Interval = DefaultEndpointReconcilerInterval
	}

	if c.ExtraConfig.MasterEndpointReconcileTTL == 0 {
		c.ExtraConfig.MasterEndpointReconcileTTL = DefaultEndpointReconcilerTTL
	}

	if c.ExtraConfig.EndpointReconcilerConfig.Reconciler == nil {
		c.ExtraConfig.EndpointReconcilerConfig.Reconciler = cfg.createEndpointReconciler()
	}

	return CompletedConfig{&c}
}

// New returns a new instance of Master from the given config.
// Certain config fields will be set to a default value if unset.
// Certain config fields must be specified, including:
//   KubeletClientConfig
func (c completedConfig) New(delegationTarget genericapiserver.DelegationTarget) (*Master, error) {
	if reflect.DeepEqual(c.ExtraConfig.KubeletClientConfig, kubeletclient.KubeletClientConfig{}) {
		return nil, fmt.Errorf("Master.New() called with empty config.KubeletClientConfig")
	}

	s, err := c.GenericConfig.New("kube-apiserver", delegationTarget)
	if err != nil {
		return nil, err
	}

	if c.ExtraConfig.EnableLogsSupport {
		routes.Logs{}.Install(s.Handler.GoRestfulContainer)
	}

	m := &Master{
		GenericAPIServer: s,
	}

	// install legacy rest storage
	if c.ExtraConfig.APIResourceConfigSource.VersionEnabled(apiv1.SchemeGroupVersion) {
		legacyRESTStorageProvider := corerest.LegacyRESTStorageProvider{
			StorageFactory:              c.ExtraConfig.StorageFactory,
			ProxyTransport:              c.ExtraConfig.ProxyTransport,
			KubeletClientConfig:         c.ExtraConfig.KubeletClientConfig,
			EventTTL:                    c.ExtraConfig.EventTTL,
			ServiceIPRange:              c.ExtraConfig.ServiceIPRange,
			ServiceNodePortRange:        c.ExtraConfig.ServiceNodePortRange,
			LoopbackClientConfig:        c.GenericConfig.LoopbackClientConfig,
			ServiceAccountIssuer:        c.ExtraConfig.ServiceAccountIssuer,
			ServiceAccountMaxExpiration: c.ExtraConfig.ServiceAccountMaxExpiration,
			APIAudiences:                c.GenericConfig.Authentication.APIAudiences,
		}
		m.InstallLegacyAPI(&c, c.GenericConfig.RESTOptionsGetter, legacyRESTStorageProvider)
	}

	// The order here is preserved in discovery.
	// If resources with identical names exist in more than one of these groups (e.g. "deployments.apps"" and "deployments.extensions"),
	// the order of this list determines which group an unqualified resource name (e.g. "deployments") should prefer.
	// This priority order is used for local discovery, but it ends up aggregated in `k8s.io/kubernetes/cmd/kube-apiserver/app/aggregator.go
	// with specific priorities.
	// TODO: describe the priority all the way down in the RESTStorageProviders and plumb it back through the various discovery
	// handlers that we have.
	restStorageProviders := []RESTStorageProvider{
		auditregistrationrest.RESTStorageProvider{},
		authenticationrest.RESTStorageProvider{Authenticator: c.GenericConfig.Authentication.Authenticator, APIAudiences: c.GenericConfig.Authentication.APIAudiences},
		authorizationrest.RESTStorageProvider{Authorizer: c.GenericConfig.Authorization.Authorizer, RuleResolver: c.GenericConfig.RuleResolver},
		autoscalingrest.RESTStorageProvider{},
		batchrest.RESTStorageProvider{},
		certificatesrest.RESTStorageProvider{},
		coordinationrest.RESTStorageProvider{},
		extensionsrest.RESTStorageProvider{},
		networkingrest.RESTStorageProvider{},
		noderest.RESTStorageProvider{},
		policyrest.RESTStorageProvider{},
		rbacrest.RESTStorageProvider{Authorizer: c.GenericConfig.Authorization.Authorizer},
		schedulingrest.RESTStorageProvider{},
		settingsrest.RESTStorageProvider{},
		storagerest.RESTStorageProvider{},
		// keep apps after extensions so legacy clients resolve the extensions versions of shared resource names.
		// See https://github.com/kubernetes/kubernetes/issues/42392
		appsrest.RESTStorageProvider{},
		admissionregistrationrest.RESTStorageProvider{},
		eventsrest.RESTStorageProvider{TTL: c.ExtraConfig.EventTTL},
	}
	m.InstallAPIs(c.ExtraConfig.APIResourceConfigSource, c.GenericConfig.RESTOptionsGetter, restStorageProviders...)

	if c.ExtraConfig.Tunneler != nil {
		m.installTunneler(c.ExtraConfig.Tunneler, corev1client.NewForConfigOrDie(c.GenericConfig.LoopbackClientConfig).Nodes())
	}

	m.GenericAPIServer.AddPostStartHookOrDie("ca-registration", c.ExtraConfig.ClientCARegistrationHook.PostStartHook)

	return m, nil
}

func (m *Master) InstallLegacyAPI(c *completedConfig, restOptionsGetter generic.RESTOptionsGetter, legacyRESTStorageProvider corerest.LegacyRESTStorageProvider) {
	legacyRESTStorage, apiGroupInfo, err := legacyRESTStorageProvider.NewLegacyRESTStorage(restOptionsGetter)
	if err != nil {
		klog.Fatalf("Error building core storage: %v", err)
	}

	controllerName := "bootstrap-controller"
	coreClient := corev1client.NewForConfigOrDie(c.GenericConfig.LoopbackClientConfig)
	bootstrapController := c.NewBootstrapController(legacyRESTStorage, coreClient, coreClient, coreClient, coreClient.RESTClient())
	m.GenericAPIServer.AddPostStartHookOrDie(controllerName, bootstrapController.PostStartHook)
	m.GenericAPIServer.AddPreShutdownHookOrDie(controllerName, bootstrapController.PreShutdownHook)

	if err := m.GenericAPIServer.InstallLegacyAPIGroup(genericapiserver.DefaultLegacyAPIPrefix, &apiGroupInfo); err != nil {
		klog.Fatalf("Error in registering group versions: %v", err)
	}
}

func (m *Master) installTunneler(nodeTunneler tunneler.Tunneler, nodeClient corev1client.NodeInterface) {
	nodeTunneler.Run(nodeAddressProvider{nodeClient}.externalAddresses)
	m.GenericAPIServer.AddHealthzChecks(healthz.NamedCheck("SSH Tunnel Check", tunneler.TunnelSyncHealthChecker(nodeTunneler)))
	prometheus.NewGaugeFunc(prometheus.GaugeOpts{
		Name: "apiserver_proxy_tunnel_sync_duration_seconds",
		Help: "The time since the last successful synchronization of the SSH tunnels for proxy requests.",
	}, func() float64 { return float64(nodeTunneler.SecondsSinceSync()) })
	prometheus.NewGaugeFunc(prometheus.GaugeOpts{
		Name: "apiserver_proxy_tunnel_sync_latency_secs",
		Help: "(Deprecated) The time since the last successful synchronization of the SSH tunnels for proxy requests.",
	}, func() float64 { return float64(nodeTunneler.SecondsSinceSync()) })
}

// RESTStorageProvider is a factory type for REST storage.
type RESTStorageProvider interface {
	GroupName() string
	NewRESTStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (genericapiserver.APIGroupInfo, bool)
}

// InstallAPIs will install the APIs for the restStorageProviders if they are enabled.
func (m *Master) InstallAPIs(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter, restStorageProviders ...RESTStorageProvider) {
	apiGroupsInfo := []*genericapiserver.APIGroupInfo{}

	for _, restStorageBuilder := range restStorageProviders {
		groupName := restStorageBuilder.GroupName()
		if !apiResourceConfigSource.AnyVersionForGroupEnabled(groupName) {
			klog.V(1).Infof("Skipping disabled API group %q.", groupName)
			continue
		}
		apiGroupInfo, enabled := restStorageBuilder.NewRESTStorage(apiResourceConfigSource, restOptionsGetter)
		if !enabled {
			klog.Warningf("Problem initializing API group %q, skipping.", groupName)
			continue
		}
		klog.V(1).Infof("Enabling API group %q.", groupName)

		if postHookProvider, ok := restStorageBuilder.(genericapiserver.PostStartHookProvider); ok {
			name, hook, err := postHookProvider.PostStartHook()
			if err != nil {
				klog.Fatalf("Error building PostStartHook: %v", err)
			}
			m.GenericAPIServer.AddPostStartHookOrDie(name, hook)
		}

		apiGroupsInfo = append(apiGroupsInfo, &apiGroupInfo)
	}

	if err := m.GenericAPIServer.InstallAPIGroups(apiGroupsInfo...); err != nil {
		klog.Fatalf("Error in registering group versions: %v", err)
	}
}

type nodeAddressProvider struct {
	nodeClient corev1client.NodeInterface
}

func (n nodeAddressProvider) externalAddresses() ([]string, error) {
	preferredAddressTypes := []apiv1.NodeAddressType{
		apiv1.NodeExternalIP,
	}
	nodes, err := n.nodeClient.List(metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	var matchErr error
	addrs := []string{}
	for ix := range nodes.Items {
		node := &nodes.Items[ix]
		addr, err := nodeutil.GetPreferredNodeAddress(node, preferredAddressTypes)
		if err != nil {
			if _, ok := err.(*nodeutil.NoMatchError); ok {
				matchErr = err
				continue
			}
			return nil, err
		}
		addrs = append(addrs, addr)
	}
	if len(addrs) == 0 && matchErr != nil {
		// We only return an error if we have items.
		// Currently we return empty list/no error if Items is empty.
		// We do this for backward compatibility reasons.
		return nil, matchErr
	}
	return addrs, nil
}

func DefaultAPIResourceConfigSource() *serverstorage.ResourceConfig {
	ret := serverstorage.NewResourceConfig()
	// NOTE: GroupVersions listed here will be enabled by default. Don't put alpha versions in the list.
	ret.EnableVersions(
		admissionregistrationv1beta1.SchemeGroupVersion,
		apiv1.SchemeGroupVersion,
		appsv1.SchemeGroupVersion,
		authenticationv1.SchemeGroupVersion,
		authenticationv1beta1.SchemeGroupVersion,
		authorizationapiv1.SchemeGroupVersion,
		authorizationapiv1beta1.SchemeGroupVersion,
		autoscalingapiv1.SchemeGroupVersion,
		autoscalingapiv2beta1.SchemeGroupVersion,
		autoscalingapiv2beta2.SchemeGroupVersion,
		batchapiv1.SchemeGroupVersion,
		batchapiv1beta1.SchemeGroupVersion,
		certificatesapiv1beta1.SchemeGroupVersion,
		coordinationapiv1.SchemeGroupVersion,
		coordinationapiv1beta1.SchemeGroupVersion,
		eventsv1beta1.SchemeGroupVersion,
		extensionsapiv1beta1.SchemeGroupVersion,
		networkingapiv1.SchemeGroupVersion,
		networkingapiv1beta1.SchemeGroupVersion,
		nodev1beta1.SchemeGroupVersion,
		policyapiv1beta1.SchemeGroupVersion,
		rbacv1.SchemeGroupVersion,
		rbacv1beta1.SchemeGroupVersion,
		storageapiv1.SchemeGroupVersion,
		storageapiv1beta1.SchemeGroupVersion,
		schedulingapiv1beta1.SchemeGroupVersion,
		schedulingapiv1.SchemeGroupVersion,
	)
	// enable non-deprecated beta resources in extensions/v1beta1 explicitly so we have a full list of what's possible to serve
	ret.EnableResources(
		extensionsapiv1beta1.SchemeGroupVersion.WithResource("ingresses"),
	)
	// enable deprecated beta resources in extensions/v1beta1 explicitly so we have a full list of what's possible to serve
	ret.EnableResources(
		extensionsapiv1beta1.SchemeGroupVersion.WithResource("daemonsets"),
		extensionsapiv1beta1.SchemeGroupVersion.WithResource("deployments"),
		extensionsapiv1beta1.SchemeGroupVersion.WithResource("networkpolicies"),
		extensionsapiv1beta1.SchemeGroupVersion.WithResource("podsecuritypolicies"),
		extensionsapiv1beta1.SchemeGroupVersion.WithResource("replicasets"),
		extensionsapiv1beta1.SchemeGroupVersion.WithResource("replicationcontrollers"),
	)
	// enable deprecated beta versions explicitly so we have a full list of what's possible to serve
	ret.EnableVersions(
		appsv1beta1.SchemeGroupVersion,
		appsv1beta2.SchemeGroupVersion,
	)
	// disable alpha versions explicitly so we have a full list of what's possible to serve
	ret.DisableVersions(
		auditregistrationv1alpha1.SchemeGroupVersion,
		batchapiv2alpha1.SchemeGroupVersion,
		nodev1alpha1.SchemeGroupVersion,
		rbacv1alpha1.SchemeGroupVersion,
		schedulingv1alpha1.SchemeGroupVersion,
		settingsv1alpha1.SchemeGroupVersion,
		storageapiv1alpha1.SchemeGroupVersion,
	)

	return ret
}
