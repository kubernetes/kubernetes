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
	"context"
	"fmt"
	"net"
	"net/http"
	"reflect"
	"strconv"
	"time"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	admissionregistrationv1beta1 "k8s.io/api/admissionregistration/v1beta1"
	appsv1 "k8s.io/api/apps/v1"
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
	discoveryv1beta1 "k8s.io/api/discovery/v1beta1"
	eventsv1beta1 "k8s.io/api/events/v1beta1"
	extensionsapiv1beta1 "k8s.io/api/extensions/v1beta1"
	flowcontrolv1alpha1 "k8s.io/api/flowcontrol/v1alpha1"
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
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/endpoints/discovery"
	"k8s.io/apiserver/pkg/registry/generic"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/dynamiccertificates"
	"k8s.io/apiserver/pkg/server/healthz"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	storagefactory "k8s.io/apiserver/pkg/storage/storagebackend/factory"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	discoveryclient "k8s.io/client-go/kubernetes/typed/discovery/v1beta1"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	kubeoptions "k8s.io/kubernetes/pkg/kubeapiserver/options"
	kubeletclient "k8s.io/kubernetes/pkg/kubelet/client"
	"k8s.io/kubernetes/pkg/master/controller/clusterauthenticationtrust"
	"k8s.io/kubernetes/pkg/master/reconcilers"
	"k8s.io/kubernetes/pkg/master/tunneler"
	"k8s.io/kubernetes/pkg/routes"
	"k8s.io/kubernetes/pkg/serviceaccount"
	nodeutil "k8s.io/kubernetes/pkg/util/node"

	"k8s.io/klog/v2"

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
	discoveryrest "k8s.io/kubernetes/pkg/registry/discovery/rest"
	eventsrest "k8s.io/kubernetes/pkg/registry/events/rest"
	extensionsrest "k8s.io/kubernetes/pkg/registry/extensions/rest"
	flowcontrolrest "k8s.io/kubernetes/pkg/registry/flowcontrol/rest"
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

// ExtraConfig defines extra configuration for the master
type ExtraConfig struct {
	ClusterAuthenticationInfo clusterauthenticationtrust.ClusterAuthenticationInfo

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

	// dual stack services, the range represents an alternative IP range for service IP
	// must be of different family than primary (ServiceIPRange)
	SecondaryServiceIPRange net.IPNet
	// the secondary IP address the GenericAPIServer service (must be inside SecondaryServiceIPRange)
	SecondaryAPIServerServiceIP net.IP

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
	ExtendExpiration            bool

	// ServiceAccountIssuerDiscovery
	ServiceAccountIssuerURL  string
	ServiceAccountJWKSURI    string
	ServiceAccountPublicKeys []interface{}

	VersionedInformers informers.SharedInformerFactory
}

// Config defines configuration for the master
type Config struct {
	GenericConfig *genericapiserver.Config
	ExtraConfig   ExtraConfig
}

type completedConfig struct {
	GenericConfig genericapiserver.CompletedConfig
	ExtraConfig   *ExtraConfig
}

// CompletedConfig embeds a private pointer that cannot be instantiated outside of this package
type CompletedConfig struct {
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

	ClusterAuthenticationInfo clusterauthenticationtrust.ClusterAuthenticationInfo
}

func (c *Config) createMasterCountReconciler() reconcilers.EndpointReconciler {
	endpointClient := corev1client.NewForConfigOrDie(c.GenericConfig.LoopbackClientConfig)
	var endpointSliceClient discoveryclient.EndpointSlicesGetter
	if utilfeature.DefaultFeatureGate.Enabled(features.EndpointSlice) {
		endpointSliceClient = discoveryclient.NewForConfigOrDie(c.GenericConfig.LoopbackClientConfig)
	}
	endpointsAdapter := reconcilers.NewEndpointsAdapter(endpointClient, endpointSliceClient)

	return reconcilers.NewMasterCountEndpointReconciler(c.ExtraConfig.MasterCount, endpointsAdapter)
}

func (c *Config) createNoneReconciler() reconcilers.EndpointReconciler {
	return reconcilers.NewNoneEndpointReconciler()
}

func (c *Config) createLeaseReconciler() reconcilers.EndpointReconciler {
	endpointClient := corev1client.NewForConfigOrDie(c.GenericConfig.LoopbackClientConfig)
	var endpointSliceClient discoveryclient.EndpointSlicesGetter
	if utilfeature.DefaultFeatureGate.Enabled(features.EndpointSlice) {
		endpointSliceClient = discoveryclient.NewForConfigOrDie(c.GenericConfig.LoopbackClientConfig)
	}
	endpointsAdapter := reconcilers.NewEndpointsAdapter(endpointClient, endpointSliceClient)

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

	return reconcilers.NewLeaseEndpointReconciler(endpointsAdapter, masterLeases)
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
func (c *Config) Complete() CompletedConfig {
	cfg := completedConfig{
		c.GenericConfig.Complete(c.ExtraConfig.VersionedInformers),
		&c.ExtraConfig,
	}

	serviceIPRange, apiServerServiceIP, err := ServiceIPRange(cfg.ExtraConfig.ServiceIPRange)
	if err != nil {
		klog.Fatalf("Error determining service IP ranges: %v", err)
	}
	if cfg.ExtraConfig.ServiceIPRange.IP == nil {
		cfg.ExtraConfig.ServiceIPRange = serviceIPRange
	}
	if cfg.ExtraConfig.APIServerServiceIP == nil {
		cfg.ExtraConfig.APIServerServiceIP = apiServerServiceIP
	}

	discoveryAddresses := discovery.DefaultAddresses{DefaultAddress: cfg.GenericConfig.ExternalAddress}
	discoveryAddresses.CIDRRules = append(discoveryAddresses.CIDRRules,
		discovery.CIDRRule{IPRange: cfg.ExtraConfig.ServiceIPRange, Address: net.JoinHostPort(cfg.ExtraConfig.APIServerServiceIP.String(), strconv.Itoa(cfg.ExtraConfig.APIServerServicePort))})
	cfg.GenericConfig.DiscoveryAddresses = discoveryAddresses

	if cfg.ExtraConfig.ServiceNodePortRange.Size == 0 {
		// TODO: Currently no way to specify an empty range (do we need to allow this?)
		// We should probably allow this for clouds that don't require NodePort to do load-balancing (GCE)
		// but then that breaks the strict nestedness of ServiceType.
		// Review post-v1
		cfg.ExtraConfig.ServiceNodePortRange = kubeoptions.DefaultServiceNodePortRange
		klog.Infof("Node port range unspecified. Defaulting to %v.", cfg.ExtraConfig.ServiceNodePortRange)
	}

	if cfg.ExtraConfig.EndpointReconcilerConfig.Interval == 0 {
		cfg.ExtraConfig.EndpointReconcilerConfig.Interval = DefaultEndpointReconcilerInterval
	}

	if cfg.ExtraConfig.MasterEndpointReconcileTTL == 0 {
		cfg.ExtraConfig.MasterEndpointReconcileTTL = DefaultEndpointReconcilerTTL
	}

	if cfg.ExtraConfig.EndpointReconcilerConfig.Reconciler == nil {
		cfg.ExtraConfig.EndpointReconcilerConfig.Reconciler = c.createEndpointReconciler()
	}

	return CompletedConfig{&cfg}
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

	if utilfeature.DefaultFeatureGate.Enabled(features.ServiceAccountIssuerDiscovery) {
		// Metadata and keys are expected to only change across restarts at present,
		// so we just marshal immediately and serve the cached JSON bytes.
		md, err := serviceaccount.NewOpenIDMetadata(
			c.ExtraConfig.ServiceAccountIssuerURL,
			c.ExtraConfig.ServiceAccountJWKSURI,
			c.GenericConfig.ExternalAddress,
			c.ExtraConfig.ServiceAccountPublicKeys,
		)
		if err != nil {
			// If there was an error, skip installing the endpoints and log the
			// error, but continue on. We don't return the error because the
			// metadata responses require additional, backwards incompatible
			// validation of command-line options.
			msg := fmt.Sprintf("Could not construct pre-rendered responses for"+
				" ServiceAccountIssuerDiscovery endpoints. Endpoints will not be"+
				" enabled. Error: %v", err)
			if c.ExtraConfig.ServiceAccountIssuerURL != "" {
				// The user likely expects this feature to be enabled if issuer URL is
				// set and the feature gate is enabled. In the future, if there is no
				// longer a feature gate and issuer URL is not set, the user may not
				// expect this feature to be enabled. We log the former case as an Error
				// and the latter case as an Info.
				klog.Error(msg)
			} else {
				klog.Info(msg)
			}
		} else {
			routes.NewOpenIDMetadataServer(md.ConfigJSON, md.PublicKeysetJSON).
				Install(s.Handler.GoRestfulContainer)
		}
	}

	m := &Master{
		GenericAPIServer:          s,
		ClusterAuthenticationInfo: c.ExtraConfig.ClusterAuthenticationInfo,
	}

	// install legacy rest storage
	if c.ExtraConfig.APIResourceConfigSource.VersionEnabled(apiv1.SchemeGroupVersion) {
		legacyRESTStorageProvider := corerest.LegacyRESTStorageProvider{
			StorageFactory:              c.ExtraConfig.StorageFactory,
			ProxyTransport:              c.ExtraConfig.ProxyTransport,
			KubeletClientConfig:         c.ExtraConfig.KubeletClientConfig,
			EventTTL:                    c.ExtraConfig.EventTTL,
			ServiceIPRange:              c.ExtraConfig.ServiceIPRange,
			SecondaryServiceIPRange:     c.ExtraConfig.SecondaryServiceIPRange,
			ServiceNodePortRange:        c.ExtraConfig.ServiceNodePortRange,
			LoopbackClientConfig:        c.GenericConfig.LoopbackClientConfig,
			ServiceAccountIssuer:        c.ExtraConfig.ServiceAccountIssuer,
			ExtendExpiration:            c.ExtraConfig.ExtendExpiration,
			ServiceAccountMaxExpiration: c.ExtraConfig.ServiceAccountMaxExpiration,
			APIAudiences:                c.GenericConfig.Authentication.APIAudiences,
		}
		if err := m.InstallLegacyAPI(&c, c.GenericConfig.RESTOptionsGetter, legacyRESTStorageProvider); err != nil {
			return nil, err
		}
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
		discoveryrest.StorageProvider{},
		extensionsrest.RESTStorageProvider{},
		networkingrest.RESTStorageProvider{},
		noderest.RESTStorageProvider{},
		policyrest.RESTStorageProvider{},
		rbacrest.RESTStorageProvider{Authorizer: c.GenericConfig.Authorization.Authorizer},
		schedulingrest.RESTStorageProvider{},
		settingsrest.RESTStorageProvider{},
		storagerest.RESTStorageProvider{},
		flowcontrolrest.RESTStorageProvider{},
		// keep apps after extensions so legacy clients resolve the extensions versions of shared resource names.
		// See https://github.com/kubernetes/kubernetes/issues/42392
		appsrest.StorageProvider{},
		admissionregistrationrest.RESTStorageProvider{},
		eventsrest.RESTStorageProvider{TTL: c.ExtraConfig.EventTTL},
	}
	if err := m.InstallAPIs(c.ExtraConfig.APIResourceConfigSource, c.GenericConfig.RESTOptionsGetter, restStorageProviders...); err != nil {
		return nil, err
	}

	if c.ExtraConfig.Tunneler != nil {
		m.installTunneler(c.ExtraConfig.Tunneler, corev1client.NewForConfigOrDie(c.GenericConfig.LoopbackClientConfig).Nodes())
	}

	m.GenericAPIServer.AddPostStartHookOrDie("start-cluster-authentication-info-controller", func(hookContext genericapiserver.PostStartHookContext) error {
		kubeClient, err := kubernetes.NewForConfig(hookContext.LoopbackClientConfig)
		if err != nil {
			return err
		}
		controller := clusterauthenticationtrust.NewClusterAuthenticationTrustController(m.ClusterAuthenticationInfo, kubeClient)

		// prime values and start listeners
		if m.ClusterAuthenticationInfo.ClientCA != nil {
			if notifier, ok := m.ClusterAuthenticationInfo.ClientCA.(dynamiccertificates.Notifier); ok {
				notifier.AddListener(controller)
			}
			if controller, ok := m.ClusterAuthenticationInfo.ClientCA.(dynamiccertificates.ControllerRunner); ok {
				// runonce to be sure that we have a value.
				if err := controller.RunOnce(); err != nil {
					runtime.HandleError(err)
				}
				go controller.Run(1, hookContext.StopCh)
			}
		}
		if m.ClusterAuthenticationInfo.RequestHeaderCA != nil {
			if notifier, ok := m.ClusterAuthenticationInfo.RequestHeaderCA.(dynamiccertificates.Notifier); ok {
				notifier.AddListener(controller)
			}
			if controller, ok := m.ClusterAuthenticationInfo.RequestHeaderCA.(dynamiccertificates.ControllerRunner); ok {
				// runonce to be sure that we have a value.
				if err := controller.RunOnce(); err != nil {
					runtime.HandleError(err)
				}
				go controller.Run(1, hookContext.StopCh)
			}
		}

		go controller.Run(1, hookContext.StopCh)
		return nil
	})

	return m, nil
}

// InstallLegacyAPI will install the legacy APIs for the restStorageProviders if they are enabled.
func (m *Master) InstallLegacyAPI(c *completedConfig, restOptionsGetter generic.RESTOptionsGetter, legacyRESTStorageProvider corerest.LegacyRESTStorageProvider) error {
	legacyRESTStorage, apiGroupInfo, err := legacyRESTStorageProvider.NewLegacyRESTStorage(restOptionsGetter)
	if err != nil {
		return fmt.Errorf("error building core storage: %v", err)
	}

	controllerName := "bootstrap-controller"
	coreClient := corev1client.NewForConfigOrDie(c.GenericConfig.LoopbackClientConfig)
	bootstrapController := c.NewBootstrapController(legacyRESTStorage, coreClient, coreClient, coreClient, coreClient.RESTClient())
	m.GenericAPIServer.AddPostStartHookOrDie(controllerName, bootstrapController.PostStartHook)
	m.GenericAPIServer.AddPreShutdownHookOrDie(controllerName, bootstrapController.PreShutdownHook)

	if err := m.GenericAPIServer.InstallLegacyAPIGroup(genericapiserver.DefaultLegacyAPIPrefix, &apiGroupInfo); err != nil {
		return fmt.Errorf("error in registering group versions: %v", err)
	}
	return nil
}

func (m *Master) installTunneler(nodeTunneler tunneler.Tunneler, nodeClient corev1client.NodeInterface) {
	nodeTunneler.Run(nodeAddressProvider{nodeClient}.externalAddresses)
	err := m.GenericAPIServer.AddHealthChecks(healthz.NamedCheck("SSH Tunnel Check", tunneler.TunnelSyncHealthChecker(nodeTunneler)))
	if err != nil {
		klog.Errorf("Failed adding ssh tunnel health check %v\n", err)
	}
}

// RESTStorageProvider is a factory type for REST storage.
type RESTStorageProvider interface {
	GroupName() string
	NewRESTStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (genericapiserver.APIGroupInfo, bool, error)
}

// InstallAPIs will install the APIs for the restStorageProviders if they are enabled.
func (m *Master) InstallAPIs(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter, restStorageProviders ...RESTStorageProvider) error {
	apiGroupsInfo := []*genericapiserver.APIGroupInfo{}

	for _, restStorageBuilder := range restStorageProviders {
		groupName := restStorageBuilder.GroupName()
		if !apiResourceConfigSource.AnyVersionForGroupEnabled(groupName) {
			klog.V(1).Infof("Skipping disabled API group %q.", groupName)
			continue
		}
		apiGroupInfo, enabled, err := restStorageBuilder.NewRESTStorage(apiResourceConfigSource, restOptionsGetter)
		if err != nil {
			return fmt.Errorf("problem initializing API group %q : %v", groupName, err)
		}
		if !enabled {
			klog.Warningf("API group %q is not enabled, skipping.", groupName)
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
		return fmt.Errorf("error in registering group versions: %v", err)
	}
	return nil
}

type nodeAddressProvider struct {
	nodeClient corev1client.NodeInterface
}

func (n nodeAddressProvider) externalAddresses() ([]string, error) {
	preferredAddressTypes := []apiv1.NodeAddressType{
		apiv1.NodeExternalIP,
	}
	nodes, err := n.nodeClient.List(context.TODO(), metav1.ListOptions{})
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

// DefaultAPIResourceConfigSource returns default configuration for an APIResource.
func DefaultAPIResourceConfigSource() *serverstorage.ResourceConfig {
	ret := serverstorage.NewResourceConfig()
	// NOTE: GroupVersions listed here will be enabled by default. Don't put alpha versions in the list.
	ret.EnableVersions(
		admissionregistrationv1.SchemeGroupVersion,
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
		discoveryv1beta1.SchemeGroupVersion,
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
	// disable alpha versions explicitly so we have a full list of what's possible to serve
	ret.DisableVersions(
		auditregistrationv1alpha1.SchemeGroupVersion,
		batchapiv2alpha1.SchemeGroupVersion,
		nodev1alpha1.SchemeGroupVersion,
		rbacv1alpha1.SchemeGroupVersion,
		schedulingv1alpha1.SchemeGroupVersion,
		settingsv1alpha1.SchemeGroupVersion,
		storageapiv1alpha1.SchemeGroupVersion,
		flowcontrolv1alpha1.SchemeGroupVersion,
	)

	return ret
}
