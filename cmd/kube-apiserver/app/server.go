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

// Package app does all of the work necessary to create a Kubernetes
// APIServer by binding together the API, master and APIServer infrastructure.
// It can be configured and called directly or via the hyperkube framework.
package app

import (
	"crypto/tls"
	"fmt"
	"net/http"
	"net/url"
	"os"

	"github.com/spf13/cobra"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	extensionsapiserver "k8s.io/apiextensions-apiserver/pkg/apiserver"
	"k8s.io/apimachinery/pkg/runtime"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/admission"
	genericapifilters "k8s.io/apiserver/pkg/endpoints/filters"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/egressselector"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/apiserver/pkg/util/notfoundhandler"
	"k8s.io/apiserver/pkg/util/webhook"
	"k8s.io/client-go/dynamic"
	clientgoinformers "k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/util/keyutil"
	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/component-base/cli/globalflag"
	"k8s.io/component-base/logs"
	logsapi "k8s.io/component-base/logs/api/v1"
	_ "k8s.io/component-base/metrics/prometheus/workqueue"
	"k8s.io/component-base/term"
	"k8s.io/component-base/version"
	"k8s.io/component-base/version/verflag"
	"k8s.io/klog/v2"
	aggregatorapiserver "k8s.io/kube-aggregator/pkg/apiserver"
	aggregatorscheme "k8s.io/kube-aggregator/pkg/apiserver/scheme"

	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/capabilities"
	"k8s.io/kubernetes/pkg/controlplane"
	controlplaneapiserver "k8s.io/kubernetes/pkg/controlplane/apiserver"
	"k8s.io/kubernetes/pkg/controlplane/reconcilers"
	"k8s.io/kubernetes/pkg/features"
	generatedopenapi "k8s.io/kubernetes/pkg/generated/openapi"
	kubeapiserveradmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
	"k8s.io/kubernetes/pkg/serviceaccount"
)

func init() {
	utilruntime.Must(logsapi.AddFeatureGates(utilfeature.DefaultMutableFeatureGate))
}

// NewAPIServerCommand creates a *cobra.Command object with default parameters
func NewAPIServerCommand() *cobra.Command {
	s := options.NewServerRunOptions()
	cmd := &cobra.Command{
		Use: "kube-apiserver",
		Long: `The Kubernetes API server validates and configures data
for the api objects which include pods, services, replicationcontrollers, and
others. The API Server services REST operations and provides the frontend to the
cluster's shared state through which all other components interact.`,

		// stop printing usage when the command errors
		SilenceUsage: true,
		PersistentPreRunE: func(*cobra.Command, []string) error {
			// silence client-go warnings.
			// kube-apiserver loopback clients should not log self-issued warnings.
			rest.SetDefaultWarningHandler(rest.NoWarnings{})
			return nil
		},
		RunE: func(cmd *cobra.Command, args []string) error {
			verflag.PrintAndExitIfRequested()
			fs := cmd.Flags()

			// Activate logging as soon as possible, after that
			// show flags with the final logging configuration.
			if err := logsapi.ValidateAndApply(s.Logs, utilfeature.DefaultFeatureGate); err != nil {
				return err
			}
			cliflag.PrintFlags(fs)

			// set default options
			completedOptions, err := s.Complete()
			if err != nil {
				return err
			}

			// validate options
			if errs := completedOptions.Validate(); len(errs) != 0 {
				return utilerrors.NewAggregate(errs)
			}
			// add feature enablement metrics
			utilfeature.DefaultMutableFeatureGate.AddMetrics()
			return Run(completedOptions, genericapiserver.SetupSignalHandler())
		},
		Args: func(cmd *cobra.Command, args []string) error {
			for _, arg := range args {
				if len(arg) > 0 {
					return fmt.Errorf("%q does not take any arguments, got %q", cmd.CommandPath(), args)
				}
			}
			return nil
		},
	}

	fs := cmd.Flags()
	namedFlagSets := s.Flags()
	verflag.AddFlags(namedFlagSets.FlagSet("global"))
	globalflag.AddGlobalFlags(namedFlagSets.FlagSet("global"), cmd.Name(), logs.SkipLoggingConfigurationFlags())
	options.AddCustomGlobalFlags(namedFlagSets.FlagSet("generic"))
	for _, f := range namedFlagSets.FlagSets {
		fs.AddFlagSet(f)
	}

	cols, _, _ := term.TerminalSize(cmd.OutOrStdout())
	cliflag.SetUsageAndHelpFunc(cmd, namedFlagSets, cols)

	return cmd
}

// Run runs the specified APIServer.  This should never exit.
func Run(opts options.CompletedOptions, stopCh <-chan struct{}) error {
	// To help debugging, immediately log version
	klog.Infof("Version: %+v", version.Get())

	klog.InfoS("Golang settings", "GOGC", os.Getenv("GOGC"), "GOMAXPROCS", os.Getenv("GOMAXPROCS"), "GOTRACEBACK", os.Getenv("GOTRACEBACK"))

	config, err := NewConfig(opts)
	if err != nil {
		return err
	}
	completed, err := config.Complete()
	if err != nil {
		return err
	}
	server, err := CreateServerChain(completed)
	if err != nil {
		return err
	}

	prepared, err := server.PrepareRun()
	if err != nil {
		return err
	}

	return prepared.Run(stopCh)
}

// CreateServerChain creates the apiservers connected via delegation.
func CreateServerChain(config CompletedConfig) (*aggregatorapiserver.APIAggregator, error) {
	notFoundHandler := notfoundhandler.New(config.KubeAPIs.ControlPlane.Generic.Serializer, genericapifilters.NoMuxAndDiscoveryIncompleteKey)
	apiExtensionsServer, err := config.ApiExtensions.New(genericapiserver.NewEmptyDelegateWithCustomHandler(notFoundHandler))
	if err != nil {
		return nil, err
	}
	crdAPIEnabled := config.ApiExtensions.GenericConfig.MergedResourceConfig.ResourceEnabled(apiextensionsv1.SchemeGroupVersion.WithResource("customresourcedefinitions"))

	kubeAPIServer, err := config.KubeAPIs.New(apiExtensionsServer.GenericAPIServer)
	if err != nil {
		return nil, err
	}

	// aggregator comes last in the chain
	aggregatorServer, err := createAggregatorServer(config.Aggregator, kubeAPIServer.GenericAPIServer, apiExtensionsServer.Informers, crdAPIEnabled)
	if err != nil {
		// we don't need special handling for innerStopCh because the aggregator server doesn't create any go routines
		return nil, err
	}

	return aggregatorServer, nil
}

// CreateProxyTransport creates the dialer infrastructure to connect to the nodes.
func CreateProxyTransport() *http.Transport {
	var proxyDialerFn utilnet.DialFunc
	// Proxying to pods and services is IP-based... don't expect to be able to verify the hostname
	proxyTLSClientConfig := &tls.Config{InsecureSkipVerify: true}
	proxyTransport := utilnet.SetTransportDefaults(&http.Transport{
		DialContext:     proxyDialerFn,
		TLSClientConfig: proxyTLSClientConfig,
	})
	return proxyTransport
}

// CreateKubeAPIServerConfig creates all the resources for running the API server, but runs none of them
func CreateKubeAPIServerConfig(opts options.CompletedOptions) (
	*controlplane.Config,
	aggregatorapiserver.ServiceResolver,
	[]admission.PluginInitializer,
	error,
) {
	proxyTransport := CreateProxyTransport()

	genericConfig, versionedInformers, storageFactory, err := controlplaneapiserver.BuildGenericConfig(
		opts.CompletedOptions,
		[]*runtime.Scheme{legacyscheme.Scheme, extensionsapiserver.Scheme, aggregatorscheme.Scheme},
		controlplane.DefaultAPIResourceConfigSource(),
		generatedopenapi.GetOpenAPIDefinitions,
	)
	if err != nil {
		return nil, nil, nil, err
	}

	capabilities.Setup(opts.AllowPrivileged, opts.MaxConnectionBytesPerSec)

	opts.Metrics.Apply()
	serviceaccount.RegisterMetrics()

	config := &controlplane.Config{
		ControlPlane: controlplaneapiserver.Config{
			Generic: genericConfig,
			Extra: controlplaneapiserver.Extra{
				APIResourceConfigSource: storageFactory.APIResourceConfigSource,
				StorageFactory:          storageFactory,
				EventTTL:                opts.EventTTL,
				EnableLogsSupport:       opts.EnableLogsHandler,
				ProxyTransport:          proxyTransport,
				SystemNamespaces:        opts.SystemNamespaces,

				ServiceAccountIssuer:        opts.ServiceAccountIssuer,
				ServiceAccountMaxExpiration: opts.ServiceAccountTokenMaxExpiration,
				ExtendExpiration:            opts.Authentication.ServiceAccounts.ExtendExpiration,

				VersionedInformers: versionedInformers,
			},
		},
		Extra: controlplane.Extra{
			KubeletClientConfig: opts.KubeletConfig,

			ServiceIPRange:          opts.PrimaryServiceClusterIPRange,
			APIServerServiceIP:      opts.APIServerServiceIP,
			SecondaryServiceIPRange: opts.SecondaryServiceClusterIPRange,

			APIServerServicePort: 443,

			ServiceNodePortRange:      opts.ServiceNodePortRange,
			KubernetesServiceNodePort: opts.KubernetesServiceNodePort,

			EndpointReconcilerType: reconcilers.Type(opts.EndpointReconcilerType),
			MasterCount:            opts.MasterCount,
		},
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.UnknownVersionInteroperabilityProxy) {
		config.ControlPlane.PeerEndpointLeaseReconciler, err = controlplaneapiserver.CreatePeerEndpointLeaseReconciler(*genericConfig, storageFactory)
		if err != nil {
			return nil, nil, nil, err
		}
		// build peer proxy config only if peer ca file exists
		if opts.PeerCAFile != "" {
			config.ControlPlane.PeerProxy, err = controlplaneapiserver.BuildPeerProxy(versionedInformers, genericConfig.StorageVersionManager, opts.ProxyClientCertFile,
				opts.ProxyClientKeyFile, opts.PeerCAFile, opts.PeerAdvertiseAddress, genericConfig.APIServerID, config.ControlPlane.Extra.PeerEndpointLeaseReconciler, config.ControlPlane.Generic.Serializer)
			if err != nil {
				return nil, nil, nil, err
			}
		}
	}

	clientCAProvider, err := opts.Authentication.ClientCert.GetClientCAContentProvider()
	if err != nil {
		return nil, nil, nil, err
	}
	config.ControlPlane.ClusterAuthenticationInfo.ClientCA = clientCAProvider

	requestHeaderConfig, err := opts.Authentication.RequestHeader.ToAuthenticationRequestHeaderConfig()
	if err != nil {
		return nil, nil, nil, err
	}
	if requestHeaderConfig != nil {
		config.ControlPlane.ClusterAuthenticationInfo.RequestHeaderCA = requestHeaderConfig.CAContentProvider
		config.ControlPlane.ClusterAuthenticationInfo.RequestHeaderAllowedNames = requestHeaderConfig.AllowedClientNames
		config.ControlPlane.ClusterAuthenticationInfo.RequestHeaderExtraHeaderPrefixes = requestHeaderConfig.ExtraHeaderPrefixes
		config.ControlPlane.ClusterAuthenticationInfo.RequestHeaderGroupHeaders = requestHeaderConfig.GroupHeaders
		config.ControlPlane.ClusterAuthenticationInfo.RequestHeaderUsernameHeaders = requestHeaderConfig.UsernameHeaders
	}

	// setup admission
	admissionConfig := &kubeapiserveradmission.Config{
		ExternalInformers:    versionedInformers,
		LoopbackClientConfig: genericConfig.LoopbackClientConfig,
		CloudConfigFile:      opts.CloudProvider.CloudConfigFile,
	}
	serviceResolver := buildServiceResolver(opts.EnableAggregatorRouting, genericConfig.LoopbackClientConfig.Host, versionedInformers)
	pluginInitializers, err := admissionConfig.New(proxyTransport, genericConfig.EgressSelector, serviceResolver, genericConfig.TracerProvider)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to create admission plugin initializer: %v", err)
	}
	clientgoExternalClient, err := clientset.NewForConfig(genericConfig.LoopbackClientConfig)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to create real client-go external client: %w", err)
	}
	dynamicExternalClient, err := dynamic.NewForConfig(genericConfig.LoopbackClientConfig)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to create real dynamic external client: %w", err)
	}
	err = opts.Admission.ApplyTo(
		genericConfig,
		versionedInformers,
		clientgoExternalClient,
		dynamicExternalClient,
		utilfeature.DefaultFeatureGate,
		pluginInitializers...)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to apply admission: %w", err)
	}

	if config.ControlPlane.Generic.EgressSelector != nil {
		// Use the config.ControlPlane.Generic.EgressSelector lookup to find the dialer to connect to the kubelet
		config.Extra.KubeletClientConfig.Lookup = config.ControlPlane.Generic.EgressSelector.Lookup

		// Use the config.ControlPlane.Generic.EgressSelector lookup as the transport used by the "proxy" subresources.
		networkContext := egressselector.Cluster.AsNetworkContext()
		dialer, err := config.ControlPlane.Generic.EgressSelector.Lookup(networkContext)
		if err != nil {
			return nil, nil, nil, err
		}
		c := proxyTransport.Clone()
		c.DialContext = dialer
		config.ControlPlane.ProxyTransport = c
	}

	// Load and set the public keys.
	var pubKeys []interface{}
	for _, f := range opts.Authentication.ServiceAccounts.KeyFiles {
		keys, err := keyutil.PublicKeysFromFile(f)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("failed to parse key file %q: %v", f, err)
		}
		pubKeys = append(pubKeys, keys...)
	}
	config.ControlPlane.ServiceAccountIssuerURL = opts.Authentication.ServiceAccounts.Issuers[0]
	config.ControlPlane.ServiceAccountJWKSURI = opts.Authentication.ServiceAccounts.JWKSURI
	config.ControlPlane.ServiceAccountPublicKeys = pubKeys

	return config, serviceResolver, pluginInitializers, nil
}

var testServiceResolver webhook.ServiceResolver

// SetServiceResolverForTests allows the service resolver to be overridden during tests.
// Tests using this function must run serially as this function is not safe to call concurrently with server start.
func SetServiceResolverForTests(resolver webhook.ServiceResolver) func() {
	if testServiceResolver != nil {
		panic("test service resolver is set: tests are either running concurrently or clean up was skipped")
	}

	testServiceResolver = resolver

	return func() {
		testServiceResolver = nil
	}
}

func buildServiceResolver(enabledAggregatorRouting bool, hostname string, informer clientgoinformers.SharedInformerFactory) webhook.ServiceResolver {
	if testServiceResolver != nil {
		return testServiceResolver
	}

	var serviceResolver webhook.ServiceResolver
	if enabledAggregatorRouting {
		serviceResolver = aggregatorapiserver.NewEndpointServiceResolver(
			informer.Core().V1().Services().Lister(),
			informer.Core().V1().Endpoints().Lister(),
		)
	} else {
		serviceResolver = aggregatorapiserver.NewClusterIPServiceResolver(
			informer.Core().V1().Services().Lister(),
		)
	}

	// resolve kubernetes.default.svc locally
	if localHost, err := url.Parse(hostname); err == nil {
		serviceResolver = aggregatorapiserver.NewLoopbackServiceResolver(serviceResolver, localHost)
	}
	return serviceResolver
}
