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
	"io/ioutil"
	"net"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/go-openapi/spec"
	"github.com/golang/glog"
	"github.com/spf13/cobra"

	extensionsapiserver "k8s.io/apiextensions-apiserver/pkg/apiserver"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/sets"
	utilwait "k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission"
	webhookinit "k8s.io/apiserver/pkg/admission/plugin/webhook/initializer"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	openapinamer "k8s.io/apiserver/pkg/endpoints/openapi"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/filters"
	serveroptions "k8s.io/apiserver/pkg/server/options"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/apiserver/pkg/storage/etcd3/preflight"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	apiserverflag "k8s.io/apiserver/pkg/util/flag"
	"k8s.io/apiserver/pkg/util/webhook"
	cacheddiscovery "k8s.io/client-go/discovery/cached"
	clientgoinformers "k8s.io/client-go/informers"
	clientgoclientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	certutil "k8s.io/client-go/util/cert"
	aggregatorapiserver "k8s.io/kube-aggregator/pkg/apiserver"
	aggregatorscheme "k8s.io/kube-aggregator/pkg/apiserver/scheme"
	openapi "k8s.io/kube-openapi/pkg/common"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/capabilities"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	"k8s.io/kubernetes/pkg/cloudprovider"
	serviceaccountcontroller "k8s.io/kubernetes/pkg/controller/serviceaccount"
	"k8s.io/kubernetes/pkg/features"
	generatedopenapi "k8s.io/kubernetes/pkg/generated/openapi"
	"k8s.io/kubernetes/pkg/kubeapiserver"
	kubeapiserveradmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
	kubeauthenticator "k8s.io/kubernetes/pkg/kubeapiserver/authenticator"
	"k8s.io/kubernetes/pkg/kubeapiserver/authorizer/modes"
	kubeoptions "k8s.io/kubernetes/pkg/kubeapiserver/options"
	kubeserver "k8s.io/kubernetes/pkg/kubeapiserver/server"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/pkg/master/reconcilers"
	"k8s.io/kubernetes/pkg/master/tunneler"
	quotainstall "k8s.io/kubernetes/pkg/quota/v1/install"
	"k8s.io/kubernetes/pkg/registry/cachesize"
	rbacrest "k8s.io/kubernetes/pkg/registry/rbac/rest"
	"k8s.io/kubernetes/pkg/serviceaccount"
	"k8s.io/kubernetes/pkg/version"
	"k8s.io/kubernetes/pkg/version/verflag"
	"k8s.io/kubernetes/plugin/pkg/auth/authenticator/token/bootstrap"

	utilflag "k8s.io/kubernetes/pkg/util/flag"
	_ "k8s.io/kubernetes/pkg/util/reflector/prometheus" // for reflector metric registration
	_ "k8s.io/kubernetes/pkg/util/workqueue/prometheus" // for workqueue metric registration
)

const etcdRetryLimit = 60
const etcdRetryInterval = 1 * time.Second

// NewAPIServerCommand creates a *cobra.Command object with default parameters
func NewAPIServerCommand(stopCh <-chan struct{}) *cobra.Command {
	s := options.NewServerRunOptions()
	cmd := &cobra.Command{
		Use: "kube-apiserver",
		Long: `The Kubernetes API server validates and configures data
for the api objects which include pods, services, replicationcontrollers, and
others. The API Server services REST operations and provides the frontend to the
cluster's shared state through which all other components interact.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			verflag.PrintAndExitIfRequested()
			utilflag.PrintFlags(cmd.Flags())

			// set default options
			completedOptions, err := Complete(s)
			if err != nil {
				return err
			}

			// validate options
			if errs := completedOptions.Validate(); len(errs) != 0 {
				return utilerrors.NewAggregate(errs)
			}

			return Run(completedOptions, stopCh)
		},
	}

	fs := cmd.Flags()
	namedFlagSets := s.Flags()
	for _, f := range namedFlagSets.FlagSets {
		fs.AddFlagSet(f)
	}

	usageFmt := "Usage:\n  %s\n"
	cols, _, _ := apiserverflag.TerminalSize(cmd.OutOrStdout())
	cmd.SetUsageFunc(func(cmd *cobra.Command) error {
		fmt.Fprintf(cmd.OutOrStderr(), usageFmt, cmd.UseLine())
		apiserverflag.PrintSections(cmd.OutOrStderr(), namedFlagSets, cols)
		return nil
	})
	cmd.SetHelpFunc(func(cmd *cobra.Command, args []string) {
		fmt.Fprintf(cmd.OutOrStdout(), "%s\n\n"+usageFmt, cmd.Long, cmd.UseLine())
		apiserverflag.PrintSections(cmd.OutOrStdout(), namedFlagSets, cols)
	})

	return cmd
}

// Run runs the specified APIServer.  This should never exit.
func Run(completeOptions completedServerRunOptions, stopCh <-chan struct{}) error {
	// To help debugging, immediately log version
	glog.Infof("Version: %+v", version.Get())

	server, err := CreateServerChain(completeOptions, stopCh)
	if err != nil {
		return err
	}

	return server.PrepareRun().Run(stopCh)
}

// CreateServerChain creates the apiservers connected via delegation.
func CreateServerChain(completedOptions completedServerRunOptions, stopCh <-chan struct{}) (*genericapiserver.GenericAPIServer, error) {
	nodeTunneler, proxyTransport, err := CreateNodeDialer(completedOptions)
	if err != nil {
		return nil, err
	}

	kubeAPIServerConfig, insecureServingInfo, serviceResolver, pluginInitializer, admissionPostStartHook, err := CreateKubeAPIServerConfig(completedOptions, nodeTunneler, proxyTransport)
	if err != nil {
		return nil, err
	}

	// If additional API servers are added, they should be gated.
	apiExtensionsConfig, err := createAPIExtensionsConfig(*kubeAPIServerConfig.GenericConfig, kubeAPIServerConfig.ExtraConfig.VersionedInformers, pluginInitializer, completedOptions.ServerRunOptions, completedOptions.MasterCount)
	if err != nil {
		return nil, err
	}
	apiExtensionsServer, err := createAPIExtensionsServer(apiExtensionsConfig, genericapiserver.NewEmptyDelegate())
	if err != nil {
		return nil, err
	}

	kubeAPIServer, err := CreateKubeAPIServer(kubeAPIServerConfig, apiExtensionsServer.GenericAPIServer, admissionPostStartHook)
	if err != nil {
		return nil, err
	}

	// otherwise go down the normal path of standing the aggregator up in front of the API server
	// this wires up openapi
	kubeAPIServer.GenericAPIServer.PrepareRun()

	// This will wire up openapi for extension api server
	apiExtensionsServer.GenericAPIServer.PrepareRun()

	// aggregator comes last in the chain
	aggregatorConfig, err := createAggregatorConfig(*kubeAPIServerConfig.GenericConfig, completedOptions.ServerRunOptions, kubeAPIServerConfig.ExtraConfig.VersionedInformers, serviceResolver, proxyTransport, pluginInitializer)
	if err != nil {
		return nil, err
	}
	aggregatorServer, err := createAggregatorServer(aggregatorConfig, kubeAPIServer.GenericAPIServer, apiExtensionsServer.Informers)
	if err != nil {
		// we don't need special handling for innerStopCh because the aggregator server doesn't create any go routines
		return nil, err
	}

	if insecureServingInfo != nil {
		insecureHandlerChain := kubeserver.BuildInsecureHandlerChain(aggregatorServer.GenericAPIServer.UnprotectedHandler(), kubeAPIServerConfig.GenericConfig)
		if err := insecureServingInfo.Serve(insecureHandlerChain, kubeAPIServerConfig.GenericConfig.RequestTimeout, stopCh); err != nil {
			return nil, err
		}
	}

	return aggregatorServer.GenericAPIServer, nil
}

// CreateKubeAPIServer creates and wires a workable kube-apiserver
func CreateKubeAPIServer(kubeAPIServerConfig *master.Config, delegateAPIServer genericapiserver.DelegationTarget, admissionPostStartHook genericapiserver.PostStartHookFunc) (*master.Master, error) {
	kubeAPIServer, err := kubeAPIServerConfig.Complete().New(delegateAPIServer)
	if err != nil {
		return nil, err
	}

	kubeAPIServer.GenericAPIServer.AddPostStartHookOrDie("start-kube-apiserver-admission-initializer", admissionPostStartHook)

	return kubeAPIServer, nil
}

// CreateNodeDialer creates the dialer infrastructure to connect to the nodes.
func CreateNodeDialer(s completedServerRunOptions) (tunneler.Tunneler, *http.Transport, error) {
	// Setup nodeTunneler if needed
	var nodeTunneler tunneler.Tunneler
	var proxyDialerFn utilnet.DialFunc
	if len(s.SSHUser) > 0 {
		// Get ssh key distribution func, if supported
		var installSSHKey tunneler.InstallSSHKey
		cloud, err := cloudprovider.InitCloudProvider(s.CloudProvider.CloudProvider, s.CloudProvider.CloudConfigFile)
		if err != nil {
			return nil, nil, fmt.Errorf("cloud provider could not be initialized: %v", err)
		}
		if cloud != nil {
			if instances, supported := cloud.Instances(); supported {
				installSSHKey = instances.AddSSHKeyToAllInstances
			}
		}
		if s.KubeletConfig.Port == 0 {
			return nil, nil, fmt.Errorf("must enable kubelet port if proxy ssh-tunneling is specified")
		}
		if s.KubeletConfig.ReadOnlyPort == 0 {
			return nil, nil, fmt.Errorf("must enable kubelet readonly port if proxy ssh-tunneling is specified")
		}
		// Set up the nodeTunneler
		// TODO(cjcullen): If we want this to handle per-kubelet ports or other
		// kubelet listen-addresses, we need to plumb through options.
		healthCheckPath := &url.URL{
			Scheme: "http",
			Host:   net.JoinHostPort("127.0.0.1", strconv.FormatUint(uint64(s.KubeletConfig.ReadOnlyPort), 10)),
			Path:   "healthz",
		}
		nodeTunneler = tunneler.New(s.SSHUser, s.SSHKeyfile, healthCheckPath, installSSHKey)

		// Use the nodeTunneler's dialer when proxying to pods, services, and nodes
		proxyDialerFn = nodeTunneler.Dial
	}
	// Proxying to pods and services is IP-based... don't expect to be able to verify the hostname
	proxyTLSClientConfig := &tls.Config{InsecureSkipVerify: true}
	proxyTransport := utilnet.SetTransportDefaults(&http.Transport{
		DialContext:     proxyDialerFn,
		TLSClientConfig: proxyTLSClientConfig,
	})
	return nodeTunneler, proxyTransport, nil
}

// CreateKubeAPIServerConfig creates all the resources for running the API server, but runs none of them
func CreateKubeAPIServerConfig(
	s completedServerRunOptions,
	nodeTunneler tunneler.Tunneler,
	proxyTransport *http.Transport,
) (
	config *master.Config,
	insecureServingInfo *genericapiserver.DeprecatedInsecureServingInfo,
	serviceResolver aggregatorapiserver.ServiceResolver,
	pluginInitializers []admission.PluginInitializer,
	admissionPostStartHook genericapiserver.PostStartHookFunc,
	lastErr error,
) {
	var genericConfig *genericapiserver.Config
	var storageFactory *serverstorage.DefaultStorageFactory
	var sharedInformers informers.SharedInformerFactory
	var versionedInformers clientgoinformers.SharedInformerFactory
	genericConfig, sharedInformers, versionedInformers, insecureServingInfo, serviceResolver, pluginInitializers, admissionPostStartHook, storageFactory, lastErr = buildGenericConfig(s.ServerRunOptions, proxyTransport)
	if lastErr != nil {
		return
	}

	if _, port, err := net.SplitHostPort(s.Etcd.StorageConfig.ServerList[0]); err == nil && port != "0" && len(port) != 0 {
		if err := utilwait.PollImmediate(etcdRetryInterval, etcdRetryLimit*etcdRetryInterval, preflight.EtcdConnection{ServerList: s.Etcd.StorageConfig.ServerList}.CheckEtcdServers); err != nil {
			lastErr = fmt.Errorf("error waiting for etcd connection: %v", err)
			return
		}
	}

	capabilities.Initialize(capabilities.Capabilities{
		AllowPrivileged: s.AllowPrivileged,
		// TODO(vmarmol): Implement support for HostNetworkSources.
		PrivilegedSources: capabilities.PrivilegedSources{
			HostNetworkSources: []string{},
			HostPIDSources:     []string{},
			HostIPCSources:     []string{},
		},
		PerConnectionBandwidthLimitBytesPerSec: s.MaxConnectionBytesPerSec,
	})

	serviceIPRange, apiServerServiceIP, lastErr := master.DefaultServiceIPRange(s.ServiceClusterIPRange)
	if lastErr != nil {
		return
	}

	clientCA, lastErr := readCAorNil(s.Authentication.ClientCert.ClientCA)
	if lastErr != nil {
		return
	}
	requestHeaderProxyCA, lastErr := readCAorNil(s.Authentication.RequestHeader.ClientCAFile)
	if lastErr != nil {
		return
	}

	var (
		issuer        serviceaccount.TokenGenerator
		apiAudiences  []string
		maxExpiration time.Duration
	)

	if s.ServiceAccountSigningKeyFile != "" ||
		s.Authentication.ServiceAccounts.Issuer != "" ||
		len(s.Authentication.ServiceAccounts.APIAudiences) > 0 {
		if !utilfeature.DefaultFeatureGate.Enabled(features.TokenRequest) {
			lastErr = fmt.Errorf("the TokenRequest feature is not enabled but --service-account-signing-key-file, --service-account-issuer and/or --service-account-api-audiences flags were passed")
			return
		}
		if s.ServiceAccountSigningKeyFile == "" ||
			s.Authentication.ServiceAccounts.Issuer == "" ||
			len(s.Authentication.ServiceAccounts.APIAudiences) == 0 ||
			len(s.Authentication.ServiceAccounts.KeyFiles) == 0 {
			lastErr = fmt.Errorf("service-account-signing-key-file, service-account-issuer, service-account-api-audiences and service-account-key-file should be specified together")
			return
		}
		sk, err := certutil.PrivateKeyFromFile(s.ServiceAccountSigningKeyFile)
		if err != nil {
			lastErr = fmt.Errorf("failed to parse service-account-issuer-key-file: %v", err)
			return
		}
		if s.Authentication.ServiceAccounts.MaxExpiration != 0 {
			lowBound := time.Hour
			upBound := time.Duration(1<<32) * time.Second
			if s.Authentication.ServiceAccounts.MaxExpiration < lowBound ||
				s.Authentication.ServiceAccounts.MaxExpiration > upBound {
				lastErr = fmt.Errorf("the serviceaccount max expiration is out of range, must be between 1 hour to 2^32 seconds")
				return
			}
		}

		issuer, err = serviceaccount.JWTTokenGenerator(s.Authentication.ServiceAccounts.Issuer, sk)
		if err != nil {
			lastErr = fmt.Errorf("failed to build token generator: %v", err)
			return
		}
		apiAudiences = s.Authentication.ServiceAccounts.APIAudiences
		maxExpiration = s.Authentication.ServiceAccounts.MaxExpiration
	}

	config = &master.Config{
		GenericConfig: genericConfig,
		ExtraConfig: master.ExtraConfig{
			ClientCARegistrationHook: master.ClientCARegistrationHook{
				ClientCA:                         clientCA,
				RequestHeaderUsernameHeaders:     s.Authentication.RequestHeader.UsernameHeaders,
				RequestHeaderGroupHeaders:        s.Authentication.RequestHeader.GroupHeaders,
				RequestHeaderExtraHeaderPrefixes: s.Authentication.RequestHeader.ExtraHeaderPrefixes,
				RequestHeaderCA:                  requestHeaderProxyCA,
				RequestHeaderAllowedNames:        s.Authentication.RequestHeader.AllowedNames,
			},

			APIResourceConfigSource: storageFactory.APIResourceConfigSource,
			StorageFactory:          storageFactory,
			EventTTL:                s.EventTTL,
			KubeletClientConfig:     s.KubeletConfig,
			EnableLogsSupport:       s.EnableLogsHandler,
			ProxyTransport:          proxyTransport,

			Tunneler: nodeTunneler,

			ServiceIPRange:       serviceIPRange,
			APIServerServiceIP:   apiServerServiceIP,
			APIServerServicePort: 443,

			ServiceNodePortRange:      s.ServiceNodePortRange,
			KubernetesServiceNodePort: s.KubernetesServiceNodePort,

			EndpointReconcilerType: reconcilers.Type(s.EndpointReconcilerType),
			MasterCount:            s.MasterCount,

			ServiceAccountIssuer:        issuer,
			ServiceAccountAPIAudiences:  apiAudiences,
			ServiceAccountMaxExpiration: maxExpiration,

			InternalInformers:  sharedInformers,
			VersionedInformers: versionedInformers,
		},
	}

	if nodeTunneler != nil {
		// Use the nodeTunneler's dialer to connect to the kubelet
		config.ExtraConfig.KubeletClientConfig.Dial = nodeTunneler.Dial
	}

	return
}

// BuildGenericConfig takes the master server options and produces the genericapiserver.Config associated with it
func buildGenericConfig(
	s *options.ServerRunOptions,
	proxyTransport *http.Transport,
) (
	genericConfig *genericapiserver.Config,
	sharedInformers informers.SharedInformerFactory,
	versionedInformers clientgoinformers.SharedInformerFactory,
	insecureServingInfo *genericapiserver.DeprecatedInsecureServingInfo,
	serviceResolver aggregatorapiserver.ServiceResolver,
	pluginInitializers []admission.PluginInitializer,
	admissionPostStartHook genericapiserver.PostStartHookFunc,
	storageFactory *serverstorage.DefaultStorageFactory,
	lastErr error,
) {
	genericConfig = genericapiserver.NewConfig(legacyscheme.Codecs)
	genericConfig.MergedResourceConfig = master.DefaultAPIResourceConfigSource()

	if lastErr = s.GenericServerRunOptions.ApplyTo(genericConfig); lastErr != nil {
		return
	}

	if lastErr = s.InsecureServing.ApplyTo(&insecureServingInfo, &genericConfig.LoopbackClientConfig); lastErr != nil {
		return
	}
	if lastErr = s.SecureServing.ApplyTo(&genericConfig.SecureServing, &genericConfig.LoopbackClientConfig); lastErr != nil {
		return
	}
	if lastErr = s.Authentication.ApplyTo(genericConfig); lastErr != nil {
		return
	}
	if lastErr = s.Audit.ApplyTo(genericConfig); lastErr != nil {
		return
	}
	if lastErr = s.Features.ApplyTo(genericConfig); lastErr != nil {
		return
	}
	if lastErr = s.APIEnablement.ApplyTo(genericConfig, master.DefaultAPIResourceConfigSource(), legacyscheme.Scheme); lastErr != nil {
		return
	}

	genericConfig.OpenAPIConfig = genericapiserver.DefaultOpenAPIConfig(generatedopenapi.GetOpenAPIDefinitions, openapinamer.NewDefinitionNamer(legacyscheme.Scheme, extensionsapiserver.Scheme, aggregatorscheme.Scheme))
	genericConfig.OpenAPIConfig.PostProcessSpec = postProcessOpenAPISpecForBackwardCompatibility
	genericConfig.OpenAPIConfig.Info.Title = "Kubernetes"
	genericConfig.SwaggerConfig = genericapiserver.DefaultSwaggerConfig()
	genericConfig.LongRunningFunc = filters.BasicLongRunningRequestCheck(
		sets.NewString("watch", "proxy"),
		sets.NewString("attach", "exec", "proxy", "log", "portforward"),
	)

	kubeVersion := version.Get()
	genericConfig.Version = &kubeVersion

	storageFactoryConfig := kubeapiserver.NewStorageFactoryConfig()
	storageFactoryConfig.ApiResourceConfig = genericConfig.MergedResourceConfig
	completedStorageFactoryConfig, err := storageFactoryConfig.Complete(s.Etcd, s.StorageSerialization)
	if err != nil {
		lastErr = err
		return
	}
	storageFactory, lastErr = completedStorageFactoryConfig.New()
	if lastErr != nil {
		return
	}
	if lastErr = s.Etcd.ApplyWithStorageFactoryTo(storageFactory, genericConfig); lastErr != nil {
		return
	}

	// Use protobufs for self-communication.
	// Since not every generic apiserver has to support protobufs, we
	// cannot default to it in generic apiserver and need to explicitly
	// set it in kube-apiserver.
	genericConfig.LoopbackClientConfig.ContentConfig.ContentType = "application/vnd.kubernetes.protobuf"

	client, err := internalclientset.NewForConfig(genericConfig.LoopbackClientConfig)
	if err != nil {
		lastErr = fmt.Errorf("failed to create clientset: %v", err)
		return
	}

	kubeClientConfig := genericConfig.LoopbackClientConfig
	sharedInformers = informers.NewSharedInformerFactory(client, 10*time.Minute)
	clientgoExternalClient, err := clientgoclientset.NewForConfig(kubeClientConfig)
	if err != nil {
		lastErr = fmt.Errorf("failed to create real external clientset: %v", err)
		return
	}
	versionedInformers = clientgoinformers.NewSharedInformerFactory(clientgoExternalClient, 10*time.Minute)

	if s.EnableAggregatorRouting {
		serviceResolver = aggregatorapiserver.NewEndpointServiceResolver(
			versionedInformers.Core().V1().Services().Lister(),
			versionedInformers.Core().V1().Endpoints().Lister(),
		)
	} else {
		serviceResolver = aggregatorapiserver.NewClusterIPServiceResolver(
			versionedInformers.Core().V1().Services().Lister(),
		)
	}
	// resolve kubernetes.default.svc locally
	localHost, err := url.Parse(genericConfig.LoopbackClientConfig.Host)
	if err != nil {
		lastErr = err
		return
	}
	serviceResolver = aggregatorapiserver.NewLoopbackServiceResolver(serviceResolver, localHost)

	genericConfig.Authentication.Authenticator, genericConfig.OpenAPIConfig.SecurityDefinitions, err = BuildAuthenticator(s, clientgoExternalClient, sharedInformers)
	if err != nil {
		lastErr = fmt.Errorf("invalid authentication config: %v", err)
		return
	}

	genericConfig.Authorization.Authorizer, genericConfig.RuleResolver, err = BuildAuthorizer(s, versionedInformers)
	if err != nil {
		lastErr = fmt.Errorf("invalid authorization config: %v", err)
		return
	}
	if !sets.NewString(s.Authorization.Modes...).Has(modes.ModeRBAC) {
		genericConfig.DisabledPostStartHooks.Insert(rbacrest.PostStartHookName)
	}

	webhookAuthResolverWrapper := func(delegate webhook.AuthenticationInfoResolver) webhook.AuthenticationInfoResolver {
		return &webhook.AuthenticationInfoResolverDelegator{
			ClientConfigForFunc: func(server string) (*rest.Config, error) {
				if server == "kubernetes.default.svc" {
					return genericConfig.LoopbackClientConfig, nil
				}
				return delegate.ClientConfigFor(server)
			},
			ClientConfigForServiceFunc: func(serviceName, serviceNamespace string) (*rest.Config, error) {
				if serviceName == "kubernetes" && serviceNamespace == "default" {
					return genericConfig.LoopbackClientConfig, nil
				}
				ret, err := delegate.ClientConfigForService(serviceName, serviceNamespace)
				if err != nil {
					return nil, err
				}
				if proxyTransport != nil && proxyTransport.DialContext != nil {
					ret.Dial = proxyTransport.DialContext
				}
				return ret, err
			},
		}
	}
	pluginInitializers, admissionPostStartHook, err = BuildAdmissionPluginInitializers(
		s,
		client,
		sharedInformers,
		serviceResolver,
		webhookAuthResolverWrapper,
	)
	if err != nil {
		lastErr = fmt.Errorf("failed to create admission plugin initializer: %v", err)
		return
	}

	err = s.Admission.ApplyTo(
		genericConfig,
		versionedInformers,
		kubeClientConfig,
		legacyscheme.Scheme,
		pluginInitializers...)
	if err != nil {
		lastErr = fmt.Errorf("failed to initialize admission: %v", err)
	}

	return
}

// BuildAdmissionPluginInitializers constructs the admission plugin initializer
func BuildAdmissionPluginInitializers(
	s *options.ServerRunOptions,
	client internalclientset.Interface,
	sharedInformers informers.SharedInformerFactory,
	serviceResolver aggregatorapiserver.ServiceResolver,
	webhookAuthWrapper webhook.AuthenticationInfoResolverWrapper,
) ([]admission.PluginInitializer, genericapiserver.PostStartHookFunc, error) {
	var cloudConfig []byte

	if s.CloudProvider.CloudConfigFile != "" {
		var err error
		cloudConfig, err = ioutil.ReadFile(s.CloudProvider.CloudConfigFile)
		if err != nil {
			glog.Fatalf("Error reading from cloud configuration file %s: %#v", s.CloudProvider.CloudConfigFile, err)
		}
	}

	// We have a functional client so we can use that to build our discovery backed REST mapper
	// Use a discovery client capable of being refreshed.
	discoveryClient := cacheddiscovery.NewMemCacheClient(client.Discovery())
	discoveryRESTMapper := restmapper.NewDeferredDiscoveryRESTMapper(discoveryClient)

	admissionPostStartHook := func(context genericapiserver.PostStartHookContext) error {
		discoveryRESTMapper.Reset()
		go utilwait.Until(discoveryRESTMapper.Reset, 30*time.Second, context.StopCh)
		return nil
	}

	quotaConfiguration := quotainstall.NewQuotaConfigurationForAdmission()

	kubePluginInitializer := kubeapiserveradmission.NewPluginInitializer(client, sharedInformers, cloudConfig, discoveryRESTMapper, quotaConfiguration)
	webhookPluginInitializer := webhookinit.NewPluginInitializer(webhookAuthWrapper, serviceResolver)

	return []admission.PluginInitializer{webhookPluginInitializer, kubePluginInitializer}, admissionPostStartHook, nil
}

// BuildAuthenticator constructs the authenticator
func BuildAuthenticator(s *options.ServerRunOptions, extclient clientgoclientset.Interface, sharedInformers informers.SharedInformerFactory) (authenticator.Request, *spec.SecurityDefinitions, error) {
	authenticatorConfig := s.Authentication.ToAuthenticationConfig()
	if s.Authentication.ServiceAccounts.Lookup {
		authenticatorConfig.ServiceAccountTokenGetter = serviceaccountcontroller.NewGetterFromClient(extclient)
	}
	authenticatorConfig.BootstrapTokenAuthenticator = bootstrap.NewTokenAuthenticator(
		sharedInformers.Core().InternalVersion().Secrets().Lister().Secrets(v1.NamespaceSystem),
	)

	return authenticatorConfig.New()
}

// BuildAuthorizer constructs the authorizer
func BuildAuthorizer(s *options.ServerRunOptions, versionedInformers clientgoinformers.SharedInformerFactory) (authorizer.Authorizer, authorizer.RuleResolver, error) {
	authorizationConfig := s.Authorization.ToAuthorizationConfig(versionedInformers)
	return authorizationConfig.New()
}

// completedServerRunOptions is a private wrapper that enforces a call of Complete() before Run can be invoked.
type completedServerRunOptions struct {
	*options.ServerRunOptions
}

// Complete set default ServerRunOptions.
// Should be called after kube-apiserver flags parsed.
func Complete(s *options.ServerRunOptions) (completedServerRunOptions, error) {
	var options completedServerRunOptions
	// set defaults
	if err := s.GenericServerRunOptions.DefaultAdvertiseAddress(s.SecureServing.SecureServingOptions); err != nil {
		return options, err
	}
	if err := kubeoptions.DefaultAdvertiseAddress(s.GenericServerRunOptions, s.InsecureServing.DeprecatedInsecureServingOptions); err != nil {
		return options, err
	}
	serviceIPRange, apiServerServiceIP, err := master.DefaultServiceIPRange(s.ServiceClusterIPRange)
	if err != nil {
		return options, fmt.Errorf("error determining service IP ranges: %v", err)
	}
	s.ServiceClusterIPRange = serviceIPRange
	if err := s.SecureServing.MaybeDefaultWithSelfSignedCerts(s.GenericServerRunOptions.AdvertiseAddress.String(), []string{"kubernetes.default.svc", "kubernetes.default", "kubernetes"}, []net.IP{apiServerServiceIP}); err != nil {
		return options, fmt.Errorf("error creating self-signed certificates: %v", err)
	}

	if len(s.GenericServerRunOptions.ExternalHost) == 0 {
		if len(s.GenericServerRunOptions.AdvertiseAddress) > 0 {
			s.GenericServerRunOptions.ExternalHost = s.GenericServerRunOptions.AdvertiseAddress.String()
		} else {
			if hostname, err := os.Hostname(); err == nil {
				s.GenericServerRunOptions.ExternalHost = hostname
			} else {
				return options, fmt.Errorf("error finding host name: %v", err)
			}
		}
		glog.Infof("external host was not specified, using %v", s.GenericServerRunOptions.ExternalHost)
	}

	s.Authentication.ApplyAuthorization(s.Authorization)

	// Use (ServiceAccountSigningKeyFile != "") as a proxy to the user enabling
	// TokenRequest functionality. This defaulting was convenient, but messed up
	// a lot of people when they rotated their serving cert with no idea it was
	// connected to their service account keys. We are taking this oppurtunity to
	// remove this problematic defaulting.
	if s.ServiceAccountSigningKeyFile == "" {
		// Default to the private server key for service account token signing
		if len(s.Authentication.ServiceAccounts.KeyFiles) == 0 && s.SecureServing.ServerCert.CertKey.KeyFile != "" {
			if kubeauthenticator.IsValidServiceAccountKeyFile(s.SecureServing.ServerCert.CertKey.KeyFile) {
				s.Authentication.ServiceAccounts.KeyFiles = []string{s.SecureServing.ServerCert.CertKey.KeyFile}
			} else {
				glog.Warning("No TLS key provided, service account token authentication disabled")
			}
		}
	}

	if s.Etcd.StorageConfig.DeserializationCacheSize == 0 {
		// When size of cache is not explicitly set, estimate its size based on
		// target memory usage.
		glog.V(2).Infof("Initializing deserialization cache size based on %dMB limit", s.GenericServerRunOptions.TargetRAMMB)

		// This is the heuristics that from memory capacity is trying to infer
		// the maximum number of nodes in the cluster and set cache sizes based
		// on that value.
		// From our documentation, we officially recommend 120GB machines for
		// 2000 nodes, and we scale from that point. Thus we assume ~60MB of
		// capacity per node.
		// TODO: We may consider deciding that some percentage of memory will
		// be used for the deserialization cache and divide it by the max object
		// size to compute its size. We may even go further and measure
		// collective sizes of the objects in the cache.
		clusterSize := s.GenericServerRunOptions.TargetRAMMB / 60
		s.Etcd.StorageConfig.DeserializationCacheSize = 25 * clusterSize
		if s.Etcd.StorageConfig.DeserializationCacheSize < 1000 {
			s.Etcd.StorageConfig.DeserializationCacheSize = 1000
		}
	}
	if s.Etcd.EnableWatchCache {
		glog.V(2).Infof("Initializing cache sizes based on %dMB limit", s.GenericServerRunOptions.TargetRAMMB)
		sizes := cachesize.NewHeuristicWatchCacheSizes(s.GenericServerRunOptions.TargetRAMMB)
		if userSpecified, err := serveroptions.ParseWatchCacheSizes(s.Etcd.WatchCacheSizes); err == nil {
			for resource, size := range userSpecified {
				sizes[resource] = size
			}
		}
		s.Etcd.WatchCacheSizes, err = serveroptions.WriteWatchCacheSizes(sizes)
		if err != nil {
			return options, err
		}
	}

	// TODO: remove when we stop supporting the legacy group version.
	if s.APIEnablement.RuntimeConfig != nil {
		for key, value := range s.APIEnablement.RuntimeConfig {
			if key == "v1" || strings.HasPrefix(key, "v1/") ||
				key == "api/v1" || strings.HasPrefix(key, "api/v1/") {
				delete(s.APIEnablement.RuntimeConfig, key)
				s.APIEnablement.RuntimeConfig["/v1"] = value
			}
			if key == "api/legacy" {
				delete(s.APIEnablement.RuntimeConfig, key)
			}
		}
	}
	options.ServerRunOptions = s
	return options, nil
}

func readCAorNil(file string) ([]byte, error) {
	if len(file) == 0 {
		return nil, nil
	}
	return ioutil.ReadFile(file)
}

// PostProcessSpec adds removed definitions for backward compatibility
func postProcessOpenAPISpecForBackwardCompatibility(s *spec.Swagger) (*spec.Swagger, error) {
	compatibilityMap := map[string]string{
		"io.k8s.kubernetes.pkg.apis.authorization.v1beta1.SelfSubjectAccessReview":                     "io.k8s.api.authorization.v1beta1.SelfSubjectAccessReview",
		"io.k8s.kubernetes.pkg.api.v1.GitRepoVolumeSource":                                             "io.k8s.api.core.v1.GitRepoVolumeSource",
		"io.k8s.kubernetes.pkg.apis.admissionregistration.v1alpha1.ValidatingWebhookConfigurationList": "io.k8s.api.admissionregistration.v1alpha1.ValidatingWebhookConfigurationList",
		"io.k8s.kubernetes.pkg.api.v1.EndpointPort":                                                    "io.k8s.api.core.v1.EndpointPort",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.SupplementalGroupsStrategyOptions":              "io.k8s.api.extensions.v1beta1.SupplementalGroupsStrategyOptions",
		"io.k8s.kubernetes.pkg.api.v1.PodStatus":                                                       "io.k8s.api.core.v1.PodStatus",
		"io.k8s.kubernetes.pkg.apis.rbac.v1beta1.RoleBindingList":                                      "io.k8s.api.rbac.v1beta1.RoleBindingList",
		"io.k8s.kubernetes.pkg.apis.policy.v1beta1.PodDisruptionBudgetSpec":                            "io.k8s.api.policy.v1beta1.PodDisruptionBudgetSpec",
		"io.k8s.kubernetes.pkg.api.v1.HTTPGetAction":                                                   "io.k8s.api.core.v1.HTTPGetAction",
		"io.k8s.kubernetes.pkg.apis.authorization.v1.ResourceAttributes":                               "io.k8s.api.authorization.v1.ResourceAttributes",
		"io.k8s.kubernetes.pkg.api.v1.PersistentVolumeList":                                            "io.k8s.api.core.v1.PersistentVolumeList",
		"io.k8s.kubernetes.pkg.apis.batch.v2alpha1.CronJobSpec":                                        "io.k8s.api.batch.v2alpha1.CronJobSpec",
		"io.k8s.kubernetes.pkg.api.v1.CephFSVolumeSource":                                              "io.k8s.api.core.v1.CephFSVolumeSource",
		"io.k8s.kubernetes.pkg.api.v1.Affinity":                                                        "io.k8s.api.core.v1.Affinity",
		"io.k8s.kubernetes.pkg.apis.rbac.v1beta1.PolicyRule":                                           "io.k8s.api.rbac.v1beta1.PolicyRule",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.DaemonSetSpec":                                  "io.k8s.api.extensions.v1beta1.DaemonSetSpec",
		"io.k8s.kubernetes.pkg.api.v1.ProjectedVolumeSource":                                           "io.k8s.api.core.v1.ProjectedVolumeSource",
		"io.k8s.kubernetes.pkg.api.v1.TCPSocketAction":                                                 "io.k8s.api.core.v1.TCPSocketAction",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.DaemonSet":                                      "io.k8s.api.extensions.v1beta1.DaemonSet",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.IngressList":                                    "io.k8s.api.extensions.v1beta1.IngressList",
		"io.k8s.kubernetes.pkg.api.v1.PodSpec":                                                         "io.k8s.api.core.v1.PodSpec",
		"io.k8s.kubernetes.pkg.apis.authentication.v1.TokenReview":                                     "io.k8s.api.authentication.v1.TokenReview",
		"io.k8s.kubernetes.pkg.apis.authorization.v1beta1.SubjectAccessReview":                         "io.k8s.api.authorization.v1beta1.SubjectAccessReview",
		"io.k8s.kubernetes.pkg.apis.rbac.v1alpha1.ClusterRoleBinding":                                  "io.k8s.api.rbac.v1alpha1.ClusterRoleBinding",
		"io.k8s.kubernetes.pkg.api.v1.Node":                                                            "io.k8s.api.core.v1.Node",
		"io.k8s.kubernetes.pkg.apis.admissionregistration.v1alpha1.ServiceReference":                   "io.k8s.api.admissionregistration.v1alpha1.ServiceReference",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.DeploymentStatus":                               "io.k8s.api.extensions.v1beta1.DeploymentStatus",
		"io.k8s.kubernetes.pkg.apis.rbac.v1beta1.RoleRef":                                              "io.k8s.api.rbac.v1beta1.RoleRef",
		"io.k8s.kubernetes.pkg.apis.apps.v1beta1.Scale":                                                "io.k8s.api.apps.v1beta1.Scale",
		"io.k8s.kubernetes.pkg.apis.admissionregistration.v1alpha1.InitializerConfiguration":           "io.k8s.api.admissionregistration.v1alpha1.InitializerConfiguration",
		"io.k8s.kubernetes.pkg.api.v1.PhotonPersistentDiskVolumeSource":                                "io.k8s.api.core.v1.PhotonPersistentDiskVolumeSource",
		"io.k8s.kubernetes.pkg.api.v1.PreferredSchedulingTerm":                                         "io.k8s.api.core.v1.PreferredSchedulingTerm",
		"io.k8s.kubernetes.pkg.apis.batch.v1.JobSpec":                                                  "io.k8s.api.batch.v1.JobSpec",
		"io.k8s.kubernetes.pkg.api.v1.EventSource":                                                     "io.k8s.api.core.v1.EventSource",
		"io.k8s.kubernetes.pkg.api.v1.Container":                                                       "io.k8s.api.core.v1.Container",
		"io.k8s.kubernetes.pkg.apis.admissionregistration.v1alpha1.AdmissionHookClientConfig":          "io.k8s.api.admissionregistration.v1alpha1.AdmissionHookClientConfig",
		"io.k8s.kubernetes.pkg.api.v1.ResourceQuota":                                                   "io.k8s.api.core.v1.ResourceQuota",
		"io.k8s.kubernetes.pkg.api.v1.SecretList":                                                      "io.k8s.api.core.v1.SecretList",
		"io.k8s.kubernetes.pkg.api.v1.NodeSystemInfo":                                                  "io.k8s.api.core.v1.NodeSystemInfo",
		"io.k8s.kubernetes.pkg.apis.rbac.v1alpha1.PolicyRule":                                          "io.k8s.api.rbac.v1alpha1.PolicyRule",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.ReplicaSetSpec":                                 "io.k8s.api.extensions.v1beta1.ReplicaSetSpec",
		"io.k8s.kubernetes.pkg.api.v1.NodeStatus":                                                      "io.k8s.api.core.v1.NodeStatus",
		"io.k8s.kubernetes.pkg.api.v1.ResourceQuotaList":                                               "io.k8s.api.core.v1.ResourceQuotaList",
		"io.k8s.kubernetes.pkg.api.v1.HostPathVolumeSource":                                            "io.k8s.api.core.v1.HostPathVolumeSource",
		"io.k8s.kubernetes.pkg.apis.certificates.v1beta1.CertificateSigningRequest":                    "io.k8s.api.certificates.v1beta1.CertificateSigningRequest",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.IngressRule":                                    "io.k8s.api.extensions.v1beta1.IngressRule",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.NetworkPolicyPeer":                              "io.k8s.api.extensions.v1beta1.NetworkPolicyPeer",
		"io.k8s.kubernetes.pkg.apis.storage.v1.StorageClass":                                           "io.k8s.api.storage.v1.StorageClass",
		"io.k8s.kubernetes.pkg.apis.networking.v1.NetworkPolicyPeer":                                   "io.k8s.api.networking.v1.NetworkPolicyPeer",
		"io.k8s.kubernetes.pkg.apis.networking.v1.NetworkPolicyIngressRule":                            "io.k8s.api.networking.v1.NetworkPolicyIngressRule",
		"io.k8s.kubernetes.pkg.api.v1.StorageOSPersistentVolumeSource":                                 "io.k8s.api.core.v1.StorageOSPersistentVolumeSource",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.NetworkPolicyIngressRule":                       "io.k8s.api.extensions.v1beta1.NetworkPolicyIngressRule",
		"io.k8s.kubernetes.pkg.api.v1.PodAffinity":                                                     "io.k8s.api.core.v1.PodAffinity",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.RollbackConfig":                                 "io.k8s.api.extensions.v1beta1.RollbackConfig",
		"io.k8s.kubernetes.pkg.api.v1.PodList":                                                         "io.k8s.api.core.v1.PodList",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.ScaleStatus":                                    "io.k8s.api.extensions.v1beta1.ScaleStatus",
		"io.k8s.kubernetes.pkg.api.v1.ComponentCondition":                                              "io.k8s.api.core.v1.ComponentCondition",
		"io.k8s.kubernetes.pkg.apis.certificates.v1beta1.CertificateSigningRequestList":                "io.k8s.api.certificates.v1beta1.CertificateSigningRequestList",
		"io.k8s.kubernetes.pkg.apis.rbac.v1alpha1.ClusterRoleBindingList":                              "io.k8s.api.rbac.v1alpha1.ClusterRoleBindingList",
		"io.k8s.kubernetes.pkg.apis.autoscaling.v2alpha1.HorizontalPodAutoscalerCondition":             "io.k8s.api.autoscaling.v2alpha1.HorizontalPodAutoscalerCondition",
		"io.k8s.kubernetes.pkg.api.v1.ServiceList":                                                     "io.k8s.api.core.v1.ServiceList",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.PodSecurityPolicy":                              "io.k8s.api.extensions.v1beta1.PodSecurityPolicy",
		"io.k8s.kubernetes.pkg.apis.batch.v1.JobCondition":                                             "io.k8s.api.batch.v1.JobCondition",
		"io.k8s.kubernetes.pkg.apis.apps.v1beta1.DeploymentStatus":                                     "io.k8s.api.apps.v1beta1.DeploymentStatus",
		"io.k8s.kubernetes.pkg.api.v1.Volume":                                                          "io.k8s.api.core.v1.Volume",
		"io.k8s.kubernetes.pkg.apis.rbac.v1alpha1.RoleBindingList":                                     "io.k8s.api.rbac.v1alpha1.RoleBindingList",
		"io.k8s.kubernetes.pkg.apis.admissionregistration.v1alpha1.Rule":                               "io.k8s.api.admissionregistration.v1alpha1.Rule",
		"io.k8s.kubernetes.pkg.apis.admissionregistration.v1alpha1.InitializerConfigurationList":       "io.k8s.api.admissionregistration.v1alpha1.InitializerConfigurationList",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.NetworkPolicy":                                  "io.k8s.api.extensions.v1beta1.NetworkPolicy",
		"io.k8s.kubernetes.pkg.apis.rbac.v1alpha1.ClusterRoleList":                                     "io.k8s.api.rbac.v1alpha1.ClusterRoleList",
		"io.k8s.kubernetes.pkg.api.v1.ObjectFieldSelector":                                             "io.k8s.api.core.v1.ObjectFieldSelector",
		"io.k8s.kubernetes.pkg.api.v1.EventList":                                                       "io.k8s.api.core.v1.EventList",
		"io.k8s.kubernetes.pkg.apis.autoscaling.v2alpha1.MetricStatus":                                 "io.k8s.api.autoscaling.v2alpha1.MetricStatus",
		"io.k8s.kubernetes.pkg.apis.networking.v1.NetworkPolicyPort":                                   "io.k8s.api.networking.v1.NetworkPolicyPort",
		"io.k8s.kubernetes.pkg.apis.rbac.v1beta1.RoleList":                                             "io.k8s.api.rbac.v1beta1.RoleList",
		"io.k8s.kubernetes.pkg.apis.rbac.v1alpha1.RoleList":                                            "io.k8s.api.rbac.v1alpha1.RoleList",
		"io.k8s.kubernetes.pkg.apis.apps.v1beta1.DeploymentStrategy":                                   "io.k8s.api.apps.v1beta1.DeploymentStrategy",
		"io.k8s.kubernetes.pkg.apis.autoscaling.v1.CrossVersionObjectReference":                        "io.k8s.api.autoscaling.v1.CrossVersionObjectReference",
		"io.k8s.kubernetes.pkg.api.v1.ConfigMapProjection":                                             "io.k8s.api.core.v1.ConfigMapProjection",
		"io.k8s.kubernetes.pkg.apis.autoscaling.v2alpha1.CrossVersionObjectReference":                  "io.k8s.api.autoscaling.v2alpha1.CrossVersionObjectReference",
		"io.k8s.kubernetes.pkg.api.v1.LoadBalancerStatus":                                              "io.k8s.api.core.v1.LoadBalancerStatus",
		"io.k8s.kubernetes.pkg.api.v1.ISCSIVolumeSource":                                               "io.k8s.api.core.v1.ISCSIVolumeSource",
		"io.k8s.kubernetes.pkg.apis.apps.v1beta1.ControllerRevisionList":                               "io.k8s.api.apps.v1beta1.ControllerRevisionList",
		"io.k8s.kubernetes.pkg.api.v1.EndpointSubset":                                                  "io.k8s.api.core.v1.EndpointSubset",
		"io.k8s.kubernetes.pkg.api.v1.SELinuxOptions":                                                  "io.k8s.api.core.v1.SELinuxOptions",
		"io.k8s.kubernetes.pkg.api.v1.PersistentVolumeClaimVolumeSource":                               "io.k8s.api.core.v1.PersistentVolumeClaimVolumeSource",
		"io.k8s.kubernetes.pkg.apis.autoscaling.v2alpha1.MetricSpec":                                   "io.k8s.api.autoscaling.v2alpha1.MetricSpec",
		"io.k8s.kubernetes.pkg.apis.apps.v1beta1.StatefulSetList":                                      "io.k8s.api.apps.v1beta1.StatefulSetList",
		"io.k8s.kubernetes.pkg.apis.authorization.v1beta1.ResourceAttributes":                          "io.k8s.api.authorization.v1beta1.ResourceAttributes",
		"io.k8s.kubernetes.pkg.api.v1.Capabilities":                                                    "io.k8s.api.core.v1.Capabilities",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.Deployment":                                     "io.k8s.api.extensions.v1beta1.Deployment",
		"io.k8s.kubernetes.pkg.api.v1.Binding":                                                         "io.k8s.api.core.v1.Binding",
		"io.k8s.kubernetes.pkg.api.v1.ReplicationControllerList":                                       "io.k8s.api.core.v1.ReplicationControllerList",
		"io.k8s.kubernetes.pkg.apis.authorization.v1.SelfSubjectAccessReview":                          "io.k8s.api.authorization.v1.SelfSubjectAccessReview",
		"io.k8s.kubernetes.pkg.apis.authentication.v1beta1.UserInfo":                                   "io.k8s.api.authentication.v1beta1.UserInfo",
		"io.k8s.kubernetes.pkg.api.v1.HostAlias":                                                       "io.k8s.api.core.v1.HostAlias",
		"io.k8s.kubernetes.pkg.apis.apps.v1beta1.StatefulSetUpdateStrategy":                            "io.k8s.api.apps.v1beta1.StatefulSetUpdateStrategy",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.IngressSpec":                                    "io.k8s.api.extensions.v1beta1.IngressSpec",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.DeploymentCondition":                            "io.k8s.api.extensions.v1beta1.DeploymentCondition",
		"io.k8s.kubernetes.pkg.api.v1.GCEPersistentDiskVolumeSource":                                   "io.k8s.api.core.v1.GCEPersistentDiskVolumeSource",
		"io.k8s.kubernetes.pkg.apis.admissionregistration.v1alpha1.Webhook":                            "io.k8s.api.admissionregistration.v1alpha1.Webhook",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.Scale":                                          "io.k8s.api.extensions.v1beta1.Scale",
		"io.k8s.kubernetes.pkg.apis.autoscaling.v2alpha1.HorizontalPodAutoscalerStatus":                "io.k8s.api.autoscaling.v2alpha1.HorizontalPodAutoscalerStatus",
		"io.k8s.kubernetes.pkg.api.v1.FlexVolumeSource":                                                "io.k8s.api.core.v1.FlexVolumeSource",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.RollingUpdateDeployment":                        "io.k8s.api.extensions.v1beta1.RollingUpdateDeployment",
		"io.k8s.kubernetes.pkg.apis.autoscaling.v2alpha1.ObjectMetricStatus":                           "io.k8s.api.autoscaling.v2alpha1.ObjectMetricStatus",
		"io.k8s.kubernetes.pkg.api.v1.Event":                                                           "io.k8s.api.core.v1.Event",
		"io.k8s.kubernetes.pkg.api.v1.ResourceQuotaSpec":                                               "io.k8s.api.core.v1.ResourceQuotaSpec",
		"io.k8s.kubernetes.pkg.api.v1.Handler":                                                         "io.k8s.api.core.v1.Handler",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.IngressBackend":                                 "io.k8s.api.extensions.v1beta1.IngressBackend",
		"io.k8s.kubernetes.pkg.apis.rbac.v1alpha1.Role":                                                "io.k8s.api.rbac.v1alpha1.Role",
		"io.k8s.kubernetes.pkg.apis.autoscaling.v2alpha1.ObjectMetricSource":                           "io.k8s.api.autoscaling.v2alpha1.ObjectMetricSource",
		"io.k8s.kubernetes.pkg.apis.autoscaling.v2alpha1.ResourceMetricStatus":                         "io.k8s.api.autoscaling.v2alpha1.ResourceMetricStatus",
		"io.k8s.kubernetes.pkg.apis.autoscaling.v1.HorizontalPodAutoscalerSpec":                        "io.k8s.api.autoscaling.v1.HorizontalPodAutoscalerSpec",
		"io.k8s.kubernetes.pkg.api.v1.Lifecycle":                                                       "io.k8s.api.core.v1.Lifecycle",
		"io.k8s.kubernetes.pkg.apis.certificates.v1beta1.CertificateSigningRequestStatus":              "io.k8s.api.certificates.v1beta1.CertificateSigningRequestStatus",
		"io.k8s.kubernetes.pkg.api.v1.ContainerStateRunning":                                           "io.k8s.api.core.v1.ContainerStateRunning",
		"io.k8s.kubernetes.pkg.api.v1.ServiceAccountList":                                              "io.k8s.api.core.v1.ServiceAccountList",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.HostPortRange":                                  "io.k8s.api.extensions.v1beta1.HostPortRange",
		"io.k8s.kubernetes.pkg.apis.apps.v1beta1.ControllerRevision":                                   "io.k8s.api.apps.v1beta1.ControllerRevision",
		"io.k8s.kubernetes.pkg.api.v1.ReplicationControllerSpec":                                       "io.k8s.api.core.v1.ReplicationControllerSpec",
		"io.k8s.kubernetes.pkg.api.v1.ContainerStateTerminated":                                        "io.k8s.api.core.v1.ContainerStateTerminated",
		"io.k8s.kubernetes.pkg.api.v1.ReplicationControllerStatus":                                     "io.k8s.api.core.v1.ReplicationControllerStatus",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.DaemonSetList":                                  "io.k8s.api.extensions.v1beta1.DaemonSetList",
		"io.k8s.kubernetes.pkg.apis.authorization.v1.SelfSubjectAccessReviewSpec":                      "io.k8s.api.authorization.v1.SelfSubjectAccessReviewSpec",
		"io.k8s.kubernetes.pkg.api.v1.ComponentStatusList":                                             "io.k8s.api.core.v1.ComponentStatusList",
		"io.k8s.kubernetes.pkg.api.v1.ContainerStateWaiting":                                           "io.k8s.api.core.v1.ContainerStateWaiting",
		"io.k8s.kubernetes.pkg.api.v1.VolumeMount":                                                     "io.k8s.api.core.v1.VolumeMount",
		"io.k8s.kubernetes.pkg.api.v1.Secret":                                                          "io.k8s.api.core.v1.Secret",
		"io.k8s.kubernetes.pkg.apis.rbac.v1beta1.ClusterRoleList":                                      "io.k8s.api.rbac.v1beta1.ClusterRoleList",
		"io.k8s.kubernetes.pkg.api.v1.ConfigMapList":                                                   "io.k8s.api.core.v1.ConfigMapList",
		"io.k8s.kubernetes.pkg.apis.storage.v1beta1.StorageClassList":                                  "io.k8s.api.storage.v1beta1.StorageClassList",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.HTTPIngressPath":                                "io.k8s.api.extensions.v1beta1.HTTPIngressPath",
		"io.k8s.kubernetes.pkg.apis.rbac.v1alpha1.ClusterRole":                                         "io.k8s.api.rbac.v1alpha1.ClusterRole",
		"io.k8s.kubernetes.pkg.apis.autoscaling.v2alpha1.ResourceMetricSource":                         "io.k8s.api.autoscaling.v2alpha1.ResourceMetricSource",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.DeploymentRollback":                             "io.k8s.api.extensions.v1beta1.DeploymentRollback",
		"io.k8s.kubernetes.pkg.api.v1.PersistentVolumeClaimSpec":                                       "io.k8s.api.core.v1.PersistentVolumeClaimSpec",
		"io.k8s.kubernetes.pkg.api.v1.ReplicationController":                                           "io.k8s.api.core.v1.ReplicationController",
		"io.k8s.kubernetes.pkg.apis.apps.v1beta1.StatefulSetSpec":                                      "io.k8s.api.apps.v1beta1.StatefulSetSpec",
		"io.k8s.kubernetes.pkg.api.v1.SecurityContext":                                                 "io.k8s.api.core.v1.SecurityContext",
		"io.k8s.kubernetes.pkg.apis.networking.v1.NetworkPolicySpec":                                   "io.k8s.api.networking.v1.NetworkPolicySpec",
		"io.k8s.kubernetes.pkg.api.v1.LocalObjectReference":                                            "io.k8s.api.core.v1.LocalObjectReference",
		"io.k8s.kubernetes.pkg.api.v1.RBDVolumeSource":                                                 "io.k8s.api.core.v1.RBDVolumeSource",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.NetworkPolicySpec":                              "io.k8s.api.extensions.v1beta1.NetworkPolicySpec",
		"io.k8s.kubernetes.pkg.api.v1.KeyToPath":                                                       "io.k8s.api.core.v1.KeyToPath",
		"io.k8s.kubernetes.pkg.api.v1.WeightedPodAffinityTerm":                                         "io.k8s.api.core.v1.WeightedPodAffinityTerm",
		"io.k8s.kubernetes.pkg.apis.autoscaling.v2alpha1.PodsMetricStatus":                             "io.k8s.api.autoscaling.v2alpha1.PodsMetricStatus",
		"io.k8s.kubernetes.pkg.api.v1.NodeAddress":                                                     "io.k8s.api.core.v1.NodeAddress",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.Ingress":                                        "io.k8s.api.extensions.v1beta1.Ingress",
		"io.k8s.kubernetes.pkg.apis.policy.v1beta1.PodDisruptionBudget":                                "io.k8s.api.policy.v1beta1.PodDisruptionBudget",
		"io.k8s.kubernetes.pkg.api.v1.ServicePort":                                                     "io.k8s.api.core.v1.ServicePort",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.IDRange":                                        "io.k8s.api.extensions.v1beta1.IDRange",
		"io.k8s.kubernetes.pkg.api.v1.SecretEnvSource":                                                 "io.k8s.api.core.v1.SecretEnvSource",
		"io.k8s.kubernetes.pkg.api.v1.NodeSelector":                                                    "io.k8s.api.core.v1.NodeSelector",
		"io.k8s.kubernetes.pkg.api.v1.PersistentVolumeClaimStatus":                                     "io.k8s.api.core.v1.PersistentVolumeClaimStatus",
		"io.k8s.kubernetes.pkg.apis.apps.v1beta1.DeploymentSpec":                                       "io.k8s.api.apps.v1beta1.DeploymentSpec",
		"io.k8s.kubernetes.pkg.apis.authorization.v1.NonResourceAttributes":                            "io.k8s.api.authorization.v1.NonResourceAttributes",
		"io.k8s.kubernetes.pkg.apis.autoscaling.v1.ScaleStatus":                                        "io.k8s.api.autoscaling.v1.ScaleStatus",
		"io.k8s.kubernetes.pkg.api.v1.PodCondition":                                                    "io.k8s.api.core.v1.PodCondition",
		"io.k8s.kubernetes.pkg.api.v1.PodTemplateSpec":                                                 "io.k8s.api.core.v1.PodTemplateSpec",
		"io.k8s.kubernetes.pkg.apis.apps.v1beta1.StatefulSet":                                          "io.k8s.api.apps.v1beta1.StatefulSet",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.NetworkPolicyPort":                              "io.k8s.api.extensions.v1beta1.NetworkPolicyPort",
		"io.k8s.kubernetes.pkg.apis.authentication.v1beta1.TokenReview":                                "io.k8s.api.authentication.v1beta1.TokenReview",
		"io.k8s.kubernetes.pkg.api.v1.LimitRangeSpec":                                                  "io.k8s.api.core.v1.LimitRangeSpec",
		"io.k8s.kubernetes.pkg.api.v1.FlockerVolumeSource":                                             "io.k8s.api.core.v1.FlockerVolumeSource",
		"io.k8s.kubernetes.pkg.apis.policy.v1beta1.Eviction":                                           "io.k8s.api.policy.v1beta1.Eviction",
		"io.k8s.kubernetes.pkg.api.v1.PersistentVolumeClaimList":                                       "io.k8s.api.core.v1.PersistentVolumeClaimList",
		"io.k8s.kubernetes.pkg.apis.certificates.v1beta1.CertificateSigningRequestCondition":           "io.k8s.api.certificates.v1beta1.CertificateSigningRequestCondition",
		"io.k8s.kubernetes.pkg.api.v1.DownwardAPIVolumeFile":                                           "io.k8s.api.core.v1.DownwardAPIVolumeFile",
		"io.k8s.kubernetes.pkg.apis.authorization.v1beta1.LocalSubjectAccessReview":                    "io.k8s.api.authorization.v1beta1.LocalSubjectAccessReview",
		"io.k8s.kubernetes.pkg.apis.apps.v1beta1.ScaleStatus":                                          "io.k8s.api.apps.v1beta1.ScaleStatus",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.HTTPIngressRuleValue":                           "io.k8s.api.extensions.v1beta1.HTTPIngressRuleValue",
		"io.k8s.kubernetes.pkg.apis.batch.v1.Job":                                                      "io.k8s.api.batch.v1.Job",
		"io.k8s.kubernetes.pkg.apis.admissionregistration.v1alpha1.ValidatingWebhookConfiguration":     "io.k8s.api.admissionregistration.v1alpha1.ValidatingWebhookConfiguration",
		"io.k8s.kubernetes.pkg.apis.rbac.v1beta1.RoleBinding":                                          "io.k8s.api.rbac.v1beta1.RoleBinding",
		"io.k8s.kubernetes.pkg.api.v1.FCVolumeSource":                                                  "io.k8s.api.core.v1.FCVolumeSource",
		"io.k8s.kubernetes.pkg.api.v1.EndpointAddress":                                                 "io.k8s.api.core.v1.EndpointAddress",
		"io.k8s.kubernetes.pkg.api.v1.ContainerPort":                                                   "io.k8s.api.core.v1.ContainerPort",
		"io.k8s.kubernetes.pkg.apis.rbac.v1beta1.ClusterRoleBinding":                                   "io.k8s.api.rbac.v1beta1.ClusterRoleBinding",
		"io.k8s.kubernetes.pkg.api.v1.GlusterfsVolumeSource":                                           "io.k8s.api.core.v1.GlusterfsVolumeSource",
		"io.k8s.kubernetes.pkg.api.v1.ResourceRequirements":                                            "io.k8s.api.core.v1.ResourceRequirements",
		"io.k8s.kubernetes.pkg.apis.apps.v1beta1.RollingUpdateDeployment":                              "io.k8s.api.apps.v1beta1.RollingUpdateDeployment",
		"io.k8s.kubernetes.pkg.api.v1.NamespaceStatus":                                                 "io.k8s.api.core.v1.NamespaceStatus",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.RunAsUserStrategyOptions":                       "io.k8s.api.extensions.v1beta1.RunAsUserStrategyOptions",
		"io.k8s.kubernetes.pkg.api.v1.Namespace":                                                       "io.k8s.api.core.v1.Namespace",
		"io.k8s.kubernetes.pkg.apis.authorization.v1.SubjectAccessReviewSpec":                          "io.k8s.api.authorization.v1.SubjectAccessReviewSpec",
		"io.k8s.kubernetes.pkg.apis.autoscaling.v2alpha1.HorizontalPodAutoscaler":                      "io.k8s.api.autoscaling.v2alpha1.HorizontalPodAutoscaler",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.ReplicaSetCondition":                            "io.k8s.api.extensions.v1beta1.ReplicaSetCondition",
		"io.k8s.kubernetes.pkg.apis.autoscaling.v1.HorizontalPodAutoscalerStatus":                      "io.k8s.api.autoscaling.v1.HorizontalPodAutoscalerStatus",
		"io.k8s.kubernetes.pkg.apis.authentication.v1.TokenReviewStatus":                               "io.k8s.api.authentication.v1.TokenReviewStatus",
		"io.k8s.kubernetes.pkg.api.v1.PersistentVolume":                                                "io.k8s.api.core.v1.PersistentVolume",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.FSGroupStrategyOptions":                         "io.k8s.api.extensions.v1beta1.FSGroupStrategyOptions",
		"io.k8s.kubernetes.pkg.api.v1.PodSecurityContext":                                              "io.k8s.api.core.v1.PodSecurityContext",
		"io.k8s.kubernetes.pkg.api.v1.PodTemplate":                                                     "io.k8s.api.core.v1.PodTemplate",
		"io.k8s.kubernetes.pkg.apis.authorization.v1.LocalSubjectAccessReview":                         "io.k8s.api.authorization.v1.LocalSubjectAccessReview",
		"io.k8s.kubernetes.pkg.api.v1.StorageOSVolumeSource":                                           "io.k8s.api.core.v1.StorageOSVolumeSource",
		"io.k8s.kubernetes.pkg.api.v1.NodeSelectorTerm":                                                "io.k8s.api.core.v1.NodeSelectorTerm",
		"io.k8s.kubernetes.pkg.apis.rbac.v1beta1.Role":                                                 "io.k8s.api.rbac.v1beta1.Role",
		"io.k8s.kubernetes.pkg.api.v1.ContainerStatus":                                                 "io.k8s.api.core.v1.ContainerStatus",
		"io.k8s.kubernetes.pkg.apis.authorization.v1.SubjectAccessReviewStatus":                        "io.k8s.api.authorization.v1.SubjectAccessReviewStatus",
		"io.k8s.kubernetes.pkg.apis.authentication.v1.TokenReviewSpec":                                 "io.k8s.api.authentication.v1.TokenReviewSpec",
		"io.k8s.kubernetes.pkg.api.v1.ConfigMap":                                                       "io.k8s.api.core.v1.ConfigMap",
		"io.k8s.kubernetes.pkg.api.v1.ServiceStatus":                                                   "io.k8s.api.core.v1.ServiceStatus",
		"io.k8s.kubernetes.pkg.apis.authorization.v1beta1.SelfSubjectAccessReviewSpec":                 "io.k8s.api.authorization.v1beta1.SelfSubjectAccessReviewSpec",
		"io.k8s.kubernetes.pkg.api.v1.CinderVolumeSource":                                              "io.k8s.api.core.v1.CinderVolumeSource",
		"io.k8s.kubernetes.pkg.apis.settings.v1alpha1.PodPresetSpec":                                   "io.k8s.api.settings.v1alpha1.PodPresetSpec",
		"io.k8s.kubernetes.pkg.apis.authorization.v1beta1.NonResourceAttributes":                       "io.k8s.api.authorization.v1beta1.NonResourceAttributes",
		"io.k8s.kubernetes.pkg.api.v1.ContainerImage":                                                  "io.k8s.api.core.v1.ContainerImage",
		"io.k8s.kubernetes.pkg.api.v1.ReplicationControllerCondition":                                  "io.k8s.api.core.v1.ReplicationControllerCondition",
		"io.k8s.kubernetes.pkg.api.v1.EmptyDirVolumeSource":                                            "io.k8s.api.core.v1.EmptyDirVolumeSource",
		"io.k8s.kubernetes.pkg.apis.autoscaling.v1.HorizontalPodAutoscalerList":                        "io.k8s.api.autoscaling.v1.HorizontalPodAutoscalerList",
		"io.k8s.kubernetes.pkg.apis.batch.v1.JobList":                                                  "io.k8s.api.batch.v1.JobList",
		"io.k8s.kubernetes.pkg.api.v1.NFSVolumeSource":                                                 "io.k8s.api.core.v1.NFSVolumeSource",
		"io.k8s.kubernetes.pkg.api.v1.Pod":                                                             "io.k8s.api.core.v1.Pod",
		"io.k8s.kubernetes.pkg.api.v1.ObjectReference":                                                 "io.k8s.api.core.v1.ObjectReference",
		"io.k8s.kubernetes.pkg.apis.apps.v1beta1.Deployment":                                           "io.k8s.api.apps.v1beta1.Deployment",
		"io.k8s.kubernetes.pkg.apis.storage.v1.StorageClassList":                                       "io.k8s.api.storage.v1.StorageClassList",
		"io.k8s.kubernetes.pkg.api.v1.AttachedVolume":                                                  "io.k8s.api.core.v1.AttachedVolume",
		"io.k8s.kubernetes.pkg.api.v1.AWSElasticBlockStoreVolumeSource":                                "io.k8s.api.core.v1.AWSElasticBlockStoreVolumeSource",
		"io.k8s.kubernetes.pkg.apis.batch.v2alpha1.CronJobList":                                        "io.k8s.api.batch.v2alpha1.CronJobList",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.DeploymentSpec":                                 "io.k8s.api.extensions.v1beta1.DeploymentSpec",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.PodSecurityPolicyList":                          "io.k8s.api.extensions.v1beta1.PodSecurityPolicyList",
		"io.k8s.kubernetes.pkg.api.v1.PodAffinityTerm":                                                 "io.k8s.api.core.v1.PodAffinityTerm",
		"io.k8s.kubernetes.pkg.api.v1.HTTPHeader":                                                      "io.k8s.api.core.v1.HTTPHeader",
		"io.k8s.kubernetes.pkg.api.v1.ConfigMapKeySelector":                                            "io.k8s.api.core.v1.ConfigMapKeySelector",
		"io.k8s.kubernetes.pkg.api.v1.SecretKeySelector":                                               "io.k8s.api.core.v1.SecretKeySelector",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.DeploymentList":                                 "io.k8s.api.extensions.v1beta1.DeploymentList",
		"io.k8s.kubernetes.pkg.apis.authentication.v1.UserInfo":                                        "io.k8s.api.authentication.v1.UserInfo",
		"io.k8s.kubernetes.pkg.api.v1.LoadBalancerIngress":                                             "io.k8s.api.core.v1.LoadBalancerIngress",
		"io.k8s.kubernetes.pkg.api.v1.DaemonEndpoint":                                                  "io.k8s.api.core.v1.DaemonEndpoint",
		"io.k8s.kubernetes.pkg.api.v1.NodeSelectorRequirement":                                         "io.k8s.api.core.v1.NodeSelectorRequirement",
		"io.k8s.kubernetes.pkg.apis.batch.v2alpha1.CronJobStatus":                                      "io.k8s.api.batch.v2alpha1.CronJobStatus",
		"io.k8s.kubernetes.pkg.apis.autoscaling.v1.Scale":                                              "io.k8s.api.autoscaling.v1.Scale",
		"io.k8s.kubernetes.pkg.api.v1.ScaleIOVolumeSource":                                             "io.k8s.api.core.v1.ScaleIOVolumeSource",
		"io.k8s.kubernetes.pkg.api.v1.PodAntiAffinity":                                                 "io.k8s.api.core.v1.PodAntiAffinity",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.PodSecurityPolicySpec":                          "io.k8s.api.extensions.v1beta1.PodSecurityPolicySpec",
		"io.k8s.kubernetes.pkg.apis.settings.v1alpha1.PodPresetList":                                   "io.k8s.api.settings.v1alpha1.PodPresetList",
		"io.k8s.kubernetes.pkg.api.v1.NodeAffinity":                                                    "io.k8s.api.core.v1.NodeAffinity",
		"io.k8s.kubernetes.pkg.apis.apps.v1beta1.DeploymentCondition":                                  "io.k8s.api.apps.v1beta1.DeploymentCondition",
		"io.k8s.kubernetes.pkg.api.v1.NodeSpec":                                                        "io.k8s.api.core.v1.NodeSpec",
		"io.k8s.kubernetes.pkg.apis.apps.v1beta1.StatefulSetStatus":                                    "io.k8s.api.apps.v1beta1.StatefulSetStatus",
		"io.k8s.kubernetes.pkg.apis.admissionregistration.v1alpha1.RuleWithOperations":                 "io.k8s.api.admissionregistration.v1alpha1.RuleWithOperations",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.IngressStatus":                                  "io.k8s.api.extensions.v1beta1.IngressStatus",
		"io.k8s.kubernetes.pkg.api.v1.LimitRangeList":                                                  "io.k8s.api.core.v1.LimitRangeList",
		"io.k8s.kubernetes.pkg.api.v1.AzureDiskVolumeSource":                                           "io.k8s.api.core.v1.AzureDiskVolumeSource",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.ReplicaSetStatus":                               "io.k8s.api.extensions.v1beta1.ReplicaSetStatus",
		"io.k8s.kubernetes.pkg.api.v1.ComponentStatus":                                                 "io.k8s.api.core.v1.ComponentStatus",
		"io.k8s.kubernetes.pkg.apis.autoscaling.v1.HorizontalPodAutoscaler":                            "io.k8s.api.autoscaling.v1.HorizontalPodAutoscaler",
		"io.k8s.kubernetes.pkg.apis.networking.v1.NetworkPolicy":                                       "io.k8s.api.networking.v1.NetworkPolicy",
		"io.k8s.kubernetes.pkg.apis.apps.v1beta1.RollbackConfig":                                       "io.k8s.api.apps.v1beta1.RollbackConfig",
		"io.k8s.kubernetes.pkg.api.v1.NodeCondition":                                                   "io.k8s.api.core.v1.NodeCondition",
		"io.k8s.kubernetes.pkg.api.v1.DownwardAPIProjection":                                           "io.k8s.api.core.v1.DownwardAPIProjection",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.SELinuxStrategyOptions":                         "io.k8s.api.extensions.v1beta1.SELinuxStrategyOptions",
		"io.k8s.kubernetes.pkg.api.v1.NamespaceSpec":                                                   "io.k8s.api.core.v1.NamespaceSpec",
		"io.k8s.kubernetes.pkg.apis.certificates.v1beta1.CertificateSigningRequestSpec":                "io.k8s.api.certificates.v1beta1.CertificateSigningRequestSpec",
		"io.k8s.kubernetes.pkg.api.v1.ServiceSpec":                                                     "io.k8s.api.core.v1.ServiceSpec",
		"io.k8s.kubernetes.pkg.apis.authorization.v1.SubjectAccessReview":                              "io.k8s.api.authorization.v1.SubjectAccessReview",
		"io.k8s.kubernetes.pkg.apis.apps.v1beta1.DeploymentList":                                       "io.k8s.api.apps.v1beta1.DeploymentList",
		"io.k8s.kubernetes.pkg.api.v1.Toleration":                                                      "io.k8s.api.core.v1.Toleration",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.NetworkPolicyList":                              "io.k8s.api.extensions.v1beta1.NetworkPolicyList",
		"io.k8s.kubernetes.pkg.apis.autoscaling.v2alpha1.PodsMetricSource":                             "io.k8s.api.autoscaling.v2alpha1.PodsMetricSource",
		"io.k8s.kubernetes.pkg.api.v1.EnvFromSource":                                                   "io.k8s.api.core.v1.EnvFromSource",
		"io.k8s.kubernetes.pkg.apis.autoscaling.v1.ScaleSpec":                                          "io.k8s.api.autoscaling.v1.ScaleSpec",
		"io.k8s.kubernetes.pkg.api.v1.PodTemplateList":                                                 "io.k8s.api.core.v1.PodTemplateList",
		"io.k8s.kubernetes.pkg.apis.autoscaling.v2alpha1.HorizontalPodAutoscalerSpec":                  "io.k8s.api.autoscaling.v2alpha1.HorizontalPodAutoscalerSpec",
		"io.k8s.kubernetes.pkg.api.v1.SecretProjection":                                                "io.k8s.api.core.v1.SecretProjection",
		"io.k8s.kubernetes.pkg.api.v1.ResourceFieldSelector":                                           "io.k8s.api.core.v1.ResourceFieldSelector",
		"io.k8s.kubernetes.pkg.api.v1.PersistentVolumeSpec":                                            "io.k8s.api.core.v1.PersistentVolumeSpec",
		"io.k8s.kubernetes.pkg.api.v1.ConfigMapVolumeSource":                                           "io.k8s.api.core.v1.ConfigMapVolumeSource",
		"io.k8s.kubernetes.pkg.apis.autoscaling.v2alpha1.HorizontalPodAutoscalerList":                  "io.k8s.api.autoscaling.v2alpha1.HorizontalPodAutoscalerList",
		"io.k8s.kubernetes.pkg.apis.authentication.v1beta1.TokenReviewStatus":                          "io.k8s.api.authentication.v1beta1.TokenReviewStatus",
		"io.k8s.kubernetes.pkg.apis.networking.v1.NetworkPolicyList":                                   "io.k8s.api.networking.v1.NetworkPolicyList",
		"io.k8s.kubernetes.pkg.api.v1.Endpoints":                                                       "io.k8s.api.core.v1.Endpoints",
		"io.k8s.kubernetes.pkg.api.v1.LimitRangeItem":                                                  "io.k8s.api.core.v1.LimitRangeItem",
		"io.k8s.kubernetes.pkg.api.v1.ServiceAccount":                                                  "io.k8s.api.core.v1.ServiceAccount",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.ScaleSpec":                                      "io.k8s.api.extensions.v1beta1.ScaleSpec",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.IngressTLS":                                     "io.k8s.api.extensions.v1beta1.IngressTLS",
		"io.k8s.kubernetes.pkg.apis.batch.v2alpha1.CronJob":                                            "io.k8s.api.batch.v2alpha1.CronJob",
		"io.k8s.kubernetes.pkg.apis.rbac.v1alpha1.Subject":                                             "io.k8s.api.rbac.v1alpha1.Subject",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.DaemonSetStatus":                                "io.k8s.api.extensions.v1beta1.DaemonSetStatus",
		"io.k8s.kubernetes.pkg.apis.policy.v1beta1.PodDisruptionBudgetList":                            "io.k8s.api.policy.v1beta1.PodDisruptionBudgetList",
		"io.k8s.kubernetes.pkg.api.v1.VsphereVirtualDiskVolumeSource":                                  "io.k8s.api.core.v1.VsphereVirtualDiskVolumeSource",
		"io.k8s.kubernetes.pkg.apis.rbac.v1alpha1.RoleRef":                                             "io.k8s.api.rbac.v1alpha1.RoleRef",
		"io.k8s.kubernetes.pkg.api.v1.PortworxVolumeSource":                                            "io.k8s.api.core.v1.PortworxVolumeSource",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.ReplicaSetList":                                 "io.k8s.api.extensions.v1beta1.ReplicaSetList",
		"io.k8s.kubernetes.pkg.api.v1.VolumeProjection":                                                "io.k8s.api.core.v1.VolumeProjection",
		"io.k8s.kubernetes.pkg.apis.storage.v1beta1.StorageClass":                                      "io.k8s.api.storage.v1beta1.StorageClass",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.ReplicaSet":                                     "io.k8s.api.extensions.v1beta1.ReplicaSet",
		"io.k8s.kubernetes.pkg.apis.apps.v1beta1.DeploymentRollback":                                   "io.k8s.api.apps.v1beta1.DeploymentRollback",
		"io.k8s.kubernetes.pkg.apis.rbac.v1alpha1.RoleBinding":                                         "io.k8s.api.rbac.v1alpha1.RoleBinding",
		"io.k8s.kubernetes.pkg.api.v1.AzureFileVolumeSource":                                           "io.k8s.api.core.v1.AzureFileVolumeSource",
		"io.k8s.kubernetes.pkg.apis.policy.v1beta1.PodDisruptionBudgetStatus":                          "io.k8s.api.policy.v1beta1.PodDisruptionBudgetStatus",
		"io.k8s.kubernetes.pkg.apis.authentication.v1beta1.TokenReviewSpec":                            "io.k8s.api.authentication.v1beta1.TokenReviewSpec",
		"io.k8s.kubernetes.pkg.api.v1.EndpointsList":                                                   "io.k8s.api.core.v1.EndpointsList",
		"io.k8s.kubernetes.pkg.api.v1.ConfigMapEnvSource":                                              "io.k8s.api.core.v1.ConfigMapEnvSource",
		"io.k8s.kubernetes.pkg.apis.batch.v2alpha1.JobTemplateSpec":                                    "io.k8s.api.batch.v2alpha1.JobTemplateSpec",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.DaemonSetUpdateStrategy":                        "io.k8s.api.extensions.v1beta1.DaemonSetUpdateStrategy",
		"io.k8s.kubernetes.pkg.apis.authorization.v1beta1.SubjectAccessReviewSpec":                     "io.k8s.api.authorization.v1beta1.SubjectAccessReviewSpec",
		"io.k8s.kubernetes.pkg.api.v1.LocalVolumeSource":                                               "io.k8s.api.core.v1.LocalVolumeSource",
		"io.k8s.kubernetes.pkg.api.v1.ContainerState":                                                  "io.k8s.api.core.v1.ContainerState",
		"io.k8s.kubernetes.pkg.api.v1.Service":                                                         "io.k8s.api.core.v1.Service",
		"io.k8s.kubernetes.pkg.api.v1.ExecAction":                                                      "io.k8s.api.core.v1.ExecAction",
		"io.k8s.kubernetes.pkg.api.v1.Taint":                                                           "io.k8s.api.core.v1.Taint",
		"io.k8s.kubernetes.pkg.apis.rbac.v1beta1.Subject":                                              "io.k8s.api.rbac.v1beta1.Subject",
		"io.k8s.kubernetes.pkg.apis.authorization.v1beta1.SubjectAccessReviewStatus":                   "io.k8s.api.authorization.v1beta1.SubjectAccessReviewStatus",
		"io.k8s.kubernetes.pkg.apis.rbac.v1beta1.ClusterRoleBindingList":                               "io.k8s.api.rbac.v1beta1.ClusterRoleBindingList",
		"io.k8s.kubernetes.pkg.api.v1.DownwardAPIVolumeSource":                                         "io.k8s.api.core.v1.DownwardAPIVolumeSource",
		"io.k8s.kubernetes.pkg.apis.batch.v1.JobStatus":                                                "io.k8s.api.batch.v1.JobStatus",
		"io.k8s.kubernetes.pkg.api.v1.ResourceQuotaStatus":                                             "io.k8s.api.core.v1.ResourceQuotaStatus",
		"io.k8s.kubernetes.pkg.api.v1.PersistentVolumeStatus":                                          "io.k8s.api.core.v1.PersistentVolumeStatus",
		"io.k8s.kubernetes.pkg.api.v1.PersistentVolumeClaim":                                           "io.k8s.api.core.v1.PersistentVolumeClaim",
		"io.k8s.kubernetes.pkg.api.v1.NodeDaemonEndpoints":                                             "io.k8s.api.core.v1.NodeDaemonEndpoints",
		"io.k8s.kubernetes.pkg.api.v1.EnvVar":                                                          "io.k8s.api.core.v1.EnvVar",
		"io.k8s.kubernetes.pkg.api.v1.SecretVolumeSource":                                              "io.k8s.api.core.v1.SecretVolumeSource",
		"io.k8s.kubernetes.pkg.api.v1.EnvVarSource":                                                    "io.k8s.api.core.v1.EnvVarSource",
		"io.k8s.kubernetes.pkg.apis.apps.v1beta1.RollingUpdateStatefulSetStrategy":                     "io.k8s.api.apps.v1beta1.RollingUpdateStatefulSetStrategy",
		"io.k8s.kubernetes.pkg.apis.rbac.v1beta1.ClusterRole":                                          "io.k8s.api.rbac.v1beta1.ClusterRole",
		"io.k8s.kubernetes.pkg.apis.admissionregistration.v1alpha1.Initializer":                        "io.k8s.api.admissionregistration.v1alpha1.Initializer",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.DeploymentStrategy":                             "io.k8s.api.extensions.v1beta1.DeploymentStrategy",
		"io.k8s.kubernetes.pkg.apis.apps.v1beta1.ScaleSpec":                                            "io.k8s.api.apps.v1beta1.ScaleSpec",
		"io.k8s.kubernetes.pkg.apis.settings.v1alpha1.PodPreset":                                       "io.k8s.api.settings.v1alpha1.PodPreset",
		"io.k8s.kubernetes.pkg.api.v1.Probe":                                                           "io.k8s.api.core.v1.Probe",
		"io.k8s.kubernetes.pkg.api.v1.NamespaceList":                                                   "io.k8s.api.core.v1.NamespaceList",
		"io.k8s.kubernetes.pkg.api.v1.QuobyteVolumeSource":                                             "io.k8s.api.core.v1.QuobyteVolumeSource",
		"io.k8s.kubernetes.pkg.api.v1.NodeList":                                                        "io.k8s.api.core.v1.NodeList",
		"io.k8s.kubernetes.pkg.apis.extensions.v1beta1.RollingUpdateDaemonSet":                         "io.k8s.api.extensions.v1beta1.RollingUpdateDaemonSet",
		"io.k8s.kubernetes.pkg.api.v1.LimitRange":                                                      "io.k8s.api.core.v1.LimitRange",
	}

	for k, v := range compatibilityMap {
		if _, found := s.Definitions[v]; !found {
			continue
		}
		s.Definitions[k] = spec.Schema{
			SchemaProps: spec.SchemaProps{
				Ref:         spec.MustCreateRef("#/definitions/" + openapi.EscapeJsonPointer(v)),
				Description: fmt.Sprintf("Deprecated. Please use %s instead.", v),
			},
		}
	}
	return s, nil
}
