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
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/go-openapi/spec"
	"github.com/golang/glog"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/sets"
	utilwait "k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/filters"
	"k8s.io/apiserver/pkg/server/options/encryptionconfig"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	aggregatorapiserver "k8s.io/kube-aggregator/pkg/apiserver"
	//aggregatorinformers "k8s.io/kube-aggregator/pkg/client/informers/internalversion"
	openapi "k8s.io/kube-openapi/pkg/common"

	"k8s.io/apiserver/pkg/storage/etcd3/preflight"
	clientgoinformers "k8s.io/client-go/informers"
	clientgoclientset "k8s.io/client-go/kubernetes"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apis/networking"
	"k8s.io/kubernetes/pkg/capabilities"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	"k8s.io/kubernetes/pkg/cloudprovider"
	serviceaccountcontroller "k8s.io/kubernetes/pkg/controller/serviceaccount"
	generatedopenapi "k8s.io/kubernetes/pkg/generated/openapi"
	"k8s.io/kubernetes/pkg/kubeapiserver"
	kubeapiserveradmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
	kubeauthenticator "k8s.io/kubernetes/pkg/kubeapiserver/authenticator"
	"k8s.io/kubernetes/pkg/kubeapiserver/authorizer/modes"
	kubeoptions "k8s.io/kubernetes/pkg/kubeapiserver/options"
	kubeserver "k8s.io/kubernetes/pkg/kubeapiserver/server"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/pkg/master/tunneler"
	quotainstall "k8s.io/kubernetes/pkg/quota/install"
	"k8s.io/kubernetes/pkg/registry/cachesize"
	rbacrest "k8s.io/kubernetes/pkg/registry/rbac/rest"
	"k8s.io/kubernetes/pkg/version"
	"k8s.io/kubernetes/plugin/pkg/auth/authenticator/token/bootstrap"

	_ "k8s.io/kubernetes/pkg/util/reflector/prometheus" // for reflector metric registration
	_ "k8s.io/kubernetes/pkg/util/workqueue/prometheus" // for workqueue metric registration
)

const etcdRetryLimit = 60
const etcdRetryInterval = 1 * time.Second

// NewAPIServerCommand creates a *cobra.Command object with default parameters
func NewAPIServerCommand() *cobra.Command {
	s := options.NewServerRunOptions()
	s.AddFlags(pflag.CommandLine)
	cmd := &cobra.Command{
		Use: "kube-apiserver",
		Long: `The Kubernetes API server validates and configures data
for the api objects which include pods, services, replicationcontrollers, and
others. The API Server services REST operations and provides the frontend to the
cluster's shared state through which all other components interact.`,
		Run: func(cmd *cobra.Command, args []string) {
		},
	}

	return cmd
}

// Run runs the specified APIServer.  This should never exit.
func Run(runOptions *options.ServerRunOptions, stopCh <-chan struct{}) error {
	// To help debugging, immediately log version
	glog.Infof("Version: %+v", version.Get())

	server, err := CreateServerChain(runOptions, stopCh)
	if err != nil {
		return err
	}

	return server.PrepareRun().Run(stopCh)
}

// CreateServerChain creates the apiservers connected via delegation.
func CreateServerChain(runOptions *options.ServerRunOptions, stopCh <-chan struct{}) (*genericapiserver.GenericAPIServer, error) {
	nodeTunneler, proxyTransport, err := CreateNodeDialer(runOptions)
	if err != nil {
		return nil, err
	}
	kubeAPIServerConfig, sharedInformers, versionedInformers, insecureServingOptions, serviceResolver, err := CreateKubeAPIServerConfig(runOptions, nodeTunneler, proxyTransport)
	if err != nil {
		return nil, err
	}

	// TPRs are enabled and not yet beta, since this these are the successor, they fall under the same enablement rule
	// If additional API servers are added, they should be gated.
	apiExtensionsConfig, err := createAPIExtensionsConfig(*kubeAPIServerConfig.GenericConfig, runOptions)
	if err != nil {
		return nil, err
	}
	apiExtensionsServer, err := createAPIExtensionsServer(apiExtensionsConfig, genericapiserver.EmptyDelegate)
	if err != nil {
		return nil, err
	}

	kubeAPIServer, err := CreateKubeAPIServer(kubeAPIServerConfig, apiExtensionsServer.GenericAPIServer, sharedInformers)
	if err != nil {
		return nil, err
	}

	// if we're starting up a hacked up version of this API server for a weird test case,
	// just start the API server as is because clients don't get built correctly when you do this
	if len(os.Getenv("KUBE_API_VERSIONS")) > 0 {
		if insecureServingOptions != nil {
			insecureHandlerChain := kubeserver.BuildInsecureHandlerChain(kubeAPIServer.GenericAPIServer.UnprotectedHandler(), kubeAPIServerConfig.GenericConfig)
			if err := kubeserver.NonBlockingRun(insecureServingOptions, insecureHandlerChain, stopCh); err != nil {
				return nil, err
			}
		}

		return kubeAPIServer.GenericAPIServer, nil
	}

	// otherwise go down the normal path of standing the aggregator up in front of the API server
	// this wires up openapi
	kubeAPIServer.GenericAPIServer.PrepareRun()

	// aggregator comes last in the chain
	aggregatorConfig, err := createAggregatorConfig(*kubeAPIServerConfig.GenericConfig, runOptions, versionedInformers, serviceResolver, proxyTransport)
	if err != nil {
		return nil, err
	}
	aggregatorConfig.ProxyTransport = proxyTransport
	aggregatorConfig.ServiceResolver = serviceResolver
	aggregatorServer, err := createAggregatorServer(aggregatorConfig, kubeAPIServer.GenericAPIServer, apiExtensionsServer.Informers)
	if err != nil {
		// we don't need special handling for innerStopCh because the aggregator server doesn't create any go routines
		return nil, err
	}

	if insecureServingOptions != nil {
		insecureHandlerChain := kubeserver.BuildInsecureHandlerChain(aggregatorServer.GenericAPIServer.UnprotectedHandler(), kubeAPIServerConfig.GenericConfig)
		if err := kubeserver.NonBlockingRun(insecureServingOptions, insecureHandlerChain, stopCh); err != nil {
			return nil, err
		}
	}

	return aggregatorServer.GenericAPIServer, nil
}

// CreateKubeAPIServer creates and wires a workable kube-apiserver
func CreateKubeAPIServer(kubeAPIServerConfig *master.Config, delegateAPIServer genericapiserver.DelegationTarget, sharedInformers informers.SharedInformerFactory) (*master.Master, error) {
	kubeAPIServer, err := kubeAPIServerConfig.Complete().New(delegateAPIServer)
	if err != nil {
		return nil, err
	}
	kubeAPIServer.GenericAPIServer.AddPostStartHook("start-kube-apiserver-informers", func(context genericapiserver.PostStartHookContext) error {
		sharedInformers.Start(context.StopCh)
		return nil
	})

	return kubeAPIServer, nil
}

// CreateNodeDialer creates the dialer infrastructure to connect to the nodes.
func CreateNodeDialer(s *options.ServerRunOptions) (tunneler.Tunneler, *http.Transport, error) {
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
		Dial:            proxyDialerFn,
		TLSClientConfig: proxyTLSClientConfig,
	})
	return nodeTunneler, proxyTransport, nil
}

// CreateKubeAPIServerConfig creates all the resources for running the API server, but runs none of them
func CreateKubeAPIServerConfig(s *options.ServerRunOptions, nodeTunneler tunneler.Tunneler, proxyTransport http.RoundTripper) (*master.Config, informers.SharedInformerFactory, clientgoinformers.SharedInformerFactory, *kubeserver.InsecureServingInfo, aggregatorapiserver.ServiceResolver, error) {
	// register all admission plugins
	RegisterAllAdmissionPlugins(s.Admission.Plugins)

	// set defaults in the options before trying to create the generic config
	if err := defaultOptions(s); err != nil {
		return nil, nil, nil, nil, nil, err
	}

	// validate options
	if errs := s.Validate(); len(errs) != 0 {
		return nil, nil, nil, nil, nil, utilerrors.NewAggregate(errs)
	}

	genericConfig, sharedInformers, versionedInformers, insecureServingOptions, serviceResolver, err := BuildGenericConfig(s)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}

	if _, port, err := net.SplitHostPort(s.Etcd.StorageConfig.ServerList[0]); err == nil && port != "0" && len(port) != 0 {
		if err := utilwait.PollImmediate(etcdRetryInterval, etcdRetryLimit*etcdRetryInterval, preflight.EtcdConnection{ServerList: s.Etcd.StorageConfig.ServerList}.CheckEtcdServers); err != nil {
			return nil, nil, nil, nil, nil, fmt.Errorf("error waiting for etcd connection: %v", err)
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

	serviceIPRange, apiServerServiceIP, err := master.DefaultServiceIPRange(s.ServiceClusterIPRange)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}

	storageFactory, err := BuildStorageFactory(s)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}

	clientCA, err := readCAorNil(s.Authentication.ClientCert.ClientCA)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	requestHeaderProxyCA, err := readCAorNil(s.Authentication.RequestHeader.ClientCAFile)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}

	config := &master.Config{
		GenericConfig: genericConfig,

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
		EnableCoreControllers:   true,
		EventTTL:                s.EventTTL,
		KubeletClientConfig:     s.KubeletConfig,
		EnableUISupport:         true,
		EnableLogsSupport:       s.EnableLogsHandler,
		ProxyTransport:          proxyTransport,

		Tunneler: nodeTunneler,

		ServiceIPRange:       serviceIPRange,
		APIServerServiceIP:   apiServerServiceIP,
		APIServerServicePort: 443,

		ServiceNodePortRange:      s.ServiceNodePortRange,
		KubernetesServiceNodePort: s.KubernetesServiceNodePort,

		MasterCount: s.MasterCount,
	}

	if nodeTunneler != nil {
		// Use the nodeTunneler's dialer to connect to the kubelet
		config.KubeletClientConfig.Dial = nodeTunneler.Dial
	}

	return config, sharedInformers, versionedInformers, insecureServingOptions, serviceResolver, nil
}

// BuildGenericConfig takes the master server options and produces the genericapiserver.Config associated with it
func BuildGenericConfig(s *options.ServerRunOptions) (*genericapiserver.Config, informers.SharedInformerFactory, clientgoinformers.SharedInformerFactory, *kubeserver.InsecureServingInfo, aggregatorapiserver.ServiceResolver, error) {
	genericConfig := genericapiserver.NewConfig(api.Codecs)
	if err := s.GenericServerRunOptions.ApplyTo(genericConfig); err != nil {
		return nil, nil, nil, nil, nil, err
	}
	insecureServingOptions, err := s.InsecureServing.ApplyTo(genericConfig)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	if err := s.SecureServing.ApplyTo(genericConfig); err != nil {
		return nil, nil, nil, nil, nil, err
	}
	if err := s.Authentication.ApplyTo(genericConfig); err != nil {
		return nil, nil, nil, nil, nil, err
	}
	if err := s.Audit.ApplyTo(genericConfig); err != nil {
		return nil, nil, nil, nil, nil, err
	}
	if err := s.Features.ApplyTo(genericConfig); err != nil {
		return nil, nil, nil, nil, nil, err
	}

	genericConfig.OpenAPIConfig = genericapiserver.DefaultOpenAPIConfig(generatedopenapi.GetOpenAPIDefinitions, api.Scheme)
	genericConfig.OpenAPIConfig.PostProcessSpec = postProcessOpenAPISpecForBackwardCompatibility
	genericConfig.OpenAPIConfig.Info.Title = "Kubernetes"
	genericConfig.SwaggerConfig = genericapiserver.DefaultSwaggerConfig()
	genericConfig.EnableMetrics = true
	genericConfig.LongRunningFunc = filters.BasicLongRunningRequestCheck(
		sets.NewString("watch", "proxy"),
		sets.NewString("attach", "exec", "proxy", "log", "portforward"),
	)

	kubeVersion := version.Get()
	genericConfig.Version = &kubeVersion

	storageFactory, err := BuildStorageFactory(s)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	if err := s.Etcd.ApplyWithStorageFactoryTo(storageFactory, genericConfig); err != nil {
		return nil, nil, nil, nil, nil, err
	}

	// Use protobufs for self-communication.
	// Since not every generic apiserver has to support protobufs, we
	// cannot default to it in generic apiserver and need to explicitly
	// set it in kube-apiserver.
	genericConfig.LoopbackClientConfig.ContentConfig.ContentType = "application/vnd.kubernetes.protobuf"

	client, err := internalclientset.NewForConfig(genericConfig.LoopbackClientConfig)
	if err != nil {
		kubeAPIVersions := os.Getenv("KUBE_API_VERSIONS")
		if len(kubeAPIVersions) == 0 {
			return nil, nil, nil, nil, nil, fmt.Errorf("failed to create clientset: %v", err)
		}

		// KUBE_API_VERSIONS is used in test-update-storage-objects.sh, disabling a number of API
		// groups. This leads to a nil client above and undefined behaviour further down.
		//
		// TODO: get rid of KUBE_API_VERSIONS or define sane behaviour if set
		glog.Errorf("Failed to create clientset with KUBE_API_VERSIONS=%q. KUBE_API_VERSIONS is only for testing. Things will break.", kubeAPIVersions)
	}
	externalClient, err := clientset.NewForConfig(genericConfig.LoopbackClientConfig)
	if err != nil {
		return nil, nil, nil, nil, nil, fmt.Errorf("failed to create external clientset: %v", err)
	}
	sharedInformers := informers.NewSharedInformerFactory(client, 10*time.Minute)

	clientgoExternalClient, err := clientgoclientset.NewForConfig(genericConfig.LoopbackClientConfig)
	if err != nil {
		return nil, nil, nil, nil, nil, fmt.Errorf("failed to create real external clientset: %v", err)
	}
	versionedInformers := clientgoinformers.NewSharedInformerFactory(clientgoExternalClient, 10*time.Minute)

	var serviceResolver aggregatorapiserver.ServiceResolver
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

	genericConfig.Authenticator, genericConfig.OpenAPIConfig.SecurityDefinitions, err = BuildAuthenticator(s, storageFactory, client, sharedInformers)
	if err != nil {
		return nil, nil, nil, nil, nil, fmt.Errorf("invalid authentication config: %v", err)
	}

	genericConfig.Authorizer, err = BuildAuthorizer(s, sharedInformers)
	if err != nil {
		return nil, nil, nil, nil, nil, fmt.Errorf("invalid authorization config: %v", err)
	}
	if !sets.NewString(s.Authorization.Modes()...).Has(modes.ModeRBAC) {
		genericConfig.DisabledPostStartHooks.Insert(rbacrest.PostStartHookName)
	}

	pluginInitializer, err := BuildAdmissionPluginInitializer(
		s,
		client,
		externalClient,
		sharedInformers,
		genericConfig.Authorizer,
		serviceResolver,
	)
	if err != nil {
		return nil, nil, nil, nil, nil, fmt.Errorf("failed to create admission plugin initializer: %v", err)
	}

	err = s.Admission.ApplyTo(
		genericConfig,
		pluginInitializer)
	if err != nil {
		return nil, nil, nil, nil, nil, fmt.Errorf("failed to initialize admission: %v", err)
	}
	return genericConfig, sharedInformers, versionedInformers, insecureServingOptions, serviceResolver, nil
}

// BuildAdmissionPluginInitializer constructs the admission plugin initializer
func BuildAdmissionPluginInitializer(s *options.ServerRunOptions, client internalclientset.Interface, externalClient clientset.Interface, sharedInformers informers.SharedInformerFactory, apiAuthorizer authorizer.Authorizer, serviceResolver aggregatorapiserver.ServiceResolver) (admission.PluginInitializer, error) {
	var cloudConfig []byte

	if s.CloudProvider.CloudConfigFile != "" {
		var err error
		cloudConfig, err = ioutil.ReadFile(s.CloudProvider.CloudConfigFile)
		if err != nil {
			glog.Fatalf("Error reading from cloud configuration file %s: %#v", s.CloudProvider.CloudConfigFile, err)
		}
	}

	// TODO: use a dynamic restmapper. See https://github.com/kubernetes/kubernetes/pull/42615.
	restMapper := api.Registry.RESTMapper()

	// NOTE: we do not provide informers to the quota registry because admission level decisions
	// do not require us to open watches for all items tracked by quota.
	quotaRegistry := quotainstall.NewRegistry(nil, nil)

	pluginInitializer := kubeapiserveradmission.NewPluginInitializer(client, externalClient, sharedInformers, apiAuthorizer, cloudConfig, restMapper, quotaRegistry)

	// Read client cert/key for plugins that need to make calls out
	if len(s.ProxyClientCertFile) > 0 && len(s.ProxyClientKeyFile) > 0 {
		certBytes, err := ioutil.ReadFile(s.ProxyClientCertFile)
		if err != nil {
			return nil, err
		}
		keyBytes, err := ioutil.ReadFile(s.ProxyClientKeyFile)
		if err != nil {
			return nil, err
		}
		pluginInitializer = pluginInitializer.SetClientCert(certBytes, keyBytes)
	}

	pluginInitializer = pluginInitializer.SetServiceResolver(serviceResolver)

	return pluginInitializer, nil
}

// BuildAuthenticator constructs the authenticator
func BuildAuthenticator(s *options.ServerRunOptions, storageFactory serverstorage.StorageFactory, client internalclientset.Interface, sharedInformers informers.SharedInformerFactory) (authenticator.Request, *spec.SecurityDefinitions, error) {
	authenticatorConfig := s.Authentication.ToAuthenticationConfig()
	if s.Authentication.ServiceAccounts.Lookup {
		// we have to go direct to storage because the clientsets fail when they're initialized with some API versions excluded
		// we should stop trying to control them like that.
		storageConfigServiceAccounts, err := storageFactory.NewConfig(api.Resource("serviceaccounts"))
		if err != nil {
			return nil, nil, fmt.Errorf("unable to get serviceaccounts storage: %v", err)
		}
		storageConfigSecrets, err := storageFactory.NewConfig(api.Resource("secrets"))
		if err != nil {
			return nil, nil, fmt.Errorf("unable to get secrets storage: %v", err)
		}
		authenticatorConfig.ServiceAccountTokenGetter = serviceaccountcontroller.NewGetterFromStorageInterface(
			storageConfigServiceAccounts,
			storageFactory.ResourcePrefix(api.Resource("serviceaccounts")),
			storageConfigSecrets,
			storageFactory.ResourcePrefix(api.Resource("secrets")),
		)
	}
	if client == nil || reflect.ValueOf(client).IsNil() {
		// TODO: Remove check once client can never be nil.
		glog.Errorf("Failed to setup bootstrap token authenticator because the loopback clientset was not setup properly.")
	} else {
		authenticatorConfig.BootstrapTokenAuthenticator = bootstrap.NewTokenAuthenticator(
			sharedInformers.Core().InternalVersion().Secrets().Lister().Secrets(v1.NamespaceSystem),
		)
	}
	return authenticatorConfig.New()
}

// BuildAuthorizer constructs the authorizer
func BuildAuthorizer(s *options.ServerRunOptions, sharedInformers informers.SharedInformerFactory) (authorizer.Authorizer, error) {
	authorizationConfig := s.Authorization.ToAuthorizationConfig(sharedInformers)
	return authorizationConfig.New()
}

// BuildStorageFactory constructs the storage factory
func BuildStorageFactory(s *options.ServerRunOptions) (*serverstorage.DefaultStorageFactory, error) {
	storageGroupsToEncodingVersion, err := s.StorageSerialization.StorageGroupsToEncodingVersion()
	if err != nil {
		return nil, fmt.Errorf("error generating storage version map: %s", err)
	}
	storageFactory, err := kubeapiserver.NewStorageFactory(
		s.Etcd.StorageConfig, s.Etcd.DefaultStorageMediaType, api.Codecs,
		serverstorage.NewDefaultResourceEncodingConfig(api.Registry), storageGroupsToEncodingVersion,
		// FIXME: this GroupVersionResource override should be configurable
		// TODO we need to update this to batch/v1beta1 when it's enabled by default
		[]schema.GroupVersionResource{batch.Resource("cronjobs").WithVersion("v2alpha1")},
		master.DefaultAPIResourceConfigSource(), s.APIEnablement.RuntimeConfig)
	if err != nil {
		return nil, fmt.Errorf("error in initializing storage factory: %s", err)
	}

	// keep Deployments, NetworkPolicies, Daemonsets and ReplicaSets in extensions for backwards compatibility, we'll have to migrate at some point, eventually
	storageFactory.AddCohabitatingResources(extensions.Resource("deployments"), apps.Resource("deployments"))
	storageFactory.AddCohabitatingResources(extensions.Resource("daemonsets"), apps.Resource("daemonsets"))
	storageFactory.AddCohabitatingResources(extensions.Resource("replicasets"), apps.Resource("replicasets"))
	storageFactory.AddCohabitatingResources(extensions.Resource("networkpolicies"), networking.Resource("networkpolicies"))
	for _, override := range s.Etcd.EtcdServersOverrides {
		tokens := strings.Split(override, "#")
		if len(tokens) != 2 {
			glog.Errorf("invalid value of etcd server overrides: %s", override)
			continue
		}

		apiresource := strings.Split(tokens[0], "/")
		if len(apiresource) != 2 {
			glog.Errorf("invalid resource definition: %s", tokens[0])
			continue
		}
		group := apiresource[0]
		resource := apiresource[1]
		groupResource := schema.GroupResource{Group: group, Resource: resource}

		servers := strings.Split(tokens[1], ";")
		storageFactory.SetEtcdLocation(groupResource, servers)
	}

	if s.Etcd.EncryptionProviderConfigFilepath != "" {
		transformerOverrides, err := encryptionconfig.GetTransformerOverrides(s.Etcd.EncryptionProviderConfigFilepath)
		if err != nil {
			return nil, err
		}
		for groupResource, transformer := range transformerOverrides {
			storageFactory.SetTransformer(groupResource, transformer)
		}
	}

	return storageFactory, nil
}

func defaultOptions(s *options.ServerRunOptions) error {
	// set defaults
	if err := s.GenericServerRunOptions.DefaultAdvertiseAddress(s.SecureServing); err != nil {
		return err
	}
	if err := kubeoptions.DefaultAdvertiseAddress(s.GenericServerRunOptions, s.InsecureServing); err != nil {
		return err
	}
	_, apiServerServiceIP, err := master.DefaultServiceIPRange(s.ServiceClusterIPRange)
	if err != nil {
		return fmt.Errorf("error determining service IP ranges: %v", err)
	}
	s.SecureServing.ForceLoopbackConfigUsage()

	if err := s.SecureServing.MaybeDefaultWithSelfSignedCerts(s.GenericServerRunOptions.AdvertiseAddress.String(), []string{"kubernetes.default.svc", "kubernetes.default", "kubernetes"}, []net.IP{apiServerServiceIP}); err != nil {
		return fmt.Errorf("error creating self-signed certificates: %v", err)
	}
	if err := s.CloudProvider.DefaultExternalHost(s.GenericServerRunOptions); err != nil {
		return fmt.Errorf("error setting the external host value: %v", err)
	}

	s.Authentication.ApplyAuthorization(s.Authorization)

	// Default to the private server key for service account token signing
	if len(s.Authentication.ServiceAccounts.KeyFiles) == 0 && s.SecureServing.ServerCert.CertKey.KeyFile != "" {
		if kubeauthenticator.IsValidServiceAccountKeyFile(s.SecureServing.ServerCert.CertKey.KeyFile) {
			s.Authentication.ServiceAccounts.KeyFiles = []string{s.SecureServing.ServerCert.CertKey.KeyFile}
		} else {
			glog.Warning("No TLS key provided, service account token authentication disabled")
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
		cachesize.InitializeWatchCacheSizes(s.GenericServerRunOptions.TargetRAMMB)
		cachesize.SetWatchCacheSizes(s.GenericServerRunOptions.WatchCacheSizes)
	}

	return nil
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
		"v1beta1.DeploymentStatus":            "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.DeploymentStatus",
		"v1beta1.ReplicaSetList":              "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.ReplicaSetList",
		"v1beta1.Eviction":                    "k8s.io/kubernetes/pkg/apis/policy/v1beta1.Eviction",
		"v1beta1.StatefulSetList":             "k8s.io/kubernetes/pkg/apis/apps/v1beta1.StatefulSetList",
		"v1beta1.RoleBinding":                 "k8s.io/kubernetes/pkg/apis/rbac/v1beta1.RoleBinding",
		"v1beta1.PodSecurityPolicyList":       "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.PodSecurityPolicyList",
		"v1.NodeSpec":                         "k8s.io/kubernetes/pkg/api/v1.NodeSpec",
		"v1.FlockerVolumeSource":              "k8s.io/kubernetes/pkg/api/v1.FlockerVolumeSource",
		"v1.ContainerState":                   "k8s.io/kubernetes/pkg/api/v1.ContainerState",
		"v1beta1.ClusterRole":                 "k8s.io/kubernetes/pkg/apis/rbac/v1beta1.ClusterRole",
		"v1beta1.StorageClass":                "k8s.io/kubernetes/pkg/apis/storage/v1beta1.StorageClass",
		"v1.FlexVolumeSource":                 "k8s.io/kubernetes/pkg/api/v1.FlexVolumeSource",
		"v1.SecretKeySelector":                "k8s.io/kubernetes/pkg/api/v1.SecretKeySelector",
		"v1.DeleteOptions":                    "k8s.io/kubernetes/pkg/api/v1.DeleteOptions",
		"v1.PodStatus":                        "k8s.io/kubernetes/pkg/api/v1.PodStatus",
		"v1.NodeStatus":                       "k8s.io/kubernetes/pkg/api/v1.NodeStatus",
		"v1.ServiceSpec":                      "k8s.io/kubernetes/pkg/api/v1.ServiceSpec",
		"v1.AttachedVolume":                   "k8s.io/kubernetes/pkg/api/v1.AttachedVolume",
		"v1.PersistentVolume":                 "k8s.io/kubernetes/pkg/api/v1.PersistentVolume",
		"v1.LimitRangeList":                   "k8s.io/kubernetes/pkg/api/v1.LimitRangeList",
		"v1alpha1.Role":                       "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1.Role",
		"v1.Affinity":                         "k8s.io/kubernetes/pkg/api/v1.Affinity",
		"v1beta1.PodDisruptionBudget":         "k8s.io/kubernetes/pkg/apis/policy/v1beta1.PodDisruptionBudget",
		"v1alpha1.RoleBindingList":            "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1.RoleBindingList",
		"v1.PodAffinity":                      "k8s.io/kubernetes/pkg/api/v1.PodAffinity",
		"v1beta1.SELinuxStrategyOptions":      "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.SELinuxStrategyOptions",
		"v1.ResourceQuotaList":                "k8s.io/kubernetes/pkg/api/v1.ResourceQuotaList",
		"v1.PodList":                          "k8s.io/kubernetes/pkg/api/v1.PodList",
		"v1.EnvVarSource":                     "k8s.io/kubernetes/pkg/api/v1.EnvVarSource",
		"v1beta1.TokenReviewStatus":           "k8s.io/kubernetes/pkg/apis/authentication/v1beta1.TokenReviewStatus",
		"v1.PersistentVolumeClaimList":        "k8s.io/kubernetes/pkg/api/v1.PersistentVolumeClaimList",
		"v1beta1.RoleList":                    "k8s.io/kubernetes/pkg/apis/rbac/v1beta1.RoleList",
		"v1.ListMeta":                         "k8s.io/apimachinery/pkg/apis/meta/v1.ListMeta",
		"v1.ObjectMeta":                       "k8s.io/apimachinery/pkg/apis/meta/v1.ObjectMeta",
		"v1.APIGroupList":                     "k8s.io/apimachinery/pkg/apis/meta/v1.APIGroupList",
		"v2alpha1.Job":                        "k8s.io/kubernetes/pkg/apis/batch/v2alpha1.Job",
		"v1.EnvFromSource":                    "k8s.io/kubernetes/pkg/api/v1.EnvFromSource",
		"v1beta1.IngressStatus":               "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.IngressStatus",
		"v1.Service":                          "k8s.io/kubernetes/pkg/api/v1.Service",
		"v1beta1.DaemonSetStatus":             "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.DaemonSetStatus",
		"v1alpha1.Subject":                    "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1.Subject",
		"v1.HorizontalPodAutoscaler":          "k8s.io/kubernetes/pkg/apis/autoscaling/v1.HorizontalPodAutoscaler",
		"v1.StatusCause":                      "k8s.io/apimachinery/pkg/apis/meta/v1.StatusCause",
		"v1.NodeSelectorRequirement":          "k8s.io/kubernetes/pkg/api/v1.NodeSelectorRequirement",
		"v1beta1.NetworkPolicyIngressRule":    "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.NetworkPolicyIngressRule",
		"v1beta1.ThirdPartyResource":          "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.ThirdPartyResource",
		"v1beta1.PodSecurityPolicy":           "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.PodSecurityPolicy",
		"v1beta1.StatefulSet":                 "k8s.io/kubernetes/pkg/apis/apps/v1beta1.StatefulSet",
		"v1.LabelSelector":                    "k8s.io/apimachinery/pkg/apis/meta/v1.LabelSelector",
		"v1.ScaleSpec":                        "k8s.io/kubernetes/pkg/apis/autoscaling/v1.ScaleSpec",
		"v1.DownwardAPIVolumeFile":            "k8s.io/kubernetes/pkg/api/v1.DownwardAPIVolumeFile",
		"v1beta1.HorizontalPodAutoscaler":     "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.HorizontalPodAutoscaler",
		"v1.AWSElasticBlockStoreVolumeSource": "k8s.io/kubernetes/pkg/api/v1.AWSElasticBlockStoreVolumeSource",
		"v1.ComponentStatus":                  "k8s.io/kubernetes/pkg/api/v1.ComponentStatus",
		"v2alpha1.JobSpec":                    "k8s.io/kubernetes/pkg/apis/batch/v2alpha1.JobSpec",
		"v1.ContainerImage":                   "k8s.io/kubernetes/pkg/api/v1.ContainerImage",
		"v1.ReplicationControllerStatus":      "k8s.io/kubernetes/pkg/api/v1.ReplicationControllerStatus",
		"v1.ResourceQuota":                    "k8s.io/kubernetes/pkg/api/v1.ResourceQuota",
		"v1beta1.NetworkPolicyList":           "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.NetworkPolicyList",
		"v1beta1.NonResourceAttributes":       "k8s.io/kubernetes/pkg/apis/authorization/v1beta1.NonResourceAttributes",
		"v1.JobCondition":                     "k8s.io/kubernetes/pkg/apis/batch/v1.JobCondition",
		"v1.LabelSelectorRequirement":         "k8s.io/apimachinery/pkg/apis/meta/v1.LabelSelectorRequirement",
		"v1beta1.Deployment":                  "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.Deployment",
		"v1.LoadBalancerIngress":              "k8s.io/kubernetes/pkg/api/v1.LoadBalancerIngress",
		"v1.SecretList":                       "k8s.io/kubernetes/pkg/api/v1.SecretList",
		"v1beta1.ReplicaSetSpec":              "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.ReplicaSetSpec",
		"v1beta1.RoleBindingList":             "k8s.io/kubernetes/pkg/apis/rbac/v1beta1.RoleBindingList",
		"v1.ServicePort":                      "k8s.io/kubernetes/pkg/api/v1.ServicePort",
		"v1.Namespace":                        "k8s.io/kubernetes/pkg/api/v1.Namespace",
		"v1beta1.NetworkPolicyPeer":           "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.NetworkPolicyPeer",
		"v1.ReplicationControllerList":        "k8s.io/kubernetes/pkg/api/v1.ReplicationControllerList",
		"v1beta1.ReplicaSetCondition":         "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.ReplicaSetCondition",
		"v1.ReplicationControllerCondition":   "k8s.io/kubernetes/pkg/api/v1.ReplicationControllerCondition",
		"v1.DaemonEndpoint":                   "k8s.io/kubernetes/pkg/api/v1.DaemonEndpoint",
		"v1beta1.NetworkPolicyPort":           "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.NetworkPolicyPort",
		"v1.NodeSystemInfo":                   "k8s.io/kubernetes/pkg/api/v1.NodeSystemInfo",
		"v1.LimitRangeItem":                   "k8s.io/kubernetes/pkg/api/v1.LimitRangeItem",
		"v1.ConfigMapVolumeSource":            "k8s.io/kubernetes/pkg/api/v1.ConfigMapVolumeSource",
		"v1beta1.ClusterRoleList":             "k8s.io/kubernetes/pkg/apis/rbac/v1beta1.ClusterRoleList",
		"v1beta1.ResourceAttributes":          "k8s.io/kubernetes/pkg/apis/authorization/v1beta1.ResourceAttributes",
		"v1.Pod":                              "k8s.io/kubernetes/pkg/api/v1.Pod",
		"v1.FCVolumeSource":                   "k8s.io/kubernetes/pkg/api/v1.FCVolumeSource",
		"v1beta1.SubresourceReference":        "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.SubresourceReference",
		"v1.ResourceQuotaStatus":              "k8s.io/kubernetes/pkg/api/v1.ResourceQuotaStatus",
		"v1alpha1.RoleBinding":                "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1.RoleBinding",
		"v1.PodCondition":                     "k8s.io/kubernetes/pkg/api/v1.PodCondition",
		"v1.GroupVersionForDiscovery":         "k8s.io/apimachinery/pkg/apis/meta/v1.GroupVersionForDiscovery",
		"v1.NamespaceStatus":                  "k8s.io/kubernetes/pkg/api/v1.NamespaceStatus",
		"v1.Job":                              "k8s.io/kubernetes/pkg/apis/batch/v1.Job",
		"v1.PersistentVolumeClaimVolumeSource":        "k8s.io/kubernetes/pkg/api/v1.PersistentVolumeClaimVolumeSource",
		"v1.Handler":                                  "k8s.io/kubernetes/pkg/api/v1.Handler",
		"v1.ComponentStatusList":                      "k8s.io/kubernetes/pkg/api/v1.ComponentStatusList",
		"v1.ServerAddressByClientCIDR":                "k8s.io/apimachinery/pkg/apis/meta/v1.ServerAddressByClientCIDR",
		"v1.PodAntiAffinity":                          "k8s.io/kubernetes/pkg/api/v1.PodAntiAffinity",
		"v1.ISCSIVolumeSource":                        "k8s.io/kubernetes/pkg/api/v1.ISCSIVolumeSource",
		"v1.ContainerStateRunning":                    "k8s.io/kubernetes/pkg/api/v1.ContainerStateRunning",
		"v1.WeightedPodAffinityTerm":                  "k8s.io/kubernetes/pkg/api/v1.WeightedPodAffinityTerm",
		"v1beta1.HostPortRange":                       "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.HostPortRange",
		"v1.HorizontalPodAutoscalerSpec":              "k8s.io/kubernetes/pkg/apis/autoscaling/v1.HorizontalPodAutoscalerSpec",
		"v1.HorizontalPodAutoscalerList":              "k8s.io/kubernetes/pkg/apis/autoscaling/v1.HorizontalPodAutoscalerList",
		"v1beta1.RoleRef":                             "k8s.io/kubernetes/pkg/apis/rbac/v1beta1.RoleRef",
		"v1.Probe":                                    "k8s.io/kubernetes/pkg/api/v1.Probe",
		"v1beta1.IngressTLS":                          "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.IngressTLS",
		"v1beta1.ThirdPartyResourceList":              "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.ThirdPartyResourceList",
		"v1beta1.DaemonSet":                           "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.DaemonSet",
		"v1.APIGroup":                                 "k8s.io/apimachinery/pkg/apis/meta/v1.APIGroup",
		"v1beta1.Subject":                             "k8s.io/kubernetes/pkg/apis/rbac/v1beta1.Subject",
		"v1beta1.DeploymentList":                      "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.DeploymentList",
		"v1.NodeAffinity":                             "k8s.io/kubernetes/pkg/api/v1.NodeAffinity",
		"v1beta1.RollingUpdateDeployment":             "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.RollingUpdateDeployment",
		"v1beta1.APIVersion":                          "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.APIVersion",
		"v1alpha1.CertificateSigningRequest":          "k8s.io/kubernetes/pkg/apis/certificates/v1alpha1.CertificateSigningRequest",
		"v1.CinderVolumeSource":                       "k8s.io/kubernetes/pkg/api/v1.CinderVolumeSource",
		"v1.NamespaceSpec":                            "k8s.io/kubernetes/pkg/api/v1.NamespaceSpec",
		"v1beta1.PodDisruptionBudgetSpec":             "k8s.io/kubernetes/pkg/apis/policy/v1beta1.PodDisruptionBudgetSpec",
		"v1.Patch":                                    "k8s.io/apimachinery/pkg/apis/meta/v1.Patch",
		"v1beta1.ClusterRoleBinding":                  "k8s.io/kubernetes/pkg/apis/rbac/v1beta1.ClusterRoleBinding",
		"v1beta1.HorizontalPodAutoscalerSpec":         "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.HorizontalPodAutoscalerSpec",
		"v1.PersistentVolumeClaimSpec":                "k8s.io/kubernetes/pkg/api/v1.PersistentVolumeClaimSpec",
		"v1.Secret":                                   "k8s.io/kubernetes/pkg/api/v1.Secret",
		"v1.NodeCondition":                            "k8s.io/kubernetes/pkg/api/v1.NodeCondition",
		"v1.LocalObjectReference":                     "k8s.io/kubernetes/pkg/api/v1.LocalObjectReference",
		"runtime.RawExtension":                        "k8s.io/apimachinery/pkg/runtime.RawExtension",
		"v1.PreferredSchedulingTerm":                  "k8s.io/kubernetes/pkg/api/v1.PreferredSchedulingTerm",
		"v1.RBDVolumeSource":                          "k8s.io/kubernetes/pkg/api/v1.RBDVolumeSource",
		"v1.KeyToPath":                                "k8s.io/kubernetes/pkg/api/v1.KeyToPath",
		"v1.ScaleStatus":                              "k8s.io/kubernetes/pkg/apis/autoscaling/v1.ScaleStatus",
		"v1alpha1.PolicyRule":                         "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1.PolicyRule",
		"v1.EndpointPort":                             "k8s.io/kubernetes/pkg/api/v1.EndpointPort",
		"v1beta1.IngressList":                         "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.IngressList",
		"v1.EndpointAddress":                          "k8s.io/kubernetes/pkg/api/v1.EndpointAddress",
		"v1.NodeSelector":                             "k8s.io/kubernetes/pkg/api/v1.NodeSelector",
		"v1beta1.StorageClassList":                    "k8s.io/kubernetes/pkg/apis/storage/v1beta1.StorageClassList",
		"v1.ServiceList":                              "k8s.io/kubernetes/pkg/api/v1.ServiceList",
		"v2alpha1.CronJobSpec":                        "k8s.io/kubernetes/pkg/apis/batch/v2alpha1.CronJobSpec",
		"v1.ContainerStateTerminated":                 "k8s.io/kubernetes/pkg/api/v1.ContainerStateTerminated",
		"v1beta1.TokenReview":                         "k8s.io/kubernetes/pkg/apis/authentication/v1beta1.TokenReview",
		"v1beta1.IngressBackend":                      "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.IngressBackend",
		"v1.Time":                                     "k8s.io/apimachinery/pkg/apis/meta/v1.Time",
		"v1beta1.IngressSpec":                         "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.IngressSpec",
		"v2alpha1.JobTemplateSpec":                    "k8s.io/kubernetes/pkg/apis/batch/v2alpha1.JobTemplateSpec",
		"v1.LimitRange":                               "k8s.io/kubernetes/pkg/api/v1.LimitRange",
		"v1beta1.UserInfo":                            "k8s.io/kubernetes/pkg/apis/authentication/v1beta1.UserInfo",
		"v1.ResourceQuotaSpec":                        "k8s.io/kubernetes/pkg/api/v1.ResourceQuotaSpec",
		"v1.ContainerPort":                            "k8s.io/kubernetes/pkg/api/v1.ContainerPort",
		"v1beta1.HTTPIngressRuleValue":                "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.HTTPIngressRuleValue",
		"v1.AzureFileVolumeSource":                    "k8s.io/kubernetes/pkg/api/v1.AzureFileVolumeSource",
		"v1beta1.NetworkPolicySpec":                   "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.NetworkPolicySpec",
		"v1.PodTemplateSpec":                          "k8s.io/kubernetes/pkg/api/v1.PodTemplateSpec",
		"v1.SecretVolumeSource":                       "k8s.io/kubernetes/pkg/api/v1.SecretVolumeSource",
		"v1.PodSpec":                                  "k8s.io/kubernetes/pkg/api/v1.PodSpec",
		"v1.CephFSVolumeSource":                       "k8s.io/kubernetes/pkg/api/v1.CephFSVolumeSource",
		"v1beta1.CPUTargetUtilization":                "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.CPUTargetUtilization",
		"v1.Volume":                                   "k8s.io/kubernetes/pkg/api/v1.Volume",
		"v1beta1.Ingress":                             "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.Ingress",
		"v1beta1.HorizontalPodAutoscalerList":         "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.HorizontalPodAutoscalerList",
		"v1.PersistentVolumeStatus":                   "k8s.io/kubernetes/pkg/api/v1.PersistentVolumeStatus",
		"v1beta1.IDRange":                             "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.IDRange",
		"v2alpha1.JobCondition":                       "k8s.io/kubernetes/pkg/apis/batch/v2alpha1.JobCondition",
		"v1beta1.IngressRule":                         "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.IngressRule",
		"v1alpha1.RoleRef":                            "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1.RoleRef",
		"v1.PodAffinityTerm":                          "k8s.io/kubernetes/pkg/api/v1.PodAffinityTerm",
		"v1.ObjectReference":                          "k8s.io/kubernetes/pkg/api/v1.ObjectReference",
		"v1.ServiceStatus":                            "k8s.io/kubernetes/pkg/api/v1.ServiceStatus",
		"v1.APIResource":                              "k8s.io/apimachinery/pkg/apis/meta/v1.APIResource",
		"v1beta1.Scale":                               "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.Scale",
		"v1.AzureDiskVolumeSource":                    "k8s.io/kubernetes/pkg/api/v1.AzureDiskVolumeSource",
		"v1beta1.SubjectAccessReviewStatus":           "k8s.io/kubernetes/pkg/apis/authorization/v1beta1.SubjectAccessReviewStatus",
		"v1.ConfigMap":                                "k8s.io/kubernetes/pkg/api/v1.ConfigMap",
		"v1.CrossVersionObjectReference":              "k8s.io/kubernetes/pkg/apis/autoscaling/v1.CrossVersionObjectReference",
		"v1.APIVersions":                              "k8s.io/apimachinery/pkg/apis/meta/v1.APIVersions",
		"v1alpha1.ClusterRoleList":                    "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1.ClusterRoleList",
		"v1.Node":                                     "k8s.io/kubernetes/pkg/api/v1.Node",
		"resource.Quantity":                           "k8s.io/kubernetes/pkg/api/resource.Quantity",
		"v1.Event":                                    "k8s.io/kubernetes/pkg/api/v1.Event",
		"v1.JobStatus":                                "k8s.io/kubernetes/pkg/apis/batch/v1.JobStatus",
		"v1.PersistentVolumeSpec":                     "k8s.io/kubernetes/pkg/api/v1.PersistentVolumeSpec",
		"v1beta1.SubjectAccessReviewSpec":             "k8s.io/kubernetes/pkg/apis/authorization/v1beta1.SubjectAccessReviewSpec",
		"v1.ResourceFieldSelector":                    "k8s.io/kubernetes/pkg/api/v1.ResourceFieldSelector",
		"v1.EndpointSubset":                           "k8s.io/kubernetes/pkg/api/v1.EndpointSubset",
		"v1alpha1.CertificateSigningRequestSpec":      "k8s.io/kubernetes/pkg/apis/certificates/v1alpha1.CertificateSigningRequestSpec",
		"v1.HostPathVolumeSource":                     "k8s.io/kubernetes/pkg/api/v1.HostPathVolumeSource",
		"v1.LoadBalancerStatus":                       "k8s.io/kubernetes/pkg/api/v1.LoadBalancerStatus",
		"v1beta1.HTTPIngressPath":                     "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.HTTPIngressPath",
		"v1beta1.Role":                                "k8s.io/kubernetes/pkg/apis/rbac/v1beta1.Role",
		"v1beta1.DeploymentStrategy":                  "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.DeploymentStrategy",
		"v1beta1.RunAsUserStrategyOptions":            "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.RunAsUserStrategyOptions",
		"v1beta1.DeploymentSpec":                      "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.DeploymentSpec",
		"v1.ExecAction":                               "k8s.io/kubernetes/pkg/api/v1.ExecAction",
		"v1beta1.PodSecurityPolicySpec":               "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.PodSecurityPolicySpec",
		"v1.HorizontalPodAutoscalerStatus":            "k8s.io/kubernetes/pkg/apis/autoscaling/v1.HorizontalPodAutoscalerStatus",
		"v1.PersistentVolumeList":                     "k8s.io/kubernetes/pkg/api/v1.PersistentVolumeList",
		"v1alpha1.ClusterRole":                        "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1.ClusterRole",
		"v1.JobSpec":                                  "k8s.io/kubernetes/pkg/apis/batch/v1.JobSpec",
		"v1beta1.DaemonSetSpec":                       "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.DaemonSetSpec",
		"v2alpha1.CronJobList":                        "k8s.io/kubernetes/pkg/apis/batch/v2alpha1.CronJobList",
		"v1.Endpoints":                                "k8s.io/kubernetes/pkg/api/v1.Endpoints",
		"v1.SELinuxOptions":                           "k8s.io/kubernetes/pkg/api/v1.SELinuxOptions",
		"v1beta1.SelfSubjectAccessReviewSpec":         "k8s.io/kubernetes/pkg/apis/authorization/v1beta1.SelfSubjectAccessReviewSpec",
		"v1beta1.ScaleStatus":                         "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.ScaleStatus",
		"v1.NodeSelectorTerm":                         "k8s.io/kubernetes/pkg/api/v1.NodeSelectorTerm",
		"v1alpha1.CertificateSigningRequestStatus":    "k8s.io/kubernetes/pkg/apis/certificates/v1alpha1.CertificateSigningRequestStatus",
		"v1.StatusDetails":                            "k8s.io/apimachinery/pkg/apis/meta/v1.StatusDetails",
		"v2alpha1.JobStatus":                          "k8s.io/kubernetes/pkg/apis/batch/v2alpha1.JobStatus",
		"v1beta1.DeploymentRollback":                  "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.DeploymentRollback",
		"v1.GlusterfsVolumeSource":                    "k8s.io/kubernetes/pkg/api/v1.GlusterfsVolumeSource",
		"v1.ServiceAccountList":                       "k8s.io/kubernetes/pkg/api/v1.ServiceAccountList",
		"v1.JobList":                                  "k8s.io/kubernetes/pkg/apis/batch/v1.JobList",
		"v1.EventList":                                "k8s.io/kubernetes/pkg/api/v1.EventList",
		"v1.ContainerStateWaiting":                    "k8s.io/kubernetes/pkg/api/v1.ContainerStateWaiting",
		"v1.APIResourceList":                          "k8s.io/apimachinery/pkg/apis/meta/v1.APIResourceList",
		"v1.ContainerStatus":                          "k8s.io/kubernetes/pkg/api/v1.ContainerStatus",
		"v2alpha1.JobList":                            "k8s.io/kubernetes/pkg/apis/batch/v2alpha1.JobList",
		"v1.ConfigMapKeySelector":                     "k8s.io/kubernetes/pkg/api/v1.ConfigMapKeySelector",
		"v1.PhotonPersistentDiskVolumeSource":         "k8s.io/kubernetes/pkg/api/v1.PhotonPersistentDiskVolumeSource",
		"v1.PodTemplateList":                          "k8s.io/kubernetes/pkg/api/v1.PodTemplateList",
		"v1.PersistentVolumeClaimStatus":              "k8s.io/kubernetes/pkg/api/v1.PersistentVolumeClaimStatus",
		"v1.ServiceAccount":                           "k8s.io/kubernetes/pkg/api/v1.ServiceAccount",
		"v1alpha1.CertificateSigningRequestList":      "k8s.io/kubernetes/pkg/apis/certificates/v1alpha1.CertificateSigningRequestList",
		"v1beta1.SupplementalGroupsStrategyOptions":   "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.SupplementalGroupsStrategyOptions",
		"v1.HTTPHeader":                               "k8s.io/kubernetes/pkg/api/v1.HTTPHeader",
		"version.Info":                                "k8s.io/apimachinery/pkg/version.Info",
		"v1.EventSource":                              "k8s.io/kubernetes/pkg/api/v1.EventSource",
		"v1alpha1.ClusterRoleBindingList":             "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1.ClusterRoleBindingList",
		"v1.OwnerReference":                           "k8s.io/apimachinery/pkg/apis/meta/v1.OwnerReference",
		"v1beta1.ClusterRoleBindingList":              "k8s.io/kubernetes/pkg/apis/rbac/v1beta1.ClusterRoleBindingList",
		"v1beta1.ScaleSpec":                           "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.ScaleSpec",
		"v1.GitRepoVolumeSource":                      "k8s.io/kubernetes/pkg/api/v1.GitRepoVolumeSource",
		"v1beta1.NetworkPolicy":                       "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.NetworkPolicy",
		"v1.ConfigMapEnvSource":                       "k8s.io/kubernetes/pkg/api/v1.ConfigMapEnvSource",
		"v1.PodTemplate":                              "k8s.io/kubernetes/pkg/api/v1.PodTemplate",
		"v1beta1.DeploymentCondition":                 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.DeploymentCondition",
		"v1beta1.PodDisruptionBudgetStatus":           "k8s.io/kubernetes/pkg/apis/policy/v1beta1.PodDisruptionBudgetStatus",
		"v1.EnvVar":                                   "k8s.io/kubernetes/pkg/api/v1.EnvVar",
		"v1.LimitRangeSpec":                           "k8s.io/kubernetes/pkg/api/v1.LimitRangeSpec",
		"v1.DownwardAPIVolumeSource":                  "k8s.io/kubernetes/pkg/api/v1.DownwardAPIVolumeSource",
		"v1.NodeDaemonEndpoints":                      "k8s.io/kubernetes/pkg/api/v1.NodeDaemonEndpoints",
		"v1.ComponentCondition":                       "k8s.io/kubernetes/pkg/api/v1.ComponentCondition",
		"v1alpha1.CertificateSigningRequestCondition": "k8s.io/kubernetes/pkg/apis/certificates/v1alpha1.CertificateSigningRequestCondition",
		"v1.SecurityContext":                          "k8s.io/kubernetes/pkg/api/v1.SecurityContext",
		"v1beta1.LocalSubjectAccessReview":            "k8s.io/kubernetes/pkg/apis/authorization/v1beta1.LocalSubjectAccessReview",
		"v1beta1.StatefulSetSpec":                     "k8s.io/kubernetes/pkg/apis/apps/v1beta1.StatefulSetSpec",
		"v1.NodeAddress":                              "k8s.io/kubernetes/pkg/api/v1.NodeAddress",
		"v1.QuobyteVolumeSource":                      "k8s.io/kubernetes/pkg/api/v1.QuobyteVolumeSource",
		"v1.Capabilities":                             "k8s.io/kubernetes/pkg/api/v1.Capabilities",
		"v1.GCEPersistentDiskVolumeSource":            "k8s.io/kubernetes/pkg/api/v1.GCEPersistentDiskVolumeSource",
		"v1beta1.ReplicaSet":                          "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.ReplicaSet",
		"v1beta1.HorizontalPodAutoscalerStatus":       "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.HorizontalPodAutoscalerStatus",
		"v1beta1.PolicyRule":                          "k8s.io/kubernetes/pkg/apis/rbac/v1beta1.PolicyRule",
		"v1.ConfigMapList":                            "k8s.io/kubernetes/pkg/api/v1.ConfigMapList",
		"v1.Lifecycle":                                "k8s.io/kubernetes/pkg/api/v1.Lifecycle",
		"v1beta1.SelfSubjectAccessReview":             "k8s.io/kubernetes/pkg/apis/authorization/v1beta1.SelfSubjectAccessReview",
		"v2alpha1.CronJob":                            "k8s.io/kubernetes/pkg/apis/batch/v2alpha1.CronJob",
		"v2alpha1.CronJobStatus":                      "k8s.io/kubernetes/pkg/apis/batch/v2alpha1.CronJobStatus",
		"v1beta1.SubjectAccessReview":                 "k8s.io/kubernetes/pkg/apis/authorization/v1beta1.SubjectAccessReview",
		"v1.Preconditions":                            "k8s.io/kubernetes/pkg/api/v1.Preconditions",
		"v1beta1.DaemonSetList":                       "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.DaemonSetList",
		"v1.PersistentVolumeClaim":                    "k8s.io/kubernetes/pkg/api/v1.PersistentVolumeClaim",
		"v1.Scale":                                    "k8s.io/kubernetes/pkg/apis/autoscaling/v1.Scale",
		"v1beta1.StatefulSetStatus":                   "k8s.io/kubernetes/pkg/apis/apps/v1beta1.StatefulSetStatus",
		"v1.NFSVolumeSource":                          "k8s.io/kubernetes/pkg/api/v1.NFSVolumeSource",
		"v1.ObjectFieldSelector":                      "k8s.io/kubernetes/pkg/api/v1.ObjectFieldSelector",
		"v1.ResourceRequirements":                     "k8s.io/kubernetes/pkg/api/v1.ResourceRequirements",
		"v1.WatchEvent":                               "k8s.io/apimachinery/pkg/apis/meta/v1.WatchEvent",
		"v1.ReplicationControllerSpec":                "k8s.io/kubernetes/pkg/api/v1.ReplicationControllerSpec",
		"v1.HTTPGetAction":                            "k8s.io/kubernetes/pkg/api/v1.HTTPGetAction",
		"v1beta1.RollbackConfig":                      "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.RollbackConfig",
		"v1beta1.TokenReviewSpec":                     "k8s.io/kubernetes/pkg/apis/authentication/v1beta1.TokenReviewSpec",
		"v1.PodSecurityContext":                       "k8s.io/kubernetes/pkg/api/v1.PodSecurityContext",
		"v1beta1.PodDisruptionBudgetList":             "k8s.io/kubernetes/pkg/apis/policy/v1beta1.PodDisruptionBudgetList",
		"v1.VolumeMount":                              "k8s.io/kubernetes/pkg/api/v1.VolumeMount",
		"v1.ReplicationController":                    "k8s.io/kubernetes/pkg/api/v1.ReplicationController",
		"v1.NamespaceList":                            "k8s.io/kubernetes/pkg/api/v1.NamespaceList",
		"v1alpha1.ClusterRoleBinding":                 "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1.ClusterRoleBinding",
		"v1.TCPSocketAction":                          "k8s.io/kubernetes/pkg/api/v1.TCPSocketAction",
		"v1.Binding":                                  "k8s.io/kubernetes/pkg/api/v1.Binding",
		"v1beta1.ReplicaSetStatus":                    "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.ReplicaSetStatus",
		"intstr.IntOrString":                          "k8s.io/kubernetes/pkg/util/intstr.IntOrString",
		"v1.EndpointsList":                            "k8s.io/kubernetes/pkg/api/v1.EndpointsList",
		"v1.Container":                                "k8s.io/kubernetes/pkg/api/v1.Container",
		"v1alpha1.RoleList":                           "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1.RoleList",
		"v1.VsphereVirtualDiskVolumeSource":           "k8s.io/kubernetes/pkg/api/v1.VsphereVirtualDiskVolumeSource",
		"v1.NodeList":                                 "k8s.io/kubernetes/pkg/api/v1.NodeList",
		"v1.EmptyDirVolumeSource":                     "k8s.io/kubernetes/pkg/api/v1.EmptyDirVolumeSource",
		"v1beta1.FSGroupStrategyOptions":              "k8s.io/kubernetes/pkg/apis/extensions/v1beta1.FSGroupStrategyOptions",
		"v1.Status":                                   "k8s.io/apimachinery/pkg/apis/meta/v1.Status",
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
