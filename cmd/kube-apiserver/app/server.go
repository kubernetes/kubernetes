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
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/golang/glog"
	"github.com/pborman/uuid"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apiserver"
	"k8s.io/kubernetes/pkg/apiserver/authenticator"
	authorizerunion "k8s.io/kubernetes/pkg/auth/authorizer/union"
	"k8s.io/kubernetes/pkg/auth/user"
	"k8s.io/kubernetes/pkg/capabilities"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller/informers"
	serviceaccountcontroller "k8s.io/kubernetes/pkg/controller/serviceaccount"
	generatedopenapi "k8s.io/kubernetes/pkg/generated/openapi"
	"k8s.io/kubernetes/pkg/genericapiserver"
	"k8s.io/kubernetes/pkg/genericapiserver/authorizer"
	genericvalidation "k8s.io/kubernetes/pkg/genericapiserver/validation"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/pkg/registry/cachesize"
	"k8s.io/kubernetes/pkg/serviceaccount"
	utilnet "k8s.io/kubernetes/pkg/util/net"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/version"
	authenticatorunion "k8s.io/kubernetes/plugin/pkg/auth/authenticator/request/union"
)

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
func Run(s *options.ServerRunOptions) error {
	genericvalidation.VerifyEtcdServersList(s.GenericServerRunOptions)
	genericapiserver.DefaultAndValidateRunOptions(s.GenericServerRunOptions)
	genericConfig := genericapiserver.NewConfig(). // create the new config
							ApplyOptions(s.GenericServerRunOptions). // apply the options selected
							Complete()                               // set default values based on the known values

	serviceIPRange, apiServerServiceIP, err := genericapiserver.DefaultServiceIPRange(s.GenericServerRunOptions.ServiceClusterIPRange)
	if err != nil {
		glog.Fatalf("Error determining service IP ranges: %v", err)
	}
	if err := genericConfig.MaybeGenerateServingCerts(apiServerServiceIP); err != nil {
		glog.Fatalf("Failed to generate service certificate: %v", err)
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

	// Setup tunneler if needed
	var tunneler genericapiserver.Tunneler
	var proxyDialerFn apiserver.ProxyDialerFunc
	if len(s.SSHUser) > 0 {
		// Get ssh key distribution func, if supported
		var installSSH genericapiserver.InstallSSHKey
		cloud, err := cloudprovider.InitCloudProvider(s.GenericServerRunOptions.CloudProvider, s.GenericServerRunOptions.CloudConfigFile)
		if err != nil {
			glog.Fatalf("Cloud provider could not be initialized: %v", err)
		}
		if cloud != nil {
			if instances, supported := cloud.Instances(); supported {
				installSSH = instances.AddSSHKeyToAllInstances
			}
		}
		if s.KubeletConfig.Port == 0 {
			glog.Fatalf("Must enable kubelet port if proxy ssh-tunneling is specified.")
		}
		// Set up the tunneler
		// TODO(cjcullen): If we want this to handle per-kubelet ports or other
		// kubelet listen-addresses, we need to plumb through options.
		healthCheckPath := &url.URL{
			Scheme: "https",
			Host:   net.JoinHostPort("127.0.0.1", strconv.FormatUint(uint64(s.KubeletConfig.Port), 10)),
			Path:   "healthz",
		}
		tunneler = genericapiserver.NewSSHTunneler(s.SSHUser, s.SSHKeyfile, healthCheckPath, installSSH)

		// Use the tunneler's dialer to connect to the kubelet
		s.KubeletConfig.Dial = tunneler.Dial
		// Use the tunneler's dialer when proxying to pods, services, and nodes
		proxyDialerFn = tunneler.Dial
	}

	// Proxying to pods and services is IP-based... don't expect to be able to verify the hostname
	proxyTLSClientConfig := &tls.Config{InsecureSkipVerify: true}

	if s.GenericServerRunOptions.StorageConfig.DeserializationCacheSize == 0 {
		// When size of cache is not explicitly set, estimate its size based on
		// target memory usage.
		glog.V(2).Infof("Initalizing deserialization cache size based on %dMB limit", s.GenericServerRunOptions.TargetRAMMB)

		// This is the heuristics that from memory capacity is trying to infer
		// the maximum number of nodes in the cluster and set cache sizes based
		// on that value.
		// From our documentation, we officially recomment 120GB machines for
		// 2000 nodes, and we scale from that point. Thus we assume ~60MB of
		// capacity per node.
		// TODO: We may consider deciding that some percentage of memory will
		// be used for the deserialization cache and divide it by the max object
		// size to compute its size. We may even go further and measure
		// collective sizes of the objects in the cache.
		clusterSize := s.GenericServerRunOptions.TargetRAMMB / 60
		s.GenericServerRunOptions.StorageConfig.DeserializationCacheSize = 25 * clusterSize
		if s.GenericServerRunOptions.StorageConfig.DeserializationCacheSize < 1000 {
			s.GenericServerRunOptions.StorageConfig.DeserializationCacheSize = 1000
		}
	}

	storageGroupsToEncodingVersion, err := s.GenericServerRunOptions.StorageGroupsToEncodingVersion()
	if err != nil {
		glog.Fatalf("error generating storage version map: %s", err)
	}
	storageFactory, err := genericapiserver.BuildDefaultStorageFactory(
		s.GenericServerRunOptions.StorageConfig, s.GenericServerRunOptions.DefaultStorageMediaType, api.Codecs,
		genericapiserver.NewDefaultResourceEncodingConfig(), storageGroupsToEncodingVersion,
		// FIXME: this GroupVersionResource override should be configurable
		[]unversioned.GroupVersionResource{batch.Resource("cronjobs").WithVersion("v2alpha1")},
		master.DefaultAPIResourceConfigSource(), s.GenericServerRunOptions.RuntimeConfig)
	if err != nil {
		glog.Fatalf("error in initializing storage factory: %s", err)
	}
	storageFactory.AddCohabitatingResources(batch.Resource("jobs"), extensions.Resource("jobs"))
	storageFactory.AddCohabitatingResources(autoscaling.Resource("horizontalpodautoscalers"), extensions.Resource("horizontalpodautoscalers"))
	for _, override := range s.GenericServerRunOptions.EtcdServersOverrides {
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
		groupResource := unversioned.GroupResource{Group: group, Resource: resource}

		servers := strings.Split(tokens[1], ";")
		storageFactory.SetEtcdLocation(groupResource, servers)
	}

	// Default to the private server key for service account token signing
	if len(s.ServiceAccountKeyFiles) == 0 && s.GenericServerRunOptions.TLSPrivateKeyFile != "" {
		if authenticator.IsValidServiceAccountKeyFile(s.GenericServerRunOptions.TLSPrivateKeyFile) {
			s.ServiceAccountKeyFiles = []string{s.GenericServerRunOptions.TLSPrivateKeyFile}
		} else {
			glog.Warning("No TLS key provided, service account token authentication disabled")
		}
	}

	var serviceAccountGetter serviceaccount.ServiceAccountTokenGetter
	if s.ServiceAccountLookup {
		// If we need to look up service accounts and tokens,
		// go directly to etcd to avoid recursive auth insanity
		storageConfig, err := storageFactory.NewConfig(api.Resource("serviceaccounts"))
		if err != nil {
			glog.Fatalf("Unable to get serviceaccounts storage: %v", err)
		}
		serviceAccountGetter = serviceaccountcontroller.NewGetterFromStorageInterface(storageConfig, storageFactory.ResourcePrefix(api.Resource("serviceaccounts")), storageFactory.ResourcePrefix(api.Resource("secrets")))
	}

	apiAuthenticator, securityDefinitions, err := authenticator.New(authenticator.AuthenticatorConfig{
		Anonymous:                   s.GenericServerRunOptions.AnonymousAuth,
		AnyToken:                    s.GenericServerRunOptions.EnableAnyToken,
		BasicAuthFile:               s.GenericServerRunOptions.BasicAuthFile,
		ClientCAFile:                s.GenericServerRunOptions.ClientCAFile,
		TokenAuthFile:               s.GenericServerRunOptions.TokenAuthFile,
		OIDCIssuerURL:               s.GenericServerRunOptions.OIDCIssuerURL,
		OIDCClientID:                s.GenericServerRunOptions.OIDCClientID,
		OIDCCAFile:                  s.GenericServerRunOptions.OIDCCAFile,
		OIDCUsernameClaim:           s.GenericServerRunOptions.OIDCUsernameClaim,
		OIDCGroupsClaim:             s.GenericServerRunOptions.OIDCGroupsClaim,
		ServiceAccountKeyFiles:      s.ServiceAccountKeyFiles,
		ServiceAccountLookup:        s.ServiceAccountLookup,
		ServiceAccountTokenGetter:   serviceAccountGetter,
		KeystoneURL:                 s.GenericServerRunOptions.KeystoneURL,
		KeystoneCAFile:              s.GenericServerRunOptions.KeystoneCAFile,
		WebhookTokenAuthnConfigFile: s.WebhookTokenAuthnConfigFile,
		WebhookTokenAuthnCacheTTL:   s.WebhookTokenAuthnCacheTTL,
		RequestHeaderConfig:         s.GenericServerRunOptions.AuthenticationRequestHeaderConfig(),
	})

	if err != nil {
		glog.Fatalf("Invalid Authentication Config: %v", err)
	}

	privilegedLoopbackToken := uuid.NewRandom().String()
	selfClientConfig, err := s.GenericServerRunOptions.NewSelfClientConfig(privilegedLoopbackToken)
	if err != nil {
		glog.Fatalf("Failed to create clientset: %v", err)
	}
	client, err := s.GenericServerRunOptions.NewSelfClient(privilegedLoopbackToken)
	if err != nil {
		glog.Errorf("Failed to create clientset: %v", err)
	}
	sharedInformers := informers.NewSharedInformerFactory(client, 10*time.Minute)

	authorizationConfig := authorizer.AuthorizationConfig{
		PolicyFile:                  s.GenericServerRunOptions.AuthorizationPolicyFile,
		WebhookConfigFile:           s.GenericServerRunOptions.AuthorizationWebhookConfigFile,
		WebhookCacheAuthorizedTTL:   s.GenericServerRunOptions.AuthorizationWebhookCacheAuthorizedTTL,
		WebhookCacheUnauthorizedTTL: s.GenericServerRunOptions.AuthorizationWebhookCacheUnauthorizedTTL,
		RBACSuperUser:               s.GenericServerRunOptions.AuthorizationRBACSuperUser,
		InformerFactory:             sharedInformers,
	}
	authorizationModeNames := strings.Split(s.GenericServerRunOptions.AuthorizationMode, ",")
	apiAuthorizer, err := authorizer.NewAuthorizerFromAuthorizationConfig(authorizationModeNames, authorizationConfig)
	if err != nil {
		glog.Fatalf("Invalid Authorization Config: %v", err)
	}

	admissionControlPluginNames := strings.Split(s.GenericServerRunOptions.AdmissionControl, ",")

	// TODO(dims): We probably need to add an option "EnableLoopbackToken"
	if apiAuthenticator != nil {
		var uid = uuid.NewRandom().String()
		tokens := make(map[string]*user.DefaultInfo)
		tokens[privilegedLoopbackToken] = &user.DefaultInfo{
			Name:   user.APIServerUser,
			UID:    uid,
			Groups: []string{user.SystemPrivilegedGroup},
		}

		tokenAuthenticator := authenticator.NewAuthenticatorFromTokens(tokens)
		apiAuthenticator = authenticatorunion.New(tokenAuthenticator, apiAuthenticator)

		tokenAuthorizer := authorizer.NewPrivilegedGroups(user.SystemPrivilegedGroup)
		apiAuthorizer = authorizerunion.New(tokenAuthorizer, apiAuthorizer)
	}

	pluginInitializer := admission.NewPluginInitializer(sharedInformers, apiAuthorizer)

	admissionController, err := admission.NewFromPlugins(client, admissionControlPluginNames, s.GenericServerRunOptions.AdmissionControlConfigFile, pluginInitializer)
	if err != nil {
		glog.Fatalf("Failed to initialize plugins: %v", err)
	}

	proxyTransport := utilnet.SetTransportDefaults(&http.Transport{
		Dial:            proxyDialerFn,
		TLSClientConfig: proxyTLSClientConfig,
	})
	kubeVersion := version.Get()

	genericConfig.Version = &kubeVersion
	genericConfig.LoopbackClientConfig = selfClientConfig
	genericConfig.Authenticator = apiAuthenticator
	genericConfig.Authorizer = apiAuthorizer
	genericConfig.AdmissionControl = admissionController
	genericConfig.APIResourceConfigSource = storageFactory.APIResourceConfigSource
	genericConfig.OpenAPIConfig.Info.Title = "Kubernetes"
	genericConfig.OpenAPIConfig.Definitions = generatedopenapi.OpenAPIDefinitions
	genericConfig.EnableOpenAPISupport = true
	genericConfig.EnableMetrics = true
	genericConfig.OpenAPIConfig.SecurityDefinitions = securityDefinitions

	config := &master.Config{
		GenericConfig: genericConfig.Config,

		StorageFactory:          storageFactory,
		EnableWatchCache:        s.GenericServerRunOptions.EnableWatchCache,
		EnableCoreControllers:   true,
		DeleteCollectionWorkers: s.GenericServerRunOptions.DeleteCollectionWorkers,
		EventTTL:                s.EventTTL,
		KubeletClientConfig:     s.KubeletConfig,
		EnableUISupport:         true,
		EnableLogsSupport:       true,
		ProxyTransport:          proxyTransport,

		Tunneler: tunneler,

		ServiceIPRange:       serviceIPRange,
		APIServerServiceIP:   apiServerServiceIP,
		APIServerServicePort: 443,

		ServiceNodePortRange:      s.GenericServerRunOptions.ServiceNodePortRange,
		KubernetesServiceNodePort: s.GenericServerRunOptions.KubernetesServiceNodePort,

		MasterCount: s.GenericServerRunOptions.MasterCount,
	}

	if s.GenericServerRunOptions.EnableWatchCache {
		glog.V(2).Infof("Initalizing cache sizes based on %dMB limit", s.GenericServerRunOptions.TargetRAMMB)
		cachesize.InitializeWatchCacheSizes(s.GenericServerRunOptions.TargetRAMMB)
		cachesize.SetWatchCacheSizes(s.GenericServerRunOptions.WatchCacheSizes)
	}

	m, err := config.Complete().New()
	if err != nil {
		return err
	}

	sharedInformers.Start(wait.NeverStop)
	m.GenericAPIServer.PrepareRun().Run(wait.NeverStop)
	return nil
}
