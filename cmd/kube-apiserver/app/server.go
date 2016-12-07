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
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apiserver"
	"k8s.io/kubernetes/pkg/apiserver/authenticator"
	"k8s.io/kubernetes/pkg/capabilities"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller/informers"
	serviceaccountcontroller "k8s.io/kubernetes/pkg/controller/serviceaccount"
	generatedopenapi "k8s.io/kubernetes/pkg/generated/openapi"
	"k8s.io/kubernetes/pkg/genericapiserver"
	"k8s.io/kubernetes/pkg/genericapiserver/authorizer"
	"k8s.io/kubernetes/pkg/genericapiserver/filters"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/pkg/registry/cachesize"
	"k8s.io/kubernetes/pkg/runtime/schema"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	utilnet "k8s.io/kubernetes/pkg/util/net"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/version"
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
	if errs := s.Etcd.Validate(); len(errs) > 0 {
		return utilerrors.NewAggregate(errs)
	}
	if err := s.GenericServerRunOptions.DefaultExternalAddress(s.SecureServing, s.InsecureServing); err != nil {
		return err
	}

	serviceIPRange, apiServerServiceIP, err := master.DefaultServiceIPRange(s.GenericServerRunOptions.ServiceClusterIPRange)
	if err != nil {
		return fmt.Errorf("error determining service IP ranges: %v", err)
	}

	if err := s.SecureServing.MaybeDefaultWithSelfSignedCerts(s.GenericServerRunOptions.AdvertiseAddress.String(), apiServerServiceIP); err != nil {
		return fmt.Errorf("error creating self-signed certificates: %v", err)
	}

	genericapiserver.DefaultAndValidateRunOptions(s.GenericServerRunOptions)

	genericConfig := genericapiserver.NewConfig(). // create the new config
							ApplyOptions(s.GenericServerRunOptions). // apply the options selected
							ApplyInsecureServingOptions(s.InsecureServing)

	if _, err := genericConfig.ApplySecureServingOptions(s.SecureServing); err != nil {
		return fmt.Errorf("failed to configure https: %s", err)
	}
	if _, err = genericConfig.ApplyAuthenticationOptions(s.Authentication); err != nil {
		return fmt.Errorf("failed to configure authentication: %s", err)
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
			return fmt.Errorf("cloud provider could not be initialized: %v", err)
		}
		if cloud != nil {
			if instances, supported := cloud.Instances(); supported {
				installSSH = instances.AddSSHKeyToAllInstances
			}
		}
		if s.KubeletConfig.Port == 0 {
			return fmt.Errorf("must enable kubelet port if proxy ssh-tunneling is specified")
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

	if s.Etcd.StorageConfig.DeserializationCacheSize == 0 {
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
		s.Etcd.StorageConfig.DeserializationCacheSize = 25 * clusterSize
		if s.Etcd.StorageConfig.DeserializationCacheSize < 1000 {
			s.Etcd.StorageConfig.DeserializationCacheSize = 1000
		}
	}

	storageGroupsToEncodingVersion, err := s.GenericServerRunOptions.StorageGroupsToEncodingVersion()
	if err != nil {
		return fmt.Errorf("error generating storage version map: %s", err)
	}
	storageFactory, err := genericapiserver.BuildDefaultStorageFactory(
		s.Etcd.StorageConfig, s.GenericServerRunOptions.DefaultStorageMediaType, api.Codecs,
		genericapiserver.NewDefaultResourceEncodingConfig(), storageGroupsToEncodingVersion,
		// FIXME: this GroupVersionResource override should be configurable
		[]schema.GroupVersionResource{batch.Resource("cronjobs").WithVersion("v2alpha1")},
		master.DefaultAPIResourceConfigSource(), s.GenericServerRunOptions.RuntimeConfig)
	if err != nil {
		return fmt.Errorf("error in initializing storage factory: %s", err)
	}
	storageFactory.AddCohabitatingResources(batch.Resource("jobs"), extensions.Resource("jobs"))
	storageFactory.AddCohabitatingResources(autoscaling.Resource("horizontalpodautoscalers"), extensions.Resource("horizontalpodautoscalers"))
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

	// Default to the private server key for service account token signing
	if len(s.Authentication.ServiceAccounts.KeyFiles) == 0 && s.SecureServing.ServerCert.CertKey.KeyFile != "" {
		if authenticator.IsValidServiceAccountKeyFile(s.SecureServing.ServerCert.CertKey.KeyFile) {
			s.Authentication.ServiceAccounts.KeyFiles = []string{s.SecureServing.ServerCert.CertKey.KeyFile}
		} else {
			glog.Warning("No TLS key provided, service account token authentication disabled")
		}
	}

	authenticatorConfig := s.Authentication.ToAuthenticationConfig()
	if s.Authentication.ServiceAccounts.Lookup {
		// If we need to look up service accounts and tokens,
		// go directly to etcd to avoid recursive auth insanity
		storageConfig, err := storageFactory.NewConfig(api.Resource("serviceaccounts"))
		if err != nil {
			return fmt.Errorf("unable to get serviceaccounts storage: %v", err)
		}
		authenticatorConfig.ServiceAccountTokenGetter = serviceaccountcontroller.NewGetterFromStorageInterface(storageConfig, storageFactory.ResourcePrefix(api.Resource("serviceaccounts")), storageFactory.ResourcePrefix(api.Resource("secrets")))
	}

	apiAuthenticator, securityDefinitions, err := authenticator.New(authenticatorConfig)
	if err != nil {
		return fmt.Errorf("invalid Authentication Config: %v", err)
	}

	privilegedLoopbackToken := uuid.NewRandom().String()
	selfClientConfig, err := genericapiserver.NewSelfClientConfig(genericConfig.SecureServingInfo, genericConfig.InsecureServingInfo, privilegedLoopbackToken)
	if err != nil {
		return fmt.Errorf("failed to create clientset: %v", err)
	}
	client, err := internalclientset.NewForConfig(selfClientConfig)
	if err != nil {
		glog.Errorf("Failed to create clientset: %v", err)
	}
	sharedInformers := informers.NewSharedInformerFactory(nil, client, 10*time.Minute)

	authorizationConfig := s.Authorization.ToAuthorizationConfig(sharedInformers)
	apiAuthorizer, err := authorizer.NewAuthorizerFromAuthorizationConfig(authorizationConfig)
	if err != nil {
		return fmt.Errorf("invalid Authorization Config: %v", err)
	}

	admissionControlPluginNames := strings.Split(s.GenericServerRunOptions.AdmissionControl, ",")
	pluginInitializer := admission.NewPluginInitializer(sharedInformers, apiAuthorizer)
	admissionController, err := admission.NewFromPlugins(client, admissionControlPluginNames, s.GenericServerRunOptions.AdmissionControlConfigFile, pluginInitializer)
	if err != nil {
		return fmt.Errorf("failed to initialize plugins: %v", err)
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
	genericConfig.OpenAPIConfig.Info.Title = "Kubernetes"
	genericConfig.OpenAPIConfig.Definitions = generatedopenapi.OpenAPIDefinitions
	genericConfig.EnableOpenAPISupport = true
	genericConfig.EnableMetrics = true
	genericConfig.OpenAPIConfig.SecurityDefinitions = securityDefinitions
	genericConfig.LongRunningFunc = filters.BasicLongRunningRequestCheck(
		sets.NewString("watch", "proxy"),
		sets.NewString("attach", "exec", "proxy", "log", "portforward"),
	)

	config := &master.Config{
		GenericConfig: genericConfig,

		APIResourceConfigSource: storageFactory.APIResourceConfigSource,
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
