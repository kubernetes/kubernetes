/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"net/url"
	"strconv"
	"strings"

	"github.com/golang/glog"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	appsapi "k8s.io/kubernetes/pkg/apis/apps/v1alpha1"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	autoscalingapiv1 "k8s.io/kubernetes/pkg/apis/autoscaling/v1"
	"k8s.io/kubernetes/pkg/apis/batch"
	batchapiv1 "k8s.io/kubernetes/pkg/apis/batch/v1"
	"k8s.io/kubernetes/pkg/apis/extensions"
	extensionsapiv1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/apiserver"
	"k8s.io/kubernetes/pkg/apiserver/authenticator"
	"k8s.io/kubernetes/pkg/capabilities"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/cloudprovider"
	serviceaccountcontroller "k8s.io/kubernetes/pkg/controller/serviceaccount"
	"k8s.io/kubernetes/pkg/genericapiserver"
	kubeletclient "k8s.io/kubernetes/pkg/kubelet/client"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/pkg/registry/cachesize"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/serviceaccount"
)

// NewAPIServerCommand creates a *cobra.Command object with default parameters
func NewAPIServerCommand() *cobra.Command {
	s := options.NewAPIServer()
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
func Run(s *options.APIServer) error {
	genericapiserver.DefaultAndValidateRunOptions(s.ServerRunOptions)

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
		cloud, err := cloudprovider.InitCloudProvider(s.CloudProvider, s.CloudConfigFile)
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

	kubeletClient, err := kubeletclient.NewStaticKubeletClient(&s.KubeletConfig)
	if err != nil {
		glog.Fatalf("Failure to start kubelet client: %v", err)
	}

	apiResourceConfigSource, err := parseRuntimeConfig(s)
	if err != nil {
		glog.Fatalf("error in parsing runtime-config: %s", err)
	}

	clientConfig := &restclient.Config{
		Host: net.JoinHostPort(s.InsecureBindAddress.String(), strconv.Itoa(s.InsecurePort)),
		// Increase QPS limits. The client is currently passed to all admission plugins,
		// and those can be throttled in case of higher load on apiserver - see #22340 and #22422
		// for more details. Once #22422 is fixed, we may want to remove it.
		QPS:   50,
		Burst: 100,
	}
	if len(s.DeprecatedStorageVersion) != 0 {
		gv, err := unversioned.ParseGroupVersion(s.DeprecatedStorageVersion)
		if err != nil {
			glog.Fatalf("error in parsing group version: %s", err)
		}
		clientConfig.GroupVersion = &gv
	}

	client, err := clientset.NewForConfig(clientConfig)
	if err != nil {
		glog.Errorf("Failed to create clientset: %v", err)
	}

	resourceEncoding := genericapiserver.NewDefaultResourceEncodingConfig()
	groupToEncoding, err := s.StorageGroupsToEncodingVersion()
	if err != nil {
		glog.Fatalf("error getting group encoding: %s", err)
	}
	for group, storageEncodingVersion := range groupToEncoding {
		resourceEncoding.SetVersionEncoding(group, storageEncodingVersion, unversioned.GroupVersion{Group: group, Version: runtime.APIVersionInternal})
	}

	storageFactory := genericapiserver.NewDefaultStorageFactory(s.StorageConfig, s.DefaultStorageMediaType, api.Codecs, resourceEncoding, apiResourceConfigSource)
	// third party resources are always serialized to storage using JSON
	storageFactory.SetSerializer(extensions.Resource("thirdpartyresources"), "application/json", nil)
	storageFactory.AddCohabitatingResources(batch.Resource("jobs"), extensions.Resource("jobs"))
	storageFactory.AddCohabitatingResources(autoscaling.Resource("horizontalpodautoscalers"), extensions.Resource("horizontalpodautoscalers"))
	for _, override := range s.EtcdServersOverrides {
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
	if s.ServiceAccountKeyFile == "" && s.TLSPrivateKeyFile != "" {
		if authenticator.IsValidServiceAccountKeyFile(s.TLSPrivateKeyFile) {
			s.ServiceAccountKeyFile = s.TLSPrivateKeyFile
		} else {
			glog.Warning("No RSA key provided, service account token authentication disabled")
		}
	}

	var serviceAccountGetter serviceaccount.ServiceAccountTokenGetter
	if s.ServiceAccountLookup {
		// If we need to look up service accounts and tokens,
		// go directly to etcd to avoid recursive auth insanity
		storage, err := storageFactory.New(api.Resource("serviceaccounts"))
		if err != nil {
			glog.Fatalf("Unable to get serviceaccounts storage: %v", err)
		}
		serviceAccountGetter = serviceaccountcontroller.NewGetterFromStorageInterface(storage)
	}

	authenticator, err := authenticator.New(authenticator.AuthenticatorConfig{
		BasicAuthFile:             s.BasicAuthFile,
		ClientCAFile:              s.ClientCAFile,
		TokenAuthFile:             s.TokenAuthFile,
		OIDCIssuerURL:             s.OIDCIssuerURL,
		OIDCClientID:              s.OIDCClientID,
		OIDCCAFile:                s.OIDCCAFile,
		OIDCUsernameClaim:         s.OIDCUsernameClaim,
		OIDCGroupsClaim:           s.OIDCGroupsClaim,
		ServiceAccountKeyFile:     s.ServiceAccountKeyFile,
		ServiceAccountLookup:      s.ServiceAccountLookup,
		ServiceAccountTokenGetter: serviceAccountGetter,
		KeystoneURL:               s.KeystoneURL,
	})

	if err != nil {
		glog.Fatalf("Invalid Authentication Config: %v", err)
	}

	authorizationModeNames := strings.Split(s.AuthorizationMode, ",")
	authorizer, err := apiserver.NewAuthorizerFromAuthorizationConfig(authorizationModeNames, s.AuthorizationConfig)
	if err != nil {
		glog.Fatalf("Invalid Authorization Config: %v", err)
	}

	admissionControlPluginNames := strings.Split(s.AdmissionControl, ",")
	admissionController := admission.NewFromPlugins(client, admissionControlPluginNames, s.AdmissionControlConfigFile)

	genericConfig := genericapiserver.NewConfig(s.ServerRunOptions)
	// TODO: Move the following to generic api server as well.
	genericConfig.StorageFactory = storageFactory
	genericConfig.Authenticator = authenticator
	genericConfig.SupportsBasicAuth = len(s.BasicAuthFile) > 0
	genericConfig.Authorizer = authorizer
	genericConfig.AdmissionControl = admissionController
	genericConfig.APIResourceConfigSource = apiResourceConfigSource
	genericConfig.MasterServiceNamespace = s.MasterServiceNamespace
	genericConfig.ProxyDialer = proxyDialerFn
	genericConfig.ProxyTLSClientConfig = proxyTLSClientConfig
	genericConfig.Serializer = api.Codecs

	config := &master.Config{
		Config:                  genericConfig,
		EnableCoreControllers:   true,
		DeleteCollectionWorkers: s.DeleteCollectionWorkers,
		EventTTL:                s.EventTTL,
		KubeletClient:           kubeletClient,

		Tunneler: tunneler,
	}

	if s.EnableWatchCache {
		cachesize.SetWatchCacheSizes(s.WatchCacheSizes)
	}

	m, err := master.New(config)
	if err != nil {
		return err
	}

	m.Run(s.ServerRunOptions)
	return nil
}

func getRuntimeConfigValue(s *options.APIServer, apiKey string, defaultValue bool) bool {
	flagValue, ok := s.RuntimeConfig[apiKey]
	if ok {
		if flagValue == "" {
			return true
		}
		boolValue, err := strconv.ParseBool(flagValue)
		if err != nil {
			glog.Fatalf("Invalid value of %s: %s, err: %v", apiKey, flagValue, err)
		}
		return boolValue
	}
	return defaultValue
}

// Parses the given runtime-config and formats it into genericapiserver.APIResourceConfigSource
func parseRuntimeConfig(s *options.APIServer) (genericapiserver.APIResourceConfigSource, error) {
	v1GroupVersionString := "api/v1"
	extensionsGroupVersionString := extensionsapiv1beta1.SchemeGroupVersion.String()
	versionToResourceSpecifier := map[unversioned.GroupVersion]string{
		apiv1.SchemeGroupVersion:                v1GroupVersionString,
		extensionsapiv1beta1.SchemeGroupVersion: extensionsGroupVersionString,
		batchapiv1.SchemeGroupVersion:           batchapiv1.SchemeGroupVersion.String(),
		autoscalingapiv1.SchemeGroupVersion:     autoscalingapiv1.SchemeGroupVersion.String(),
		appsapi.SchemeGroupVersion:              appsapi.SchemeGroupVersion.String(),
	}

	resourceConfig := master.DefaultAPIResourceConfigSource()

	// "api/all=false" allows users to selectively enable specific api versions.
	enableAPIByDefault := true
	allAPIFlagValue, ok := s.RuntimeConfig["api/all"]
	if ok && allAPIFlagValue == "false" {
		enableAPIByDefault = false
	}

	// "api/legacy=false" allows users to disable legacy api versions.
	disableLegacyAPIs := false
	legacyAPIFlagValue, ok := s.RuntimeConfig["api/legacy"]
	if ok && legacyAPIFlagValue == "false" {
		disableLegacyAPIs = true
	}
	_ = disableLegacyAPIs // hush the compiler while we don't have legacy APIs to disable.

	// "<resourceSpecifier>={true|false} allows users to enable/disable API.
	// This takes preference over api/all and api/legacy, if specified.
	for version, resourceSpecifier := range versionToResourceSpecifier {
		enableVersion := getRuntimeConfigValue(s, resourceSpecifier, enableAPIByDefault)
		if enableVersion {
			resourceConfig.EnableVersions(version)
		} else {
			resourceConfig.DisableVersions(version)
		}
	}

	for key := range s.RuntimeConfig {
		tokens := strings.Split(key, "/")
		if len(tokens) != 3 {
			continue
		}

		switch {
		case strings.HasPrefix(key, extensionsGroupVersionString+"/"):
			if !resourceConfig.AnyResourcesForVersionEnabled(extensionsapiv1beta1.SchemeGroupVersion) {
				return nil, fmt.Errorf("%v is disabled, you cannot configure its resources individually", extensionsapiv1beta1.SchemeGroupVersion)
			}

			resource := strings.TrimPrefix(key, extensionsGroupVersionString+"/")
			if getRuntimeConfigValue(s, key, false) {
				resourceConfig.EnableResources(extensionsapiv1beta1.SchemeGroupVersion.WithResource(resource))
			} else {
				resourceConfig.DisableResources(extensionsapiv1beta1.SchemeGroupVersion.WithResource(resource))
			}

		default:
			// TODO enable individual resource capability for all GroupVersionResources
			return nil, fmt.Errorf("%v resources cannot be enabled/disabled individually", key)
		}
	}
	return resourceConfig, nil
}
