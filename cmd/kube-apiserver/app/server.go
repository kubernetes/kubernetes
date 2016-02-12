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
	"os"
	"strconv"
	"strings"

	"github.com/golang/glog"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/extensions"
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
	"k8s.io/kubernetes/pkg/runtime/serializer/versioning"
	"k8s.io/kubernetes/pkg/serviceaccount"
	"k8s.io/kubernetes/pkg/storage"
	etcdstorage "k8s.io/kubernetes/pkg/storage/etcd"
	utilnet "k8s.io/kubernetes/pkg/util/net"
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

// TODO: Longer term we should read this from some config store, rather than a flag.
func verifyClusterIPFlags(s *options.APIServer) {
	if s.ServiceClusterIPRange.IP == nil {
		glog.Fatal("No --service-cluster-ip-range specified")
	}
	var ones, bits = s.ServiceClusterIPRange.Mask.Size()
	if bits-ones > 20 {
		glog.Fatal("Specified --service-cluster-ip-range is too large")
	}
}

// For testing.
type newEtcdFunc func([]string, runtime.NegotiatedSerializer, string, string, string, bool) (storage.Interface, error)

func newEtcd(etcdServerList []string, ns runtime.NegotiatedSerializer, storageGroupVersionString, memoryGroupVersionString, pathPrefix string, quorum bool) (etcdStorage storage.Interface, err error) {
	if storageGroupVersionString == "" {
		return etcdStorage, fmt.Errorf("storageVersion is required to create a etcd storage")
	}
	storageVersion, err := unversioned.ParseGroupVersion(storageGroupVersionString)
	if err != nil {
		return nil, fmt.Errorf("couldn't understand storage version %v: %v", storageGroupVersionString, err)
	}
	memoryVersion, err := unversioned.ParseGroupVersion(memoryGroupVersionString)
	if err != nil {
		return nil, fmt.Errorf("couldn't understand memory version %v: %v", memoryGroupVersionString, err)
	}

	var storageConfig etcdstorage.EtcdConfig
	storageConfig.ServerList = etcdServerList
	storageConfig.Prefix = pathPrefix
	storageConfig.Quorum = quorum
	s, ok := ns.SerializerForMediaType("application/json", nil)
	if !ok {
		return nil, fmt.Errorf("unable to find serializer for JSON")
	}
	glog.Infof("constructing etcd storage interface.\n  sv: %v\n  mv: %v\n", storageVersion, memoryVersion)
	encoder := ns.EncoderForVersion(s, storageVersion)
	decoder := ns.DecoderToVersion(s, memoryVersion)
	if memoryVersion.Group != storageVersion.Group {
		// Allow this codec to translate between groups.
		if err = versioning.EnableCrossGroupEncoding(encoder, memoryVersion.Group, storageVersion.Group); err != nil {
			return nil, fmt.Errorf("error setting up encoder for %v: %v", storageGroupVersionString, err)
		}
		if err = versioning.EnableCrossGroupDecoding(decoder, storageVersion.Group, memoryVersion.Group); err != nil {
			return nil, fmt.Errorf("error setting up decoder for %v: %v", storageGroupVersionString, err)
		}
	}
	storageConfig.Codec = runtime.NewCodec(encoder, decoder)
	return storageConfig.NewStorage()
}

// parse the value of --etcd-servers-overrides and update given storageDestinations.
func updateEtcdOverrides(overrides []string, storageVersions map[string]string, prefix string, quorum bool, storageDestinations *genericapiserver.StorageDestinations, newEtcdFn newEtcdFunc) {
	if len(overrides) == 0 {
		return
	}
	for _, override := range overrides {
		tokens := strings.Split(override, "#")
		if len(tokens) != 2 {
			glog.Errorf("invalid value of etcd server overrides: %s", override)
			continue
		}

		apiresource := strings.Split(tokens[0], "/")
		if len(apiresource) != 2 {
			glog.Errorf("invalid resource definition: %s", tokens[0])
		}
		group := apiresource[0]
		resource := apiresource[1]

		apigroup, err := registered.Group(group)
		if err != nil {
			glog.Errorf("invalid api group %s: %v", group, err)
			continue
		}
		if _, found := storageVersions[apigroup.GroupVersion.Group]; !found {
			glog.Errorf("Couldn't find the storage version for group %s", apigroup.GroupVersion.Group)
			continue
		}

		servers := strings.Split(tokens[1], ";")
		// Note, internalGV will be wrong for things like batch or
		// autoscalers, but they shouldn't be using the override
		// storage.
		internalGV := apigroup.GroupVersion.Group + "/__internal"
		etcdOverrideStorage, err := newEtcdFn(servers, api.Codecs, storageVersions[apigroup.GroupVersion.Group], internalGV, prefix, quorum)
		if err != nil {
			glog.Fatalf("Invalid storage version or misconfigured etcd for %s: %v", tokens[0], err)
		}

		storageDestinations.AddStorageOverride(group, resource, etcdOverrideStorage)
	}
}

// Run runs the specified APIServer.  This should never exit.
func Run(s *options.APIServer) error {
	verifyClusterIPFlags(s)

	// If advertise-address is not specified, use bind-address. If bind-address
	// is not usable (unset, 0.0.0.0, or loopback), we will use the host's default
	// interface as valid public addr for master (see: util/net#ValidPublicAddrForMaster)
	if s.AdvertiseAddress == nil || s.AdvertiseAddress.IsUnspecified() {
		hostIP, err := utilnet.ChooseBindAddress(s.BindAddress)
		if err != nil {
			glog.Fatalf("Unable to find suitable network address.error='%v' . "+
				"Try to set the AdvertiseAddress directly or provide a valid BindAddress to fix this.", err)
		}
		s.AdvertiseAddress = hostIP
	}
	glog.Infof("Will report %v as public IP address.", s.AdvertiseAddress)

	if len(s.EtcdServerList) == 0 {
		glog.Fatalf("--etcd-servers must be specified")
	}

	if s.KubernetesServiceNodePort > 0 && !s.ServiceNodePortRange.Contains(s.KubernetesServiceNodePort) {
		glog.Fatalf("Kubernetes service port range %v doesn't contain %v", s.ServiceNodePortRange, (s.KubernetesServiceNodePort))
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

	cloud, err := cloudprovider.InitCloudProvider(s.CloudProvider, s.CloudConfigFile)
	if err != nil {
		glog.Fatalf("Cloud provider could not be initialized: %v", err)
	}

	// Setup tunneler if needed
	var tunneler master.Tunneler
	var proxyDialerFn apiserver.ProxyDialerFunc
	if len(s.SSHUser) > 0 {
		// Get ssh key distribution func, if supported
		var installSSH master.InstallSSHKey
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
		tunneler = master.NewSSHTunneler(s.SSHUser, s.SSHKeyfile, healthCheckPath, installSSH)

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

	apiGroupVersionOverrides, err := parseRuntimeConfig(s)
	if err != nil {
		glog.Fatalf("error in parsing runtime-config: %s", err)
	}

	clientConfig := &restclient.Config{
		Host: net.JoinHostPort(s.InsecureBindAddress.String(), strconv.Itoa(s.InsecurePort)),
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

	legacyV1Group, err := registered.Group(api.GroupName)
	if err != nil {
		return err
	}

	storageDestinations := genericapiserver.NewStorageDestinations()

	storageVersions := s.StorageGroupsToGroupVersions()
	if _, found := storageVersions[legacyV1Group.GroupVersion.Group]; !found {
		glog.Fatalf("Couldn't find the storage version for group: %q in storageVersions: %v", legacyV1Group.GroupVersion.Group, storageVersions)
	}
	etcdStorage, err := newEtcd(s.EtcdServerList, api.Codecs, storageVersions[legacyV1Group.GroupVersion.Group], "/__internal", s.EtcdPathPrefix, s.EtcdQuorumRead)
	if err != nil {
		glog.Fatalf("Invalid storage version or misconfigured etcd: %v", err)
	}
	storageDestinations.AddAPIGroup("", etcdStorage)

	if !apiGroupVersionOverrides["extensions/v1beta1"].Disable {
		glog.Infof("Configuring extensions/v1beta1 storage destination")
		expGroup, err := registered.Group(extensions.GroupName)
		if err != nil {
			glog.Fatalf("Extensions API is enabled in runtime config, but not enabled in the environment variable KUBE_API_VERSIONS. Error: %v", err)
		}
		if _, found := storageVersions[expGroup.GroupVersion.Group]; !found {
			glog.Fatalf("Couldn't find the storage version for group: %q in storageVersions: %v", expGroup.GroupVersion.Group, storageVersions)
		}
		expEtcdStorage, err := newEtcd(s.EtcdServerList, api.Codecs, storageVersions[expGroup.GroupVersion.Group], "extensions/__internal", s.EtcdPathPrefix, s.EtcdQuorumRead)
		if err != nil {
			glog.Fatalf("Invalid extensions storage version or misconfigured etcd: %v", err)
		}
		storageDestinations.AddAPIGroup(extensions.GroupName, expEtcdStorage)

		// Since HPA has been moved to the autoscaling group, we need to make
		// sure autoscaling has a storage destination. If the autoscaling group
		// itself is on, it will overwrite this decision below.
		storageDestinations.AddAPIGroup(autoscaling.GroupName, expEtcdStorage)

		// Since Job has been moved to the batch group, we need to make
		// sure batch has a storage destination. If the batch group
		// itself is on, it will overwrite this decision below.
		storageDestinations.AddAPIGroup(batch.GroupName, expEtcdStorage)
	}

	// autoscaling/v1/horizontalpodautoscalers is a move from extensions/v1beta1/horizontalpodautoscalers.
	// The storage version needs to be either extensions/v1beta1 or autoscaling/v1.
	// Users must roll forward while using 1.2, because we will require the latter for 1.3.
	if !apiGroupVersionOverrides["autoscaling/v1"].Disable {
		glog.Infof("Configuring autoscaling/v1 storage destination")
		autoscalingGroup, err := registered.Group(autoscaling.GroupName)
		if err != nil {
			glog.Fatalf("Autoscaling API is enabled in runtime config, but not enabled in the environment variable KUBE_API_VERSIONS. Error: %v", err)
		}
		// Figure out what storage group/version we should use.
		storageGroupVersion, found := storageVersions[autoscalingGroup.GroupVersion.Group]
		if !found {
			glog.Fatalf("Couldn't find the storage version for group: %q in storageVersions: %v", autoscalingGroup.GroupVersion.Group, storageVersions)
		}

		if storageGroupVersion != "autoscaling/v1" && storageGroupVersion != "extensions/v1beta1" {
			glog.Fatalf("The storage version for autoscaling must be either 'autoscaling/v1' or 'extensions/v1beta1'")
		}
		glog.Infof("Using %v for autoscaling group storage version", storageGroupVersion)
		autoscalingEtcdStorage, err := newEtcd(s.EtcdServerList, api.Codecs, storageGroupVersion, "extensions/__internal", s.EtcdPathPrefix, s.EtcdQuorumRead)
		if err != nil {
			glog.Fatalf("Invalid extensions storage version or misconfigured etcd: %v", err)
		}
		storageDestinations.AddAPIGroup(autoscaling.GroupName, autoscalingEtcdStorage)
	}

	// batch/v1/job is a move from extensions/v1beta1/job. The storage
	// version needs to be either extensions/v1beta1 or batch/v1. Users
	// must roll forward while using 1.2, because we will require the
	// latter for 1.3.
	if !apiGroupVersionOverrides["batch/v1"].Disable {
		glog.Infof("Configuring batch/v1 storage destination")
		batchGroup, err := registered.Group(batch.GroupName)
		if err != nil {
			glog.Fatalf("Batch API is enabled in runtime config, but not enabled in the environment variable KUBE_API_VERSIONS. Error: %v", err)
		}
		// Figure out what storage group/version we should use.
		storageGroupVersion, found := storageVersions[batchGroup.GroupVersion.Group]
		if !found {
			glog.Fatalf("Couldn't find the storage version for group: %q in storageVersions: %v", batchGroup.GroupVersion.Group, storageVersions)
		}

		if storageGroupVersion != "batch/v1" && storageGroupVersion != "extensions/v1beta1" {
			glog.Fatalf("The storage version for batch must be either 'batch/v1' or 'extensions/v1beta1'")
		}
		glog.Infof("Using %v for batch group storage version", storageGroupVersion)
		batchEtcdStorage, err := newEtcd(s.EtcdServerList, api.Codecs, storageGroupVersion, "extensions/__internal", s.EtcdPathPrefix, s.EtcdQuorumRead)
		if err != nil {
			glog.Fatalf("Invalid extensions storage version or misconfigured etcd: %v", err)
		}
		storageDestinations.AddAPIGroup(batch.GroupName, batchEtcdStorage)
	}

	updateEtcdOverrides(s.EtcdServersOverrides, storageVersions, s.EtcdPathPrefix, s.EtcdQuorumRead, &storageDestinations, newEtcd)

	n := s.ServiceClusterIPRange

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
		serviceAccountGetter = serviceaccountcontroller.NewGetterFromStorageInterface(etcdStorage)
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

	if len(s.ExternalHost) == 0 {
		// TODO: extend for other providers
		if s.CloudProvider == "gce" {
			instances, supported := cloud.Instances()
			if !supported {
				glog.Fatalf("GCE cloud provider has no instances.  this shouldn't happen. exiting.")
			}
			name, err := os.Hostname()
			if err != nil {
				glog.Fatalf("Failed to get hostname: %v", err)
			}
			addrs, err := instances.NodeAddresses(name)
			if err != nil {
				glog.Warningf("Unable to obtain external host address from cloud provider: %v", err)
			} else {
				for _, addr := range addrs {
					if addr.Type == api.NodeExternalIP {
						s.ExternalHost = addr.Address
					}
				}
			}
		}
	}

	config := &master.Config{
		Config: &genericapiserver.Config{
			StorageDestinations:       storageDestinations,
			StorageVersions:           storageVersions,
			ServiceClusterIPRange:     &n,
			EnableLogsSupport:         s.EnableLogsSupport,
			EnableUISupport:           true,
			EnableSwaggerSupport:      true,
			EnableProfiling:           s.EnableProfiling,
			EnableWatchCache:          s.EnableWatchCache,
			EnableIndex:               true,
			APIPrefix:                 s.APIPrefix,
			APIGroupPrefix:            s.APIGroupPrefix,
			CorsAllowedOriginList:     s.CorsAllowedOriginList,
			ReadWritePort:             s.SecurePort,
			PublicAddress:             s.AdvertiseAddress,
			Authenticator:             authenticator,
			SupportsBasicAuth:         len(s.BasicAuthFile) > 0,
			Authorizer:                authorizer,
			AdmissionControl:          admissionController,
			APIGroupVersionOverrides:  apiGroupVersionOverrides,
			MasterServiceNamespace:    s.MasterServiceNamespace,
			MasterCount:               s.MasterCount,
			ExternalHost:              s.ExternalHost,
			MinRequestTimeout:         s.MinRequestTimeout,
			ProxyDialer:               proxyDialerFn,
			ProxyTLSClientConfig:      proxyTLSClientConfig,
			ServiceNodePortRange:      s.ServiceNodePortRange,
			KubernetesServiceNodePort: s.KubernetesServiceNodePort,
			Serializer:                api.Codecs,
		},
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

// Parses the given runtime-config and formats it into map[string]ApiGroupVersionOverride
func parseRuntimeConfig(s *options.APIServer) (map[string]genericapiserver.APIGroupVersionOverride, error) {
	// "api/all=false" allows users to selectively enable specific api versions.
	disableAllAPIs := false
	allAPIFlagValue, ok := s.RuntimeConfig["api/all"]
	if ok && allAPIFlagValue == "false" {
		disableAllAPIs = true
	}

	// "api/legacy=false" allows users to disable legacy api versions.
	disableLegacyAPIs := false
	legacyAPIFlagValue, ok := s.RuntimeConfig["api/legacy"]
	if ok && legacyAPIFlagValue == "false" {
		disableLegacyAPIs = true
	}
	_ = disableLegacyAPIs // hush the compiler while we don't have legacy APIs to disable.

	// "api/v1={true|false} allows users to enable/disable v1 API.
	// This takes preference over api/all and api/legacy, if specified.
	disableV1 := disableAllAPIs
	v1GroupVersion := "api/v1"
	disableV1 = !getRuntimeConfigValue(s, v1GroupVersion, !disableV1)
	apiGroupVersionOverrides := map[string]genericapiserver.APIGroupVersionOverride{}
	if disableV1 {
		apiGroupVersionOverrides[v1GroupVersion] = genericapiserver.APIGroupVersionOverride{
			Disable: true,
		}
	}

	// "extensions/v1beta1={true|false} allows users to enable/disable the extensions API.
	// This takes preference over api/all, if specified.
	disableExtensions := disableAllAPIs
	extensionsGroupVersion := "extensions/v1beta1"
	// TODO: Make this a loop over all group/versions when there are more of them.
	disableExtensions = !getRuntimeConfigValue(s, extensionsGroupVersion, !disableExtensions)
	if disableExtensions {
		apiGroupVersionOverrides[extensionsGroupVersion] = genericapiserver.APIGroupVersionOverride{
			Disable: true,
		}
	}

	disableAutoscaling := disableAllAPIs
	autoscalingGroupVersion := "autoscaling/v1"
	disableAutoscaling = !getRuntimeConfigValue(s, autoscalingGroupVersion, !disableAutoscaling)
	if disableAutoscaling {
		apiGroupVersionOverrides[autoscalingGroupVersion] = genericapiserver.APIGroupVersionOverride{
			Disable: true,
		}
	}
	disableBatch := disableAllAPIs
	batchGroupVersion := "batch/v1"
	disableBatch = !getRuntimeConfigValue(s, batchGroupVersion, !disableBatch)
	if disableBatch {
		apiGroupVersionOverrides[batchGroupVersion] = genericapiserver.APIGroupVersionOverride{
			Disable: true,
		}
	}

	for key := range s.RuntimeConfig {
		if strings.HasPrefix(key, v1GroupVersion+"/") {
			return nil, fmt.Errorf("api/v1 resources cannot be enabled/disabled individually")
		} else if strings.HasPrefix(key, extensionsGroupVersion+"/") {
			resource := strings.TrimPrefix(key, extensionsGroupVersion+"/")

			apiGroupVersionOverride := apiGroupVersionOverrides[extensionsGroupVersion]
			if apiGroupVersionOverride.ResourceOverrides == nil {
				apiGroupVersionOverride.ResourceOverrides = map[string]bool{}
			}
			apiGroupVersionOverride.ResourceOverrides[resource] = getRuntimeConfigValue(s, key, false)
			apiGroupVersionOverrides[extensionsGroupVersion] = apiGroupVersionOverride
		}
	}
	return apiGroupVersionOverrides, nil
}
