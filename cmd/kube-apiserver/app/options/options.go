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

// Package options contains flags and options for initializing an apiserver
package options

import (
	"net"
	"time"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	genericoptions "k8s.io/apiserver/pkg/server/options"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/validation"
	kubeoptions "k8s.io/kubernetes/pkg/kubeapiserver/options"
	kubeletclient "k8s.io/kubernetes/pkg/kubelet/client"
	"k8s.io/kubernetes/pkg/master/ports"

	// add the kubernetes feature gates
	_ "k8s.io/kubernetes/pkg/features"

	"github.com/spf13/pflag"
)

// DefaultServiceNodePortRange is the default port range for NodePort services.
var DefaultServiceNodePortRange = utilnet.PortRange{Base: 30000, Size: 2768}

// ServerRunOptions runs a kubernetes api server.
type ServerRunOptions struct {
	GenericServerRunOptions *genericoptions.ServerRunOptions
	Etcd                    *genericoptions.EtcdOptions
	SecureServing           *genericoptions.SecureServingOptions
	InsecureServing         *kubeoptions.InsecureServingOptions
	Audit                   *genericoptions.AuditLogOptions
	Features                *genericoptions.FeatureOptions
	Authentication          *kubeoptions.BuiltInAuthenticationOptions
	Authorization           *kubeoptions.BuiltInAuthorizationOptions
	CloudProvider           *kubeoptions.CloudProviderOptions
	StorageSerialization    *kubeoptions.StorageSerializationOptions
	APIEnablement           *kubeoptions.APIEnablementOptions

	AllowPrivileged           bool
	EventTTL                  time.Duration
	KubeletConfig             kubeletclient.KubeletClientConfig
	KubernetesServiceNodePort int
	MasterCount               int
	MaxConnectionBytesPerSec  int64
	ServiceClusterIPRange     net.IPNet // TODO: make this a list
	ServiceNodePortRange      utilnet.PortRange
	SSHKeyfile                string
	SSHUser                   string

	ProxyClientCertFile string
	ProxyClientKeyFile  string
}

// NewServerRunOptions creates a new ServerRunOptions object with default parameters
func NewServerRunOptions() *ServerRunOptions {
	s := ServerRunOptions{
		GenericServerRunOptions: genericoptions.NewServerRunOptions(),
		Etcd:                 genericoptions.NewEtcdOptions(storagebackend.NewDefaultConfig(kubeoptions.DefaultEtcdPathPrefix, api.Scheme, nil)),
		SecureServing:        kubeoptions.NewSecureServingOptions(),
		InsecureServing:      kubeoptions.NewInsecureServingOptions(),
		Audit:                genericoptions.NewAuditLogOptions(),
		Features:             genericoptions.NewFeatureOptions(),
		Authentication:       kubeoptions.NewBuiltInAuthenticationOptions().WithAll(),
		Authorization:        kubeoptions.NewBuiltInAuthorizationOptions(),
		CloudProvider:        kubeoptions.NewCloudProviderOptions(),
		StorageSerialization: kubeoptions.NewStorageSerializationOptions(),
		APIEnablement:        kubeoptions.NewAPIEnablementOptions(),

		EventTTL:    1 * time.Hour,
		MasterCount: 1,
		KubeletConfig: kubeletclient.KubeletClientConfig{
			Port:         ports.KubeletPort,
			ReadOnlyPort: ports.KubeletReadOnlyPort,
			PreferredAddressTypes: []string{
				// --override-hostname
				string(api.NodeHostName),

				// internal, preferring DNS if reported
				string(api.NodeInternalDNS),
				string(api.NodeInternalIP),

				// external, preferring DNS if reported
				string(api.NodeExternalDNS),
				string(api.NodeExternalIP),

				string(api.NodeLegacyHostIP),
			},
			EnableHttps: true,
			HTTPTimeout: time.Duration(5) * time.Second,
		},
		ServiceNodePortRange: DefaultServiceNodePortRange,
	}
	// Overwrite the default for storage data format.
	s.Etcd.DefaultStorageMediaType = "application/vnd.kubernetes.protobuf"
	return &s
}

// AddFlags adds flags for a specific APIServer to the specified FlagSet
func (s *ServerRunOptions) AddFlags(fs *pflag.FlagSet) {
	// Add the generic flags.
	s.GenericServerRunOptions.AddUniversalFlags(fs)
	s.Etcd.AddFlags(fs)
	s.SecureServing.AddFlags(fs)
	s.SecureServing.AddDeprecatedFlags(fs)
	s.InsecureServing.AddFlags(fs)
	s.InsecureServing.AddDeprecatedFlags(fs)
	s.Audit.AddFlags(fs)
	s.Features.AddFlags(fs)
	s.Authentication.AddFlags(fs)
	s.Authorization.AddFlags(fs)
	s.CloudProvider.AddFlags(fs)
	s.StorageSerialization.AddFlags(fs)
	s.APIEnablement.AddFlags(fs)

	// Note: the weird ""+ in below lines seems to be the only way to get gofmt to
	// arrange these text blocks sensibly. Grrr.

	fs.DurationVar(&s.EventTTL, "event-ttl", s.EventTTL,
		"Amount of time to retain events. Default is 1h.")

	fs.BoolVar(&s.AllowPrivileged, "allow-privileged", s.AllowPrivileged,
		"If true, allow privileged containers.")

	fs.StringVar(&s.SSHUser, "ssh-user", s.SSHUser,
		"If non-empty, use secure SSH proxy to the nodes, using this user name")

	fs.StringVar(&s.SSHKeyfile, "ssh-keyfile", s.SSHKeyfile,
		"If non-empty, use secure SSH proxy to the nodes, using this user keyfile")

	fs.Int64Var(&s.MaxConnectionBytesPerSec, "max-connection-bytes-per-sec", s.MaxConnectionBytesPerSec, ""+
		"If non-zero, throttle each user connection to this number of bytes/sec. "+
		"Currently only applies to long-running requests.")

	fs.IntVar(&s.MasterCount, "apiserver-count", s.MasterCount,
		"The number of apiservers running in the cluster, must be a positive number.")

	// See #14282 for details on how to test/try this option out.
	// TODO: remove this comment once this option is tested in CI.
	fs.IntVar(&s.KubernetesServiceNodePort, "kubernetes-service-node-port", s.KubernetesServiceNodePort, ""+
		"If non-zero, the Kubernetes master service (which apiserver creates/maintains) will be "+
		"of type NodePort, using this as the value of the port. If zero, the Kubernetes master "+
		"service will be of type ClusterIP.")

	fs.IPNetVar(&s.ServiceClusterIPRange, "service-cluster-ip-range", s.ServiceClusterIPRange, ""+
		"A CIDR notation IP range from which to assign service cluster IPs. This must not "+
		"overlap with any IP ranges assigned to nodes for pods.")

	fs.IPNetVar(&s.ServiceClusterIPRange, "portal-net", s.ServiceClusterIPRange,
		"DEPRECATED: see --service-cluster-ip-range instead.")
	fs.MarkDeprecated("portal-net", "see --service-cluster-ip-range instead")

	fs.Var(&s.ServiceNodePortRange, "service-node-port-range", ""+
		"A port range to reserve for services with NodePort visibility. "+
		"Example: '30000-32767'. Inclusive at both ends of the range.")
	fs.Var(&s.ServiceNodePortRange, "service-node-ports", "DEPRECATED: see --service-node-port-range instead")
	fs.MarkDeprecated("service-node-ports", "see --service-node-port-range instead")

	// Kubelet related flags:
	fs.BoolVar(&s.KubeletConfig.EnableHttps, "kubelet-https", s.KubeletConfig.EnableHttps,
		"Use https for kubelet connections.")

	fs.StringSliceVar(&s.KubeletConfig.PreferredAddressTypes, "kubelet-preferred-address-types", s.KubeletConfig.PreferredAddressTypes,
		"List of the preferred NodeAddressTypes to use for kubelet connections.")

	fs.UintVar(&s.KubeletConfig.Port, "kubelet-port", s.KubeletConfig.Port,
		"DEPRECATED: kubelet port.")
	fs.MarkDeprecated("kubelet-port", "kubelet-port is deprecated and will be removed.")

	fs.UintVar(&s.KubeletConfig.ReadOnlyPort, "kubelet-read-only-port", s.KubeletConfig.ReadOnlyPort,
		"DEPRECATED: kubelet port.")

	fs.DurationVar(&s.KubeletConfig.HTTPTimeout, "kubelet-timeout", s.KubeletConfig.HTTPTimeout,
		"Timeout for kubelet operations.")

	fs.StringVar(&s.KubeletConfig.CertFile, "kubelet-client-certificate", s.KubeletConfig.CertFile,
		"Path to a client cert file for TLS.")

	fs.StringVar(&s.KubeletConfig.KeyFile, "kubelet-client-key", s.KubeletConfig.KeyFile,
		"Path to a client key file for TLS.")

	fs.StringVar(&s.KubeletConfig.CAFile, "kubelet-certificate-authority", s.KubeletConfig.CAFile,
		"Path to a cert file for the certificate authority.")

	// TODO: delete this flag as soon as we identify and fix all clients that send malformed updates, like #14126.
	fs.BoolVar(&validation.RepairMalformedUpdates, "repair-malformed-updates", validation.RepairMalformedUpdates, ""+
		"If true, server will do its best to fix the update request to pass the validation, "+
		"e.g., setting empty UID in update request to its existing value. This flag can be turned off "+
		"after we fix all the clients that send malformed updates.")

	fs.StringVar(&s.ProxyClientCertFile, "proxy-client-cert-file", s.ProxyClientCertFile,
		"client certificate used to prove the identity of the aggragator or kube-apiserver when it proxies requests to a user api-server")
	fs.StringVar(&s.ProxyClientKeyFile, "proxy-client-key-file", s.ProxyClientKeyFile,
		"client certificate key used to prove the identity of the aggragator or kube-apiserver when it proxies requests to a user api-server")

}
