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
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/validation"
	genericoptions "k8s.io/kubernetes/pkg/genericapiserver/options"
	kubeletclient "k8s.io/kubernetes/pkg/kubelet/client"
	"k8s.io/kubernetes/pkg/master/ports"

	"github.com/spf13/pflag"
)

// ServerRunOptions runs a kubernetes api server.
type ServerRunOptions struct {
	GenericServerRunOptions     *genericoptions.ServerRunOptions
	AllowPrivileged             bool
	EventTTL                    time.Duration
	KubeletConfig               kubeletclient.KubeletClientConfig
	MaxConnectionBytesPerSec    int64
	SSHKeyfile                  string
	SSHUser                     string
	ServiceAccountKeyFiles      []string
	ServiceAccountLookup        bool
	WebhookTokenAuthnConfigFile string
	WebhookTokenAuthnCacheTTL   time.Duration
}

// NewServerRunOptions creates a new ServerRunOptions object with default parameters
func NewServerRunOptions() *ServerRunOptions {
	s := ServerRunOptions{
		GenericServerRunOptions: genericoptions.NewServerRunOptions().WithEtcdOptions(),
		EventTTL:                1 * time.Hour,
		KubeletConfig: kubeletclient.KubeletClientConfig{
			Port: ports.KubeletPort,
			PreferredAddressTypes: []string{
				string(api.NodeHostName),
				string(api.NodeInternalIP),
				string(api.NodeExternalIP),
				string(api.NodeLegacyHostIP),
			},
			EnableHttps: true,
			HTTPTimeout: time.Duration(5) * time.Second,
		},
		WebhookTokenAuthnCacheTTL: 2 * time.Minute,
	}
	return &s
}

// AddFlags adds flags for a specific APIServer to the specified FlagSet
func (s *ServerRunOptions) AddFlags(fs *pflag.FlagSet) {
	// Add the generic flags.
	s.GenericServerRunOptions.AddUniversalFlags(fs)
	//Add etcd specific flags.
	s.GenericServerRunOptions.AddEtcdStorageFlags(fs)
	// Note: the weird ""+ in below lines seems to be the only way to get gofmt to
	// arrange these text blocks sensibly. Grrr.

	fs.DurationVar(&s.EventTTL, "event-ttl", s.EventTTL,
		"Amount of time to retain events. Default is 1h.")

	fs.StringArrayVar(&s.ServiceAccountKeyFiles, "service-account-key-file", s.ServiceAccountKeyFiles, ""+
		"File containing PEM-encoded x509 RSA or ECDSA private or public keys, used to verify "+
		"ServiceAccount tokens. If unspecified, --tls-private-key-file is used. "+
		"The specified file can contain multiple keys, and the flag can be specified multiple times with different files.")

	fs.BoolVar(&s.ServiceAccountLookup, "service-account-lookup", s.ServiceAccountLookup,
		"If true, validate ServiceAccount tokens exist in etcd as part of authentication.")

	fs.StringVar(&s.WebhookTokenAuthnConfigFile, "authentication-token-webhook-config-file", s.WebhookTokenAuthnConfigFile, ""+
		"File with webhook configuration for token authentication in kubeconfig format. "+
		"The API server will query the remote service to determine authentication for bearer tokens.")

	fs.DurationVar(&s.WebhookTokenAuthnCacheTTL, "authentication-token-webhook-cache-ttl", s.WebhookTokenAuthnCacheTTL,
		"The duration to cache responses from the webhook token authenticator. Default is 2m.")

	fs.BoolVar(&s.AllowPrivileged, "allow-privileged", s.AllowPrivileged,
		"If true, allow privileged containers.")

	fs.StringVar(&s.SSHUser, "ssh-user", s.SSHUser,
		"If non-empty, use secure SSH proxy to the nodes, using this user name")

	fs.StringVar(&s.SSHKeyfile, "ssh-keyfile", s.SSHKeyfile,
		"If non-empty, use secure SSH proxy to the nodes, using this user keyfile")

	fs.Int64Var(&s.MaxConnectionBytesPerSec, "max-connection-bytes-per-sec", s.MaxConnectionBytesPerSec, ""+
		"If non-zero, throttle each user connection to this number of bytes/sec. "+
		"Currently only applies to long-running requests.")

	// Kubelet related flags:
	fs.BoolVar(&s.KubeletConfig.EnableHttps, "kubelet-https", s.KubeletConfig.EnableHttps,
		"Use https for kubelet connections.")

	fs.StringSliceVar(&s.KubeletConfig.PreferredAddressTypes, "kubelet-preferred-address-types", s.KubeletConfig.PreferredAddressTypes,
		"List of the preferred NodeAddressTypes to use for kubelet connections.")

	fs.UintVar(&s.KubeletConfig.Port, "kubelet-port", s.KubeletConfig.Port,
		"DEPRECATED: kubelet port.")
	fs.MarkDeprecated("kubelet-port", "kubelet-port is deprecated and will be removed.")

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
}
