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

// Package options contains flags and options for initializing an apiserver
package options

import (
	"net"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/apiserver"
	"k8s.io/kubernetes/pkg/genericapiserver"
	kubeletclient "k8s.io/kubernetes/pkg/kubelet/client"
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/pkg/util"

	"github.com/spf13/pflag"
)

const (
	// Maximum duration before timing out read/write requests
	// Set to a value larger than the timeouts in each watch server.
	ReadWriteTimeout = time.Minute * 60
	// TODO: This can be tightened up. It still matches objects named watch or proxy.
	defaultLongRunningRequestRE = "(/|^)((watch|proxy)(/|$)|(logs?|portforward|exec|attach)/?$)"
)

// APIServer runs a kubernetes api server.
type APIServer struct {
	InsecureBindAddress        net.IP
	InsecurePort               int
	BindAddress                net.IP
	AdvertiseAddress           net.IP
	SecurePort                 int
	ExternalHost               string
	TLSCertFile                string
	TLSPrivateKeyFile          string
	CertDirectory              string
	APIPrefix                  string
	APIGroupPrefix             string
	DeprecatedStorageVersion   string
	StorageVersions            string
	CloudProvider              string
	CloudConfigFile            string
	EventTTL                   time.Duration
	BasicAuthFile              string
	ClientCAFile               string
	TokenAuthFile              string
	OIDCIssuerURL              string
	OIDCClientID               string
	OIDCCAFile                 string
	OIDCUsernameClaim          string
	ServiceAccountKeyFile      string
	ServiceAccountLookup       bool
	KeystoneURL                string
	AuthorizationMode          string
	AuthorizationPolicyFile    string
	AdmissionControl           string
	AdmissionControlConfigFile string
	EtcdServerList             []string
	EtcdServersOverrides       []string
	EtcdPathPrefix             string
	CorsAllowedOriginList      []string
	AllowPrivileged            bool
	ServiceClusterIPRange      net.IPNet // TODO: make this a list
	ServiceNodePortRange       util.PortRange
	EnableLogsSupport          bool
	MasterServiceNamespace     string
	MasterCount                int
	RuntimeConfig              util.ConfigurationMap
	KubeletConfig              kubeletclient.KubeletClientConfig
	EnableProfiling            bool
	EnableWatchCache           bool
	MaxRequestsInFlight        int
	MinRequestTimeout          int
	LongRunningRequestRE       string
	SSHUser                    string
	SSHKeyfile                 string
	MaxConnectionBytesPerSec   int64
	KubernetesServiceNodePort  int
	PamPassword                bool
}

// NewAPIServer creates a new APIServer object with default parameters
func NewAPIServer() *APIServer {
	s := APIServer{
		InsecurePort:           8080,
		InsecureBindAddress:    net.ParseIP("127.0.0.1"),
		BindAddress:            net.ParseIP("0.0.0.0"),
		SecurePort:             6443,
		APIPrefix:              "/api",
		APIGroupPrefix:         "/apis",
		EventTTL:               1 * time.Hour,
		AuthorizationMode:      "AlwaysAllow",
		AdmissionControl:       "AlwaysAdmit",
		EtcdPathPrefix:         genericapiserver.DefaultEtcdPathPrefix,
		EnableLogsSupport:      true,
		MasterServiceNamespace: api.NamespaceDefault,
		MasterCount:            1,
		CertDirectory:          "/var/run/kubernetes",
		StorageVersions:        latest.AllPreferredGroupVersions(),
		LongRunningRequestRE:   defaultLongRunningRequestRE,

		RuntimeConfig: make(util.ConfigurationMap),
		KubeletConfig: kubeletclient.KubeletClientConfig{
			Port:        ports.KubeletPort,
			EnableHttps: true,
			HTTPTimeout: time.Duration(5) * time.Second,
		},
	}

	return &s
}

// AddFlags adds flags for a specific APIServer to the specified FlagSet
func (s *APIServer) AddFlags(fs *pflag.FlagSet) {
	// Note: the weird ""+ in below lines seems to be the only way to get gofmt to
	// arrange these text blocks sensibly. Grrr.
	fs.IntVar(&s.InsecurePort, "insecure-port", s.InsecurePort, ""+
		"The port on which to serve unsecured, unauthenticated access. Default 8080. It is assumed "+
		"that firewall rules are set up such that this port is not reachable from outside of "+
		"the cluster and that port 443 on the cluster's public address is proxied to this "+
		"port. This is performed by nginx in the default setup.")
	fs.IntVar(&s.InsecurePort, "port", s.InsecurePort, "DEPRECATED: see --insecure-port instead")
	fs.MarkDeprecated("port", "see --insecure-port instead")
	fs.IPVar(&s.InsecureBindAddress, "insecure-bind-address", s.InsecureBindAddress, ""+
		"The IP address on which to serve the --insecure-port (set to 0.0.0.0 for all interfaces). "+
		"Defaults to localhost.")
	fs.IPVar(&s.InsecureBindAddress, "address", s.InsecureBindAddress, "DEPRECATED: see --insecure-bind-address instead")
	fs.MarkDeprecated("address", "see --insecure-bind-address instead")
	fs.IPVar(&s.BindAddress, "bind-address", s.BindAddress, ""+
		"The IP address on which to listen for the --secure-port port. The "+
		"associated interface(s) must be reachable by the rest of the cluster, and by CLI/web "+
		"clients. If blank, all interfaces will be used (0.0.0.0).")
	fs.IPVar(&s.AdvertiseAddress, "advertise-address", s.AdvertiseAddress, ""+
		"The IP address on which to advertise the apiserver to members of the cluster. This "+
		"address must be reachable by the rest of the cluster. If blank, the --bind-address "+
		"will be used. If --bind-address is unspecified, the host's default interface will "+
		"be used.")
	fs.IPVar(&s.BindAddress, "public-address-override", s.BindAddress, "DEPRECATED: see --bind-address instead")
	fs.MarkDeprecated("public-address-override", "see --bind-address instead")
	fs.IntVar(&s.SecurePort, "secure-port", s.SecurePort, ""+
		"The port on which to serve HTTPS with authentication and authorization. If 0, "+
		"don't serve HTTPS at all.")
	fs.StringVar(&s.TLSCertFile, "tls-cert-file", s.TLSCertFile, ""+
		"File containing x509 Certificate for HTTPS.  (CA cert, if any, concatenated after server cert). "+
		"If HTTPS serving is enabled, and --tls-cert-file and --tls-private-key-file are not provided, "+
		"a self-signed certificate and key are generated for the public address and saved to /var/run/kubernetes.")
	fs.StringVar(&s.TLSPrivateKeyFile, "tls-private-key-file", s.TLSPrivateKeyFile, "File containing x509 private key matching --tls-cert-file.")
	fs.StringVar(&s.CertDirectory, "cert-dir", s.CertDirectory, "The directory where the TLS certs are located (by default /var/run/kubernetes). "+
		"If --tls-cert-file and --tls-private-key-file are provided, this flag will be ignored.")
	fs.StringVar(&s.APIPrefix, "api-prefix", s.APIPrefix, "The prefix for API requests on the server. Default '/api'.")
	fs.MarkDeprecated("api-prefix", "--api-prefix is deprecated and will be removed when the v1 API is retired.")
	fs.StringVar(&s.DeprecatedStorageVersion, "storage-version", s.DeprecatedStorageVersion, "The version to store the legacy v1 resources with. Defaults to server preferred")
	fs.MarkDeprecated("storage-version", "--storage-version is deprecated and will be removed when the v1 API is retired. See --storage-versions instead.")
	fs.StringVar(&s.StorageVersions, "storage-versions", s.StorageVersions, "The versions to store resources with. "+
		"Different groups may be stored in different versions. Specified in the format \"group1/version1,group2/version2...\". "+
		"This flag expects a complete list of storage versions of ALL groups registered in the server. "+
		"It defaults to a list of preferred versions of all registered groups, which is derived from the KUBE_API_VERSIONS environment variable.")
	fs.StringVar(&s.CloudProvider, "cloud-provider", s.CloudProvider, "The provider for cloud services.  Empty string for no provider.")
	fs.StringVar(&s.CloudConfigFile, "cloud-config", s.CloudConfigFile, "The path to the cloud provider configuration file.  Empty string for no configuration file.")
	fs.DurationVar(&s.EventTTL, "event-ttl", s.EventTTL, "Amount of time to retain events. Default 1 hour.")
	fs.StringVar(&s.BasicAuthFile, "basic-auth-file", s.BasicAuthFile, "If set, the file that will be used to admit requests to the secure port of the API server via http basic authentication.")
	fs.StringVar(&s.ClientCAFile, "client-ca-file", s.ClientCAFile, "If set, any request presenting a client certificate signed by one of the authorities in the client-ca-file is authenticated with an identity corresponding to the CommonName of the client certificate.")
	fs.StringVar(&s.TokenAuthFile, "token-auth-file", s.TokenAuthFile, "If set, the file that will be used to secure the secure port of the API server via token authentication.")
	fs.StringVar(&s.OIDCIssuerURL, "oidc-issuer-url", s.OIDCIssuerURL, "The URL of the OpenID issuer, only HTTPS scheme will be accepted. If set, it will be used to verify the OIDC JSON Web Token (JWT)")
	fs.StringVar(&s.OIDCClientID, "oidc-client-id", s.OIDCClientID, "The client ID for the OpenID Connect client, must be set if oidc-issuer-url is set")
	fs.StringVar(&s.OIDCCAFile, "oidc-ca-file", s.OIDCCAFile, "If set, the OpenID server's certificate will be verified by one of the authorities in the oidc-ca-file, otherwise the host's root CA set will be used")
	fs.StringVar(&s.OIDCUsernameClaim, "oidc-username-claim", "sub", ""+
		"The OpenID claim to use as the user name. Note that claims other than the default ('sub') is not "+
		"guaranteed to be unique and immutable. This flag is experimental, please see the authentication documentation for further details.")
	fs.StringVar(&s.ServiceAccountKeyFile, "service-account-key-file", s.ServiceAccountKeyFile, "File containing PEM-encoded x509 RSA private or public key, used to verify ServiceAccount tokens. If unspecified, --tls-private-key-file is used.")
	fs.BoolVar(&s.ServiceAccountLookup, "service-account-lookup", s.ServiceAccountLookup, "If true, validate ServiceAccount tokens exist in etcd as part of authentication.")
	fs.StringVar(&s.KeystoneURL, "experimental-keystone-url", s.KeystoneURL, "If passed, activates the keystone authentication plugin")
	fs.StringVar(&s.AuthorizationMode, "authorization-mode", s.AuthorizationMode, "Ordered list of plug-ins to do authorization on secure port. Comma-delimited list of: "+strings.Join(apiserver.AuthorizationModeChoices, ","))
	fs.StringVar(&s.AuthorizationPolicyFile, "authorization-policy-file", s.AuthorizationPolicyFile, "File with authorization policy in csv format, used with --authorization-mode=ABAC, on the secure port.")
	fs.StringVar(&s.AdmissionControl, "admission-control", s.AdmissionControl, "Ordered list of plug-ins to do admission control of resources into cluster. Comma-delimited list of: "+strings.Join(admission.GetPlugins(), ", "))
	fs.StringVar(&s.AdmissionControlConfigFile, "admission-control-config-file", s.AdmissionControlConfigFile, "File with admission control configuration.")
	fs.StringSliceVar(&s.EtcdServerList, "etcd-servers", s.EtcdServerList, "List of etcd servers to watch (http://ip:port), comma separated. Mutually exclusive with -etcd-config")
	fs.StringSliceVar(&s.EtcdServersOverrides, "etcd-servers-overrides", s.EtcdServersOverrides, "Per-resource etcd servers overrides, comma separated. The individual override format: group/resource#servers, where servers are http://ip:port, semicolon separated.")
	fs.StringVar(&s.EtcdPathPrefix, "etcd-prefix", s.EtcdPathPrefix, "The prefix for all resource paths in etcd.")
	fs.StringSliceVar(&s.CorsAllowedOriginList, "cors-allowed-origins", s.CorsAllowedOriginList, "List of allowed origins for CORS, comma separated.  An allowed origin can be a regular expression to support subdomain matching.  If this list is empty CORS will not be enabled.")
	fs.BoolVar(&s.AllowPrivileged, "allow-privileged", s.AllowPrivileged, "If true, allow privileged containers.")
	fs.IPNetVar(&s.ServiceClusterIPRange, "service-cluster-ip-range", s.ServiceClusterIPRange, "A CIDR notation IP range from which to assign service cluster IPs. This must not overlap with any IP ranges assigned to nodes for pods.")
	fs.IPNetVar(&s.ServiceClusterIPRange, "portal-net", s.ServiceClusterIPRange, "Deprecated: see --service-cluster-ip-range instead.")
	fs.MarkDeprecated("portal-net", "see --service-cluster-ip-range instead.")
	fs.Var(&s.ServiceNodePortRange, "service-node-port-range", "A port range to reserve for services with NodePort visibility.  Example: '30000-32767'.  Inclusive at both ends of the range.")
	fs.Var(&s.ServiceNodePortRange, "service-node-ports", "Deprecated: see --service-node-port-range instead.")
	fs.MarkDeprecated("service-node-ports", "see --service-node-port-range instead.")
	fs.StringVar(&s.MasterServiceNamespace, "master-service-namespace", s.MasterServiceNamespace, "The namespace from which the kubernetes master services should be injected into pods")
	fs.IntVar(&s.MasterCount, "apiserver-count", s.MasterCount, "The number of apiservers running in the cluster")
	fs.Var(&s.RuntimeConfig, "runtime-config", "A set of key=value pairs that describe runtime configuration that may be passed to apiserver. apis/<groupVersion> key can be used to turn on/off specific api versions. apis/<groupVersion>/<resource> can be used to turn on/off specific resources. api/all and api/legacy are special keys to control all and legacy api versions respectively.")
	fs.BoolVar(&s.EnableProfiling, "profiling", true, "Enable profiling via web interface host:port/debug/pprof/")
	// TODO: enable cache in integration tests.
	fs.BoolVar(&s.EnableWatchCache, "watch-cache", true, "Enable watch caching in the apiserver")
	fs.StringVar(&s.ExternalHost, "external-hostname", "", "The hostname to use when generating externalized URLs for this master (e.g. Swagger API Docs.)")
	fs.IntVar(&s.MaxRequestsInFlight, "max-requests-inflight", 400, "The maximum number of requests in flight at a given time.  When the server exceeds this, it rejects requests.  Zero for no limit.")
	fs.IntVar(&s.MinRequestTimeout, "min-request-timeout", 1800, "An optional field indicating the minimum number of seconds a handler must keep a request open before timing it out. Currently only honored by the watch request handler, which picks a randomized value above this number as the connection timeout, to spread out load.")
	fs.StringVar(&s.LongRunningRequestRE, "long-running-request-regexp", s.LongRunningRequestRE, "A regular expression matching long running requests which should be excluded from maximum inflight request handling.")
	fs.StringVar(&s.SSHUser, "ssh-user", "", "If non-empty, use secure SSH proxy to the nodes, using this user name")
	fs.StringVar(&s.SSHKeyfile, "ssh-keyfile", "", "If non-empty, use secure SSH proxy to the nodes, using this user keyfile")
	fs.Int64Var(&s.MaxConnectionBytesPerSec, "max-connection-bytes-per-sec", 0, "If non-zero, throttle each user connection to this number of bytes/sec.  Currently only applies to long-running requests")
	// Kubelet related flags:
	fs.BoolVar(&s.KubeletConfig.EnableHttps, "kubelet-https", s.KubeletConfig.EnableHttps, "Use https for kubelet connections")
	fs.UintVar(&s.KubeletConfig.Port, "kubelet-port", s.KubeletConfig.Port, "Kubelet port")
	fs.MarkDeprecated("kubelet-port", "kubelet-port is deprecated and will be removed")
	fs.DurationVar(&s.KubeletConfig.HTTPTimeout, "kubelet-timeout", s.KubeletConfig.HTTPTimeout, "Timeout for kubelet operations")
	fs.StringVar(&s.KubeletConfig.CertFile, "kubelet-client-certificate", s.KubeletConfig.CertFile, "Path to a client cert file for TLS.")
	fs.StringVar(&s.KubeletConfig.KeyFile, "kubelet-client-key", s.KubeletConfig.KeyFile, "Path to a client key file for TLS.")
	fs.StringVar(&s.KubeletConfig.CAFile, "kubelet-certificate-authority", s.KubeletConfig.CAFile, "Path to a cert. file for the certificate authority.")
	// See #14282 for details on how to test/try this option out.  TODO remove this comment once this option is tested in CI.
	fs.IntVar(&s.KubernetesServiceNodePort, "kubernetes-service-node-port", 0, "If non-zero, the Kubernetes master service (which apiserver creates/maintains) will be of type NodePort, using this as the value of the port. If zero, the Kubernetes master service will be of type ClusterIP.")
	// TODO: delete this flag as soon as we identify and fix all clients that send malformed updates, like #14126.
	fs.BoolVar(&validation.RepairMalformedUpdates, "repair-malformed-updates", true, "If true, server will do its best to fix the update request to pass the validation, e.g., setting empty UID in update request to its existing value. This flag can be turned off after we fix all the clients that send malformed updates.")
	fs.BoolVar(&s.PamPassword, "pam-password", false, "Enable use of PAM authentication for validating user credentials.")
}
