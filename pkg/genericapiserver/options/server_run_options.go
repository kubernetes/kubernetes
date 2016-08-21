/*
Copyright 2016 The Kubernetes Authors.

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

package options

import (
	"net"
	"strconv"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	apiutil "k8s.io/kubernetes/pkg/api/util"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/storage/storagebackend"
	"k8s.io/kubernetes/pkg/util/config"
	utilnet "k8s.io/kubernetes/pkg/util/net"

	"github.com/golang/glog"
	"github.com/spf13/pflag"
)

const (
	DefaultDeserializationCacheSize = 50000

	// TODO: This can be tightened up. It still matches objects named watch or proxy.
	defaultLongRunningRequestRE = "(/|^)((watch|proxy)(/|$)|(logs?|portforward|exec|attach)/?$)"
)

var DefaultServiceNodePortRange = utilnet.PortRange{Base: 30000, Size: 2768}

const (
	ModeAlwaysAllow string = "AlwaysAllow"
	ModeAlwaysDeny  string = "AlwaysDeny"
	ModeABAC        string = "ABAC"
	ModeWebhook     string = "Webhook"
	ModeRBAC        string = "RBAC"
)

var AuthorizationModeChoices = []string{ModeAlwaysAllow, ModeAlwaysDeny, ModeABAC, ModeWebhook, ModeRBAC}

// ServerRunOptions contains the options while running a generic api server.
type ServerRunOptions struct {
	APIGroupPrefix             string
	APIPrefix                  string
	AdmissionControl           string
	AdmissionControlConfigFile string
	AdvertiseAddress           net.IP

	// Authorization mode and associated flags.
	AuthorizationMode                        string
	AuthorizationPolicyFile                  string
	AuthorizationWebhookConfigFile           string
	AuthorizationWebhookCacheAuthorizedTTL   time.Duration
	AuthorizationWebhookCacheUnauthorizedTTL time.Duration
	AuthorizationRBACSuperUser               string

	BasicAuthFile           string
	BindAddress             net.IP
	CertDirectory           string
	ClientCAFile            string
	CloudConfigFile         string
	CloudProvider           string
	CorsAllowedOriginList   []string
	DefaultStorageMediaType string
	DeleteCollectionWorkers int
	// Used to specify the storage version that should be used for the legacy v1 api group.
	DeprecatedStorageVersion  string
	AuditLogPath              string
	AuditLogMaxAge            int
	AuditLogMaxBackups        int
	AuditLogMaxSize           int
	EnableLogsSupport         bool
	EnableProfiling           bool
	EnableSwaggerUI           bool
	EnableWatchCache          bool
	EtcdServersOverrides      []string
	StorageConfig             storagebackend.Config
	ExternalHost              string
	InsecureBindAddress       net.IP
	InsecurePort              int
	KeystoneURL               string
	KubernetesServiceNodePort int
	LongRunningRequestRE      string
	MasterCount               int
	MasterServiceNamespace    string
	MaxRequestsInFlight       int
	MinRequestTimeout         int
	OIDCCAFile                string
	OIDCClientID              string
	OIDCIssuerURL             string
	OIDCUsernameClaim         string
	OIDCGroupsClaim           string
	RuntimeConfig             config.ConfigurationMap
	SecurePort                int
	ServiceClusterIPRange     net.IPNet // TODO: make this a list
	ServiceNodePortRange      utilnet.PortRange
	StorageVersions           string
	// The default values for StorageVersions. StorageVersions overrides
	// these; you can change this if you want to change the defaults (e.g.,
	// for testing). This is not actually exposed as a flag.
	DefaultStorageVersions string
	TargetRAMMB            int
	TLSCertFile            string
	TLSPrivateKeyFile      string
	TokenAuthFile          string
	WatchCacheSizes        []string
}

func NewServerRunOptions() *ServerRunOptions {
	return &ServerRunOptions{
		APIGroupPrefix:                           "/apis",
		APIPrefix:                                "/api",
		AdmissionControl:                         "AlwaysAdmit",
		AuthorizationMode:                        "AlwaysAllow",
		AuthorizationWebhookCacheAuthorizedTTL:   5 * time.Minute,
		AuthorizationWebhookCacheUnauthorizedTTL: 30 * time.Second,
		BindAddress:                              net.ParseIP("0.0.0.0"),
		CertDirectory:                            "/var/run/kubernetes",
		DefaultStorageMediaType:                  "application/json",
		DefaultStorageVersions:                   registered.AllPreferredGroupVersions(),
		DeleteCollectionWorkers:                  1,
		EnableLogsSupport:                        true,
		EnableProfiling:                          true,
		EnableWatchCache:                         true,
		InsecureBindAddress:                      net.ParseIP("127.0.0.1"),
		InsecurePort:                             8080,
		LongRunningRequestRE:                     defaultLongRunningRequestRE,
		MasterCount:                              1,
		MasterServiceNamespace:                   api.NamespaceDefault,
		MaxRequestsInFlight:                      400,
		MinRequestTimeout:                        1800,
		RuntimeConfig:                            make(config.ConfigurationMap),
		SecurePort:                               6443,
		ServiceNodePortRange:                     DefaultServiceNodePortRange,
		StorageVersions:                          registered.AllPreferredGroupVersions(),
	}
}

func (o *ServerRunOptions) WithEtcdOptions() *ServerRunOptions {
	o.StorageConfig = storagebackend.Config{
		Prefix: DefaultEtcdPathPrefix,
		DeserializationCacheSize: DefaultDeserializationCacheSize,
	}
	return o
}

// StorageGroupsToEncodingVersion returns a map from group name to group version,
// computed from the s.DeprecatedStorageVersion and s.StorageVersions flags.
func (s *ServerRunOptions) StorageGroupsToEncodingVersion() (map[string]unversioned.GroupVersion, error) {
	storageVersionMap := map[string]unversioned.GroupVersion{}
	if s.DeprecatedStorageVersion != "" {
		storageVersionMap[""] = unversioned.GroupVersion{Group: apiutil.GetGroup(s.DeprecatedStorageVersion), Version: apiutil.GetVersion(s.DeprecatedStorageVersion)}
	}

	// First, get the defaults.
	if err := mergeGroupVersionIntoMap(s.DefaultStorageVersions, storageVersionMap); err != nil {
		return nil, err
	}
	// Override any defaults with the user settings.
	if err := mergeGroupVersionIntoMap(s.StorageVersions, storageVersionMap); err != nil {
		return nil, err
	}

	return storageVersionMap, nil
}

// dest must be a map of group to groupVersion.
func mergeGroupVersionIntoMap(gvList string, dest map[string]unversioned.GroupVersion) error {
	for _, gvString := range strings.Split(gvList, ",") {
		if gvString == "" {
			continue
		}
		// We accept two formats. "group/version" OR
		// "group=group/version". The latter is used when types
		// move between groups.
		if !strings.Contains(gvString, "=") {
			gv, err := unversioned.ParseGroupVersion(gvString)
			if err != nil {
				return err
			}
			dest[gv.Group] = gv

		} else {
			parts := strings.SplitN(gvString, "=", 2)
			gv, err := unversioned.ParseGroupVersion(parts[1])
			if err != nil {
				return err
			}
			dest[parts[0]] = gv
		}
	}

	return nil
}

// Returns a clientset which can be used to talk to this apiserver.
func (s *ServerRunOptions) NewSelfClient() (clientset.Interface, error) {
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

	return clientset.NewForConfig(clientConfig)
}

// AddFlags adds flags for a specific APIServer to the specified FlagSet
func (s *ServerRunOptions) AddUniversalFlags(fs *pflag.FlagSet) {
	// Note: the weird ""+ in below lines seems to be the only way to get gofmt to
	// arrange these text blocks sensibly. Grrr.

	fs.StringVar(&s.AdmissionControl, "admission-control", s.AdmissionControl, ""+
		"Ordered list of plug-ins to do admission control of resources into cluster. "+
		"Comma-delimited list of: "+strings.Join(admission.GetPlugins(), ", ")+".")

	fs.StringVar(&s.AdmissionControlConfigFile, "admission-control-config-file", s.AdmissionControlConfigFile,
		"File with admission control configuration.")

	fs.IPVar(&s.AdvertiseAddress, "advertise-address", s.AdvertiseAddress, ""+
		"The IP address on which to advertise the apiserver to members of the cluster. This "+
		"address must be reachable by the rest of the cluster. If blank, the --bind-address "+
		"will be used. If --bind-address is unspecified, the host's default interface will "+
		"be used.")

	fs.StringVar(&s.AuthorizationMode, "authorization-mode", s.AuthorizationMode, ""+
		"Ordered list of plug-ins to do authorization on secure port. Comma-delimited list of: "+
		strings.Join(AuthorizationModeChoices, ",")+".")

	fs.StringVar(&s.AuthorizationPolicyFile, "authorization-policy-file", s.AuthorizationPolicyFile, ""+
		"File with authorization policy in csv format, used with --authorization-mode=ABAC, on the secure port.")

	fs.StringVar(&s.AuthorizationWebhookConfigFile, "authorization-webhook-config-file", s.AuthorizationWebhookConfigFile, ""+
		"File with webhook configuration in kubeconfig format, used with --authorization-mode=Webhook. "+
		"The API server will query the remote service to determine access on the API server's secure port.")

	fs.DurationVar(&s.AuthorizationWebhookCacheAuthorizedTTL, "authorization-webhook-cache-authorized-ttl",
		s.AuthorizationWebhookCacheAuthorizedTTL,
		"The duration to cache 'authorized' responses from the webhook authorizer. Default is 5m.")

	fs.DurationVar(&s.AuthorizationWebhookCacheUnauthorizedTTL,
		"authorization-webhook-cache-unauthorized-ttl", s.AuthorizationWebhookCacheUnauthorizedTTL,
		"The duration to cache 'unauthorized' responses from the webhook authorizer. Default is 30s.")

	fs.StringVar(&s.AuthorizationRBACSuperUser, "authorization-rbac-super-user", s.AuthorizationRBACSuperUser, ""+
		"If specified, a username which avoids RBAC authorization checks and role binding "+
		"privilege escalation checks, to be used with --authorization-mode=RBAC.")

	fs.StringVar(&s.BasicAuthFile, "basic-auth-file", s.BasicAuthFile, ""+
		"If set, the file that will be used to admit requests to the secure port of the API server "+
		"via http basic authentication.")

	fs.IPVar(&s.BindAddress, "public-address-override", s.BindAddress,
		"DEPRECATED: see --bind-address instead.")
	fs.MarkDeprecated("public-address-override", "see --bind-address instead.")

	fs.IPVar(&s.BindAddress, "bind-address", s.BindAddress, ""+
		"The IP address on which to listen for the --secure-port port. The "+
		"associated interface(s) must be reachable by the rest of the cluster, and by CLI/web "+
		"clients. If blank, all interfaces will be used (0.0.0.0).")

	fs.StringVar(&s.CertDirectory, "cert-dir", s.CertDirectory, ""+
		"The directory where the TLS certs are located (by default /var/run/kubernetes). "+
		"If --tls-cert-file and --tls-private-key-file are provided, this flag will be ignored.")

	fs.StringVar(&s.ClientCAFile, "client-ca-file", s.ClientCAFile, ""+
		"If set, any request presenting a client certificate signed by one of "+
		"the authorities in the client-ca-file is authenticated with an identity "+
		"corresponding to the CommonName of the client certificate.")

	fs.StringVar(&s.CloudProvider, "cloud-provider", s.CloudProvider,
		"The provider for cloud services. Empty string for no provider.")

	fs.StringVar(&s.CloudConfigFile, "cloud-config", s.CloudConfigFile,
		"The path to the cloud provider configuration file. Empty string for no configuration file.")

	fs.StringSliceVar(&s.CorsAllowedOriginList, "cors-allowed-origins", s.CorsAllowedOriginList, ""+
		"List of allowed origins for CORS, comma separated.  An allowed origin can be a regular "+
		"expression to support subdomain matching. If this list is empty CORS will not be enabled.")

	fs.StringVar(&s.DefaultStorageMediaType, "storage-media-type", s.DefaultStorageMediaType, ""+
		"The media type to use to store objects in storage. Defaults to application/json. "+
		"Some resources may only support a specific media type and will ignore this setting.")

	fs.IntVar(&s.DeleteCollectionWorkers, "delete-collection-workers", s.DeleteCollectionWorkers,
		"Number of workers spawned for DeleteCollection call. These are used to speed up namespace cleanup.")

	fs.StringVar(&s.AuditLogPath, "audit-log-path", s.AuditLogPath,
		"If set, all requests coming to the apiserver will be logged to this file.")
	fs.IntVar(&s.AuditLogMaxAge, "audit-log-maxage", s.AuditLogMaxBackups,
		"The maximum number of days to retain old audit log files based on the timestamp encoded in their filename.")
	fs.IntVar(&s.AuditLogMaxBackups, "audit-log-maxbackup", s.AuditLogMaxBackups,
		"The maximum number of old audit log files to retain.")
	fs.IntVar(&s.AuditLogMaxSize, "audit-log-maxsize", s.AuditLogMaxSize,
		"The maximum size in megabytes of the audit log file before it gets rotated. Defaults to 100MB.")

	fs.BoolVar(&s.EnableProfiling, "profiling", s.EnableProfiling,
		"Enable profiling via web interface host:port/debug/pprof/")

	fs.BoolVar(&s.EnableSwaggerUI, "enable-swagger-ui", s.EnableSwaggerUI,
		"Enables swagger ui on the apiserver at /swagger-ui")

	// TODO: enable cache in integration tests.
	fs.BoolVar(&s.EnableWatchCache, "watch-cache", s.EnableWatchCache,
		"Enable watch caching in the apiserver")

	fs.IntVar(&s.TargetRAMMB, "target-ram-mb", s.TargetRAMMB,
		"Memory limit for apiserver in MB (used to configure sizes of caches, etc.)")

	fs.StringVar(&s.ExternalHost, "external-hostname", s.ExternalHost,
		"The hostname to use when generating externalized URLs for this master (e.g. Swagger API Docs).")

	fs.IPVar(&s.InsecureBindAddress, "insecure-bind-address", s.InsecureBindAddress, ""+
		"The IP address on which to serve the --insecure-port (set to 0.0.0.0 for all interfaces). "+
		"Defaults to localhost.")
	fs.IPVar(&s.InsecureBindAddress, "address", s.InsecureBindAddress,
		"DEPRECATED: see --insecure-bind-address instead.")
	fs.MarkDeprecated("address", "see --insecure-bind-address instead.")

	fs.IntVar(&s.InsecurePort, "insecure-port", s.InsecurePort, ""+
		"The port on which to serve unsecured, unauthenticated access. Default 8080. It is assumed "+
		"that firewall rules are set up such that this port is not reachable from outside of "+
		"the cluster and that port 443 on the cluster's public address is proxied to this "+
		"port. This is performed by nginx in the default setup.")

	fs.IntVar(&s.InsecurePort, "port", s.InsecurePort, "DEPRECATED: see --insecure-port instead.")
	fs.MarkDeprecated("port", "see --insecure-port instead.")

	fs.StringVar(&s.KeystoneURL, "experimental-keystone-url", s.KeystoneURL,
		"If passed, activates the keystone authentication plugin.")

	// See #14282 for details on how to test/try this option out.
	// TODO: remove this comment once this option is tested in CI.
	fs.IntVar(&s.KubernetesServiceNodePort, "kubernetes-service-node-port", s.KubernetesServiceNodePort, ""+
		"If non-zero, the Kubernetes master service (which apiserver creates/maintains) will be "+
		"of type NodePort, using this as the value of the port. If zero, the Kubernetes master "+
		"service will be of type ClusterIP.")

	fs.StringVar(&s.LongRunningRequestRE, "long-running-request-regexp", s.LongRunningRequestRE, ""+
		"A regular expression matching long running requests which should "+
		"be excluded from maximum inflight request handling.")

	fs.IntVar(&s.MasterCount, "apiserver-count", s.MasterCount,
		"The number of apiservers running in the cluster.")

	fs.StringVar(&s.MasterServiceNamespace, "master-service-namespace", s.MasterServiceNamespace, ""+
		"The namespace from which the kubernetes master services should be injected into pods.")

	fs.IntVar(&s.MaxRequestsInFlight, "max-requests-inflight", s.MaxRequestsInFlight, ""+
		"The maximum number of requests in flight at a given time. When the server exceeds this, "+
		"it rejects requests. Zero for no limit.")

	fs.IntVar(&s.MinRequestTimeout, "min-request-timeout", s.MinRequestTimeout, ""+
		"An optional field indicating the minimum number of seconds a handler must keep "+
		"a request open before timing it out. Currently only honored by the watch request "+
		"handler, which picks a randomized value above this number as the connection timeout, "+
		"to spread out load.")

	fs.StringVar(&s.OIDCIssuerURL, "oidc-issuer-url", s.OIDCIssuerURL, ""+
		"The URL of the OpenID issuer, only HTTPS scheme will be accepted. "+
		"If set, it will be used to verify the OIDC JSON Web Token (JWT).")

	fs.StringVar(&s.OIDCClientID, "oidc-client-id", s.OIDCClientID,
		"The client ID for the OpenID Connect client, must be set if oidc-issuer-url is set.")

	fs.StringVar(&s.OIDCCAFile, "oidc-ca-file", s.OIDCCAFile, ""+
		"If set, the OpenID server's certificate will be verified by one of the authorities "+
		"in the oidc-ca-file, otherwise the host's root CA set will be used.")

	fs.StringVar(&s.OIDCUsernameClaim, "oidc-username-claim", "sub", ""+
		"The OpenID claim to use as the user name. Note that claims other than the default ('sub') "+
		"is not guaranteed to be unique and immutable. This flag is experimental, please see "+
		"the authentication documentation for further details.")

	fs.StringVar(&s.OIDCGroupsClaim, "oidc-groups-claim", "", ""+
		"If provided, the name of a custom OpenID Connect claim for specifying user groups. "+
		"The claim value is expected to be an array of strings. This flag is experimental, "+
		"please see the authentication documentation for further details.")

	fs.Var(&s.RuntimeConfig, "runtime-config", ""+
		"A set of key=value pairs that describe runtime configuration that may be passed "+
		"to apiserver. apis/<groupVersion> key can be used to turn on/off specific api versions. "+
		"apis/<groupVersion>/<resource> can be used to turn on/off specific resources. api/all and "+
		"api/legacy are special keys to control all and legacy api versions respectively.")

	fs.IntVar(&s.SecurePort, "secure-port", s.SecurePort, ""+
		"The port on which to serve HTTPS with authentication and authorization. If 0, "+
		"don't serve HTTPS at all.")

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

	fs.StringVar(&s.StorageConfig.Type, "storage-backend", s.StorageConfig.Type,
		"The storage backend for persistence. Options: 'etcd2' (default), 'etcd3'.")

	fs.IntVar(&s.StorageConfig.DeserializationCacheSize, "deserialization-cache-size", s.StorageConfig.DeserializationCacheSize,
		"Number of deserialized json objects to cache in memory.")

	fs.StringVar(&s.DeprecatedStorageVersion, "storage-version", s.DeprecatedStorageVersion,
		"DEPRECATED: the version to store the legacy v1 resources with. Defaults to server preferred.")
	fs.MarkDeprecated("storage-version", "--storage-version is deprecated and will be removed when the v1 API "+
		"is retired. See --storage-versions instead.")

	fs.StringVar(&s.StorageVersions, "storage-versions", s.StorageVersions, ""+
		"The per-group version to store resources in. "+
		"Specified in the format \"group1/version1,group2/version2,...\". "+
		"In the case where objects are moved from one group to the other, "+
		"you may specify the format \"group1=group2/v1beta1,group3/v1beta1,...\". "+
		"You only need to pass the groups you wish to change from the defaults. "+
		"It defaults to a list of preferred versions of all registered groups, "+
		"which is derived from the KUBE_API_VERSIONS environment variable.")

	fs.StringVar(&s.TLSCertFile, "tls-cert-file", s.TLSCertFile, ""+
		"File containing x509 Certificate for HTTPS. (CA cert, if any, concatenated "+
		"after server cert). If HTTPS serving is enabled, and --tls-cert-file and "+
		"--tls-private-key-file are not provided, a self-signed certificate and key "+
		"are generated for the public address and saved to /var/run/kubernetes.")

	fs.StringVar(&s.TLSPrivateKeyFile, "tls-private-key-file", s.TLSPrivateKeyFile,
		"File containing x509 private key matching --tls-cert-file.")

	fs.StringVar(&s.TokenAuthFile, "token-auth-file", s.TokenAuthFile, ""+
		"If set, the file that will be used to secure the secure port of the API server "+
		"via token authentication.")

	fs.StringSliceVar(&s.WatchCacheSizes, "watch-cache-sizes", s.WatchCacheSizes, ""+
		"List of watch cache sizes for every resource (pods, nodes, etc.), comma separated. "+
		"The individual override format: resource#size, where size is a number. It takes effect "+
		"when watch-cache is enabled.")

	config.DefaultFeatureGate.AddFlag(fs)
}
