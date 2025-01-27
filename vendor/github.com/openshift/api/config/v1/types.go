package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// ConfigMapFileReference references a config map in a specific namespace.
// The namespace must be specified at the point of use.
type ConfigMapFileReference struct {
	Name string `json:"name"`
	// key allows pointing to a specific key/value inside of the configmap.  This is useful for logical file references.
	Key string `json:"key,omitempty"`
}

// ConfigMapNameReference references a config map in a specific namespace.
// The namespace must be specified at the point of use.
type ConfigMapNameReference struct {
	// name is the metadata.name of the referenced config map
	// +required
	Name string `json:"name"`
}

// SecretNameReference references a secret in a specific namespace.
// The namespace must be specified at the point of use.
type SecretNameReference struct {
	// name is the metadata.name of the referenced secret
	// +required
	Name string `json:"name"`
}

// HTTPServingInfo holds configuration for serving HTTP
type HTTPServingInfo struct {
	// ServingInfo is the HTTP serving information
	ServingInfo `json:",inline"`
	// maxRequestsInFlight is the number of concurrent requests allowed to the server. If zero, no limit.
	MaxRequestsInFlight int64 `json:"maxRequestsInFlight"`
	// requestTimeoutSeconds is the number of seconds before requests are timed out. The default is 60 minutes, if
	// -1 there is no limit on requests.
	RequestTimeoutSeconds int64 `json:"requestTimeoutSeconds"`
}

// ServingInfo holds information about serving web pages
type ServingInfo struct {
	// bindAddress is the ip:port to serve on
	BindAddress string `json:"bindAddress"`
	// bindNetwork is the type of network to bind to - defaults to "tcp4", accepts "tcp",
	// "tcp4", and "tcp6"
	BindNetwork string `json:"bindNetwork"`
	// CertInfo is the TLS cert info for serving secure traffic.
	// this is anonymous so that we can inline it for serialization
	CertInfo `json:",inline"`
	// clientCA is the certificate bundle for all the signers that you'll recognize for incoming client certificates
	// +optional
	ClientCA string `json:"clientCA,omitempty"`
	// namedCertificates is a list of certificates to use to secure requests to specific hostnames
	NamedCertificates []NamedCertificate `json:"namedCertificates,omitempty"`
	// minTLSVersion is the minimum TLS version supported.
	// Values must match version names from https://golang.org/pkg/crypto/tls/#pkg-constants
	MinTLSVersion string `json:"minTLSVersion,omitempty"`
	// cipherSuites contains an overridden list of ciphers for the server to support.
	// Values must match cipher suite IDs from https://golang.org/pkg/crypto/tls/#pkg-constants
	CipherSuites []string `json:"cipherSuites,omitempty"`
}

// CertInfo relates a certificate with a private key
type CertInfo struct {
	// certFile is a file containing a PEM-encoded certificate
	CertFile string `json:"certFile"`
	// keyFile is a file containing a PEM-encoded private key for the certificate specified by CertFile
	KeyFile string `json:"keyFile"`
}

// NamedCertificate specifies a certificate/key, and the names it should be served for
type NamedCertificate struct {
	// names is a list of DNS names this certificate should be used to secure
	// A name can be a normal DNS name, or can contain leading wildcard segments.
	Names []string `json:"names,omitempty"`
	// CertInfo is the TLS cert info for serving secure traffic
	CertInfo `json:",inline"`
}

// LeaderElection provides information to elect a leader
type LeaderElection struct {
	// disable allows leader election to be suspended while allowing a fully defaulted "normal" startup case.
	Disable bool `json:"disable,omitempty"`
	// namespace indicates which namespace the resource is in
	Namespace string `json:"namespace,omitempty"`
	// name indicates what name to use for the resource
	Name string `json:"name,omitempty"`

	// leaseDuration is the duration that non-leader candidates will wait
	// after observing a leadership renewal until attempting to acquire
	// leadership of a led but unrenewed leader slot. This is effectively the
	// maximum duration that a leader can be stopped before it is replaced
	// by another candidate. This is only applicable if leader election is
	// enabled.
	// +nullable
	LeaseDuration metav1.Duration `json:"leaseDuration"`
	// renewDeadline is the interval between attempts by the acting master to
	// renew a leadership slot before it stops leading. This must be less
	// than or equal to the lease duration. This is only applicable if leader
	// election is enabled.
	// +nullable
	RenewDeadline metav1.Duration `json:"renewDeadline"`
	// retryPeriod is the duration the clients should wait between attempting
	// acquisition and renewal of a leadership. This is only applicable if
	// leader election is enabled.
	// +nullable
	RetryPeriod metav1.Duration `json:"retryPeriod"`
}

// StringSource allows specifying a string inline, or externally via env var or file.
// When it contains only a string value, it marshals to a simple JSON string.
type StringSource struct {
	// StringSourceSpec specifies the string value, or external location
	StringSourceSpec `json:",inline"`
}

// StringSourceSpec specifies a string value, or external location
type StringSourceSpec struct {
	// value specifies the cleartext value, or an encrypted value if keyFile is specified.
	Value string `json:"value"`

	// env specifies an envvar containing the cleartext value, or an encrypted value if the keyFile is specified.
	Env string `json:"env"`

	// file references a file containing the cleartext value, or an encrypted value if a keyFile is specified.
	File string `json:"file"`

	// keyFile references a file containing the key to use to decrypt the value.
	KeyFile string `json:"keyFile"`
}

// RemoteConnectionInfo holds information necessary for establishing a remote connection
type RemoteConnectionInfo struct {
	// url is the remote URL to connect to
	URL string `json:"url"`
	// ca is the CA for verifying TLS connections
	CA string `json:"ca"`
	// CertInfo is the TLS client cert information to present
	// this is anonymous so that we can inline it for serialization
	CertInfo `json:",inline"`
}

type AdmissionConfig struct {
	PluginConfig map[string]AdmissionPluginConfig `json:"pluginConfig,omitempty"`

	// enabledPlugins is a list of admission plugins that must be on in addition to the default list.
	// Some admission plugins are disabled by default, but certain configurations require them.  This is fairly uncommon
	// and can result in performance penalties and unexpected behavior.
	EnabledAdmissionPlugins []string `json:"enabledPlugins,omitempty"`

	// disabledPlugins is a list of admission plugins that must be off.  Putting something in this list
	// is almost always a mistake and likely to result in cluster instability.
	DisabledAdmissionPlugins []string `json:"disabledPlugins,omitempty"`
}

// AdmissionPluginConfig holds the necessary configuration options for admission plugins
type AdmissionPluginConfig struct {
	// location is the path to a configuration file that contains the plugin's
	// configuration
	Location string `json:"location"`

	// configuration is an embedded configuration object to be used as the plugin's
	// configuration. If present, it will be used instead of the path to the configuration file.
	// +nullable
	// +kubebuilder:pruning:PreserveUnknownFields
	Configuration runtime.RawExtension `json:"configuration"`
}

type LogFormatType string

type WebHookModeType string

const (
	// LogFormatLegacy saves event in 1-line text format.
	LogFormatLegacy LogFormatType = "legacy"
	// LogFormatJson saves event in structured json format.
	LogFormatJson LogFormatType = "json"

	// WebHookModeBatch indicates that the webhook should buffer audit events
	// internally, sending batch updates either once a certain number of
	// events have been received or a certain amount of time has passed.
	WebHookModeBatch WebHookModeType = "batch"
	// WebHookModeBlocking causes the webhook to block on every attempt to process
	// a set of events. This causes requests to the API server to wait for a
	// round trip to the external audit service before sending a response.
	WebHookModeBlocking WebHookModeType = "blocking"
)

// AuditConfig holds configuration for the audit capabilities
type AuditConfig struct {
	// If this flag is set, audit log will be printed in the logs.
	// The logs contains, method, user and a requested URL.
	Enabled bool `json:"enabled"`
	// All requests coming to the apiserver will be logged to this file.
	AuditFilePath string `json:"auditFilePath"`
	// Maximum number of days to retain old log files based on the timestamp encoded in their filename.
	MaximumFileRetentionDays int32 `json:"maximumFileRetentionDays"`
	// Maximum number of old log files to retain.
	MaximumRetainedFiles int32 `json:"maximumRetainedFiles"`
	// Maximum size in megabytes of the log file before it gets rotated. Defaults to 100MB.
	MaximumFileSizeMegabytes int32 `json:"maximumFileSizeMegabytes"`

	// policyFile is a path to the file that defines the audit policy configuration.
	PolicyFile string `json:"policyFile"`
	// policyConfiguration is an embedded policy configuration object to be used
	// as the audit policy configuration. If present, it will be used instead of
	// the path to the policy file.
	// +nullable
	// +kubebuilder:pruning:PreserveUnknownFields
	PolicyConfiguration runtime.RawExtension `json:"policyConfiguration"`

	// Format of saved audits (legacy or json).
	LogFormat LogFormatType `json:"logFormat"`

	// Path to a .kubeconfig formatted file that defines the audit webhook configuration.
	WebHookKubeConfig string `json:"webHookKubeConfig"`
	// Strategy for sending audit events (block or batch).
	WebHookMode WebHookModeType `json:"webHookMode"`
}

// EtcdConnectionInfo holds information necessary for connecting to an etcd server
type EtcdConnectionInfo struct {
	// urls are the URLs for etcd
	URLs []string `json:"urls,omitempty"`
	// ca is a file containing trusted roots for the etcd server certificates
	CA string `json:"ca"`
	// CertInfo is the TLS client cert information for securing communication to etcd
	// this is anonymous so that we can inline it for serialization
	CertInfo `json:",inline"`
}

type EtcdStorageConfig struct {
	EtcdConnectionInfo `json:",inline"`

	// storagePrefix is the path within etcd that the OpenShift resources will
	// be rooted under. This value, if changed, will mean existing objects in etcd will
	// no longer be located.
	StoragePrefix string `json:"storagePrefix"`
}

// GenericAPIServerConfig is an inline-able struct for aggregated apiservers that need to store data in etcd
type GenericAPIServerConfig struct {
	// servingInfo describes how to start serving
	ServingInfo HTTPServingInfo `json:"servingInfo"`

	// corsAllowedOrigins
	CORSAllowedOrigins []string `json:"corsAllowedOrigins"`

	// auditConfig describes how to configure audit information
	AuditConfig AuditConfig `json:"auditConfig"`

	// storageConfig contains information about how to use
	StorageConfig EtcdStorageConfig `json:"storageConfig"`

	// admissionConfig holds information about how to configure admission.
	AdmissionConfig AdmissionConfig `json:"admission"`

	KubeClientConfig KubeClientConfig `json:"kubeClientConfig"`
}

type KubeClientConfig struct {
	// kubeConfig is a .kubeconfig filename for going to the owning kube-apiserver.  Empty uses an in-cluster-config
	KubeConfig string `json:"kubeConfig"`

	// connectionOverrides specifies client overrides for system components to loop back to this master.
	ConnectionOverrides ClientConnectionOverrides `json:"connectionOverrides"`
}

type ClientConnectionOverrides struct {
	// acceptContentTypes defines the Accept header sent by clients when connecting to a server, overriding the
	// default value of 'application/json'. This field will control all connections to the server used by a particular
	// client.
	AcceptContentTypes string `json:"acceptContentTypes"`
	// contentType is the content type used when sending data to the server from this client.
	ContentType string `json:"contentType"`

	// qps controls the number of queries per second allowed for this connection.
	QPS float32 `json:"qps"`
	// burst allows extra queries to accumulate when a client is exceeding its rate.
	Burst int32 `json:"burst"`
}

// GenericControllerConfig provides information to configure a controller
type GenericControllerConfig struct {
	// servingInfo is the HTTP serving information for the controller's endpoints
	ServingInfo HTTPServingInfo `json:"servingInfo"`

	// leaderElection provides information to elect a leader. Only override this if you have a specific need
	LeaderElection LeaderElection `json:"leaderElection"`

	// authentication allows configuration of authentication for the endpoints
	Authentication DelegatedAuthentication `json:"authentication"`
	// authorization allows configuration of authentication for the endpoints
	Authorization DelegatedAuthorization `json:"authorization"`
}

// DelegatedAuthentication allows authentication to be disabled.
type DelegatedAuthentication struct {
	// disabled indicates that authentication should be disabled.  By default it will use delegated authentication.
	Disabled bool `json:"disabled,omitempty"`
}

// DelegatedAuthorization allows authorization to be disabled.
type DelegatedAuthorization struct {
	// disabled indicates that authorization should be disabled.  By default it will use delegated authorization.
	Disabled bool `json:"disabled,omitempty"`
}
type RequiredHSTSPolicy struct {
	// namespaceSelector specifies a label selector such that the policy applies only to those routes that
	// are in namespaces with labels that match the selector, and are in one of the DomainPatterns.
	// Defaults to the empty LabelSelector, which matches everything.
	// +optional
	NamespaceSelector *metav1.LabelSelector `json:"namespaceSelector,omitempty"`

	// domainPatterns is a list of domains for which the desired HSTS annotations are required.
	// If domainPatterns is specified and a route is created with a spec.host matching one of the domains,
	// the route must specify the HSTS Policy components described in the matching RequiredHSTSPolicy.
	//
	// The use of wildcards is allowed like this: *.foo.com matches everything under foo.com.
	// foo.com only matches foo.com, so to cover foo.com and everything under it, you must specify *both*.
	// +kubebuilder:validation:MinItems=1
	// +required
	DomainPatterns []string `json:"domainPatterns"`

	// maxAge is the delta time range in seconds during which hosts are regarded as HSTS hosts.
	// If set to 0, it negates the effect, and hosts are removed as HSTS hosts.
	// If set to 0 and includeSubdomains is specified, all subdomains of the host are also removed as HSTS hosts.
	// maxAge is a time-to-live value, and if this policy is not refreshed on a client, the HSTS
	// policy will eventually expire on that client.
	MaxAge MaxAgePolicy `json:"maxAge"`

	// preloadPolicy directs the client to include hosts in its host preload list so that
	// it never needs to do an initial load to get the HSTS header (note that this is not defined
	// in RFC 6797 and is therefore client implementation-dependent).
	// +optional
	PreloadPolicy PreloadPolicy `json:"preloadPolicy,omitempty"`

	// includeSubDomainsPolicy means the HSTS Policy should apply to any subdomains of the host's
	// domain name.  Thus, for the host bar.foo.com, if includeSubDomainsPolicy was set to RequireIncludeSubDomains:
	// - the host app.bar.foo.com would inherit the HSTS Policy of bar.foo.com
	// - the host bar.foo.com would inherit the HSTS Policy of bar.foo.com
	// - the host foo.com would NOT inherit the HSTS Policy of bar.foo.com
	// - the host def.foo.com would NOT inherit the HSTS Policy of bar.foo.com
	// +optional
	IncludeSubDomainsPolicy IncludeSubDomainsPolicy `json:"includeSubDomainsPolicy,omitempty"`
}

// MaxAgePolicy contains a numeric range for specifying a compliant HSTS max-age for the enclosing RequiredHSTSPolicy
type MaxAgePolicy struct {
	// The largest allowed value (in seconds) of the RequiredHSTSPolicy max-age
	// This value can be left unspecified, in which case no upper limit is enforced.
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=2147483647
	LargestMaxAge *int32 `json:"largestMaxAge,omitempty"`

	// The smallest allowed value (in seconds) of the RequiredHSTSPolicy max-age
	// Setting max-age=0 allows the deletion of an existing HSTS header from a host.  This is a necessary
	// tool for administrators to quickly correct mistakes.
	// This value can be left unspecified, in which case no lower limit is enforced.
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=2147483647
	SmallestMaxAge *int32 `json:"smallestMaxAge,omitempty"`
}

// PreloadPolicy contains a value for specifying a compliant HSTS preload policy for the enclosing RequiredHSTSPolicy
// +kubebuilder:validation:Enum=RequirePreload;RequireNoPreload;NoOpinion
type PreloadPolicy string

const (
	// RequirePreloadPolicy means HSTS "preload" is required by the RequiredHSTSPolicy
	RequirePreloadPolicy PreloadPolicy = "RequirePreload"

	// RequireNoPreloadPolicy means HSTS "preload" is forbidden by the RequiredHSTSPolicy
	RequireNoPreloadPolicy PreloadPolicy = "RequireNoPreload"

	// NoOpinionPreloadPolicy means HSTS "preload" doesn't matter to the RequiredHSTSPolicy
	NoOpinionPreloadPolicy PreloadPolicy = "NoOpinion"
)

// IncludeSubDomainsPolicy contains a value for specifying a compliant HSTS includeSubdomains policy
// for the enclosing RequiredHSTSPolicy
// +kubebuilder:validation:Enum=RequireIncludeSubDomains;RequireNoIncludeSubDomains;NoOpinion
type IncludeSubDomainsPolicy string

const (
	// RequireIncludeSubDomains means HSTS "includeSubDomains" is required by the RequiredHSTSPolicy
	RequireIncludeSubDomains IncludeSubDomainsPolicy = "RequireIncludeSubDomains"

	// RequireNoIncludeSubDomains means HSTS "includeSubDomains" is forbidden by the RequiredHSTSPolicy
	RequireNoIncludeSubDomains IncludeSubDomainsPolicy = "RequireNoIncludeSubDomains"

	// NoOpinionIncludeSubDomains means HSTS "includeSubDomains" doesn't matter to the RequiredHSTSPolicy
	NoOpinionIncludeSubDomains IncludeSubDomainsPolicy = "NoOpinion"
)

// IBMCloudServiceName contains a value specifying the name of an IBM Cloud Service,
// which are used by MAPI, CIRO, CIO, Installer, etc.
// +kubebuilder:validation:Enum=CIS;COS;COSConfig;DNSServices;GlobalCatalog;GlobalSearch;GlobalTagging;HyperProtect;IAM;KeyProtect;ResourceController;ResourceManager;VPC
type IBMCloudServiceName string

const (
	// IBMCloudServiceCIS is the name for IBM Cloud CIS.
	IBMCloudServiceCIS IBMCloudServiceName = "CIS"
	// IBMCloudServiceCOS is the name for IBM Cloud COS.
	IBMCloudServiceCOS IBMCloudServiceName = "COS"
	// IBMCloudServiceCOSConfig is the name for IBM Cloud COS Config service.
	IBMCloudServiceCOSConfig IBMCloudServiceName = "COSConfig"
	// IBMCloudServiceDNSServices is the name for IBM Cloud DNS Services.
	IBMCloudServiceDNSServices IBMCloudServiceName = "DNSServices"
	// IBMCloudServiceGlobalCatalog is the name for IBM Cloud Global Catalog service.
	IBMCloudServiceGlobalCatalog IBMCloudServiceName = "GlobalCatalog"
	// IBMCloudServiceGlobalSearch is the name for IBM Cloud Global Search.
	IBMCloudServiceGlobalSearch IBMCloudServiceName = "GlobalSearch"
	// IBMCloudServiceGlobalTagging is the name for IBM Cloud Global Tagging.
	IBMCloudServiceGlobalTagging IBMCloudServiceName = "GlobalTagging"
	// IBMCloudServiceHyperProtect is the name for IBM Cloud Hyper Protect.
	IBMCloudServiceHyperProtect IBMCloudServiceName = "HyperProtect"
	// IBMCloudServiceIAM is the name for IBM Cloud IAM.
	IBMCloudServiceIAM IBMCloudServiceName = "IAM"
	// IBMCloudServiceKeyProtect is the name for IBM Cloud Key Protect.
	IBMCloudServiceKeyProtect IBMCloudServiceName = "KeyProtect"
	// IBMCloudServiceResourceController is the name for IBM Cloud Resource Controller.
	IBMCloudServiceResourceController IBMCloudServiceName = "ResourceController"
	// IBMCloudServiceResourceManager is the name for IBM Cloud Resource Manager.
	IBMCloudServiceResourceManager IBMCloudServiceName = "ResourceManager"
	// IBMCloudServiceVPC is the name for IBM Cloud VPC.
	IBMCloudServiceVPC IBMCloudServiceName = "VPC"
)
