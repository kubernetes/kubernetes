package v1

import (
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	configv1 "github.com/openshift/api/config/v1"
	osinv1 "github.com/openshift/api/osin/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +openshift:compatibility-gen:level=4
// +openshift:compatibility-gen:internal
type KubeAPIServerConfig struct {
	metav1.TypeMeta `json:",inline"`

	// provides the standard apiserver configuration
	configv1.GenericAPIServerConfig `json:",inline"`

	// authConfig configures authentication options in addition to the standard
	// oauth token and client certificate authenticators
	AuthConfig MasterAuthConfig `json:"authConfig"`

	// aggregatorConfig has options for configuring the aggregator component of the API server.
	AggregatorConfig AggregatorConfig `json:"aggregatorConfig"`

	// kubeletClientInfo contains information about how to connect to kubelets
	KubeletClientInfo KubeletConnectionInfo `json:"kubeletClientInfo"`

	// servicesSubnet is the subnet to use for assigning service IPs
	ServicesSubnet string `json:"servicesSubnet"`
	// servicesNodePortRange is the range to use for assigning service public ports on a host.
	ServicesNodePortRange string `json:"servicesNodePortRange"`

	// DEPRECATED: consolePublicURL has been deprecated and setting it has no effect.
	ConsolePublicURL string `json:"consolePublicURL"`

	// userAgentMatchingConfig controls how API calls from *voluntarily* identifying clients will be handled.  THIS DOES NOT DEFEND AGAINST MALICIOUS CLIENTS!
	// TODO I think we should just drop this feature.
	UserAgentMatchingConfig UserAgentMatchingConfig `json:"userAgentMatchingConfig"`

	// imagePolicyConfig feeds the image policy admission plugin
	// TODO make it an admission plugin config
	ImagePolicyConfig KubeAPIServerImagePolicyConfig `json:"imagePolicyConfig"`

	// projectConfig feeds an admission plugin
	// TODO make it an admission plugin config
	ProjectConfig KubeAPIServerProjectConfig `json:"projectConfig"`

	// serviceAccountPublicKeyFiles is a list of files, each containing a PEM-encoded public RSA key.
	// (If any file contains a private key, the public portion of the key is used)
	// The list of public keys is used to verify presented service account tokens.
	// Each key is tried in order until the list is exhausted or verification succeeds.
	// If no keys are specified, no service account authentication will be available.
	ServiceAccountPublicKeyFiles []string `json:"serviceAccountPublicKeyFiles"`

	// oauthConfig, if present start the /oauth endpoint in this process
	OAuthConfig *osinv1.OAuthConfig `json:"oauthConfig"`

	// TODO this needs to be removed.
	APIServerArguments map[string]Arguments `json:"apiServerArguments"`

	// minimumKubeletVersion is the lowest version of a kubelet that can join the cluster.
	// Specifically, the apiserver will deny most authorization requests of kubelets that are older
	// than the specified version, only allowing the kubelet to get and update its node object, and perform
	// subjectaccessreviews.
	// This means any kubelet that attempts to join the cluster will not be able to run any assigned workloads,
	// and will eventually be marked as not ready.
	// Its max length is 8, so maximum version allowed is either "9.999.99" or "99.99.99".
	// Since the kubelet reports the version of the kubernetes release, not Openshift, this field references
	// the underlying kubernetes version this version of Openshift is based off of.
	// In other words: if an admin wishes to ensure no nodes run an older version than Openshift 4.17, then
	// they should set the minimumKubeletVersion to 1.30.0.
	// When comparing versions, the kubelet's version is stripped of any contents outside of major.minor.patch version.
	// Thus, a kubelet with version "1.0.0-ec.0" will be compatible with minimumKubeletVersion "1.0.0" or earlier.
	// +kubebuilder:validation:XValidation:rule="self == \"\" || self.matches('^[0-9]*.[0-9]*.[0-9]*$')",message="minmumKubeletVersion must be in a semver compatible format of x.y.z, or empty"
	// +kubebuilder:validation:MaxLength:=8
	// +openshift:enable:FeatureGate=MinimumKubeletVersion
	// +optional
	MinimumKubeletVersion string `json:"minimumKubeletVersion"`
}

// Arguments masks the value so protobuf can generate
// +protobuf.nullable=true
// +protobuf.options.(gogoproto.goproto_stringer)=false
type Arguments []string

func (t Arguments) String() string {
	return fmt.Sprintf("%v", []string(t))
}

type KubeAPIServerImagePolicyConfig struct {
	// internalRegistryHostname sets the hostname for the default internal image
	// registry. The value must be in "hostname[:port]" format.
	InternalRegistryHostname string `json:"internalRegistryHostname"`
	// externalRegistryHostnames provides the hostnames for the default external image
	// registry. The external hostname should be set only when the image registry
	// is exposed externally. The first value is used in 'publicDockerImageRepository'
	// field in ImageStreams. The value must be in "hostname[:port]" format.
	ExternalRegistryHostnames []string `json:"externalRegistryHostnames"`
}

type KubeAPIServerProjectConfig struct {
	// defaultNodeSelector holds default project node label selector
	DefaultNodeSelector string `json:"defaultNodeSelector"`
}

// KubeletConnectionInfo holds information necessary for connecting to a kubelet
type KubeletConnectionInfo struct {
	// port is the port to connect to kubelets on
	Port uint32 `json:"port"`
	// ca is the CA for verifying TLS connections to kubelets
	CA string `json:"ca"`
	// CertInfo is the TLS client cert information for securing communication to kubelets
	// this is anonymous so that we can inline it for serialization
	configv1.CertInfo `json:",inline"`
}

// UserAgentMatchingConfig controls how API calls from *voluntarily* identifying clients will be handled.  THIS DOES NOT DEFEND AGAINST MALICIOUS CLIENTS!
type UserAgentMatchingConfig struct {
	// requiredClients if this list is non-empty, then a User-Agent must match one of the UserAgentRegexes to be allowed
	RequiredClients []UserAgentMatchRule `json:"requiredClients"`

	// deniedClients if this list is non-empty, then a User-Agent must not match any of the UserAgentRegexes
	DeniedClients []UserAgentDenyRule `json:"deniedClients"`

	// defaultRejectionMessage is the message shown when rejecting a client.  If it is not a set, a generic message is given.
	DefaultRejectionMessage string `json:"defaultRejectionMessage"`
}

// UserAgentMatchRule describes how to match a given request based on User-Agent and HTTPVerb
type UserAgentMatchRule struct {
	// regex is a regex that is checked against the User-Agent.
	// Known variants of oc clients
	// 1. oc accessing kube resources: oc/v1.2.0 (linux/amd64) kubernetes/bc4550d
	// 2. oc accessing openshift resources: oc/v1.1.3 (linux/amd64) openshift/b348c2f
	// 3. openshift kubectl accessing kube resources:  openshift/v1.2.0 (linux/amd64) kubernetes/bc4550d
	// 4. openshift kubectl accessing openshift resources: openshift/v1.1.3 (linux/amd64) openshift/b348c2f
	// 5. oadm accessing kube resources: oadm/v1.2.0 (linux/amd64) kubernetes/bc4550d
	// 6. oadm accessing openshift resources: oadm/v1.1.3 (linux/amd64) openshift/b348c2f
	// 7. openshift cli accessing kube resources: openshift/v1.2.0 (linux/amd64) kubernetes/bc4550d
	// 8. openshift cli accessing openshift resources: openshift/v1.1.3 (linux/amd64) openshift/b348c2f
	Regex string `json:"regex"`

	// httpVerbs specifies which HTTP verbs should be matched.  An empty list means "match all verbs".
	HTTPVerbs []string `json:"httpVerbs"`
}

// UserAgentDenyRule adds a rejection message that can be used to help a user figure out how to get an approved client
type UserAgentDenyRule struct {
	UserAgentMatchRule `json:",inline"`

	// rejectionMessage is the message shown when rejecting a client.  If it is not a set, the default message is used.
	RejectionMessage string `json:"rejectionMessage"`
}

// MasterAuthConfig configures authentication options in addition to the standard
// oauth token and client certificate authenticators
type MasterAuthConfig struct {
	// requestHeader holds options for setting up a front proxy against the API.  It is optional.
	RequestHeader *RequestHeaderAuthenticationOptions `json:"requestHeader"`
	// webhookTokenAuthenticators, if present configures remote token reviewers
	WebhookTokenAuthenticators []WebhookTokenAuthenticator `json:"webhookTokenAuthenticators"`
	// oauthMetadataFile is a path to a file containing the discovery endpoint for OAuth 2.0 Authorization
	// Server Metadata for an external OAuth server.
	// See IETF Draft: // https://tools.ietf.org/html/draft-ietf-oauth-discovery-04#section-2
	// This option is mutually exclusive with OAuthConfig
	OAuthMetadataFile string `json:"oauthMetadataFile"`
}

// WebhookTokenAuthenticators holds the necessary configuation options for
// external token authenticators
type WebhookTokenAuthenticator struct {
	// configFile is a path to a Kubeconfig file with the webhook configuration
	ConfigFile string `json:"configFile"`
	// cacheTTL indicates how long an authentication result should be cached.
	// It takes a valid time duration string (e.g. "5m").
	// If empty, you get a default timeout of 2 minutes.
	// If zero (e.g. "0m"), caching is disabled
	CacheTTL string `json:"cacheTTL"`
}

// RequestHeaderAuthenticationOptions provides options for setting up a front proxy against the entire
// API instead of against the /oauth endpoint.
type RequestHeaderAuthenticationOptions struct {
	// clientCA is a file with the trusted signer certs.  It is required.
	ClientCA string `json:"clientCA"`
	// clientCommonNames is a required list of common names to require a match from.
	ClientCommonNames []string `json:"clientCommonNames"`

	// usernameHeaders is the list of headers to check for user information.  First hit wins.
	UsernameHeaders []string `json:"usernameHeaders"`
	// groupHeaders is the set of headers to check for group information.  All are unioned.
	GroupHeaders []string `json:"groupHeaders"`
	// extraHeaderPrefixes is the set of request header prefixes to inspect for user extra. X-Remote-Extra- is suggested.
	ExtraHeaderPrefixes []string `json:"extraHeaderPrefixes"`
}

// AggregatorConfig holds information required to make the aggregator function.
type AggregatorConfig struct {
	// proxyClientInfo specifies the client cert/key to use when proxying to aggregated API servers
	ProxyClientInfo configv1.CertInfo `json:"proxyClientInfo"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +openshift:compatibility-gen:level=4
// +openshift:compatibility-gen:internal
type KubeControllerManagerConfig struct {
	metav1.TypeMeta `json:",inline"`

	// serviceServingCert provides support for the old alpha service serving cert signer CA bundle
	ServiceServingCert ServiceServingCert `json:"serviceServingCert"`

	// projectConfig is an optimization for the daemonset controller
	ProjectConfig KubeControllerManagerProjectConfig `json:"projectConfig"`

	// extendedArguments is used to configure the kube-controller-manager
	ExtendedArguments map[string]Arguments `json:"extendedArguments"`
}

type KubeControllerManagerProjectConfig struct {
	// defaultNodeSelector holds default project node label selector
	DefaultNodeSelector string `json:"defaultNodeSelector"`
}

// ServiceServingCert holds configuration for service serving cert signer which creates cert/key pairs for
// pods fulfilling a service to serve with.
type ServiceServingCert struct {
	// certFile is a file containing a PEM-encoded certificate
	CertFile string `json:"certFile"`
}
