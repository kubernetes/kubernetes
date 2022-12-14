package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"

	corev1 "k8s.io/api/core/v1"

	configv1 "github.com/openshift/api/config/v1"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:subresource:scale:specpath=.spec.replicas,statuspath=.status.availableReplicas,selectorpath=.status.selector

// IngressController describes a managed ingress controller for the cluster. The
// controller can service OpenShift Route and Kubernetes Ingress resources.
//
// When an IngressController is created, a new ingress controller deployment is
// created to allow external traffic to reach the services that expose Ingress
// or Route resources. Updating this resource may lead to disruption for public
// facing network connections as a new ingress controller revision may be rolled
// out.
//
// https://kubernetes.io/docs/concepts/services-networking/ingress-controllers
//
// Whenever possible, sensible defaults for the platform are used. See each
// field for more details.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type IngressController struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec is the specification of the desired behavior of the IngressController.
	Spec IngressControllerSpec `json:"spec,omitempty"`
	// status is the most recently observed status of the IngressController.
	Status IngressControllerStatus `json:"status,omitempty"`
}

// IngressControllerSpec is the specification of the desired behavior of the
// IngressController.
type IngressControllerSpec struct {
	// domain is a DNS name serviced by the ingress controller and is used to
	// configure multiple features:
	//
	// * For the LoadBalancerService endpoint publishing strategy, domain is
	//   used to configure DNS records. See endpointPublishingStrategy.
	//
	// * When using a generated default certificate, the certificate will be valid
	//   for domain and its subdomains. See defaultCertificate.
	//
	// * The value is published to individual Route statuses so that end-users
	//   know where to target external DNS records.
	//
	// domain must be unique among all IngressControllers, and cannot be
	// updated.
	//
	// If empty, defaults to ingress.config.openshift.io/cluster .spec.domain.
	//
	// +optional
	Domain string `json:"domain,omitempty"`

	// httpErrorCodePages specifies a configmap with custom error pages.
	// The administrator must create this configmap in the openshift-config namespace.
	// This configmap should have keys in the format "error-page-<error code>.http",
	// where <error code> is an HTTP error code.
	// For example, "error-page-503.http" defines an error page for HTTP 503 responses.
	// Currently only error pages for 503 and 404 responses can be customized.
	// Each value in the configmap should be the full response, including HTTP headers.
	// Eg- https://raw.githubusercontent.com/openshift/router/fadab45747a9b30cc3f0a4b41ad2871f95827a93/images/router/haproxy/conf/error-page-503.http
	// If this field is empty, the ingress controller uses the default error pages.
	HttpErrorCodePages configv1.ConfigMapNameReference `json:"httpErrorCodePages,omitempty"`

	// replicas is the desired number of ingress controller replicas. If unset,
	// the default depends on the value of the defaultPlacement field in the
	// cluster config.openshift.io/v1/ingresses status.
	//
	// The value of replicas is set based on the value of a chosen field in the
	// Infrastructure CR. If defaultPlacement is set to ControlPlane, the
	// chosen field will be controlPlaneTopology. If it is set to Workers the
	// chosen field will be infrastructureTopology. Replicas will then be set to 1
	// or 2 based whether the chosen field's value is SingleReplica or
	// HighlyAvailable, respectively.
	//
	// These defaults are subject to change.
	//
	// +optional
	Replicas *int32 `json:"replicas,omitempty"`

	// endpointPublishingStrategy is used to publish the ingress controller
	// endpoints to other networks, enable load balancer integrations, etc.
	//
	// If unset, the default is based on
	// infrastructure.config.openshift.io/cluster .status.platform:
	//
	//   AWS:          LoadBalancerService (with External scope)
	//   Azure:        LoadBalancerService (with External scope)
	//   GCP:          LoadBalancerService (with External scope)
	//   IBMCloud:     LoadBalancerService (with External scope)
	//   AlibabaCloud: LoadBalancerService (with External scope)
	//   Libvirt:      HostNetwork
	//
	// Any other platform types (including None) default to HostNetwork.
	//
	// endpointPublishingStrategy cannot be updated.
	//
	// +optional
	EndpointPublishingStrategy *EndpointPublishingStrategy `json:"endpointPublishingStrategy,omitempty"`

	// defaultCertificate is a reference to a secret containing the default
	// certificate served by the ingress controller. When Routes don't specify
	// their own certificate, defaultCertificate is used.
	//
	// The secret must contain the following keys and data:
	//
	//   tls.crt: certificate file contents
	//   tls.key: key file contents
	//
	// If unset, a wildcard certificate is automatically generated and used. The
	// certificate is valid for the ingress controller domain (and subdomains) and
	// the generated certificate's CA will be automatically integrated with the
	// cluster's trust store.
	//
	// If a wildcard certificate is used and shared by multiple
	// HTTP/2 enabled routes (which implies ALPN) then clients
	// (i.e., notably browsers) are at liberty to reuse open
	// connections. This means a client can reuse a connection to
	// another route and that is likely to fail. This behaviour is
	// generally known as connection coalescing.
	//
	// The in-use certificate (whether generated or user-specified) will be
	// automatically integrated with OpenShift's built-in OAuth server.
	//
	// +optional
	DefaultCertificate *corev1.LocalObjectReference `json:"defaultCertificate,omitempty"`

	// namespaceSelector is used to filter the set of namespaces serviced by the
	// ingress controller. This is useful for implementing shards.
	//
	// If unset, the default is no filtering.
	//
	// +optional
	NamespaceSelector *metav1.LabelSelector `json:"namespaceSelector,omitempty"`

	// routeSelector is used to filter the set of Routes serviced by the ingress
	// controller. This is useful for implementing shards.
	//
	// If unset, the default is no filtering.
	//
	// +optional
	RouteSelector *metav1.LabelSelector `json:"routeSelector,omitempty"`

	// nodePlacement enables explicit control over the scheduling of the ingress
	// controller.
	//
	// If unset, defaults are used. See NodePlacement for more details.
	//
	// +optional
	NodePlacement *NodePlacement `json:"nodePlacement,omitempty"`

	// tlsSecurityProfile specifies settings for TLS connections for ingresscontrollers.
	//
	// If unset, the default is based on the apiservers.config.openshift.io/cluster resource.
	//
	// Note that when using the Old, Intermediate, and Modern profile types, the effective
	// profile configuration is subject to change between releases. For example, given
	// a specification to use the Intermediate profile deployed on release X.Y.Z, an upgrade
	// to release X.Y.Z+1 may cause a new profile configuration to be applied to the ingress
	// controller, resulting in a rollout.
	//
	// +optional
	TLSSecurityProfile *configv1.TLSSecurityProfile `json:"tlsSecurityProfile,omitempty"`

	// clientTLS specifies settings for requesting and verifying client
	// certificates, which can be used to enable mutual TLS for
	// edge-terminated and reencrypt routes.
	//
	// +optional
	ClientTLS ClientTLS `json:"clientTLS"`

	// routeAdmission defines a policy for handling new route claims (for example,
	// to allow or deny claims across namespaces).
	//
	// If empty, defaults will be applied. See specific routeAdmission fields
	// for details about their defaults.
	//
	// +optional
	RouteAdmission *RouteAdmissionPolicy `json:"routeAdmission,omitempty"`

	// logging defines parameters for what should be logged where.  If this
	// field is empty, operational logs are enabled but access logs are
	// disabled.
	//
	// +optional
	Logging *IngressControllerLogging `json:"logging,omitempty"`

	// httpHeaders defines policy for HTTP headers.
	//
	// If this field is empty, the default values are used.
	//
	// +optional
	HTTPHeaders *IngressControllerHTTPHeaders `json:"httpHeaders,omitempty"`

	// httpEmptyRequestsPolicy describes how HTTP connections should be
	// handled if the connection times out before a request is received.
	// Allowed values for this field are "Respond" and "Ignore".  If the
	// field is set to "Respond", the ingress controller sends an HTTP 400
	// or 408 response, logs the connection (if access logging is enabled),
	// and counts the connection in the appropriate metrics.  If the field
	// is set to "Ignore", the ingress controller closes the connection
	// without sending a response, logging the connection, or incrementing
	// metrics.  The default value is "Respond".
	//
	// Typically, these connections come from load balancers' health probes
	// or Web browsers' speculative connections ("preconnect") and can be
	// safely ignored.  However, these requests may also be caused by
	// network errors, and so setting this field to "Ignore" may impede
	// detection and diagnosis of problems.  In addition, these requests may
	// be caused by port scans, in which case logging empty requests may aid
	// in detecting intrusion attempts.
	//
	// +optional
	// +kubebuilder:default:="Respond"
	HTTPEmptyRequestsPolicy HTTPEmptyRequestsPolicy `json:"httpEmptyRequestsPolicy,omitempty"`

	// tuningOptions defines parameters for adjusting the performance of
	// ingress controller pods. All fields are optional and will use their
	// respective defaults if not set. See specific tuningOptions fields for
	// more details.
	//
	// Setting fields within tuningOptions is generally not recommended. The
	// default values are suitable for most configurations.
	//
	// +optional
	TuningOptions IngressControllerTuningOptions `json:"tuningOptions,omitempty"`

	// unsupportedConfigOverrides allows specifying unsupported
	// configuration options.  Its use is unsupported.
	//
	// +optional
	// +nullable
	// +kubebuilder:pruning:PreserveUnknownFields
	UnsupportedConfigOverrides runtime.RawExtension `json:"unsupportedConfigOverrides"`

	// httpCompression defines a policy for HTTP traffic compression.
	// By default, there is no HTTP compression.
	//
	// +optional
	HTTPCompression HTTPCompressionPolicy `json:"httpCompression,omitempty"`
}

// httpCompressionPolicy turns on compression for the specified MIME types.
//
// This field is optional, and its absence implies that compression should not be enabled
// globally in HAProxy.
//
// If httpCompressionPolicy exists, compression should be enabled only for the specified
// MIME types.
type HTTPCompressionPolicy struct {
	// mimeTypes is a list of MIME types that should have compression applied.
	// This list can be empty, in which case the ingress controller does not apply compression.
	//
	// Note: Not all MIME types benefit from compression, but HAProxy will still use resources
	// to try to compress if instructed to.  Generally speaking, text (html, css, js, etc.)
	// formats benefit from compression, but formats that are already compressed (image,
	// audio, video, etc.) benefit little in exchange for the time and cpu spent on compressing
	// again. See https://joehonton.medium.com/the-gzip-penalty-d31bd697f1a2
	//
	// +listType=set
	MimeTypes []CompressionMIMEType `json:"mimeTypes,omitempty"`
}

// CompressionMIMEType defines the format of a single MIME type.
// E.g. "text/css; charset=utf-8", "text/html", "text/*", "image/svg+xml",
// "application/octet-stream", "X-custom/customsub", etc.
//
// The format should follow the Content-Type definition in RFC 1341:
// Content-Type := type "/" subtype *[";" parameter]
// - The type in Content-Type can be one of:
//   application, audio, image, message, multipart, text, video, or a custom
//   type preceded by "X-" and followed by a token as defined below.
// - The token is a string of at least one character, and not containing white
//   space, control characters, or any of the characters in the tspecials set.
// - The tspecials set contains the characters ()<>@,;:\"/[]?.=
// - The subtype in Content-Type is also a token.
// - The optional parameter/s following the subtype are defined as:
//   token "=" (token / quoted-string)
// - The quoted-string, as defined in RFC 822, is surrounded by double quotes
//   and can contain white space plus any character EXCEPT \, ", and CR.
//   It can also contain any single ASCII character as long as it is escaped by \.
//
// +kubebuilder:validation:Pattern=`^(?i)(x-[^][ ()\\<>@,;:"/?.=\x00-\x1F\x7F]+|application|audio|image|message|multipart|text|video)/[^][ ()\\<>@,;:"/?.=\x00-\x1F\x7F]+(; *[^][ ()\\<>@,;:"/?.=\x00-\x1F\x7F]+=([^][ ()\\<>@,;:"/?.=\x00-\x1F\x7F]+|"(\\[\x00-\x7F]|[^\x0D"\\])*"))*$`
type CompressionMIMEType string

// NodePlacement describes node scheduling configuration for an ingress
// controller.
type NodePlacement struct {
	// nodeSelector is the node selector applied to ingress controller
	// deployments.
	//
	// If set, the specified selector is used and replaces the default.
	//
	// If unset, the default depends on the value of the defaultPlacement
	// field in the cluster config.openshift.io/v1/ingresses status.
	//
	// When defaultPlacement is Workers, the default is:
	//
	//   kubernetes.io/os: linux
	//   node-role.kubernetes.io/worker: ''
	//
	// When defaultPlacement is ControlPlane, the default is:
	//
	//   kubernetes.io/os: linux
	//   node-role.kubernetes.io/master: ''
	//
	// These defaults are subject to change.
	//
	// Note that using nodeSelector.matchExpressions is not supported.  Only
	// nodeSelector.matchLabels may be used.  This is a limitation of the
	// Kubernetes API: the pod spec does not allow complex expressions for
	// node selectors.
	//
	// +optional
	NodeSelector *metav1.LabelSelector `json:"nodeSelector,omitempty"`

	// tolerations is a list of tolerations applied to ingress controller
	// deployments.
	//
	// The default is an empty list.
	//
	// See https://kubernetes.io/docs/concepts/configuration/taint-and-toleration/
	//
	// +optional
	Tolerations []corev1.Toleration `json:"tolerations,omitempty"`
}

// EndpointPublishingStrategyType is a way to publish ingress controller endpoints.
// +kubebuilder:validation:Enum=LoadBalancerService;HostNetwork;Private;NodePortService
type EndpointPublishingStrategyType string

const (
	// LoadBalancerService publishes the ingress controller using a Kubernetes
	// LoadBalancer Service.
	LoadBalancerServiceStrategyType EndpointPublishingStrategyType = "LoadBalancerService"

	// HostNetwork publishes the ingress controller on node ports where the
	// ingress controller is deployed.
	HostNetworkStrategyType EndpointPublishingStrategyType = "HostNetwork"

	// Private does not publish the ingress controller.
	PrivateStrategyType EndpointPublishingStrategyType = "Private"

	// NodePortService publishes the ingress controller using a Kubernetes NodePort Service.
	NodePortServiceStrategyType EndpointPublishingStrategyType = "NodePortService"
)

// LoadBalancerScope is the scope at which a load balancer is exposed.
// +kubebuilder:validation:Enum=Internal;External
type LoadBalancerScope string

var (
	// InternalLoadBalancer is a load balancer that is exposed only on the
	// cluster's private network.
	InternalLoadBalancer LoadBalancerScope = "Internal"

	// ExternalLoadBalancer is a load balancer that is exposed on the
	// cluster's public network (which is typically on the Internet).
	ExternalLoadBalancer LoadBalancerScope = "External"
)

// CIDR is an IP address range in CIDR notation (for example, "10.0.0.0/8"
// or "fd00::/8").
// +kubebuilder:validation:Pattern=`(^(([0-9]|[0-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[0-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])/([0-9]|[12][0-9]|3[0-2])$)|(^s*((([0-9A-Fa-f]{1,4}:){7}([0-9A-Fa-f]{1,4}|:))|(([0-9A-Fa-f]{1,4}:){6}(:[0-9A-Fa-f]{1,4}|((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3})|:))|(([0-9A-Fa-f]{1,4}:){5}(((:[0-9A-Fa-f]{1,4}){1,2})|:((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3})|:))|(([0-9A-Fa-f]{1,4}:){4}(((:[0-9A-Fa-f]{1,4}){1,3})|((:[0-9A-Fa-f]{1,4})?:((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){3}(((:[0-9A-Fa-f]{1,4}){1,4})|((:[0-9A-Fa-f]{1,4}){0,2}:((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){2}(((:[0-9A-Fa-f]{1,4}){1,5})|((:[0-9A-Fa-f]{1,4}){0,3}:((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){1}(((:[0-9A-Fa-f]{1,4}){1,6})|((:[0-9A-Fa-f]{1,4}){0,4}:((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3}))|:))|(:(((:[0-9A-Fa-f]{1,4}){1,7})|((:[0-9A-Fa-f]{1,4}){0,5}:((25[0-5]|2[0-4]d|1dd|[1-9]?d)(.(25[0-5]|2[0-4]d|1dd|[1-9]?d)){3}))|:)))(%.+)?s*(\/(12[0-8]|1[0-1][0-9]|[1-9][0-9]|[0-9]))$)`
// + ---
// + The regex for the IPv4 CIDR range was taken from other CIDR fields in the OpenShift API
// + and the one for the IPv6 CIDR range was taken from
// + https://blog.markhatton.co.uk/2011/03/15/regular-expressions-for-ip-addresses-cidr-ranges-and-hostnames/
// + The resulting regex is an OR of both regexes.
type CIDR string

// LoadBalancerStrategy holds parameters for a load balancer.
type LoadBalancerStrategy struct {
	// scope indicates the scope at which the load balancer is exposed.
	// Possible values are "External" and "Internal".
	//
	// +kubebuilder:validation:Required
	// +required
	Scope LoadBalancerScope `json:"scope"`

	// allowedSourceRanges specifies an allowlist of IP address ranges to which
	// access to the load balancer should be restricted.  Each range must be
	// specified using CIDR notation (e.g. "10.0.0.0/8" or "fd00::/8"). If no range is
	// specified, "0.0.0.0/0" for IPv4 and "::/0" for IPv6 are used by default,
	// which allows all source addresses.
	//
	// To facilitate migration from earlier versions of OpenShift that did
	// not have the allowedSourceRanges field, you may set the
	// service.beta.kubernetes.io/load-balancer-source-ranges annotation on
	// the "router-<ingresscontroller name>" service in the
	// "openshift-ingress" namespace, and this annotation will take
	// effect if allowedSourceRanges is empty on OpenShift 4.12.
	//
	// +nullable
	// +optional
	AllowedSourceRanges []CIDR `json:"allowedSourceRanges,omitempty"`

	// providerParameters holds desired load balancer information specific to
	// the underlying infrastructure provider.
	//
	// If empty, defaults will be applied. See specific providerParameters
	// fields for details about their defaults.
	//
	// +optional
	ProviderParameters *ProviderLoadBalancerParameters `json:"providerParameters,omitempty"`

	// dnsManagementPolicy indicates if the lifecycle of the wildcard DNS record
	// associated with the load balancer service will be managed by
	// the ingress operator. It defaults to Managed.
	// Valid values are: Managed and Unmanaged.
	//
	// +kubebuilder:default:="Managed"
	// +kubebuilder:validation:Required
	// +default="Managed"
	DNSManagementPolicy LoadBalancerDNSManagementPolicy `json:"dnsManagementPolicy,omitempty"`
}

// LoadBalancerDNSManagementPolicy is a policy for configuring how
// ingresscontrollers manage DNS.
//
// +kubebuilder:validation:Enum=Managed;Unmanaged
type LoadBalancerDNSManagementPolicy string

const (
	// ManagedLoadBalancerDNS specifies that the operator manages
	// a wildcard DNS record for the ingresscontroller.
	ManagedLoadBalancerDNS LoadBalancerDNSManagementPolicy = "Managed"
	// UnmanagedLoadBalancerDNS specifies that the operator does not manage
	// any wildcard DNS record for the ingresscontroller.
	UnmanagedLoadBalancerDNS LoadBalancerDNSManagementPolicy = "Unmanaged"
)

// ProviderLoadBalancerParameters holds desired load balancer information
// specific to the underlying infrastructure provider.
// +union
type ProviderLoadBalancerParameters struct {
	// type is the underlying infrastructure provider for the load balancer.
	// Allowed values are "AWS", "Azure", "BareMetal", "GCP", "Nutanix",
	// "OpenStack", and "VSphere".
	//
	// +unionDiscriminator
	// +kubebuilder:validation:Required
	// +required
	Type LoadBalancerProviderType `json:"type"`

	// aws provides configuration settings that are specific to AWS
	// load balancers.
	//
	// If empty, defaults will be applied. See specific aws fields for
	// details about their defaults.
	//
	// +optional
	AWS *AWSLoadBalancerParameters `json:"aws,omitempty"`

	// gcp provides configuration settings that are specific to GCP
	// load balancers.
	//
	// If empty, defaults will be applied. See specific gcp fields for
	// details about their defaults.
	//
	// +optional
	GCP *GCPLoadBalancerParameters `json:"gcp,omitempty"`
}

// LoadBalancerProviderType is the underlying infrastructure provider for the
// load balancer. Allowed values are "AWS", "Azure", "BareMetal", "GCP", "Nutanix",
// "OpenStack", and "VSphere".
//
// +kubebuilder:validation:Enum=AWS;Azure;BareMetal;GCP;Nutanix;OpenStack;VSphere;IBM
type LoadBalancerProviderType string

const (
	AWSLoadBalancerProvider          LoadBalancerProviderType = "AWS"
	AzureLoadBalancerProvider        LoadBalancerProviderType = "Azure"
	GCPLoadBalancerProvider          LoadBalancerProviderType = "GCP"
	OpenStackLoadBalancerProvider    LoadBalancerProviderType = "OpenStack"
	VSphereLoadBalancerProvider      LoadBalancerProviderType = "VSphere"
	IBMLoadBalancerProvider          LoadBalancerProviderType = "IBM"
	BareMetalLoadBalancerProvider    LoadBalancerProviderType = "BareMetal"
	AlibabaCloudLoadBalancerProvider LoadBalancerProviderType = "AlibabaCloud"
	NutanixLoadBalancerProvider      LoadBalancerProviderType = "Nutanix"
)

// AWSLoadBalancerParameters provides configuration settings that are
// specific to AWS load balancers.
// +union
type AWSLoadBalancerParameters struct {
	// type is the type of AWS load balancer to instantiate for an ingresscontroller.
	//
	// Valid values are:
	//
	// * "Classic": A Classic Load Balancer that makes routing decisions at either
	//   the transport layer (TCP/SSL) or the application layer (HTTP/HTTPS). See
	//   the following for additional details:
	//
	//     https://docs.aws.amazon.com/AmazonECS/latest/developerguide/load-balancer-types.html#clb
	//
	// * "NLB": A Network Load Balancer that makes routing decisions at the
	//   transport layer (TCP/SSL). See the following for additional details:
	//
	//     https://docs.aws.amazon.com/AmazonECS/latest/developerguide/load-balancer-types.html#nlb
	//
	// +unionDiscriminator
	// +kubebuilder:validation:Required
	// +required
	Type AWSLoadBalancerType `json:"type"`

	// classicLoadBalancerParameters holds configuration parameters for an AWS
	// classic load balancer. Present only if type is Classic.
	//
	// +optional
	ClassicLoadBalancerParameters *AWSClassicLoadBalancerParameters `json:"classicLoadBalancer,omitempty"`

	// networkLoadBalancerParameters holds configuration parameters for an AWS
	// network load balancer. Present only if type is NLB.
	//
	// +optional
	NetworkLoadBalancerParameters *AWSNetworkLoadBalancerParameters `json:"networkLoadBalancer,omitempty"`
}

// AWSLoadBalancerType is the type of AWS load balancer to instantiate.
// +kubebuilder:validation:Enum=Classic;NLB
type AWSLoadBalancerType string

const (
	AWSClassicLoadBalancer AWSLoadBalancerType = "Classic"
	AWSNetworkLoadBalancer AWSLoadBalancerType = "NLB"
)

// GCPLoadBalancerParameters provides configuration settings that are
// specific to GCP load balancers.
type GCPLoadBalancerParameters struct {
	// clientAccess describes how client access is restricted for internal
	// load balancers.
	//
	// Valid values are:
	// * "Global": Specifying an internal load balancer with Global client access
	//   allows clients from any region within the VPC to communicate with the load
	//   balancer.
	//
	//     https://cloud.google.com/kubernetes-engine/docs/how-to/internal-load-balancing#global_access
	//
	// * "Local": Specifying an internal load balancer with Local client access
	//   means only clients within the same region (and VPC) as the GCP load balancer
	//   can communicate with the load balancer. Note that this is the default behavior.
	//
	//     https://cloud.google.com/load-balancing/docs/internal#client_access
	//
	// +optional
	ClientAccess GCPClientAccess `json:"clientAccess,omitempty"`
}

// GCPClientAccess describes how client access is restricted for internal
// load balancers.
// +kubebuilder:validation:Enum=Global;Local
type GCPClientAccess string

const (
	GCPGlobalAccess GCPClientAccess = "Global"
	GCPLocalAccess  GCPClientAccess = "Local"
)

// AWSClassicLoadBalancerParameters holds configuration parameters for an
// AWS Classic load balancer.
type AWSClassicLoadBalancerParameters struct {
	// connectionIdleTimeout specifies the maximum time period that a
	// connection may be idle before the load balancer closes the
	// connection.  The value must be parseable as a time duration value;
	// see <https://pkg.go.dev/time#ParseDuration>.  A nil or zero value
	// means no opinion, in which case a default value is used.  The default
	// value for this field is 60s.  This default is subject to change.
	//
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Format=duration
	// +optional
	ConnectionIdleTimeout metav1.Duration `json:"connectionIdleTimeout,omitempty"`
}

// AWSNetworkLoadBalancerParameters holds configuration parameters for an
// AWS Network load balancer.
type AWSNetworkLoadBalancerParameters struct {
}

// HostNetworkStrategy holds parameters for the HostNetwork endpoint publishing
// strategy.
type HostNetworkStrategy struct {
	// protocol specifies whether the IngressController expects incoming
	// connections to use plain TCP or whether the IngressController expects
	// PROXY protocol.
	//
	// PROXY protocol can be used with load balancers that support it to
	// communicate the source addresses of client connections when
	// forwarding those connections to the IngressController.  Using PROXY
	// protocol enables the IngressController to report those source
	// addresses instead of reporting the load balancer's address in HTTP
	// headers and logs.  Note that enabling PROXY protocol on the
	// IngressController will cause connections to fail if you are not using
	// a load balancer that uses PROXY protocol to forward connections to
	// the IngressController.  See
	// http://www.haproxy.org/download/2.2/doc/proxy-protocol.txt for
	// information about PROXY protocol.
	//
	// The following values are valid for this field:
	//
	// * The empty string.
	// * "TCP".
	// * "PROXY".
	//
	// The empty string specifies the default, which is TCP without PROXY
	// protocol.  Note that the default is subject to change.
	//
	// +kubebuilder:validation:Optional
	// +optional
	Protocol IngressControllerProtocol `json:"protocol,omitempty"`

	// httpPort is the port on the host which should be used to listen for
	// HTTP requests. This field should be set when port 80 is already in use.
	// The value should not coincide with the NodePort range of the cluster.
	// When the value is 0 or is not specified it defaults to 80.
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Maximum=65535
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:default=80
	// +optional
	HTTPPort int32 `json:"httpPort,omitempty"`

	// httpsPort is the port on the host which should be used to listen for
	// HTTPS requests. This field should be set when port 443 is already in use.
	// The value should not coincide with the NodePort range of the cluster.
	// When the value is 0 or is not specified it defaults to 443.
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Maximum=65535
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:default=443
	// +optional
	HTTPSPort int32 `json:"httpsPort,omitempty"`

	// statsPort is the port on the host where the stats from the router are
	// published. The value should not coincide with the NodePort range of the
	// cluster. If an external load balancer is configured to forward connections
	// to this IngressController, the load balancer should use this port for
	// health checks. The load balancer can send HTTP probes on this port on a
	// given node, with the path /healthz/ready to determine if the ingress
	// controller is ready to receive traffic on the node. For proper operation
	// the load balancer must not forward traffic to a node until the health
	// check reports ready. The load balancer should also stop forwarding requests
	// within a maximum of 45 seconds after /healthz/ready starts reporting
	// not-ready. Probing every 5 to 10 seconds, with a 5-second timeout and with
	// a threshold of two successful or failed requests to become healthy or
	// unhealthy respectively, are well-tested values. When the value is 0 or
	// is not specified it defaults to 1936.
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Maximum=65535
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:default=1936
	// +optional
	StatsPort int32 `json:"statsPort,omitempty"`
}

// PrivateStrategy holds parameters for the Private endpoint publishing
// strategy.
type PrivateStrategy struct {
	// protocol specifies whether the IngressController expects incoming
	// connections to use plain TCP or whether the IngressController expects
	// PROXY protocol.
	//
	// PROXY protocol can be used with load balancers that support it to
	// communicate the source addresses of client connections when
	// forwarding those connections to the IngressController.  Using PROXY
	// protocol enables the IngressController to report those source
	// addresses instead of reporting the load balancer's address in HTTP
	// headers and logs.  Note that enabling PROXY protocol on the
	// IngressController will cause connections to fail if you are not using
	// a load balancer that uses PROXY protocol to forward connections to
	// the IngressController.  See
	// http://www.haproxy.org/download/2.2/doc/proxy-protocol.txt for
	// information about PROXY protocol.
	//
	// The following values are valid for this field:
	//
	// * The empty string.
	// * "TCP".
	// * "PROXY".
	//
	// The empty string specifies the default, which is TCP without PROXY
	// protocol.  Note that the default is subject to change.
	//
	// +kubebuilder:validation:Optional
	// +optional
	Protocol IngressControllerProtocol `json:"protocol,omitempty"`
}

// NodePortStrategy holds parameters for the NodePortService endpoint publishing strategy.
type NodePortStrategy struct {
	// protocol specifies whether the IngressController expects incoming
	// connections to use plain TCP or whether the IngressController expects
	// PROXY protocol.
	//
	// PROXY protocol can be used with load balancers that support it to
	// communicate the source addresses of client connections when
	// forwarding those connections to the IngressController.  Using PROXY
	// protocol enables the IngressController to report those source
	// addresses instead of reporting the load balancer's address in HTTP
	// headers and logs.  Note that enabling PROXY protocol on the
	// IngressController will cause connections to fail if you are not using
	// a load balancer that uses PROXY protocol to forward connections to
	// the IngressController.  See
	// http://www.haproxy.org/download/2.2/doc/proxy-protocol.txt for
	// information about PROXY protocol.
	//
	// The following values are valid for this field:
	//
	// * The empty string.
	// * "TCP".
	// * "PROXY".
	//
	// The empty string specifies the default, which is TCP without PROXY
	// protocol.  Note that the default is subject to change.
	//
	// +kubebuilder:validation:Optional
	// +optional
	Protocol IngressControllerProtocol `json:"protocol,omitempty"`
}

// IngressControllerProtocol specifies whether PROXY protocol is enabled or not.
// +kubebuilder:validation:Enum="";TCP;PROXY
type IngressControllerProtocol string

const (
	DefaultProtocol IngressControllerProtocol = ""
	TCPProtocol     IngressControllerProtocol = "TCP"
	ProxyProtocol   IngressControllerProtocol = "PROXY"
)

// EndpointPublishingStrategy is a way to publish the endpoints of an
// IngressController, and represents the type and any additional configuration
// for a specific type.
// +union
type EndpointPublishingStrategy struct {
	// type is the publishing strategy to use. Valid values are:
	//
	// * LoadBalancerService
	//
	// Publishes the ingress controller using a Kubernetes LoadBalancer Service.
	//
	// In this configuration, the ingress controller deployment uses container
	// networking. A LoadBalancer Service is created to publish the deployment.
	//
	// See: https://kubernetes.io/docs/concepts/services-networking/service/#loadbalancer
	//
	// If domain is set, a wildcard DNS record will be managed to point at the
	// LoadBalancer Service's external name. DNS records are managed only in DNS
	// zones defined by dns.config.openshift.io/cluster .spec.publicZone and
	// .spec.privateZone.
	//
	// Wildcard DNS management is currently supported only on the AWS, Azure,
	// and GCP platforms.
	//
	// * HostNetwork
	//
	// Publishes the ingress controller on node ports where the ingress controller
	// is deployed.
	//
	// In this configuration, the ingress controller deployment uses host
	// networking, bound to node ports 80 and 443. The user is responsible for
	// configuring an external load balancer to publish the ingress controller via
	// the node ports.
	//
	// * Private
	//
	// Does not publish the ingress controller.
	//
	// In this configuration, the ingress controller deployment uses container
	// networking, and is not explicitly published. The user must manually publish
	// the ingress controller.
	//
	// * NodePortService
	//
	// Publishes the ingress controller using a Kubernetes NodePort Service.
	//
	// In this configuration, the ingress controller deployment uses container
	// networking. A NodePort Service is created to publish the deployment. The
	// specific node ports are dynamically allocated by OpenShift; however, to
	// support static port allocations, user changes to the node port
	// field of the managed NodePort Service will preserved.
	//
	// +unionDiscriminator
	// +kubebuilder:validation:Required
	// +required
	Type EndpointPublishingStrategyType `json:"type"`

	// loadBalancer holds parameters for the load balancer. Present only if
	// type is LoadBalancerService.
	// +optional
	LoadBalancer *LoadBalancerStrategy `json:"loadBalancer,omitempty"`

	// hostNetwork holds parameters for the HostNetwork endpoint publishing
	// strategy. Present only if type is HostNetwork.
	// +optional
	HostNetwork *HostNetworkStrategy `json:"hostNetwork,omitempty"`

	// private holds parameters for the Private endpoint publishing
	// strategy. Present only if type is Private.
	// +optional
	Private *PrivateStrategy `json:"private,omitempty"`

	// nodePort holds parameters for the NodePortService endpoint publishing strategy.
	// Present only if type is NodePortService.
	// +optional
	NodePort *NodePortStrategy `json:"nodePort,omitempty"`
}

// ClientCertificatePolicy describes the policy for client certificates.
// +kubebuilder:validation:Enum="";Required;Optional
type ClientCertificatePolicy string

const (
	// ClientCertificatePolicyRequired indicates that a client certificate
	// should be required.
	ClientCertificatePolicyRequired ClientCertificatePolicy = "Required"

	// ClientCertificatePolicyOptional indicates that a client certificate
	// should be requested but not required.
	ClientCertificatePolicyOptional ClientCertificatePolicy = "Optional"
)

// ClientTLS specifies TLS configuration to enable client-to-server
// authentication, which can be used for mutual TLS.
type ClientTLS struct {
	// clientCertificatePolicy specifies whether the ingress controller
	// requires clients to provide certificates.  This field accepts the
	// values "Required" or "Optional".
	//
	// Note that the ingress controller only checks client certificates for
	// edge-terminated and reencrypt TLS routes; it cannot check
	// certificates for cleartext HTTP or passthrough TLS routes.
	//
	// +kubebuilder:validation:Required
	// +required
	ClientCertificatePolicy ClientCertificatePolicy `json:"clientCertificatePolicy"`

	// clientCA specifies a configmap containing the PEM-encoded CA
	// certificate bundle that should be used to verify a client's
	// certificate.  The administrator must create this configmap in the
	// openshift-config namespace.
	//
	// +kubebuilder:validation:Required
	// +required
	ClientCA configv1.ConfigMapNameReference `json:"clientCA"`

	// allowedSubjectPatterns specifies a list of regular expressions that
	// should be matched against the distinguished name on a valid client
	// certificate to filter requests.  The regular expressions must use
	// PCRE syntax.  If this list is empty, no filtering is performed.  If
	// the list is nonempty, then at least one pattern must match a client
	// certificate's distinguished name or else the ingress controller
	// rejects the certificate and denies the connection.
	//
	// +listType=atomic
	// +optional
	AllowedSubjectPatterns []string `json:"allowedSubjectPatterns,omitempty"`
}

// RouteAdmissionPolicy is an admission policy for allowing new route claims.
type RouteAdmissionPolicy struct {
	// namespaceOwnership describes how host name claims across namespaces should
	// be handled.
	//
	// Value must be one of:
	//
	// - Strict: Do not allow routes in different namespaces to claim the same host.
	//
	// - InterNamespaceAllowed: Allow routes to claim different paths of the same
	//   host name across namespaces.
	//
	// If empty, the default is Strict.
	// +optional
	NamespaceOwnership NamespaceOwnershipCheck `json:"namespaceOwnership,omitempty"`
	// wildcardPolicy describes how routes with wildcard policies should
	// be handled for the ingress controller. WildcardPolicy controls use
	// of routes [1] exposed by the ingress controller based on the route's
	// wildcard policy.
	//
	// [1] https://github.com/openshift/api/blob/master/route/v1/types.go
	//
	// Note: Updating WildcardPolicy from WildcardsAllowed to WildcardsDisallowed
	// will cause admitted routes with a wildcard policy of Subdomain to stop
	// working. These routes must be updated to a wildcard policy of None to be
	// readmitted by the ingress controller.
	//
	// WildcardPolicy supports WildcardsAllowed and WildcardsDisallowed values.
	//
	// If empty, defaults to "WildcardsDisallowed".
	//
	WildcardPolicy WildcardPolicy `json:"wildcardPolicy,omitempty"`
}

// WildcardPolicy is a route admission policy component that describes how
// routes with a wildcard policy should be handled.
// +kubebuilder:validation:Enum=WildcardsAllowed;WildcardsDisallowed
type WildcardPolicy string

const (
	// WildcardPolicyAllowed indicates routes with any wildcard policy are
	// admitted by the ingress controller.
	WildcardPolicyAllowed WildcardPolicy = "WildcardsAllowed"

	// WildcardPolicyDisallowed indicates only routes with a wildcard policy
	// of None are admitted by the ingress controller.
	WildcardPolicyDisallowed WildcardPolicy = "WildcardsDisallowed"
)

// NamespaceOwnershipCheck is a route admission policy component that describes
// how host name claims across namespaces should be handled.
// +kubebuilder:validation:Enum=InterNamespaceAllowed;Strict
type NamespaceOwnershipCheck string

const (
	// InterNamespaceAllowedOwnershipCheck allows routes to claim different paths of the same host name across namespaces.
	InterNamespaceAllowedOwnershipCheck NamespaceOwnershipCheck = "InterNamespaceAllowed"

	// StrictNamespaceOwnershipCheck does not allow routes to claim the same host name across namespaces.
	StrictNamespaceOwnershipCheck NamespaceOwnershipCheck = "Strict"
)

// LoggingDestinationType is a type of destination to which to send log
// messages.
//
// +kubebuilder:validation:Enum=Container;Syslog
type LoggingDestinationType string

const (
	// Container sends log messages to a sidecar container.
	ContainerLoggingDestinationType LoggingDestinationType = "Container"

	// Syslog sends log messages to a syslog endpoint.
	SyslogLoggingDestinationType LoggingDestinationType = "Syslog"

	// ContainerLoggingSidecarContainerName is the name of the container
	// with the log output in an ingress controller pod when container
	// logging is used.
	ContainerLoggingSidecarContainerName = "logs"
)

// SyslogLoggingDestinationParameters describes parameters for the Syslog
// logging destination type.
type SyslogLoggingDestinationParameters struct {
	// address is the IP address of the syslog endpoint that receives log
	// messages.
	//
	// +kubebuilder:validation:Required
	// +required
	Address string `json:"address"`

	// port is the UDP port number of the syslog endpoint that receives log
	// messages.
	//
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=65535
	// +required
	Port uint32 `json:"port"`

	// facility specifies the syslog facility of log messages.
	//
	// If this field is empty, the facility is "local1".
	//
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Enum=kern;user;mail;daemon;auth;syslog;lpr;news;uucp;cron;auth2;ftp;ntp;audit;alert;cron2;local0;local1;local2;local3;local4;local5;local6;local7
	// +optional
	Facility string `json:"facility,omitempty"`

	// maxLength is the maximum length of the syslog message
	//
	// If this field is empty, the maxLength is set to "1024".
	//
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Maximum=4096
	// +kubebuilder:validation:Minimum=480
	// +kubebuilder:default=1024
	// +optional
	MaxLength uint32 `json:"maxLength,omitempty"`
}

// ContainerLoggingDestinationParameters describes parameters for the Container
// logging destination type.
type ContainerLoggingDestinationParameters struct {
}

// LoggingDestination describes a destination for log messages.
// +union
type LoggingDestination struct {
	// type is the type of destination for logs.  It must be one of the
	// following:
	//
	// * Container
	//
	// The ingress operator configures the sidecar container named "logs" on
	// the ingress controller pod and configures the ingress controller to
	// write logs to the sidecar.  The logs are then available as container
	// logs.  The expectation is that the administrator configures a custom
	// logging solution that reads logs from this sidecar.  Note that using
	// container logs means that logs may be dropped if the rate of logs
	// exceeds the container runtime's or the custom logging solution's
	// capacity.
	//
	// * Syslog
	//
	// Logs are sent to a syslog endpoint.  The administrator must specify
	// an endpoint that can receive syslog messages.  The expectation is
	// that the administrator has configured a custom syslog instance.
	//
	// +unionDiscriminator
	// +kubebuilder:validation:Required
	// +required
	Type LoggingDestinationType `json:"type"`

	// syslog holds parameters for a syslog endpoint.  Present only if
	// type is Syslog.
	//
	// +optional
	Syslog *SyslogLoggingDestinationParameters `json:"syslog,omitempty"`

	// container holds parameters for the Container logging destination.
	// Present only if type is Container.
	//
	// +optional
	Container *ContainerLoggingDestinationParameters `json:"container,omitempty"`
}

// IngressControllerCaptureHTTPHeader describes an HTTP header that should be
// captured.
type IngressControllerCaptureHTTPHeader struct {
	// name specifies a header name.  Its value must be a valid HTTP header
	// name as defined in RFC 2616 section 4.2.
	//
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Pattern="^[-!#$%&'*+.0-9A-Z^_`a-z|~]+$"
	// +required
	Name string `json:"name"`

	// maxLength specifies a maximum length for the header value.  If a
	// header value exceeds this length, the value will be truncated in the
	// log message.  Note that the ingress controller may impose a separate
	// bound on the total length of HTTP headers in a request.
	//
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Minimum=1
	// +required
	MaxLength int `json:"maxLength"`
}

// IngressControllerCaptureHTTPHeaders specifies which HTTP headers the
// IngressController captures.
type IngressControllerCaptureHTTPHeaders struct {
	// request specifies which HTTP request headers to capture.
	//
	// If this field is empty, no request headers are captured.
	//
	// +nullable
	// +optional
	Request []IngressControllerCaptureHTTPHeader `json:"request,omitempty"`

	// response specifies which HTTP response headers to capture.
	//
	// If this field is empty, no response headers are captured.
	//
	// +nullable
	// +optional
	Response []IngressControllerCaptureHTTPHeader `json:"response,omitempty"`
}

// CookieMatchType indicates the type of matching used against cookie names to
// select a cookie for capture.
// +kubebuilder:validation:Enum=Exact;Prefix
type CookieMatchType string

const (
	// CookieMatchTypeExact indicates that an exact string match should be
	// performed.
	CookieMatchTypeExact CookieMatchType = "Exact"
	// CookieMatchTypePrefix indicates that a string prefix match should be
	// performed.
	CookieMatchTypePrefix CookieMatchType = "Prefix"
)

// IngressControllerCaptureHTTPCookie describes an HTTP cookie that should be
// captured.
type IngressControllerCaptureHTTPCookie struct {
	IngressControllerCaptureHTTPCookieUnion `json:",inline"`

	// maxLength specifies a maximum length of the string that will be
	// logged, which includes the cookie name, cookie value, and
	// one-character delimiter.  If the log entry exceeds this length, the
	// value will be truncated in the log message.  Note that the ingress
	// controller may impose a separate bound on the total length of HTTP
	// headers in a request.
	//
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=1024
	// +required
	MaxLength int `json:"maxLength"`
}

// IngressControllerCaptureHTTPCookieUnion describes optional fields of an HTTP cookie that should be captured.
// +union
type IngressControllerCaptureHTTPCookieUnion struct {
	// matchType specifies the type of match to be performed on the cookie
	// name.  Allowed values are "Exact" for an exact string match and
	// "Prefix" for a string prefix match.  If "Exact" is specified, a name
	// must be specified in the name field.  If "Prefix" is provided, a
	// prefix must be specified in the namePrefix field.  For example,
	// specifying matchType "Prefix" and namePrefix "foo" will capture a
	// cookie named "foo" or "foobar" but not one named "bar".  The first
	// matching cookie is captured.
	//
	// +unionDiscriminator
	// +kubebuilder:validation:Required
	// +required
	MatchType CookieMatchType `json:"matchType,omitempty"`

	// name specifies a cookie name.  Its value must be a valid HTTP cookie
	// name as defined in RFC 6265 section 4.1.
	//
	// +kubebuilder:validation:Pattern="^[-!#$%&'*+.0-9A-Z^_`a-z|~]*$"
	// +kubebuilder:validation:MinLength=0
	// +kubebuilder:validation:MaxLength=1024
	// +optional
	Name string `json:"name"`

	// namePrefix specifies a cookie name prefix.  Its value must be a valid
	// HTTP cookie name as defined in RFC 6265 section 4.1.
	//
	// +kubebuilder:validation:Pattern="^[-!#$%&'*+.0-9A-Z^_`a-z|~]*$"
	// +kubebuilder:validation:MinLength=0
	// +kubebuilder:validation:MaxLength=1024
	// +optional
	NamePrefix string `json:"namePrefix"`
}

// LoggingPolicy indicates how an event should be logged.
// +kubebuilder:validation:Enum=Log;Ignore
type LoggingPolicy string

const (
	// LoggingPolicyLog indicates that an event should be logged.
	LoggingPolicyLog LoggingPolicy = "Log"
	// LoggingPolicyIgnore indicates that an event should not be logged.
	LoggingPolicyIgnore LoggingPolicy = "Ignore"
)

// AccessLogging describes how client requests should be logged.
type AccessLogging struct {
	// destination is where access logs go.
	//
	// +kubebuilder:validation:Required
	// +required
	Destination LoggingDestination `json:"destination"`

	// httpLogFormat specifies the format of the log message for an HTTP
	// request.
	//
	// If this field is empty, log messages use the implementation's default
	// HTTP log format.  For HAProxy's default HTTP log format, see the
	// HAProxy documentation:
	// http://cbonte.github.io/haproxy-dconv/2.0/configuration.html#8.2.3
	//
	// Note that this format only applies to cleartext HTTP connections
	// and to secure HTTP connections for which the ingress controller
	// terminates encryption (that is, edge-terminated or reencrypt
	// connections).  It does not affect the log format for TLS passthrough
	// connections.
	//
	// +optional
	HttpLogFormat string `json:"httpLogFormat,omitempty"`

	// httpCaptureHeaders defines HTTP headers that should be captured in
	// access logs.  If this field is empty, no headers are captured.
	//
	// Note that this option only applies to cleartext HTTP connections
	// and to secure HTTP connections for which the ingress controller
	// terminates encryption (that is, edge-terminated or reencrypt
	// connections).  Headers cannot be captured for TLS passthrough
	// connections.
	//
	// +optional
	HTTPCaptureHeaders IngressControllerCaptureHTTPHeaders `json:"httpCaptureHeaders,omitempty"`

	// httpCaptureCookies specifies HTTP cookies that should be captured in
	// access logs.  If this field is empty, no cookies are captured.
	//
	// +nullable
	// +optional
	// +kubebuilder:validation:MaxItems=1
	HTTPCaptureCookies []IngressControllerCaptureHTTPCookie `json:"httpCaptureCookies,omitempty"`

	// logEmptyRequests specifies how connections on which no request is
	// received should be logged.  Typically, these empty requests come from
	// load balancers' health probes or Web browsers' speculative
	// connections ("preconnect"), in which case logging these requests may
	// be undesirable.  However, these requests may also be caused by
	// network errors, in which case logging empty requests may be useful
	// for diagnosing the errors.  In addition, these requests may be caused
	// by port scans, in which case logging empty requests may aid in
	// detecting intrusion attempts.  Allowed values for this field are
	// "Log" and "Ignore".  The default value is "Log".
	//
	// +optional
	// +kubebuilder:default:="Log"
	LogEmptyRequests LoggingPolicy `json:"logEmptyRequests,omitempty"`
}

// IngressControllerLogging describes what should be logged where.
type IngressControllerLogging struct {
	// access describes how the client requests should be logged.
	//
	// If this field is empty, access logging is disabled.
	//
	// +optional
	Access *AccessLogging `json:"access,omitempty"`
}

// IngressControllerHTTPHeaderPolicy is a policy for setting HTTP headers.
//
// +kubebuilder:validation:Enum=Append;Replace;IfNone;Never
type IngressControllerHTTPHeaderPolicy string

const (
	// AppendHTTPHeaderPolicy appends the header, preserving any existing header.
	AppendHTTPHeaderPolicy IngressControllerHTTPHeaderPolicy = "Append"
	// ReplaceHTTPHeaderPolicy sets the header, removing any existing header.
	ReplaceHTTPHeaderPolicy IngressControllerHTTPHeaderPolicy = "Replace"
	// IfNoneHTTPHeaderPolicy sets the header if it is not already set.
	IfNoneHTTPHeaderPolicy IngressControllerHTTPHeaderPolicy = "IfNone"
	// NeverHTTPHeaderPolicy never sets the header, preserving any existing
	// header.
	NeverHTTPHeaderPolicy IngressControllerHTTPHeaderPolicy = "Never"
)

// IngressControllerHTTPUniqueIdHeaderPolicy describes configuration for a
// unique id header.
type IngressControllerHTTPUniqueIdHeaderPolicy struct {
	// name specifies the name of the HTTP header (for example, "unique-id")
	// that the ingress controller should inject into HTTP requests.  The
	// field's value must be a valid HTTP header name as defined in RFC 2616
	// section 4.2.  If the field is empty, no header is injected.
	//
	// +optional
	// +kubebuilder:validation:Pattern="^$|^[-!#$%&'*+.0-9A-Z^_`a-z|~]+$"
	// +kubebuilder:validation:MinLength=0
	// +kubebuilder:validation:MaxLength=1024
	Name string `json:"name,omitempty"`

	// format specifies the format for the injected HTTP header's value.
	// This field has no effect unless name is specified.  For the
	// HAProxy-based ingress controller implementation, this format uses the
	// same syntax as the HTTP log format.  If the field is empty, the
	// default value is "%{+X}o\\ %ci:%cp_%fi:%fp_%Ts_%rt:%pid"; see the
	// corresponding HAProxy documentation:
	// http://cbonte.github.io/haproxy-dconv/2.0/configuration.html#8.2.3
	//
	// +optional
	// +kubebuilder:validation:Pattern="^(%(%|(\\{[-+]?[QXE](,[-+]?[QXE])*\\})?([A-Za-z]+|\\[[.0-9A-Z_a-z]+(\\([^)]+\\))?(,[.0-9A-Z_a-z]+(\\([^)]+\\))?)*\\]))|[^%[:cntrl:]])*$"
	// +kubebuilder:validation:MinLength=0
	// +kubebuilder:validation:MaxLength=1024
	Format string `json:"format,omitempty"`
}

// IngressControllerHTTPHeaderNameCaseAdjustment is the name of an HTTP header
// (for example, "X-Forwarded-For") in the desired capitalization.  The value
// must be a valid HTTP header name as defined in RFC 2616 section 4.2.
//
// +optional
// +kubebuilder:validation:Pattern="^$|^[-!#$%&'*+.0-9A-Z^_`a-z|~]+$"
// +kubebuilder:validation:MinLength=0
// +kubebuilder:validation:MaxLength=1024
type IngressControllerHTTPHeaderNameCaseAdjustment string

// IngressControllerHTTPHeaders specifies how the IngressController handles
// certain HTTP headers.
type IngressControllerHTTPHeaders struct {
	// forwardedHeaderPolicy specifies when and how the IngressController
	// sets the Forwarded, X-Forwarded-For, X-Forwarded-Host,
	// X-Forwarded-Port, X-Forwarded-Proto, and X-Forwarded-Proto-Version
	// HTTP headers.  The value may be one of the following:
	//
	// * "Append", which specifies that the IngressController appends the
	//   headers, preserving existing headers.
	//
	// * "Replace", which specifies that the IngressController sets the
	//   headers, replacing any existing Forwarded or X-Forwarded-* headers.
	//
	// * "IfNone", which specifies that the IngressController sets the
	//   headers if they are not already set.
	//
	// * "Never", which specifies that the IngressController never sets the
	//   headers, preserving any existing headers.
	//
	// By default, the policy is "Append".
	//
	// +optional
	ForwardedHeaderPolicy IngressControllerHTTPHeaderPolicy `json:"forwardedHeaderPolicy,omitempty"`

	// uniqueId describes configuration for a custom HTTP header that the
	// ingress controller should inject into incoming HTTP requests.
	// Typically, this header is configured to have a value that is unique
	// to the HTTP request.  The header can be used by applications or
	// included in access logs to facilitate tracing individual HTTP
	// requests.
	//
	// If this field is empty, no such header is injected into requests.
	//
	// +optional
	UniqueId IngressControllerHTTPUniqueIdHeaderPolicy `json:"uniqueId,omitempty"`

	// headerNameCaseAdjustments specifies case adjustments that can be
	// applied to HTTP header names.  Each adjustment is specified as an
	// HTTP header name with the desired capitalization.  For example,
	// specifying "X-Forwarded-For" indicates that the "x-forwarded-for"
	// HTTP header should be adjusted to have the specified capitalization.
	//
	// These adjustments are only applied to cleartext, edge-terminated, and
	// re-encrypt routes, and only when using HTTP/1.
	//
	// For request headers, these adjustments are applied only for routes
	// that have the haproxy.router.openshift.io/h1-adjust-case=true
	// annotation.  For response headers, these adjustments are applied to
	// all HTTP responses.
	//
	// If this field is empty, no request headers are adjusted.
	//
	// +nullable
	// +optional
	HeaderNameCaseAdjustments []IngressControllerHTTPHeaderNameCaseAdjustment `json:"headerNameCaseAdjustments,omitempty"`
}

// IngressControllerTuningOptions specifies options for tuning the performance
// of ingress controller pods
type IngressControllerTuningOptions struct {
	// headerBufferBytes describes how much memory should be reserved
	// (in bytes) for IngressController connection sessions.
	// Note that this value must be at least 16384 if HTTP/2 is
	// enabled for the IngressController (https://tools.ietf.org/html/rfc7540).
	// If this field is empty, the IngressController will use a default value
	// of 32768 bytes.
	//
	// Setting this field is generally not recommended as headerBufferBytes
	// values that are too small may break the IngressController and
	// headerBufferBytes values that are too large could cause the
	// IngressController to use significantly more memory than necessary.
	//
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Minimum=16384
	// +optional
	HeaderBufferBytes int32 `json:"headerBufferBytes,omitempty"`

	// headerBufferMaxRewriteBytes describes how much memory should be reserved
	// (in bytes) from headerBufferBytes for HTTP header rewriting
	// and appending for IngressController connection sessions.
	// Note that incoming HTTP requests will be limited to
	// (headerBufferBytes - headerBufferMaxRewriteBytes) bytes, meaning
	// headerBufferBytes must be greater than headerBufferMaxRewriteBytes.
	// If this field is empty, the IngressController will use a default value
	// of 8192 bytes.
	//
	// Setting this field is generally not recommended as
	// headerBufferMaxRewriteBytes values that are too small may break the
	// IngressController and headerBufferMaxRewriteBytes values that are too
	// large could cause the IngressController to use significantly more memory
	// than necessary.
	//
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Minimum=4096
	// +optional
	HeaderBufferMaxRewriteBytes int32 `json:"headerBufferMaxRewriteBytes,omitempty"`

	// threadCount defines the number of threads created per HAProxy process.
	// Creating more threads allows each ingress controller pod to handle more
	// connections, at the cost of more system resources being used. HAProxy
	// currently supports up to 64 threads. If this field is empty, the
	// IngressController will use the default value.  The current default is 4
	// threads, but this may change in future releases.
	//
	// Setting this field is generally not recommended. Increasing the number
	// of HAProxy threads allows ingress controller pods to utilize more CPU
	// time under load, potentially starving other pods if set too high.
	// Reducing the number of threads may cause the ingress controller to
	// perform poorly.
	//
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=64
	// +optional
	ThreadCount int32 `json:"threadCount,omitempty"`

	// clientTimeout defines how long a connection will be held open while
	// waiting for a client response.
	//
	// If unset, the default timeout is 30s
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Format=duration
	// +optional
	ClientTimeout *metav1.Duration `json:"clientTimeout,omitempty"`

	// clientFinTimeout defines how long a connection will be held open while
	// waiting for the client response to the server/backend closing the
	// connection.
	//
	// If unset, the default timeout is 1s
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Format=duration
	// +optional
	ClientFinTimeout *metav1.Duration `json:"clientFinTimeout,omitempty"`

	// serverTimeout defines how long a connection will be held open while
	// waiting for a server/backend response.
	//
	// If unset, the default timeout is 30s
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Format=duration
	// +optional
	ServerTimeout *metav1.Duration `json:"serverTimeout,omitempty"`

	// serverFinTimeout defines how long a connection will be held open while
	// waiting for the server/backend response to the client closing the
	// connection.
	//
	// If unset, the default timeout is 1s
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Format=duration
	// +optional
	ServerFinTimeout *metav1.Duration `json:"serverFinTimeout,omitempty"`

	// tunnelTimeout defines how long a tunnel connection (including
	// websockets) will be held open while the tunnel is idle.
	//
	// If unset, the default timeout is 1h
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Format=duration
	// +optional
	TunnelTimeout *metav1.Duration `json:"tunnelTimeout,omitempty"`

	// tlsInspectDelay defines how long the router can hold data to find a
	// matching route.
	//
	// Setting this too short can cause the router to fall back to the default
	// certificate for edge-terminated or reencrypt routes even when a better
	// matching certificate could be used.
	//
	// If unset, the default inspect delay is 5s
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Format=duration
	// +optional
	TLSInspectDelay *metav1.Duration `json:"tlsInspectDelay,omitempty"`

	// healthCheckInterval defines how long the router waits between two consecutive
	// health checks on its configured backends.  This value is applied globally as
	// a default for all routes, but may be overridden per-route by the route annotation
	// "router.openshift.io/haproxy.health.check.interval".
	//
	// Expects an unsigned duration string of decimal numbers, each with optional
	// fraction and a unit suffix, eg "300ms", "1.5h" or "2h45m".
	// Valid time units are "ns", "us" (or "µs" U+00B5 or "μs" U+03BC), "ms", "s", "m", "h".
	//
	// Setting this to less than 5s can cause excess traffic due to too frequent
	// TCP health checks and accompanying SYN packet storms.  Alternatively, setting
	// this too high can result in increased latency, due to backend servers that are no
	// longer available, but haven't yet been detected as such.
	//
	// An empty or zero healthCheckInterval means no opinion and IngressController chooses
	// a default, which is subject to change over time.
	// Currently the default healthCheckInterval value is 5s.
	//
	// Currently the minimum allowed value is 1s and the maximum allowed value is
	// 2147483647ms (24.85 days).  Both are subject to change over time.
	//
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Pattern=^(0|([0-9]+(\.[0-9]+)?(ns|us|µs|μs|ms|s|m|h))+)$
	// +kubebuilder:validation:Type:=string
	// +optional
	HealthCheckInterval *metav1.Duration `json:"healthCheckInterval,omitempty"`

	// maxConnections defines the maximum number of simultaneous
	// connections that can be established per HAProxy process.
	// Increasing this value allows each ingress controller pod to
	// handle more connections but at the cost of additional
	// system resources being consumed.
	//
	// Permitted values are: empty, 0, -1, and the range
	// 2000-2000000.
	//
	// If this field is empty or 0, the IngressController will use
	// the default value of 50000, but the default is subject to
	// change in future releases.
	//
	// If the value is -1 then HAProxy will dynamically compute a
	// maximum value based on the available ulimits in the running
	// container. Selecting -1 (i.e., auto) will result in a large
	// value being computed (~520000 on OpenShift >=4.10 clusters)
	// and therefore each HAProxy process will incur significant
	// memory usage compared to the current default of 50000.
	//
	// Setting a value that is greater than the current operating
	// system limit will prevent the HAProxy process from
	// starting.
	//
	// If you choose a discrete value (e.g., 750000) and the
	// router pod is migrated to a new node, there's no guarantee
	// that that new node has identical ulimits configured. In
	// such a scenario the pod would fail to start. If you have
	// nodes with different ulimits configured (e.g., different
	// tuned profiles) and you choose a discrete value then the
	// guidance is to use -1 and let the value be computed
	// dynamically at runtime.
	//
	// You can monitor memory usage for router containers with the
	// following metric:
	// 'container_memory_working_set_bytes{container="router",namespace="openshift-ingress"}'.
	//
	// You can monitor memory usage of individual HAProxy
	// processes in router containers with the following metric:
	// 'container_memory_working_set_bytes{container="router",namespace="openshift-ingress"}/container_processes{container="router",namespace="openshift-ingress"}'.
	//
	// +kubebuilder:validation:Optional
	// +optional
	MaxConnections int32 `json:"maxConnections,omitempty"`

	// reloadInterval defines the minimum interval at which the router is allowed to reload
	// to accept new changes. Increasing this value can prevent the accumulation of
	// HAProxy processes, depending on the scenario. Increasing this interval can
	// also lessen load imbalance on a backend's servers when using the roundrobin
	// balancing algorithm. Alternatively, decreasing this value may decrease latency
	// since updates to HAProxy's configuration can take effect more quickly.
	//
	// The value must be a time duration value; see <https://pkg.go.dev/time#ParseDuration>.
	// Currently, the minimum value allowed is 1s, and the maximum allowed value is
	// 120s. Minimum and maximum allowed values may change in future versions of OpenShift.
	// Note that if a duration outside of these bounds is provided, the value of reloadInterval
	// will be capped/floored and not rejected (e.g. a duration of over 120s will be capped to
	// 120s; the IngressController will not reject and replace this disallowed value with
	// the default).
	//
	// A zero value for reloadInterval tells the IngressController to choose the default,
	// which is currently 5s and subject to change without notice.
	//
	// This field expects an unsigned duration string of decimal numbers, each with optional
	// fraction and a unit suffix, e.g. "300ms", "1.5h" or "2h45m".
	// Valid time units are "ns", "us" (or "µs" U+00B5 or "μs" U+03BC), "ms", "s", "m", "h".
	//
	// Note: Setting a value significantly larger than the default of 5s can cause latency
	// in observing updates to routes and their endpoints. HAProxy's configuration will
	// be reloaded less frequently, and newly created routes will not be served until the
	// subsequent reload.
	//
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Pattern=^(0|([0-9]+(\.[0-9]+)?(ns|us|µs|μs|ms|s|m|h))+)$
	// +kubebuilder:validation:Type:=string
	// +optional
	ReloadInterval metav1.Duration `json:"reloadInterval,omitempty"`
}

// HTTPEmptyRequestsPolicy indicates how HTTP connections for which no request
// is received should be handled.
// +kubebuilder:validation:Enum=Respond;Ignore
type HTTPEmptyRequestsPolicy string

const (
	// HTTPEmptyRequestsPolicyRespond indicates that the ingress controller
	// should respond to empty requests.
	HTTPEmptyRequestsPolicyRespond HTTPEmptyRequestsPolicy = "Respond"
	// HTTPEmptyRequestsPolicyIgnore indicates that the ingress controller
	// should ignore empty requests.
	HTTPEmptyRequestsPolicyIgnore HTTPEmptyRequestsPolicy = "Ignore"
)

var (
	// Available indicates the ingress controller deployment is available.
	IngressControllerAvailableConditionType = "Available"
	// LoadBalancerManaged indicates the management status of any load balancer
	// service associated with an ingress controller.
	LoadBalancerManagedIngressConditionType = "LoadBalancerManaged"
	// LoadBalancerReady indicates the ready state of any load balancer service
	// associated with an ingress controller.
	LoadBalancerReadyIngressConditionType = "LoadBalancerReady"
	// DNSManaged indicates the management status of any DNS records for the
	// ingress controller.
	DNSManagedIngressConditionType = "DNSManaged"
	// DNSReady indicates the ready state of any DNS records for the ingress
	// controller.
	DNSReadyIngressConditionType = "DNSReady"
)

// IngressControllerStatus defines the observed status of the IngressController.
type IngressControllerStatus struct {
	// availableReplicas is number of observed available replicas according to the
	// ingress controller deployment.
	AvailableReplicas int32 `json:"availableReplicas"`

	// selector is a label selector, in string format, for ingress controller pods
	// corresponding to the IngressController. The number of matching pods should
	// equal the value of availableReplicas.
	Selector string `json:"selector"`

	// domain is the actual domain in use.
	Domain string `json:"domain"`

	// endpointPublishingStrategy is the actual strategy in use.
	EndpointPublishingStrategy *EndpointPublishingStrategy `json:"endpointPublishingStrategy,omitempty"`

	// conditions is a list of conditions and their status.
	//
	// Available means the ingress controller deployment is available and
	// servicing route and ingress resources (i.e, .status.availableReplicas
	// equals .spec.replicas)
	//
	// There are additional conditions which indicate the status of other
	// ingress controller features and capabilities.
	//
	//   * LoadBalancerManaged
	//   - True if the following conditions are met:
	//     * The endpoint publishing strategy requires a service load balancer.
	//   - False if any of those conditions are unsatisfied.
	//
	//   * LoadBalancerReady
	//   - True if the following conditions are met:
	//     * A load balancer is managed.
	//     * The load balancer is ready.
	//   - False if any of those conditions are unsatisfied.
	//
	//   * DNSManaged
	//   - True if the following conditions are met:
	//     * The endpoint publishing strategy and platform support DNS.
	//     * The ingress controller domain is set.
	//     * dns.config.openshift.io/cluster configures DNS zones.
	//   - False if any of those conditions are unsatisfied.
	//
	//   * DNSReady
	//   - True if the following conditions are met:
	//     * DNS is managed.
	//     * DNS records have been successfully created.
	//   - False if any of those conditions are unsatisfied.
	Conditions []OperatorCondition `json:"conditions,omitempty"`

	// tlsProfile is the TLS connection configuration that is in effect.
	// +optional
	TLSProfile *configv1.TLSProfileSpec `json:"tlsProfile,omitempty"`

	// observedGeneration is the most recent generation observed.
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// namespaceSelector is the actual namespaceSelector in use.
	// +optional
	NamespaceSelector *metav1.LabelSelector `json:"namespaceSelector,omitempty"`

	// routeSelector is the actual routeSelector in use.
	// +optional
	RouteSelector *metav1.LabelSelector `json:"routeSelector,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:object:root=true

// IngressControllerList contains a list of IngressControllers.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type IngressControllerList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []IngressController `json:"items"`
}
