package v1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// DNS holds cluster-wide information about DNS. The canonical name is `cluster`
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
// +openshift:api-approved.openshift.io=https://github.com/openshift/api/pull/470
// +openshift:file-pattern=cvoRunLevel=0000_10,operatorName=config-operator,operatorOrdering=01
// +kubebuilder:object:root=true
// +kubebuilder:resource:path=dnses,scope=Cluster
// +kubebuilder:subresource:status
// +kubebuilder:metadata:annotations=release.openshift.io/bootstrap-required=true
type DNS struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec holds user settable values for configuration
	// +required
	Spec DNSSpec `json:"spec"`
	// status holds observed values from the cluster. They may not be overridden.
	// +optional
	Status DNSStatus `json:"status"`
}

type DNSSpec struct {
	// baseDomain is the base domain of the cluster. All managed DNS records will
	// be sub-domains of this base.
	//
	// For example, given the base domain `openshift.example.com`, an API server
	// DNS record may be created for `cluster-api.openshift.example.com`.
	//
	// Once set, this field cannot be changed.
	BaseDomain string `json:"baseDomain"`
	// publicZone is the location where all the DNS records that are publicly accessible to
	// the internet exist.
	//
	// If this field is nil, no public records should be created.
	//
	// Once set, this field cannot be changed.
	//
	// +optional
	PublicZone *DNSZone `json:"publicZone,omitempty"`
	// privateZone is the location where all the DNS records that are only available internally
	// to the cluster exist.
	//
	// If this field is nil, no private records should be created.
	//
	// Once set, this field cannot be changed.
	//
	// +optional
	PrivateZone *DNSZone `json:"privateZone,omitempty"`
	// platform holds configuration specific to the underlying
	// infrastructure provider for DNS.
	// When omitted, this means the user has no opinion and the platform is left
	// to choose reasonable defaults. These defaults are subject to change over time.
	// +optional
	Platform DNSPlatformSpec `json:"platform,omitempty"`
}

// DNSZone is used to define a DNS hosted zone.
// A zone can be identified by an ID or tags.
type DNSZone struct {
	// id is the identifier that can be used to find the DNS hosted zone.
	//
	// on AWS zone can be fetched using `ID` as id in [1]
	// on Azure zone can be fetched using `ID` as a pre-determined name in [2],
	// on GCP zone can be fetched using `ID` as a pre-determined name in [3].
	//
	// [1]: https://docs.aws.amazon.com/cli/latest/reference/route53/get-hosted-zone.html#options
	// [2]: https://docs.microsoft.com/en-us/cli/azure/network/dns/zone?view=azure-cli-latest#az-network-dns-zone-show
	// [3]: https://cloud.google.com/dns/docs/reference/v1/managedZones/get
	// +optional
	ID string `json:"id,omitempty"`

	// tags can be used to query the DNS hosted zone.
	//
	// on AWS, resourcegroupstaggingapi [1] can be used to fetch a zone using `Tags` as tag-filters,
	//
	// [1]: https://docs.aws.amazon.com/cli/latest/reference/resourcegroupstaggingapi/get-resources.html#options
	// +optional
	Tags map[string]string `json:"tags,omitempty"`
}

type DNSStatus struct {
	// dnsSuffix (service-ca amongst others)
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type DNSList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata"`

	Items []DNS `json:"items"`
}

// DNSPlatformSpec holds cloud-provider-specific configuration
// for DNS administration.
// +union
// +kubebuilder:validation:XValidation:rule="has(self.type) && self.type == 'AWS' ?  has(self.aws) : !has(self.aws)",message="aws configuration is required when platform is AWS, and forbidden otherwise"
type DNSPlatformSpec struct {
	// type is the underlying infrastructure provider for the cluster.
	// Allowed values: "", "AWS".
	//
	// Individual components may not support all platforms,
	// and must handle unrecognized platforms with best-effort defaults.
	//
	// +unionDiscriminator
	// +required
	// +kubebuilder:validation:XValidation:rule="self in ['','AWS']",message="allowed values are '' and 'AWS'"
	Type PlatformType `json:"type"`

	// aws contains DNS configuration specific to the Amazon Web Services cloud provider.
	// +optional
	AWS *AWSDNSSpec `json:"aws"`
}

// AWSDNSSpec contains DNS configuration specific to the Amazon Web Services cloud provider.
type AWSDNSSpec struct {
	// privateZoneIAMRole contains the ARN of an IAM role that should be assumed when performing
	// operations on the cluster's private hosted zone specified in the cluster DNS config.
	// When left empty, no role should be assumed.
	// +kubebuilder:validation:Pattern:=`^arn:(aws|aws-cn|aws-us-gov):iam::[0-9]{12}:role\/.*$`
	// +optional
	PrivateZoneIAMRole string `json:"privateZoneIAMRole"`
}
