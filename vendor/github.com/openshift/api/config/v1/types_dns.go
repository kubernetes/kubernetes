package v1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// DNS holds cluster-wide information about DNS. The canonical name is `cluster`
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type DNS struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec holds user settable values for configuration
	// +kubebuilder:validation:Required
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
	metav1.ListMeta `json:"metadata"`

	Items []DNS `json:"items"`
}
