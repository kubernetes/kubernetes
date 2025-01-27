package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:path=dnsnameresolvers,scope=Namespaced
// +openshift:api-approved.openshift.io=https://github.com/openshift/api/pull/1524
// +openshift:file-pattern=cvoRunLevel=0000_70,operatorName=dns,operatorOrdering=00
// +openshift:compatibility-gen:level=4
// +openshift:enable:FeatureGate=DNSNameResolver

// DNSNameResolver stores the DNS name resolution information of a DNS name. It can be enabled by the TechPreviewNoUpgrade feature set.
// It can also be enabled by the feature gate DNSNameResolver when using CustomNoUpgrade feature set.
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
type DNSNameResolver struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec is the specification of the desired behavior of the DNSNameResolver.
	// +required
	Spec DNSNameResolverSpec `json:"spec"`
	// status is the most recently observed status of the DNSNameResolver.
	// +optional
	Status DNSNameResolverStatus `json:"status,omitempty"`
}

// DNSName is used for validation of a DNS name.
// +kubebuilder:validation:Pattern=`^(\*\.)?([a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?\.){2,}$`
// +kubebuilder:validation:MaxLength=254
type DNSName string

// DNSNameResolverSpec is a desired state description of DNSNameResolver.
type DNSNameResolverSpec struct {
	// name is the DNS name for which the DNS name resolution information will be stored.
	// For a regular DNS name, only the DNS name resolution information of the regular DNS
	// name will be stored. For a wildcard DNS name, the DNS name resolution information
	// of all the DNS names that match the wildcard DNS name will be stored.
	// For a wildcard DNS name, the '*' will match only one label. Additionally, only a single
	// '*' can be used at the beginning of the wildcard DNS name. For example, '*.example.com.'
	// will match 'sub1.example.com.' but won't match 'sub2.sub1.example.com.'
	// +required
	// +kubebuilder:validation:XValidation:rule="self == oldSelf",message="spec.name is immutable"
	Name DNSName `json:"name"`
}

// DNSNameResolverStatus defines the observed status of DNSNameResolver.
type DNSNameResolverStatus struct {
	// resolvedNames contains a list of matching DNS names and their corresponding IP addresses
	// along with their TTL and last DNS lookup times.
	// +listType=map
	// +listMapKey=dnsName
	// +patchMergeKey=dnsName
	// +patchStrategy=merge
	// +optional
	ResolvedNames []DNSNameResolverResolvedName `json:"resolvedNames,omitempty" patchStrategy:"merge" patchMergeKey:"dnsName"`
}

// DNSNameResolverResolvedName describes the details of a resolved DNS name.
type DNSNameResolverResolvedName struct {
	// conditions provide information about the state of the DNS name.
	// Known .status.conditions.type is: "Degraded".
	// "Degraded" is true when the last resolution failed for the DNS name,
	// and false otherwise.
	// +optional
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// dnsName is the resolved DNS name matching the name field of DNSNameResolverSpec. This field can
	// store both regular and wildcard DNS names which match the spec.name field. When the spec.name
	// field contains a regular DNS name, this field will store the same regular DNS name after it is
	// successfully resolved. When the spec.name field contains a wildcard DNS name, each resolvedName.dnsName
	// will store the regular DNS names which match the wildcard DNS name and have been successfully resolved.
	// If the wildcard DNS name can also be successfully resolved, then this field will store the wildcard
	// DNS name as well.
	// +required
	DNSName DNSName `json:"dnsName"`

	// resolvedAddresses gives the list of associated IP addresses and their corresponding TTLs and last
	// lookup times for the dnsName.
	// +required
	// +listType=map
	// +listMapKey=ip
	ResolvedAddresses []DNSNameResolverResolvedAddress `json:"resolvedAddresses"`

	// resolutionFailures keeps the count of how many consecutive times the DNS resolution failed
	// for the dnsName. If the DNS resolution succeeds then the field will be set to zero. Upon
	// every failure, the value of the field will be incremented by one. The details about the DNS
	// name will be removed, if the value of resolutionFailures reaches 5 and the TTL of all the
	// associated IP addresses have expired.
	ResolutionFailures int32 `json:"resolutionFailures,omitempty"`
}

// DNSNameResolverResolvedAddress describes the details of an IP address for a resolved DNS name.
type DNSNameResolverResolvedAddress struct {
	// ip is an IP address associated with the dnsName. The validity of the IP address expires after
	// lastLookupTime + ttlSeconds. To refresh the information, a DNS lookup will be performed upon
	// the expiration of the IP address's validity. If the information is not refreshed then it will
	// be removed with a grace period after the expiration of the IP address's validity.
	// +required
	IP string `json:"ip"`

	// ttlSeconds is the time-to-live value of the IP address. The validity of the IP address expires after
	// lastLookupTime + ttlSeconds. On a successful DNS lookup the value of this field will be updated with
	// the current time-to-live value. If the information is not refreshed then it will be removed with a
	// grace period after the expiration of the IP address's validity.
	// +required
	TTLSeconds int32 `json:"ttlSeconds"`

	// lastLookupTime is the timestamp when the last DNS lookup was completed successfully. The validity of
	// the IP address expires after lastLookupTime + ttlSeconds. The value of this field will be updated to
	// the current time on a successful DNS lookup. If the information is not refreshed then it will be
	// removed with a grace period after the expiration of the IP address's validity.
	// +required
	LastLookupTime *metav1.Time `json:"lastLookupTime"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +openshift:compatibility-gen:level=4

// DNSNameResolverList contains a list of DNSNameResolvers.
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
type DNSNameResolverList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata,omitempty"`

	// items gives the list of DNSNameResolvers.
	Items []DNSNameResolver `json:"items"`
}
