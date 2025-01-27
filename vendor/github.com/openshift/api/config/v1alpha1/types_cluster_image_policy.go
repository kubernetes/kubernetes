package v1alpha1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ClusterImagePolicy holds cluster-wide configuration for image signature verification
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +kubebuilder:object:root=true
// +kubebuilder:resource:path=clusterimagepolicies,scope=Cluster
// +kubebuilder:subresource:status
// +openshift:api-approved.openshift.io=https://github.com/openshift/api/pull/1457
// +openshift:file-pattern=cvoRunLevel=0000_10,operatorName=config-operator,operatorOrdering=01
// +openshift:enable:FeatureGate=SigstoreImageVerification
// +openshift:compatibility-gen:level=4
type ClusterImagePolicy struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec contains the configuration for the cluster image policy.
	// +required
	Spec ClusterImagePolicySpec `json:"spec"`
	// status contains the observed state of the resource.
	// +optional
	Status ClusterImagePolicyStatus `json:"status,omitempty"`
}

// CLusterImagePolicySpec is the specification of the ClusterImagePolicy custom resource.
type ClusterImagePolicySpec struct {
	// scopes defines the list of image identities assigned to a policy. Each item refers to a scope in a registry implementing the "Docker Registry HTTP API V2".
	// Scopes matching individual images are named Docker references in the fully expanded form, either using a tag or digest. For example, docker.io/library/busybox:latest (not busybox:latest).
	// More general scopes are prefixes of individual-image scopes, and specify a repository (by omitting the tag or digest), a repository
	// namespace, or a registry host (by only specifying the host name and possibly a port number) or a wildcard expression starting with `*.`, for matching all subdomains (not including a port number).
	// Wildcards are only supported for subdomain matching, and may not be used in the middle of the host, i.e.  *.example.com is a valid case, but example*.*.com is not.
	// If multiple scopes match a given image, only the policy requirements for the most specific scope apply. The policy requirements for more general scopes are ignored.
	// In addition to setting a policy appropriate for your own deployed applications, make sure that a policy on the OpenShift image repositories
	// quay.io/openshift-release-dev/ocp-release, quay.io/openshift-release-dev/ocp-v4.0-art-dev (or on a more general scope) allows deployment of the OpenShift images required for cluster operation.
	// If a scope is configured in both the ClusterImagePolicy and the ImagePolicy, or if the scope in ImagePolicy is nested under one of the scopes from the ClusterImagePolicy, only the policy from the ClusterImagePolicy will be applied.
	// For additional details about the format, please refer to the document explaining the docker transport field,
	// which can be found at: https://github.com/containers/image/blob/main/docs/containers-policy.json.5.md#docker
	// +required
	// +kubebuilder:validation:MaxItems=256
	// +listType=set
	Scopes []ImageScope `json:"scopes"`
	// policy contains configuration to allow scopes to be verified, and defines how
	// images not matching the verification policy will be treated.
	// +required
	Policy Policy `json:"policy"`
}

// +k8s:deepcopy-gen=true
type ClusterImagePolicyStatus struct {
	// conditions provide details on the status of this API Resource.
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ClusterImagePolicyList is a list of ClusterImagePolicy resources
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +openshift:compatibility-gen:level=4
type ClusterImagePolicyList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata"`

	Items []ClusterImagePolicy `json:"items"`
}
