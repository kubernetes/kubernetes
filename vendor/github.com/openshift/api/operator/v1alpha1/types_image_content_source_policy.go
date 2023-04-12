package v1alpha1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ImageContentSourcePolicy holds cluster-wide information about how to handle registry mirror rules.
// When multiple policies are defined, the outcome of the behavior is defined on each field.
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +openshift:compatibility-gen:level=4
type ImageContentSourcePolicy struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec holds user settable values for configuration
	// +kubebuilder:validation:Required
	// +required
	Spec ImageContentSourcePolicySpec `json:"spec"`
}

// ImageContentSourcePolicySpec is the specification of the ImageContentSourcePolicy CRD.
type ImageContentSourcePolicySpec struct {
	// repositoryDigestMirrors allows images referenced by image digests in pods to be
	// pulled from alternative mirrored repository locations. The image pull specification
	// provided to the pod will be compared to the source locations described in RepositoryDigestMirrors
	// and the image may be pulled down from any of the mirrors in the list instead of the
	// specified repository allowing administrators to choose a potentially faster mirror.
	// Only image pull specifications that have an image digest will have this behavior applied
	// to them - tags will continue to be pulled from the specified repository in the pull spec.
	//
	// Each “source” repository is treated independently; configurations for different “source”
	// repositories don’t interact.
	//
	// When multiple policies are defined for the same “source” repository, the sets of defined
	// mirrors will be merged together, preserving the relative order of the mirrors, if possible.
	// For example, if policy A has mirrors `a, b, c` and policy B has mirrors `c, d, e`, the
	// mirrors will be used in the order `a, b, c, d, e`.  If the orders of mirror entries conflict
	// (e.g. `a, b` vs. `b, a`) the configuration is not rejected but the resulting order is unspecified.
	// +optional
	RepositoryDigestMirrors []RepositoryDigestMirrors `json:"repositoryDigestMirrors"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ImageContentSourcePolicyList lists the items in the ImageContentSourcePolicy CRD.
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +openshift:compatibility-gen:level=4
type ImageContentSourcePolicyList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata"`

	Items []ImageContentSourcePolicy `json:"items"`
}

// RepositoryDigestMirrors holds cluster-wide information about how to handle mirros in the registries config.
// Note: the mirrors only work when pulling the images that are referenced by their digests.
type RepositoryDigestMirrors struct {
	// source is the repository that users refer to, e.g. in image pull specifications.
	// +required
	Source string `json:"source"`
	// mirrors is one or more repositories that may also contain the same images.
	// The order of mirrors in this list is treated as the user's desired priority, while source
	// is by default considered lower priority than all mirrors. Other cluster configuration,
	// including (but not limited to) other repositoryDigestMirrors objects,
	// may impact the exact order mirrors are contacted in, or some mirrors may be contacted
	// in parallel, so this should be considered a preference rather than a guarantee of ordering.
	// +optional
	Mirrors []string `json:"mirrors"`
}
