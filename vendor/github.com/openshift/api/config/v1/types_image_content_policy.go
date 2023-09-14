package v1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ImageContentPolicy holds cluster-wide information about how to handle registry mirror rules.
// When multiple policies are defined, the outcome of the behavior is defined on each field.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type ImageContentPolicy struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec holds user settable values for configuration
	// +kubebuilder:validation:Required
	// +required
	Spec ImageContentPolicySpec `json:"spec"`
}

// ImageContentPolicySpec is the specification of the ImageContentPolicy CRD.
type ImageContentPolicySpec struct {
	// repositoryDigestMirrors allows images referenced by image digests in pods to be
	// pulled from alternative mirrored repository locations. The image pull specification
	// provided to the pod will be compared to the source locations described in RepositoryDigestMirrors
	// and the image may be pulled down from any of the mirrors in the list instead of the
	// specified repository allowing administrators to choose a potentially faster mirror.
	// To pull image from mirrors by tags, should set the "allowMirrorByTags".
	//
	// Each “source” repository is treated independently; configurations for different “source”
	// repositories don’t interact.
	//
	// If the "mirrors" is not specified, the image will continue to be pulled from the specified
	// repository in the pull spec.
	//
	// When multiple policies are defined for the same “source” repository, the sets of defined
	// mirrors will be merged together, preserving the relative order of the mirrors, if possible.
	// For example, if policy A has mirrors `a, b, c` and policy B has mirrors `c, d, e`, the
	// mirrors will be used in the order `a, b, c, d, e`.  If the orders of mirror entries conflict
	// (e.g. `a, b` vs. `b, a`) the configuration is not rejected but the resulting order is unspecified.
	// +optional
	// +listType=map
	// +listMapKey=source
	RepositoryDigestMirrors []RepositoryDigestMirrors `json:"repositoryDigestMirrors"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ImageContentPolicyList lists the items in the ImageContentPolicy CRD.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type ImageContentPolicyList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata"`

	Items []ImageContentPolicy `json:"items"`
}

// RepositoryDigestMirrors holds cluster-wide information about how to handle mirrors in the registries config.
type RepositoryDigestMirrors struct {
	// source is the repository that users refer to, e.g. in image pull specifications.
	// +required
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Pattern=`^(([a-zA-Z]|[a-zA-Z][a-zA-Z0-9\-]*[a-zA-Z0-9])\.)*([A-Za-z]|[A-Za-z][A-Za-z0-9\-]*[A-Za-z0-9])(:[0-9]+)?(\/[^\/:\n]+)*(\/[^\/:\n]+((:[^\/:\n]+)|(@[^\n]+)))?$`
	Source string `json:"source"`
	// allowMirrorByTags if true, the mirrors can be used to pull the images that are referenced by their tags. Default is false, the mirrors only work when pulling the images that are referenced by their digests.
	// Pulling images by tag can potentially yield different images, depending on which endpoint
	// we pull from. Forcing digest-pulls for mirrors avoids that issue.
	// +optional
	AllowMirrorByTags bool `json:"allowMirrorByTags,omitempty"`
	// mirrors is zero or more repositories that may also contain the same images.
	// If the "mirrors" is not specified, the image will continue to be pulled from the specified
	// repository in the pull spec. No mirror will be configured.
	// The order of mirrors in this list is treated as the user's desired priority, while source
	// is by default considered lower priority than all mirrors. Other cluster configuration,
	// including (but not limited to) other repositoryDigestMirrors objects,
	// may impact the exact order mirrors are contacted in, or some mirrors may be contacted
	// in parallel, so this should be considered a preference rather than a guarantee of ordering.
	// +optional
	// +listType=set
	Mirrors []Mirror `json:"mirrors,omitempty"`
}

// +kubebuilder:validation:Pattern=`^(([a-zA-Z]|[a-zA-Z][a-zA-Z0-9\-]*[a-zA-Z0-9])\.)*([A-Za-z]|[A-Za-z][A-Za-z0-9\-]*[A-Za-z0-9])(:[0-9]+)?(\/[^\/:\n]+)*(\/[^\/:\n]+((:[^\/:\n]+)|(@[^\n]+)))?$`
type Mirror string
