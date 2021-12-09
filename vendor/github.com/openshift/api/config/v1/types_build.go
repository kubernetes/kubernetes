package v1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Build configures the behavior of OpenShift builds for the entire cluster.
// This includes default settings that can be overridden in BuildConfig objects, and overrides which are applied to all builds.
//
// The canonical name is "cluster"
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type Build struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// Spec holds user-settable values for the build controller configuration
	// +kubebuilder:validation:Required
	// +required
	Spec BuildSpec `json:"spec"`
}

type BuildSpec struct {
	// AdditionalTrustedCA is a reference to a ConfigMap containing additional CAs that
	// should be trusted for image pushes and pulls during builds.
	// The namespace for this config map is openshift-config.
	//
	// DEPRECATED: Additional CAs for image pull and push should be set on
	// image.config.openshift.io/cluster instead.
	//
	// +optional
	AdditionalTrustedCA ConfigMapNameReference `json:"additionalTrustedCA"`
	// BuildDefaults controls the default information for Builds
	// +optional
	BuildDefaults BuildDefaults `json:"buildDefaults"`
	// BuildOverrides controls override settings for builds
	// +optional
	BuildOverrides BuildOverrides `json:"buildOverrides"`
}

type BuildDefaults struct {
	// DefaultProxy contains the default proxy settings for all build operations, including image pull/push
	// and source download.
	//
	// Values can be overrode by setting the `HTTP_PROXY`, `HTTPS_PROXY`, and `NO_PROXY` environment variables
	// in the build config's strategy.
	// +optional
	DefaultProxy *ProxySpec `json:"defaultProxy,omitempty"`

	// GitProxy contains the proxy settings for git operations only. If set, this will override
	// any Proxy settings for all git commands, such as git clone.
	//
	// Values that are not set here will be inherited from DefaultProxy.
	// +optional
	GitProxy *ProxySpec `json:"gitProxy,omitempty"`

	// Env is a set of default environment variables that will be applied to the
	// build if the specified variables do not exist on the build
	// +optional
	Env []corev1.EnvVar `json:"env,omitempty"`

	// ImageLabels is a list of docker labels that are applied to the resulting image.
	// User can override a default label by providing a label with the same name in their
	// Build/BuildConfig.
	// +optional
	ImageLabels []ImageLabel `json:"imageLabels,omitempty"`

	// Resources defines resource requirements to execute the build.
	// +optional
	Resources corev1.ResourceRequirements `json:"resources"`
}

type ImageLabel struct {
	// Name defines the name of the label. It must have non-zero length.
	Name string `json:"name"`

	// Value defines the literal value of the label.
	// +optional
	Value string `json:"value,omitempty"`
}

type BuildOverrides struct {
	// ImageLabels is a list of docker labels that are applied to the resulting image.
	// If user provided a label in their Build/BuildConfig with the same name as one in this
	// list, the user's label will be overwritten.
	// +optional
	ImageLabels []ImageLabel `json:"imageLabels,omitempty"`

	// NodeSelector is a selector which must be true for the build pod to fit on a node
	// +optional
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`

	// Tolerations is a list of Tolerations that will override any existing
	// tolerations set on a build pod.
	// +optional
	Tolerations []corev1.Toleration `json:"tolerations,omitempty"`

	// ForcePull overrides, if set, the equivalent value in the builds,
	// i.e. false disables force pull for all builds,
	// true enables force pull for all builds,
	// independently of what each build specifies itself
	// +optional
	ForcePull *bool `json:"forcePull,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type BuildList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`

	Items []Build `json:"items"`
}
