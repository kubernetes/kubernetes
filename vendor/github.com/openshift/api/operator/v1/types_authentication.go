package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Authentication provides information to configure an operator to manage authentication.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type Authentication struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// +kubebuilder:validation:Required
	// +required
	Spec AuthenticationSpec `json:"spec,omitempty"`
	// +optional
	Status AuthenticationStatus `json:"status,omitempty"`
}

type AuthenticationSpec struct {
	OperatorSpec `json:",inline"`
}

type AuthenticationStatus struct {
	// OAuthAPIServer holds status specific only to oauth-apiserver
	// +optional
	OAuthAPIServer OAuthAPIServerStatus `json:"oauthAPIServer,omitempty"`

	OperatorStatus `json:",inline"`
}

type OAuthAPIServerStatus struct {
	// LatestAvailableRevision is the latest revision used as suffix of revisioned
	// secrets like encryption-config. A new revision causes a new deployment of pods.
	// +optional
	// +kubebuilder:validation:Minimum=0
	LatestAvailableRevision int32 `json:"latestAvailableRevision,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// AuthenticationList is a collection of items
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type AuthenticationList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`

	Items []Authentication `json:"items"`
}
