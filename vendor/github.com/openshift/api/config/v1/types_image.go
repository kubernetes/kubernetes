package v1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Image governs policies related to imagestream imports and runtime configuration
// for external registries. It allows cluster admins to configure which registries
// OpenShift is allowed to import images from, extra CA trust bundles for external
// registries, and policies to block or allow registry hostnames.
// When exposing OpenShift's image registry to the public, this also lets cluster
// admins specify the external hostname.
type Image struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec holds user settable values for configuration
	// +kubebuilder:validation:Required
	// +required
	Spec ImageSpec `json:"spec"`
	// status holds observed values from the cluster. They may not be overridden.
	// +optional
	Status ImageStatus `json:"status"`
}

type ImageSpec struct {
	// allowedRegistriesForImport limits the container image registries that normal users may import
	// images from. Set this list to the registries that you trust to contain valid Docker
	// images and that you want applications to be able to import from. Users with
	// permission to create Images or ImageStreamMappings via the API are not affected by
	// this policy - typically only administrators or system integrations will have those
	// permissions.
	// +optional
	AllowedRegistriesForImport []RegistryLocation `json:"allowedRegistriesForImport,omitempty"`

	// externalRegistryHostnames provides the hostnames for the default external image
	// registry. The external hostname should be set only when the image registry
	// is exposed externally. The first value is used in 'publicDockerImageRepository'
	// field in ImageStreams. The value must be in "hostname[:port]" format.
	// +optional
	ExternalRegistryHostnames []string `json:"externalRegistryHostnames,omitempty"`

	// additionalTrustedCA is a reference to a ConfigMap containing additional CAs that
	// should be trusted during imagestream import, pod image pull, build image pull, and
	// imageregistry pullthrough.
	// The namespace for this config map is openshift-config.
	// +optional
	AdditionalTrustedCA ConfigMapNameReference `json:"additionalTrustedCA"`

	// registrySources contains configuration that determines how the container runtime
	// should treat individual registries when accessing images for builds+pods. (e.g.
	// whether or not to allow insecure access).  It does not contain configuration for the
	// internal cluster registry.
	// +optional
	RegistrySources RegistrySources `json:"registrySources"`
}

type ImageStatus struct {

	// internalRegistryHostname sets the hostname for the default internal image
	// registry. The value must be in "hostname[:port]" format.
	// This value is set by the image registry operator which controls the internal registry
	// hostname. For backward compatibility, users can still use OPENSHIFT_DEFAULT_REGISTRY
	// environment variable but this setting overrides the environment variable.
	// +optional
	InternalRegistryHostname string `json:"internalRegistryHostname,omitempty"`

	// externalRegistryHostnames provides the hostnames for the default external image
	// registry. The external hostname should be set only when the image registry
	// is exposed externally. The first value is used in 'publicDockerImageRepository'
	// field in ImageStreams. The value must be in "hostname[:port]" format.
	// +optional
	ExternalRegistryHostnames []string `json:"externalRegistryHostnames,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

type ImageList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`

	Items []Image `json:"items"`
}

// RegistryLocation contains a location of the registry specified by the registry domain
// name. The domain name might include wildcards, like '*' or '??'.
type RegistryLocation struct {
	// domainName specifies a domain name for the registry
	// In case the registry use non-standard (80 or 443) port, the port should be included
	// in the domain name as well.
	DomainName string `json:"domainName"`
	// insecure indicates whether the registry is secure (https) or insecure (http)
	// By default (if not specified) the registry is assumed as secure.
	// +optional
	Insecure bool `json:"insecure,omitempty"`
}

// RegistrySources holds cluster-wide information about how to handle the registries config.
type RegistrySources struct {
	// insecureRegistries are registries which do not have a valid TLS certificates or only support HTTP connections.
	// +optional
	InsecureRegistries []string `json:"insecureRegistries,omitempty"`
	// blockedRegistries cannot be used for image pull and push actions. All other registries are permitted.
	//
	// Only one of BlockedRegistries or AllowedRegistries may be set.
	// +optional
	BlockedRegistries []string `json:"blockedRegistries,omitempty"`
	// allowedRegistries are the only registries permitted for image pull and push actions. All other registries are denied.
	//
	// Only one of BlockedRegistries or AllowedRegistries may be set.
	// +optional
	AllowedRegistries []string `json:"allowedRegistries,omitempty"`
}
