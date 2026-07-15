package v1alpha1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// CRIOCredentialProviderConfig holds cluster-wide singleton resource configurations for CRI-O credential provider, the name of this instance is "cluster". CRI-O credential provider is a binary shipped with CRI-O that provides a way to obtain container image pull credentials from external sources.
// For example, it can be used to fetch mirror registry credentials from secrets resources in the cluster within the same namespace the pod will be running in.
// CRIOCredentialProviderConfig configuration specifies the pod image sources registries that should trigger the CRI-O credential provider execution, which will resolve the CRI-O mirror configurations and obtain the necessary credentials for pod creation.
// Note: Configuration changes will only take effect after the kubelet restarts, which is automatically managed by the cluster during rollout.
//
// The resource is a singleton named "cluster".
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +kubebuilder:object:root=true
// +kubebuilder:resource:path=criocredentialproviderconfigs,scope=Cluster
// +kubebuilder:subresource:status
// +openshift:api-approved.openshift.io=https://github.com/openshift/api/pull/2557
// +openshift:file-pattern=cvoRunLevel=0000_10,operatorName=config-operator,operatorOrdering=01
// +openshift:enable:FeatureGate=CRIOCredentialProviderConfig
// +openshift:compatibility-gen:level=4
// +kubebuilder:validation:XValidation:rule="self.metadata.name == 'cluster'",message="criocredentialproviderconfig is a singleton, .metadata.name must be 'cluster'"
type CRIOCredentialProviderConfig struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitzero"`

	// spec defines the desired configuration of the CRI-O Credential Provider.
	// This field is required and must be provided when creating the resource.
	// +required
	Spec *CRIOCredentialProviderConfigSpec `json:"spec,omitempty,omitzero"`

	// status represents the current state of the CRIOCredentialProviderConfig.
	// When omitted or nil, it indicates that the status has not yet been set by the controller.
	// The controller will populate this field with validation conditions and operational state.
	// +optional
	Status CRIOCredentialProviderConfigStatus `json:"status,omitzero,omitempty"`
}

// CRIOCredentialProviderConfigSpec defines the desired configuration of the CRI-O Credential Provider.
// +kubebuilder:validation:MinProperties=0
type CRIOCredentialProviderConfigSpec struct {
	// matchImages is a list of string patterns used to determine whether
	// the CRI-O credential provider should be invoked for a given image. This list is
	// passed to the kubelet CredentialProviderConfig, and if any pattern matches
	// the requested image, CRI-O credential provider will be invoked to obtain credentials for pulling
	// that image or its mirrors.
	// Depending on the platform, the CRI-O credential provider may be installed alongside an existing platform specific provider.
	// Conflicts between the existing platform specific provider image match configuration and this list will be handled by
	// the following precedence rule: credentials from built-in kubelet providers (e.g., ECR, GCR, ACR) take precedence over those
	// from the CRIOCredentialProviderConfig when both match the same image.
	// To avoid uncertainty, it is recommended to avoid configuring your private image patterns to overlap with
	// existing platform specific provider config(e.g., the entries from https://github.com/openshift/machine-config-operator/blob/main/templates/common/aws/files/etc-kubernetes-credential-providers-ecr-credential-provider.yaml).
	// You can check the resource's Status conditions
	// to see if any entries were ignored due to exact matches with known built-in provider patterns.
	//
	// This field is optional, the items of the list must contain between 1 and 50 entries.
	// The list is treated as a set, so duplicate entries are not allowed.
	//
	// For more details, see:
	// https://kubernetes.io/docs/tasks/administer-cluster/kubelet-credential-provider/
	// https://github.com/cri-o/crio-credential-provider#architecture
	//
	// Each entry in matchImages is a pattern which can optionally contain a port and a path. Each entry must be no longer than 512 characters.
	// Wildcards ('*') are supported for full subdomain labels, such as '*.k8s.io' or 'k8s.*.io',
	// and for top-level domains, such as 'k8s.*' (which matches 'k8s.io' or 'k8s.net').
	// A global wildcard '*' (matching any domain) is not allowed.
	// Wildcards may replace an entire hostname label (e.g., *.example.com), but they cannot appear within a label (e.g., f*oo.example.com) and are not allowed in the port or path.
	// For example, 'example.*.com' is valid, but 'exa*mple.*.com' is not.
	// Each wildcard matches only a single domain label,
	// so '*.io' does **not** match '*.k8s.io'.
	//
	// A match exists between an image and a matchImage when all of the below are true:
	// Both contain the same number of domain parts and each part matches.
	// The URL path of an matchImages must be a prefix of the target image URL path.
	// If the matchImages contains a port, then the port must match in the image as well.
	//
	// Example values of matchImages:
	// - 123456789.dkr.ecr.us-east-1.amazonaws.com
	// - *.azurecr.io
	// - gcr.io
	// - *.*.registry.io
	// - registry.io:8080/path
	//
	// +kubebuilder:validation:MaxItems=50
	// +kubebuilder:validation:MinItems=1
	// +listType=set
	// +optional
	MatchImages []MatchImage `json:"matchImages,omitempty"`
}

// MatchImage is a string pattern used to match container image registry addresses.
// It must be a valid fully qualified domain name with optional wildcard, port, and path.
// The maximum length is 512 characters.
//
// Wildcards ('*') are supported for full subdomain labels and top-level domains.
// Each entry can optionally contain a port (e.g., :8080) and a path (e.g., /path).
// Wildcards are not allowed in the port or path portions.
//
// Examples:
// - "registry.io" - matches exactly registry.io
// - "*.azurecr.io" - matches any single subdomain of azurecr.io
// - "registry.io:8080/path" - matches with specific port and path prefix
//
// +kubebuilder:validation:MaxLength=512
// +kubebuilder:validation:MinLength=1
// +kubebuilder:validation:XValidation:rule="self != '*'",message="global wildcard '*' is not allowed"
// +kubebuilder:validation:XValidation:rule=`self.matches('^((\\*|[a-z0-9]([a-z0-9-]*[a-z0-9])?)(\\.(\\*|[a-z0-9]([a-z0-9-]*[a-z0-9])?))*)(:[0-9]+)?(/[-a-z0-9._/]*)?$')`,message="invalid matchImages value, must be a valid fully qualified domain name in lowercase with optional wildcard, port, and path"
type MatchImage string

// +k8s:deepcopy-gen=true
// CRIOCredentialProviderConfigStatus defines the observed state of CRIOCredentialProviderConfig
// +kubebuilder:validation:MinProperties=1
type CRIOCredentialProviderConfigStatus struct {
	// conditions represent the latest available observations of the configuration state.
	// When omitted, it indicates that no conditions have been reported yet.
	// The maximum number of conditions is 16.
	// Conditions are stored as a map keyed by condition type, ensuring uniqueness.
	//
	// Expected condition types include:
	// "Validated": indicates whether the matchImages configuration is valid
	// +optional
	// +kubebuilder:validation:MaxItems=16
	// +kubebuilder:validation:MinItems=1
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// CRIOCredentialProviderConfigList contains a list of CRIOCredentialProviderConfig resources
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +openshift:compatibility-gen:level=4
type CRIOCredentialProviderConfigList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata"`

	Items []CRIOCredentialProviderConfig `json:"items"`
}

const (
	// ConditionTypeValidated is a condition type that indicates whether the CRIOCredentialProviderConfig
	// matchImages configuration has been validated successfully.
	// When True, all matchImage patterns are valid and have been applied.
	// When False, the configuration contains errors (see Reason for details).
	// Possible reasons for False status:
	// - ValidationFailed: matchImages contains invalid patterns
	// - ConfigurationPartiallyApplied: some matchImage entries were ignored due to conflicts
	ConditionTypeValidated = "Validated"

	// ReasonValidationFailed is a condition reason used with ConditionTypeValidated=False
	// to indicate that the matchImages configuration contains one or more invalid registry patterns
	// that do not conform to the required format (valid FQDN with optional wildcard, port, and path).
	ReasonValidationFailed = "ValidationFailed"

	// ReasonConfigurationPartiallyApplied is a condition reason used with ConditionTypeValidated=False
	// to indicate that some matchImage entries were ignored due to conflicts or overlapping patterns.
	// The condition message will contain details about which entries were ignored and why.
	ReasonConfigurationPartiallyApplied = "ConfigurationPartiallyApplied"

	// ConditionTypeMachineConfigRendered is a condition type that indicates whether
	// the CRIOCredentialProviderConfig has been successfully rendered into a
	// MachineConfig object.
	// When True, the corresponding MachineConfig is present in the cluster.
	// When False, rendering failed.
	ConditionTypeMachineConfigRendered = "MachineConfigRendered"

	// ReasonMachineConfigRenderingSucceeded is a condition reason used with ConditionTypeMachineConfigRendered=True
	// to indicate that the MachineConfig was successfully created/updated in the API server.
	ReasonMachineConfigRenderingSucceeded = "MachineConfigRenderingSucceeded"

	// ReasonMachineConfigRenderingFailed is a condition reason used with ConditionTypeMachineConfigRendered=False
	// to indicate that the MachineConfig creation/update failed.
	// The condition message will contain details about the failure.
	ReasonMachineConfigRenderingFailed = "MachineConfigRenderingFailed"
)
