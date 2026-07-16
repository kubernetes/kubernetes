package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +kubebuilder:validation:Enum=Healthy;Unhealthy;Error
type KMSPluginHealthStatus string

const (
	KMSPluginHealthStatusHealthy KMSPluginHealthStatus = "Healthy"

	KMSPluginHealthStatusUnhealthy KMSPluginHealthStatus = "Unhealthy"

	KMSPluginHealthStatusError KMSPluginHealthStatus = "Error"
)

// +openshift:compatibility-gen:level=1
type KMSPluginHealthReport struct {

	// nodeName is the name of the node this instance of the plugin runs on.
	// The combination of nodeName and keyId makes this health report unique.
	// The value must be a valid Kubernetes node name: a lowercase RFC 1123 subdomain
	// consisting of lowercase alphanumeric characters, '-' or '.', starting and ending with
	// an alphanumeric character, and be at most 253 characters in length.
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=253
	// +kubebuilder:validation:XValidation:rule="!format.dns1123Subdomain().validate(self).hasValue()",message="nodeName must be a lowercase RFC 1123 subdomain consisting of lowercase alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character"
	// +required
	NodeName string `json:"nodeName,omitempty"`

	// keyId is the encryption-key-secret id (kms-{keyId}.sock), a unique identifier of the plugin on that node.
	// This is not a cryptographic key used to encrypt/decrypt any resources.
	// The value must be between 1 and 512 characters.
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=512
	// +required
	KeyId string `json:"keyId,omitempty"`

	// status contains a health indicator for the respective KMS plugin
	// The field can have three states: healthy, unhealthy, error.
	// With error and unhealthy containing additional information in Detail.
	// +required
	Status KMSPluginHealthStatus `json:"status,omitempty"`

	// lastCheckedTime is a timestamp of when the probe was last checked.
	// +required
	LastCheckedTime metav1.Time `json:"lastCheckedTime,omitempty"`

	// kekId refers to the remote KEK id from KMS v2 StatusResponse.key_id.
	// This is not a cryptographic key, but a unique representation of the KEK.
	// The value must be between 1 and 1024 characters.
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=1024
	// +required
	KEKId string `json:"kekId,omitempty"`

	// detail contains additional error/health information for the respective KMS plugin.
	// When omitted, no additional error or health information is provided.
	// When set, the value must be between 1 and 1024 characters.
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=1024
	// +optional
	Detail string `json:"detail,omitempty"`
}

// +openshift:compatibility-gen:level=1
// +kubebuilder:validation:MinProperties=1
type KMSEncryptionStatus struct {
	// healthReports contains all KMS plugin health reports.
	// When omitted, no health reports are available.
	// Each entry must have a unique combination of nodeName and keyId.
	// +optional
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:MaxItems=200
	// +listType=map
	// +listMapKey=nodeName
	// +listMapKey=keyId
	HealthReports []KMSPluginHealthReport `json:"healthReports,omitempty"`
}
