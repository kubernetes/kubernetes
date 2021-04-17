package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// CloudCredential provides a means to configure an operator to manage CredentialsRequests.
type CloudCredential struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// +kubebuilder:validation:Required
	// +required
	Spec CloudCredentialSpec `json:"spec"`
	// +optional
	Status CloudCredentialStatus `json:"status"`
}

// CloudCredentialsMode is the specified mode the cloud-credential-operator
// should reconcile CredentialsRequest with
// +kubebuilder:validation:Enum="";Manual;Mint;Passthrough
type CloudCredentialsMode string

const (
	// CloudCredentialsModeManual tells cloud-credential-operator to not reconcile any CredentialsRequests
	// (primarily used for the disconnected VPC use-cases).
	CloudCredentialsModeManual CloudCredentialsMode = "Manual"

	// CloudCredentialsModeMint tells cloud-credential-operator to reconcile all CredentialsRequests
	// by minting new users/credentials.
	CloudCredentialsModeMint CloudCredentialsMode = "Mint"

	// CloudCredentialsModePassthrough tells cloud-credential-operator to reconcile all CredentialsRequests
	// by copying the cloud-specific secret data.
	CloudCredentialsModePassthrough CloudCredentialsMode = "Passthrough"

	// CloudCredentialsModeDefault puts CCO into the default mode of operation (per-cloud/platform defaults):
	// AWS/Azure/GCP: dynamically determine cluster's cloud credential capabilities to affect
	// processing of CredentialsRequests
	// All other clouds/platforms (OpenStack, oVirt, vSphere, etc): run in "passthrough" mode
	CloudCredentialsModeDefault CloudCredentialsMode = ""
)

// CloudCredentialSpec is the specification of the desired behavior of the cloud-credential-operator.
type CloudCredentialSpec struct {
	OperatorSpec `json:",inline"`
	// CredentialsMode allows informing CCO that it should not attempt to dynamically
	// determine the root cloud credentials capabilities, and it should just run in
	// the specified mode.
	// It also allows putting the operator into "manual" mode if desired.
	// Leaving the field in default mode runs CCO so that the cluster's cloud credentials
	// will be dynamically probed for capabilities (on supported clouds/platforms).
	// Supported modes:
	//   AWS/Azure/GCP: "" (Default), "Mint", "Passthrough", "Manual"
	//   Others: Do not set value as other platforms only support running in "Passthrough"
	// +optional
	CredentialsMode CloudCredentialsMode `json:"credentialsMode,omitempty"`
}

// CloudCredentialStatus defines the observed status of the cloud-credential-operator.
type CloudCredentialStatus struct {
	OperatorStatus `json:",inline"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

type CloudCredentialList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`

	Items []CloudCredential `json:"items"`
}
