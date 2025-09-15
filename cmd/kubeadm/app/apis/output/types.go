/*
Copyright 2019 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package output

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	bootstraptokenv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/bootstraptoken/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// BootstrapToken represents information for the bootstrap token output produced by kubeadm
type BootstrapToken struct {
	metav1.TypeMeta

	bootstraptokenv1.BootstrapToken
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Images represents information for the output produced by 'kubeadm config images list'
type Images struct {
	metav1.TypeMeta

	Images []string
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ComponentUpgradePlan represents information about upgrade plan for one component
type ComponentUpgradePlan struct {
	metav1.TypeMeta

	Name           string
	CurrentVersion string
	NewVersion     string
	NodeName       string
}

// ComponentConfigVersionState describes the current and desired version of a component config
type ComponentConfigVersionState struct {
	// Group points to the Kubernetes API group that covers the config
	Group string

	// CurrentVersion is the currently active component config version
	// NOTE: This can be empty in case the config was not found on the cluster or it was unsupported
	// kubeadm generated version
	CurrentVersion string

	// PreferredVersion is the component config version that is currently preferred by kubeadm for use.
	// NOTE: As of today, this is the only version supported by kubeadm.
	PreferredVersion string

	// ManualUpgradeRequired indicates if users need to manually upgrade their component config versions. This happens if
	// the CurrentVersion of the config is user supplied (or modified) and no longer supported. Users should upgrade
	// their component configs to PreferredVersion or any other supported component config version.
	ManualUpgradeRequired bool
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// AvailableUpgrade represents information for a single available upgrade.
type AvailableUpgrade struct {
	metav1.TypeMeta

	Description string

	Components []ComponentUpgradePlan
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// UpgradePlan represents information about upgrade plan for the output
// produced by 'kubeadm upgrade plan'
type UpgradePlan struct {
	metav1.TypeMeta

	AvailableUpgrades []AvailableUpgrade

	ConfigVersions []ComponentConfigVersionState
}

// Certificate represents information for a certificate or a certificate authority when using the check-expiration command.
type Certificate struct {
	// Name of the certificate.
	Name string

	// ExpirationDate defines certificate expiration date in UTC following the RFC3339 format.
	ExpirationDate metav1.Time

	// ResidualTimeSeconds represents the duration in seconds relative to the residual time before expiration.
	ResidualTimeSeconds int64

	// ExternallyManaged defines if the certificate is externally managed.
	ExternallyManaged bool

	// CAName represents the name of the CA that signed the certificate.
	// This field is empty for self-signed, root CA certificates.
	CAName string

	// Missing represents if the certificate is missing.
	Missing bool
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// CertificateExpirationInfo represents information for the output produced by 'kubeadm certs check-expiration'.
type CertificateExpirationInfo struct {
	metav1.TypeMeta

	// Certificates holds a list of certificates to show expiration information for.
	Certificates []Certificate

	// CertificateAuthorities holds a list of certificate authorities to show expiration information for.
	CertificateAuthorities []Certificate
}
