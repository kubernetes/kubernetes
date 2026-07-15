package v1alpha1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// PKI configures cryptographic parameters for certificates generated
// internally by OpenShift components.
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
//
// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:object:root=true
// +kubebuilder:resource:path=pkis,scope=Cluster
// +openshift:api-approved.openshift.io=https://github.com/openshift/api/pull/2645
// +openshift:file-pattern=cvoRunLevel=0000_10,operatorName=config-operator,operatorOrdering=01
// +openshift:enable:FeatureGate=ConfigurablePKI
// +openshift:compatibility-gen:level=4
type PKI struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec holds user settable values for configuration
	// +required
	Spec PKISpec `json:"spec,omitzero"`
}

// PKISpec holds the specification for PKI configuration.
type PKISpec struct {
	// certificateManagement specifies how PKI configuration is managed for internally-generated certificates.
	// This controls the certificate generation approach for all OpenShift components that create
	// certificates internally, including certificate authorities, serving certificates, and client certificates.
	//
	// +required
	CertificateManagement PKICertificateManagement `json:"certificateManagement,omitzero"`
}

// PKICertificateManagement determines whether components use hardcoded defaults (Unmanaged), follow
// OpenShift best practices (Default), or use administrator-specified cryptographic parameters (Custom).
// This provides flexibility for organizations with specific compliance requirements or security policies
// while maintaining backwards compatibility for existing clusters.
//
// +kubebuilder:validation:XValidation:rule="self.mode == 'Custom' ? has(self.custom) : !has(self.custom)",message="custom is required when mode is Custom, and forbidden otherwise"
// +union
type PKICertificateManagement struct {
	// mode determines how PKI configuration is managed.
	// Valid values are "Unmanaged", "Default", and "Custom".
	//
	// When set to Unmanaged, components use their existing hardcoded certificate
	// generation behavior, exactly as if this feature did not exist. Each component
	// generates certificates using whatever parameters it was using before this
	// feature. While most components use RSA 2048, some may use different
	// parameters. Use of this mode might prevent upgrading to the next major
	// OpenShift release.
	//
	// When set to Default, OpenShift-recommended best practices for certificate
	// generation are applied. The specific parameters may evolve across OpenShift
	// releases to adopt improved cryptographic standards. In the initial release,
	// this matches Unmanaged behavior for each component. In future releases, this
	// may adopt ECDSA or larger RSA keys based on industry best practices.
	// Recommended for most customers who want to benefit from security improvements
	// automatically.
	//
	// When set to Custom, the certificate management parameters can be set
	// explicitly. Use the custom field to specify certificate generation parameters.
	//
	// +required
	// +unionDiscriminator
	Mode PKICertificateManagementMode `json:"mode,omitempty"`

	// custom contains administrator-specified cryptographic configuration.
	// Use the defaults and category override fields
	// to specify certificate generation parameters.
	// Required when mode is Custom, and forbidden otherwise.
	//
	// +optional
	// +unionMember
	Custom CustomPKIPolicy `json:"custom,omitzero"`
}

// CustomPKIPolicy contains administrator-specified cryptographic configuration.
// Administrators must specify defaults for all certificates and may optionally
// override specific categories of certificates.
//
// +kubebuilder:validation:MinProperties=1
type CustomPKIPolicy struct {
	PKIProfile `json:",inline"`
}

// PKICertificateManagementMode specifies the mode for PKI certificate management.
//
// +kubebuilder:validation:Enum=Unmanaged;Default;Custom
type PKICertificateManagementMode string

const (
	// PKICertificateManagementModeUnmanaged uses each component's existing hardcoded defaults.
	// Most components currently use RSA 2048, but parameters may differ by component.
	PKICertificateManagementModeUnmanaged PKICertificateManagementMode = "Unmanaged"

	// PKICertificateManagementModeDefault uses OpenShift-recommended best practices.
	// Specific parameters may evolve across OpenShift releases.
	PKICertificateManagementModeDefault PKICertificateManagementMode = "Default"

	// PKICertificateManagementModeCustom uses administrator-specified configuration.
	PKICertificateManagementModeCustom PKICertificateManagementMode = "Custom"
)

// PKIProfile defines the certificate generation parameters that OpenShift
// components use to create certificates. Category overrides take precedence
// over defaults.
type PKIProfile struct {
	// defaults specifies the default certificate configuration that applies
	// to all certificates unless overridden by a category override.
	//
	// +required
	Defaults DefaultCertificateConfig `json:"defaults,omitzero"`

	// signerCertificates optionally overrides certificate parameters for
	// certificate authority (CA) certificates that sign other certificates.
	// When set, these parameters take precedence over defaults for all signer certificates.
	// When omitted, the defaults are used for signer certificates.
	//
	// +optional
	SignerCertificates CertificateConfig `json:"signerCertificates,omitempty,omitzero"`

	// servingCertificates optionally overrides certificate parameters for
	// TLS server certificates used to serve HTTPS endpoints.
	// When set, these parameters take precedence over defaults for all serving certificates.
	// When omitted, the defaults are used for serving certificates.
	//
	// +optional
	ServingCertificates CertificateConfig `json:"servingCertificates,omitempty,omitzero"`

	// clientCertificates optionally overrides certificate parameters for
	// client authentication certificates used to authenticate to servers.
	// When set, these parameters take precedence over defaults for all client certificates.
	// When omitted, the defaults are used for client certificates.
	//
	// +optional
	ClientCertificates CertificateConfig `json:"clientCertificates,omitempty,omitzero"`
}

// DefaultCertificateConfig specifies the default certificate configuration
// parameters. All fields are required to ensure that defaults are fully
// specified for all certificates.
type DefaultCertificateConfig struct {
	// key specifies the cryptographic parameters for the certificate's key pair.
	// This field is required in defaults to ensure all certificates have a
	// well-defined key configuration.
	// +required
	Key KeyConfig `json:"key,omitzero"`
}

// CertificateConfig specifies configuration parameters for certificates.
// At least one property must be specified.
// +kubebuilder:validation:MinProperties=1
type CertificateConfig struct {
	// key specifies the cryptographic parameters for the certificate's key pair.
	// Currently this is the only configurable parameter. When omitted in an
	// overrides entry, the key configuration from defaults is used.
	// +optional
	Key KeyConfig `json:"key,omitzero"`
}

// KeyConfig specifies cryptographic parameters for key generation.
//
// +kubebuilder:validation:XValidation:rule="has(self.algorithm) && self.algorithm == 'RSA' ?  has(self.rsa) : !has(self.rsa)",message="rsa is required when algorithm is RSA, and forbidden otherwise"
// +kubebuilder:validation:XValidation:rule="has(self.algorithm) && self.algorithm == 'ECDSA' ?  has(self.ecdsa) : !has(self.ecdsa)",message="ecdsa is required when algorithm is ECDSA, and forbidden otherwise"
// +union
type KeyConfig struct {
	// algorithm specifies the key generation algorithm.
	// Valid values are "RSA" and "ECDSA".
	//
	// When set to RSA, the rsa field must be specified and the generated key
	// will be an RSA key with the configured key size.
	//
	// When set to ECDSA, the ecdsa field must be specified and the generated key
	// will be an ECDSA key using the configured elliptic curve.
	//
	// +required
	// +unionDiscriminator
	Algorithm KeyAlgorithm `json:"algorithm,omitempty"`

	// rsa specifies RSA key parameters.
	// Required when algorithm is RSA, and forbidden otherwise.
	// +optional
	// +unionMember
	RSA RSAKeyConfig `json:"rsa,omitzero"`

	// ecdsa specifies ECDSA key parameters.
	// Required when algorithm is ECDSA, and forbidden otherwise.
	// +optional
	// +unionMember
	ECDSA ECDSAKeyConfig `json:"ecdsa,omitzero"`
}

// RSAKeyConfig specifies parameters for RSA key generation.
type RSAKeyConfig struct {
	// keySize specifies the size of RSA keys in bits.
	// Valid values are multiples of 1024 from 2048 to 8192.
	// +required
	// +kubebuilder:validation:Minimum=2048
	// +kubebuilder:validation:Maximum=8192
	// +kubebuilder:validation:MultipleOf=1024
	KeySize int32 `json:"keySize,omitempty"`
}

// ECDSAKeyConfig specifies parameters for ECDSA key generation.
type ECDSAKeyConfig struct {
	// curve specifies the NIST elliptic curve for ECDSA keys.
	// Valid values are "P256", "P384", and "P521".
	//
	// When set to P256, the NIST P-256 curve (also known as secp256r1) is used,
	// providing 128-bit security.
	//
	// When set to P384, the NIST P-384 curve (also known as secp384r1) is used,
	// providing 192-bit security.
	//
	// When set to P521, the NIST P-521 curve (also known as secp521r1) is used,
	// providing 256-bit security.
	//
	// +required
	Curve ECDSACurve `json:"curve,omitempty"`
}

// KeyAlgorithm specifies the cryptographic algorithm used for key generation.
//
// +kubebuilder:validation:Enum=RSA;ECDSA
type KeyAlgorithm string

const (
	// KeyAlgorithmRSA specifies the RSA (Rivest-Shamir-Adleman) algorithm for key generation.
	KeyAlgorithmRSA KeyAlgorithm = "RSA"

	// KeyAlgorithmECDSA specifies the ECDSA (Elliptic Curve Digital Signature Algorithm) for key generation.
	KeyAlgorithmECDSA KeyAlgorithm = "ECDSA"
)

// ECDSACurve specifies the elliptic curve used for ECDSA key generation.
//
// +kubebuilder:validation:Enum=P256;P384;P521
type ECDSACurve string

const (
	// ECDSACurveP256 specifies the NIST P-256 curve (also known as secp256r1), providing 128-bit security.
	ECDSACurveP256 ECDSACurve = "P256"

	// ECDSACurveP384 specifies the NIST P-384 curve (also known as secp384r1), providing 192-bit security.
	ECDSACurveP384 ECDSACurve = "P384"

	// ECDSACurveP521 specifies the NIST P-521 curve (also known as secp521r1), providing 256-bit security.
	ECDSACurveP521 ECDSACurve = "P521"
)

// PKIList is a collection of PKI resources.
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
//
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +openshift:compatibility-gen:level=4
type PKIList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata,omitempty"`

	// items is a list of PKI resources
	Items []PKI `json:"items"`
}
