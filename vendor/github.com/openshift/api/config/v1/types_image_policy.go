package v1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ImagePolicy holds namespace-wide configuration for image signature verification
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +kubebuilder:object:root=true
// +kubebuilder:resource:path=imagepolicies,scope=Namespaced
// +kubebuilder:subresource:status
// +openshift:api-approved.openshift.io=https://github.com/openshift/api/pull/2310
// +openshift:file-pattern=cvoRunLevel=0000_10,operatorName=config-operator,operatorOrdering=01
// +openshift:enable:FeatureGate=SigstoreImageVerification
// +openshift:compatibility-gen:level=1
type ImagePolicy struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata"`

	// spec holds user settable values for configuration
	// +required
	Spec ImagePolicySpec `json:"spec"`
	// status contains the observed state of the resource.
	// +optional
	Status ImagePolicyStatus `json:"status"`
}

// ImagePolicySpec is the specification of the ImagePolicy CRD.
type ImagePolicySpec struct {
	// scopes is a required field that defines the list of image identities assigned to a policy. Each item refers to a scope in a registry implementing the "Docker Registry HTTP API V2".
	// Scopes matching individual images are named Docker references in the fully expanded form, either using a tag or digest. For example, docker.io/library/busybox:latest (not busybox:latest).
	// More general scopes are prefixes of individual-image scopes, and specify a repository (by omitting the tag or digest), a repository
	// namespace, or a registry host (by only specifying the host name and possibly a port number) or a wildcard expression starting with `*.`, for matching all subdomains (not including a port number).
	// Wildcards are only supported for subdomain matching, and may not be used in the middle of the host, i.e.  *.example.com is a valid case, but example*.*.com is not.
	// This support no more than 256 scopes in one object. If multiple scopes match a given image, only the policy requirements for the most specific scope apply. The policy requirements for more general scopes are ignored.
	// In addition to setting a policy appropriate for your own deployed applications, make sure that a policy on the OpenShift image repositories
	// quay.io/openshift-release-dev/ocp-release, quay.io/openshift-release-dev/ocp-v4.0-art-dev (or on a more general scope) allows deployment of the OpenShift images required for cluster operation.
	// If a scope is configured in both the ClusterImagePolicy and the ImagePolicy, or if the scope in ImagePolicy is nested under one of the scopes from the ClusterImagePolicy, only the policy from the ClusterImagePolicy will be applied.
	// For additional details about the format, please refer to the document explaining the docker transport field,
	// which can be found at: https://github.com/containers/image/blob/main/docs/containers-policy.json.5.md#docker
	// +required
	// +kubebuilder:validation:MaxItems=256
	// +listType=set
	Scopes []ImageScope `json:"scopes"`
	// policy is a required field that contains configuration to allow scopes to be verified, and defines how
	// images not matching the verification policy will be treated.
	// +required
	Policy Policy `json:"policy"`
}

// +kubebuilder:validation:XValidation:rule="size(self.split('/')[0].split('.')) == 1 ? self.split('/')[0].split('.')[0].split(':')[0] == 'localhost' : true",message="invalid image scope format, scope must contain a fully qualified domain name or 'localhost'"
// +kubebuilder:validation:XValidation:rule=`self.contains('*') ? self.matches('^\\*(?:\\.(?:[a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]))+$') : true`,message="invalid image scope with wildcard, a wildcard can only be at the start of the domain and is only supported for subdomain matching, not path matching"
// +kubebuilder:validation:XValidation:rule=`!self.contains('*') ? self.matches('^((((?:[a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9])(?:\\.(?:[a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]))+(?::[0-9]+)?)|(localhost(?::[0-9]+)?))(?:(?:/[a-z0-9]+(?:(?:(?:[._]|__|[-]*)[a-z0-9]+)+)?)+)?)(?::([\\w][\\w.-]{0,127}))?(?:@([A-Za-z][A-Za-z0-9]*(?:[-_+.][A-Za-z][A-Za-z0-9]*)*[:][[:xdigit:]]{32,}))?$') : true`,message="invalid repository namespace or image specification in the image scope"
// +kubebuilder:validation:MaxLength=512
type ImageScope string

// Policy defines the verification policy for the items in the scopes list.
type Policy struct {
	// rootOfTrust is a required field that defines the root of trust for verifying image signatures during retrieval.
	// This allows image consumers to specify policyType and corresponding configuration of the policy, matching how the policy was generated.
	// +required
	RootOfTrust PolicyRootOfTrust `json:"rootOfTrust"`
	// signedIdentity is an optional field specifies what image identity the signature claims about the image. This is useful when the image identity in the signature differs from the original image spec, such as when mirror registry is configured for the image scope, the signature from the mirror registry contains the image identity of the mirror instead of the original scope.
	// The required matchPolicy field specifies the approach used in the verification process to verify the identity in the signature and the actual image identity, the default matchPolicy is "MatchRepoDigestOrExact".
	// +optional
	SignedIdentity *PolicyIdentity `json:"signedIdentity,omitempty"`
}

// PolicyRootOfTrust defines the root of trust based on the selected policyType.
// +union
// +kubebuilder:validation:XValidation:rule="has(self.policyType) && self.policyType == 'PublicKey' ? has(self.publicKey) : !has(self.publicKey)",message="publicKey is required when policyType is PublicKey, and forbidden otherwise"
// +kubebuilder:validation:XValidation:rule="has(self.policyType) && self.policyType == 'FulcioCAWithRekor' ? has(self.fulcioCAWithRekor) : !has(self.fulcioCAWithRekor)",message="fulcioCAWithRekor is required when policyType is FulcioCAWithRekor, and forbidden otherwise"
// +openshift:validation:FeatureGateAwareXValidation:featureGate=SigstoreImageVerificationPKI,rule="has(self.policyType) && self.policyType == 'PKI' ? has(self.pki) : !has(self.pki)",message="pki is required when policyType is PKI, and forbidden otherwise"
type PolicyRootOfTrust struct {
	// policyType is a required field specifies the type of the policy for verification. This field must correspond to how the policy was generated.
	// Allowed values are "PublicKey", "FulcioCAWithRekor", and "PKI".
	// When set to "PublicKey", the policy relies on a sigstore publicKey and may optionally use a Rekor verification.
	// When set to "FulcioCAWithRekor", the policy is based on the Fulcio certification and incorporates a Rekor verification.
	// When set to "PKI", the policy is based on the certificates from Bring Your Own Public Key Infrastructure (BYOPKI). This value is enabled by turning on the SigstoreImageVerificationPKI feature gate.
	// +unionDiscriminator
	// +required
	PolicyType PolicyType `json:"policyType"`
	// publicKey defines the root of trust configuration based on a sigstore public key. Optionally include a Rekor public key for Rekor verification.
	// publicKey is required when policyType is PublicKey, and forbidden otherwise.
	// +optional
	PublicKey *PublicKey `json:"publicKey,omitempty"`
	// fulcioCAWithRekor defines the root of trust configuration based on the Fulcio certificate and the Rekor public key.
	// fulcioCAWithRekor is required when policyType is FulcioCAWithRekor, and forbidden otherwise
	// For more information about Fulcio and Rekor, please refer to the document at:
	// https://github.com/sigstore/fulcio and https://github.com/sigstore/rekor
	// +optional
	FulcioCAWithRekor *FulcioCAWithRekor `json:"fulcioCAWithRekor,omitempty"`
	// pki defines the root of trust configuration based on Bring Your Own Public Key Infrastructure (BYOPKI) Root CA(s) and corresponding intermediate certificates.
	// pki is required when policyType is PKI, and forbidden otherwise.
	// +optional
	// +openshift:enable:FeatureGate=SigstoreImageVerificationPKI
	PKI *PKI `json:"pki,omitempty"`
}

// +openshift:validation:FeatureGateAwareEnum:featureGate="",enum=PublicKey;FulcioCAWithRekor
// +openshift:validation:FeatureGateAwareEnum:featureGate=SigstoreImageVerificationPKI,enum=PublicKey;FulcioCAWithRekor;PKI
type PolicyType string

const (
	PublicKeyRootOfTrust         PolicyType = "PublicKey"
	FulcioCAWithRekorRootOfTrust PolicyType = "FulcioCAWithRekor"
	PKIRootOfTrust               PolicyType = "PKI"
)

// PublicKey defines the root of trust based on a sigstore public key.
type PublicKey struct {
	// keyData is a required field contains inline base64-encoded data for the PEM format public key.
	// keyData must be at most 8192 characters.
	// +required
	// +kubebuilder:validation:MaxLength=8192
	// +kubebuilder:validation:MinLength=68
	// +kubebuilder:validation:XValidation:rule="string(self).startsWith('-----BEGIN PUBLIC KEY-----')",message="the keyData must start with base64 encoding of '-----BEGIN PUBLIC KEY-----'."
	// +kubebuilder:validation:XValidation:rule="string(self).endsWith('-----END PUBLIC KEY-----\\n') || string(self).endsWith('-----END PUBLIC KEY-----')",message="the keyData must end with base64 encoding of '-----END PUBLIC KEY-----'."
	KeyData []byte `json:"keyData"`
	// rekorKeyData is an optional field contains inline base64-encoded data for the PEM format from the Rekor public key.
	// rekorKeyData must be at most 8192 characters.
	// +optional
	// +kubebuilder:validation:MaxLength=8192
	// +kubebuilder:validation:XValidation:rule="string(self).startsWith('-----BEGIN PUBLIC KEY-----')",message="the rekorKeyData must start with base64 encoding of '-----BEGIN PUBLIC KEY-----'."
	// +kubebuilder:validation:XValidation:rule="string(self).endsWith('-----END PUBLIC KEY-----\\n') || string(self).endsWith('-----END PUBLIC KEY-----')",message="the rekorKeyData must end with base64 encoding of '-----END PUBLIC KEY-----'."
	RekorKeyData []byte `json:"rekorKeyData,omitempty"`
}

// FulcioCAWithRekor defines the root of trust based on the Fulcio certificate and the Rekor public key.
type FulcioCAWithRekor struct {
	// fulcioCAData is a required field contains inline base64-encoded data for the PEM format fulcio CA.
	// fulcioCAData must be at most 8192 characters.
	// +required
	// +kubebuilder:validation:MaxLength=8192
	// +kubebuilder:validation:XValidation:rule="string(self).startsWith('-----BEGIN CERTIFICATE-----')",message="the fulcioCAData must start with base64 encoding of '-----BEGIN CERTIFICATE-----'."
	// +kubebuilder:validation:XValidation:rule="string(self).endsWith('-----END CERTIFICATE-----\\n') || string(self).endsWith('-----END CERTIFICATE-----')",message="the fulcioCAData must end with base64 encoding of '-----END CERTIFICATE-----'."
	FulcioCAData []byte `json:"fulcioCAData"`
	// rekorKeyData is a required field contains inline base64-encoded data for the PEM format from the Rekor public key.
	// rekorKeyData must be at most 8192 characters.
	// +required
	// +kubebuilder:validation:MaxLength=8192
	// +kubebuilder:validation:XValidation:rule="string(self).startsWith('-----BEGIN PUBLIC KEY-----')",message="the rekorKeyData must start with base64 encoding of '-----BEGIN PUBLIC KEY-----'."
	// +kubebuilder:validation:XValidation:rule="string(self).endsWith('-----END PUBLIC KEY-----\\n') || string(self).endsWith('-----END PUBLIC KEY-----')",message="the rekorKeyData must end with base64 encoding of '-----END PUBLIC KEY-----'."
	RekorKeyData []byte `json:"rekorKeyData"`
	// fulcioSubject is a required field specifies OIDC issuer and the email of the Fulcio authentication configuration.
	// +required
	FulcioSubject PolicyFulcioSubject `json:"fulcioSubject"`
}

// PolicyFulcioSubject defines the OIDC issuer and the email of the Fulcio authentication configuration.
type PolicyFulcioSubject struct {
	// oidcIssuer is a required filed contains the expected OIDC issuer. The oidcIssuer must be a valid URL and at most 2048 characters in length.
	// It will be verified that the Fulcio-issued certificate contains a (Fulcio-defined) certificate extension pointing at this OIDC issuer URL.
	// When Fulcio issues certificates, it includes a value based on an URL inside the client-provided ID token.
	// Example: "https://expected.OIDC.issuer/"
	// +required
	// +kubebuilder:validation:MaxLength=2048
	// +kubebuilder:validation:XValidation:rule="isURL(self)",message="oidcIssuer must be a valid URL"
	OIDCIssuer string `json:"oidcIssuer"`
	// signedEmail is a required field holds the email address that the Fulcio certificate is issued for.
	// The signedEmail must be a valid email address and at most 320 characters in length.
	// Example: "expected-signing-user@example.com"
	// +required
	// +kubebuilder:validation:MaxLength=320
	// +kubebuilder:validation:XValidation:rule=`self.matches('^\\S+@\\S+$')`,message="invalid email address"
	SignedEmail string `json:"signedEmail"`
}

// PKI defines the root of trust based on Root CA(s) and corresponding intermediate certificates.
type PKI struct {
	// caRootsData contains base64-encoded data of a certificate bundle PEM file, which contains one or more CA roots in the PEM format. The total length of the data must not exceed 8192 characters.
	// +required
	// +kubebuilder:validation:MaxLength=8192
	// +kubebuilder:validation:MinLength=72
	// +kubebuilder:validation:XValidation:rule="string(self).startsWith('-----BEGIN CERTIFICATE-----')",message="the caRootsData must start with base64 encoding of '-----BEGIN CERTIFICATE-----'."
	// +kubebuilder:validation:XValidation:rule="string(self).endsWith('-----END CERTIFICATE-----\\n') || string(self).endsWith('-----END CERTIFICATE-----')",message="the caRootsData must end with base64 encoding of '-----END CERTIFICATE-----'."
	// +kubebuilder:validation:XValidation:rule="string(self).findAll('-----BEGIN CERTIFICATE-----').size() == string(self).findAll('-----END CERTIFICATE-----').size()",message="caRootsData must be base64 encoding of valid PEM format data contain the same number of '-----BEGIN CERTIFICATE-----' and '-----END CERTIFICATE-----' markers."
	CertificateAuthorityRootsData []byte `json:"caRootsData"`
	// caIntermediatesData contains base64-encoded data of a certificate bundle PEM file, which contains one or more intermediate certificates in the PEM format. The total length of the data must not exceed 8192 characters.
	// caIntermediatesData requires caRootsData to be set.
	// +optional
	// +kubebuilder:validation:XValidation:rule="string(self).startsWith('-----BEGIN CERTIFICATE-----')",message="the caIntermediatesData must start with base64 encoding of '-----BEGIN CERTIFICATE-----'."
	// +kubebuilder:validation:XValidation:rule="string(self).endsWith('-----END CERTIFICATE-----\\n') || string(self).endsWith('-----END CERTIFICATE-----')",message="the caIntermediatesData must end with base64 encoding of '-----END CERTIFICATE-----'."
	// +kubebuilder:validation:XValidation:rule="string(self).findAll('-----BEGIN CERTIFICATE-----').size() == string(self).findAll('-----END CERTIFICATE-----').size()",message="caIntermediatesData must be base64 encoding of valid PEM format data contain the same number of '-----BEGIN CERTIFICATE-----' and '-----END CERTIFICATE-----' markers."
	// +kubebuilder:validation:MaxLength=8192
	// +kubebuilder:validation:MinLength=72
	CertificateAuthorityIntermediatesData []byte `json:"caIntermediatesData,omitempty"`

	// pkiCertificateSubject defines the requirements imposed on the subject to which the certificate was issued.
	// +required
	PKICertificateSubject PKICertificateSubject `json:"pkiCertificateSubject"`
}

// PKICertificateSubject defines the requirements imposed on the subject to which the certificate was issued.
// +kubebuilder:validation:XValidation:rule="has(self.email) || has(self.hostname)", message="at least one of email or hostname must be set in pkiCertificateSubject"
// +openshift:enable:FeatureGate=SigstoreImageVerificationPKI
type PKICertificateSubject struct {
	// email specifies the expected email address imposed on the subject to which the certificate was issued, and must match the email address listed in the Subject Alternative Name (SAN) field of the certificate.
	// The email must be a valid email address and at most 320 characters in length.
	// +optional
	// +kubebuilder:validation:MaxLength:=320
	// +kubebuilder:validation:XValidation:rule=`self.matches('^\\S+@\\S+$')`,message="invalid email address"
	Email string `json:"email,omitempty"`
	// hostname specifies the expected hostname imposed on the subject to which the certificate was issued, and it must match the hostname listed in the Subject Alternative Name (SAN) DNS field of the certificate.
	// The hostname must be a valid dns 1123 subdomain name, optionally prefixed by '*.', and at most 253 characters in length.
	// It must consist only of lowercase alphanumeric characters, hyphens, periods and the optional preceding asterisk.
	// +optional
	// +kubebuilder:validation:MaxLength:=253
	// +kubebuilder:validation:XValidation:rule="self.startsWith('*.') ? !format.dns1123Subdomain().validate(self.replace('*.', '', 1)).hasValue() : !format.dns1123Subdomain().validate(self).hasValue()",message="hostname must be a valid dns 1123 subdomain name, optionally prefixed by '*.'. It must consist only of lowercase alphanumeric characters, hyphens, periods and the optional preceding asterisk."
	Hostname string `json:"hostname,omitempty"`
}

// PolicyIdentity defines image identity the signature claims about the image. When omitted, the default matchPolicy is "MatchRepoDigestOrExact".
// +kubebuilder:validation:XValidation:rule="(has(self.matchPolicy) && self.matchPolicy == 'ExactRepository') ? has(self.exactRepository) : !has(self.exactRepository)",message="exactRepository is required when matchPolicy is ExactRepository, and forbidden otherwise"
// +kubebuilder:validation:XValidation:rule="(has(self.matchPolicy) && self.matchPolicy == 'RemapIdentity') ? has(self.remapIdentity) : !has(self.remapIdentity)",message="remapIdentity is required when matchPolicy is RemapIdentity, and forbidden otherwise"
// +union
type PolicyIdentity struct {
	// matchPolicy is a required filed specifies matching strategy to verify the image identity in the signature against the image scope.
	// Allowed values are "MatchRepoDigestOrExact", "MatchRepository", "ExactRepository", "RemapIdentity". When omitted, the default value is "MatchRepoDigestOrExact".
	// When set to "MatchRepoDigestOrExact", the identity in the signature must be in the same repository as the image identity if the image identity is referenced by a digest. Otherwise, the identity in the signature must be the same as the image identity.
	// When set to "MatchRepository", the identity in the signature must be in the same repository as the image identity.
	// When set to "ExactRepository", the exactRepository must be specified. The identity in the signature must be in the same repository as a specific identity specified by "repository".
	// When set to "RemapIdentity", the remapIdentity must be specified. The signature must be in the same as the remapped image identity. Remapped image identity is obtained by replacing the "prefix" with the specified “signedPrefix” if the the image identity matches the specified remapPrefix.
	// +unionDiscriminator
	// +required
	MatchPolicy IdentityMatchPolicy `json:"matchPolicy"`
	// exactRepository specifies the repository that must be exactly matched by the identity in the signature.
	// exactRepository is required if matchPolicy is set to "ExactRepository". It is used to verify that the signature claims an identity matching this exact repository, rather than the original image identity.
	// +optional
	PolicyMatchExactRepository *PolicyMatchExactRepository `json:"exactRepository,omitempty"`
	// remapIdentity specifies the prefix remapping rule for verifying image identity.
	// remapIdentity is required if matchPolicy is set to "RemapIdentity". It is used to verify that the signature claims a different registry/repository prefix than the original image.
	// +optional
	PolicyMatchRemapIdentity *PolicyMatchRemapIdentity `json:"remapIdentity,omitempty"`
}

// +kubebuilder:validation:MaxLength=512
// +kubebuilder:validation:XValidation:rule=`self.matches('.*:([\\w][\\w.-]{0,127})$')? self.matches('^(localhost:[0-9]+)$'): true`,message="invalid repository or prefix in the signedIdentity, should not include the tag or digest"
// +kubebuilder:validation:XValidation:rule=`self.matches('^(((?:[a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9])(?:\\.(?:[a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]))+(?::[0-9]+)?)|(localhost(?::[0-9]+)?))(?:(?:/[a-z0-9]+(?:(?:(?:[._]|__|[-]*)[a-z0-9]+)+)?)+)?$')`,message="invalid repository or prefix in the signedIdentity. The repository or prefix must starts with 'localhost' or a valid '.' separated domain. If contains registry paths, the path component names must start with at least one letter or number, with following parts able to be separated by one period, one or two underscore and multiple dashes."
type IdentityRepositoryPrefix string

type PolicyMatchExactRepository struct {
	// repository is the reference of the image identity to be matched.
	// repository is required if matchPolicy is set to "ExactRepository".
	// The value should be a repository name (by omitting the tag or digest) in a registry implementing the "Docker Registry HTTP API V2". For example, docker.io/library/busybox
	// +required
	Repository IdentityRepositoryPrefix `json:"repository"`
}

type PolicyMatchRemapIdentity struct {
	// prefix is required if matchPolicy is set to "RemapIdentity".
	// prefix is the prefix of the image identity to be matched.
	// If the image identity matches the specified prefix, that prefix is replaced by the specified “signedPrefix” (otherwise it is used as unchanged and no remapping takes place).
	// This is useful when verifying signatures for a mirror of some other repository namespace that preserves the vendor’s repository structure.
	// The prefix and signedPrefix values can be either host[:port] values (matching exactly the same host[:port], string), repository namespaces,
	// or repositories (i.e. they must not contain tags/digests), and match as prefixes of the fully expanded form.
	// For example, docker.io/library/busybox (not busybox) to specify that single repository, or docker.io/library (not an empty string) to specify the parent namespace of docker.io/library/busybox.
	// +required
	Prefix IdentityRepositoryPrefix `json:"prefix"`
	// signedPrefix is required if matchPolicy is set to "RemapIdentity".
	// signedPrefix is the prefix of the image identity to be matched in the signature. The format is the same as "prefix". The values can be either host[:port] values (matching exactly the same host[:port], string), repository namespaces,
	// or repositories (i.e. they must not contain tags/digests), and match as prefixes of the fully expanded form.
	// For example, docker.io/library/busybox (not busybox) to specify that single repository, or docker.io/library (not an empty string) to specify the parent namespace of docker.io/library/busybox.
	// +required
	SignedPrefix IdentityRepositoryPrefix `json:"signedPrefix"`
}

// IdentityMatchPolicy defines the type of matching for "matchPolicy".
// +kubebuilder:validation:Enum=MatchRepoDigestOrExact;MatchRepository;ExactRepository;RemapIdentity
type IdentityMatchPolicy string

const (
	IdentityMatchPolicyMatchRepoDigestOrExact IdentityMatchPolicy = "MatchRepoDigestOrExact"
	IdentityMatchPolicyMatchRepository        IdentityMatchPolicy = "MatchRepository"
	IdentityMatchPolicyExactRepository        IdentityMatchPolicy = "ExactRepository"
	IdentityMatchPolicyRemapIdentity          IdentityMatchPolicy = "RemapIdentity"
)

// +k8s:deepcopy-gen=true
type ImagePolicyStatus struct {
	// conditions provide details on the status of this API Resource.
	// condition type 'Pending' indicates that the customer resource contains a policy that cannot take effect. It is either overwritten by a global policy or the image scope is not valid.
	// +kubebuilder:validation:MaxItems=8
	// +kubebuilder:validation:MinItems=1
	// +listType=map
	// +listMapKey=type
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ImagePolicyList is a list of ImagePolicy resources
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type ImagePolicyList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +required
	metav1.ListMeta `json:"metadata"`

	// items is a list of ImagePolicies
	// +kubebuilder:validation:MaxItems=1000
	// +required
	Items []ImagePolicy `json:"items"`
}

const (
	// ImagePolicyPending indicates that the customer resource contains a policy that cannot take effect. It is either overwritten by a global policy or the image scope is not valid.
	ImagePolicyPending = "Pending"
	// ImagePolicyApplied indicates that the policy has been applied
	ImagePolicyApplied = "Applied"
)
