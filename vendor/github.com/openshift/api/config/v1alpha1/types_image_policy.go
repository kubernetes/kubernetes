package v1alpha1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ImagePolicy holds namespace-wide configuration for image signature verification
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +kubebuilder:object:root=true
// +kubebuilder:resource:path=imagepolicies,scope=Namespaced
// +kubebuilder:subresource:status
// +openshift:api-approved.openshift.io=https://github.com/openshift/api/pull/1457
// +openshift:file-pattern=cvoRunLevel=0000_10,operatorName=config-operator,operatorOrdering=01
// +openshift:enable:FeatureGate=ImagePolicy
// +openshift:compatibility-gen:level=4
type ImagePolicy struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec holds user settable values for configuration
	// +kubebuilder:validation:Required
	Spec ImagePolicySpec `json:"spec"`
	// status contains the observed state of the resource.
	// +optional
	Status ImagePolicyStatus `json:"status,omitempty"`
}

// ImagePolicySpec is the specification of the ImagePolicy CRD.
type ImagePolicySpec struct {
	// scopes defines the list of image identities assigned to a policy. Each item refers to a scope in a registry implementing the "Docker Registry HTTP API V2".
	// Scopes matching individual images are named Docker references in the fully expanded form, either using a tag or digest. For example, docker.io/library/busybox:latest (not busybox:latest).
	// More general scopes are prefixes of individual-image scopes, and specify a repository (by omitting the tag or digest), a repository
	// namespace, or a registry host (by only specifying the host name and possibly a port number) or a wildcard expression starting with `*.`, for matching all subdomains (not including a port number).
	// Wildcards are only supported for subdomain matching, and may not be used in the middle of the host, i.e.  *.example.com is a valid case, but example*.*.com is not.
	// Please be aware that the scopes should not be nested under the repositories of OpenShift Container Platform images.
	// If configured, the policies for OpenShift Container Platform repositories will not be in effect.
	// For additional details about the format, please refer to the document explaining the docker transport field,
	// which can be found at: https://github.com/containers/image/blob/main/docs/containers-policy.json.5.md#docker
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MaxItems=256
	// +listType=set
	Scopes []ImageScope `json:"scopes"`
	// policy contains configuration to allow scopes to be verified, and defines how
	// images not matching the verification policy will be treated.
	// +kubebuilder:validation:Required
	Policy Policy `json:"policy"`
}

// +kubebuilder:validation:XValidation:rule="size(self.split('/')[0].split('.')) == 1 ? self.split('/')[0].split('.')[0].split(':')[0] == 'localhost' : true",message="invalid image scope format, scope must contain a fully qualified domain name or 'localhost'"
// +kubebuilder:validation:XValidation:rule=`self.contains('*') ? self.matches('^\\*(?:\\.(?:[a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]))+$') : true`,message="invalid image scope with wildcard, a wildcard can only be at the start of the domain and is only supported for subdomain matching, not path matching"
// +kubebuilder:validation:XValidation:rule=`!self.contains('*') ? self.matches('^((((?:[a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9])(?:\\.(?:[a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]))+(?::[0-9]+)?)|(localhost(?::[0-9]+)?))(?:(?:/[a-z0-9]+(?:(?:(?:[._]|__|[-]*)[a-z0-9]+)+)?)+)?)(?::([\\w][\\w.-]{0,127}))?(?:@([A-Za-z][A-Za-z0-9]*(?:[-_+.][A-Za-z][A-Za-z0-9]*)*[:][[:xdigit:]]{32,}))?$') : true`,message="invalid repository namespace or image specification in the image scope"
// +kubebuilder:validation:MaxLength=512
type ImageScope string

// Policy defines the verification policy for the items in the scopes list.
type Policy struct {
	// rootOfTrust specifies the root of trust for the policy.
	// +kubebuilder:validation:Required
	RootOfTrust PolicyRootOfTrust `json:"rootOfTrust"`
	// signedIdentity specifies what image identity the signature claims about the image. The required matchPolicy field specifies the approach used in the verification process to verify the identity in the signature and the actual image identity, the default matchPolicy is "MatchRepoDigestOrExact".
	// +optional
	SignedIdentity PolicyIdentity `json:"signedIdentity,omitempty"`
}

// PolicyRootOfTrust defines the root of trust based on the selected policyType.
// +union
// +kubebuilder:validation:XValidation:rule="has(self.policyType) && self.policyType == 'PublicKey' ? has(self.publicKey) : !has(self.publicKey)",message="publicKey is required when policyType is PublicKey, and forbidden otherwise"
// +kubebuilder:validation:XValidation:rule="has(self.policyType) && self.policyType == 'FulcioCAWithRekor' ? has(self.fulcioCAWithRekor) : !has(self.fulcioCAWithRekor)",message="fulcioCAWithRekor is required when policyType is FulcioCAWithRekor, and forbidden otherwise"
type PolicyRootOfTrust struct {
	// policyType serves as the union's discriminator. Users are required to assign a value to this field, choosing one of the policy types that define the root of trust.
	// "PublicKey" indicates that the policy relies on a sigstore publicKey and may optionally use a Rekor verification.
	// "FulcioCAWithRekor" indicates that the policy is based on the Fulcio certification and incorporates a Rekor verification.
	// +unionDiscriminator
	// +kubebuilder:validation:Required
	PolicyType PolicyType `json:"policyType"`
	// publicKey defines the root of trust based on a sigstore public key.
	// +optional
	PublicKey *PublicKey `json:"publicKey,omitempty"`
	// fulcioCAWithRekor defines the root of trust based on the Fulcio certificate and the Rekor public key.
	// For more information about Fulcio and Rekor, please refer to the document at:
	// https://github.com/sigstore/fulcio and https://github.com/sigstore/rekor
	// +optional
	FulcioCAWithRekor *FulcioCAWithRekor `json:"fulcioCAWithRekor,omitempty"`
}

// +kubebuilder:validation:Enum=PublicKey;FulcioCAWithRekor
type PolicyType string

const (
	PublicKeyRootOfTrust         PolicyType = "PublicKey"
	FulcioCAWithRekorRootOfTrust PolicyType = "FulcioCAWithRekor"
)

// PublicKey defines the root of trust based on a sigstore public key.
type PublicKey struct {
	// keyData contains inline base64-encoded data for the PEM format public key.
	// KeyData must be at most 8192 characters.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MaxLength=8192
	KeyData []byte `json:"keyData"`
	// rekorKeyData contains inline base64-encoded data for the PEM format from the Rekor public key.
	// rekorKeyData must be at most 8192 characters.
	// +optional
	// +kubebuilder:validation:MaxLength=8192
	RekorKeyData []byte `json:"rekorKeyData,omitempty"`
}

// FulcioCAWithRekor defines the root of trust based on the Fulcio certificate and the Rekor public key.
type FulcioCAWithRekor struct {
	// fulcioCAData contains inline base64-encoded data for the PEM format fulcio CA.
	// fulcioCAData must be at most 8192 characters.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MaxLength=8192
	FulcioCAData []byte `json:"fulcioCAData"`
	// rekorKeyData contains inline base64-encoded data for the PEM format from the Rekor public key.
	// rekorKeyData must be at most 8192 characters.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MaxLength=8192
	RekorKeyData []byte `json:"rekorKeyData"`
	// fulcioSubject specifies OIDC issuer and the email of the Fulcio authentication configuration.
	// +kubebuilder:validation:Required
	FulcioSubject PolicyFulcioSubject `json:"fulcioSubject,omitempty"`
}

// PolicyFulcioSubject defines the OIDC issuer and the email of the Fulcio authentication configuration.
type PolicyFulcioSubject struct {
	// oidcIssuer contains the expected OIDC issuer. It will be verified that the Fulcio-issued certificate contains a (Fulcio-defined) certificate extension pointing at this OIDC issuer URL. When Fulcio issues certificates, it includes a value based on an URL inside the client-provided ID token.
	// Example: "https://expected.OIDC.issuer/"
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:XValidation:rule="isURL(self)",message="oidcIssuer must be a valid URL"
	OIDCIssuer string `json:"oidcIssuer"`
	// signedEmail holds the email address the the Fulcio certificate is issued for.
	// Example: "expected-signing-user@example.com"
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:XValidation:rule=`self.matches('^\\S+@\\S+$')`,message="invalid email address"
	SignedEmail string `json:"signedEmail"`
}

// PolicyIdentity defines image identity the signature claims about the image. When omitted, the default matchPolicy is "MatchRepoDigestOrExact".
// +kubebuilder:validation:XValidation:rule="(has(self.matchPolicy) && self.matchPolicy == 'ExactRepository') ? has(self.exactRepository) : !has(self.exactRepository)",message="exactRepository is required when matchPolicy is ExactRepository, and forbidden otherwise"
// +kubebuilder:validation:XValidation:rule="(has(self.matchPolicy) && self.matchPolicy == 'RemapIdentity') ? has(self.remapIdentity) : !has(self.remapIdentity)",message="remapIdentity is required when matchPolicy is RemapIdentity, and forbidden otherwise"
// +union
type PolicyIdentity struct {
	// matchPolicy sets the type of matching to be used.
	// Valid values are "MatchRepoDigestOrExact", "MatchRepository", "ExactRepository", "RemapIdentity". When omitted, the default value is "MatchRepoDigestOrExact".
	// If set matchPolicy to ExactRepository, then the exactRepository must be specified.
	// If set matchPolicy to RemapIdentity, then the remapIdentity must be specified.
	// "MatchRepoDigestOrExact" means that the identity in the signature must be in the same repository as the image identity if the image identity is referenced by a digest. Otherwise, the identity in the signature must be the same as the image identity.
	// "MatchRepository" means that the identity in the signature must be in the same repository as the image identity.
	// "ExactRepository" means that the identity in the signature must be in the same repository as a specific identity specified by "repository".
	// "RemapIdentity" means that the signature must be in the same as the remapped image identity. Remapped image identity is obtained by replacing the "prefix" with the specified “signedPrefix” if the the image identity matches the specified remapPrefix.
	// +unionDiscriminator
	// +kubebuilder:validation:Required
	MatchPolicy IdentityMatchPolicy `json:"matchPolicy"`
	// exactRepository is required if matchPolicy is set to "ExactRepository".
	// +optional
	PolicyMatchExactRepository *PolicyMatchExactRepository `json:"exactRepository,omitempty"`
	// remapIdentity is required if matchPolicy is set to "RemapIdentity".
	// +optional
	PolicyMatchRemapIdentity *PolicyMatchRemapIdentity `json:"remapIdentity,omitempty"`
}

// +kubebuilder:validation:MaxLength=512
// +kubebuilder:validation:XValidation:rule=`self.matches('.*:([\\w][\\w.-]{0,127})$')? self.matches('^(localhost:[0-9]+)$'): true`,message="invalid repository or prefix in the signedIdentity, should not include the tag or digest"
// +kubebuilder:validation:XValidation:rule=`self.matches('^(((?:[a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9])(?:\\.(?:[a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]))+(?::[0-9]+)?)|(localhost(?::[0-9]+)?))(?:(?:/[a-z0-9]+(?:(?:(?:[._]|__|[-]*)[a-z0-9]+)+)?)+)?$')`,message="invalid repository or prefix in the signedIdentity"
type IdentityRepositoryPrefix string

type PolicyMatchExactRepository struct {
	// repository is the reference of the image identity to be matched.
	// The value should be a repository name (by omitting the tag or digest) in a registry implementing the "Docker Registry HTTP API V2". For example, docker.io/library/busybox
	// +kubebuilder:validation:Required
	Repository IdentityRepositoryPrefix `json:"repository"`
}

type PolicyMatchRemapIdentity struct {
	// prefix is the prefix of the image identity to be matched.
	// If the image identity matches the specified prefix, that prefix is replaced by the specified “signedPrefix” (otherwise it is used as unchanged and no remapping takes place).
	// This useful when verifying signatures for a mirror of some other repository namespace that preserves the vendor’s repository structure.
	// The prefix and signedPrefix values can be either host[:port] values (matching exactly the same host[:port], string), repository namespaces,
	// or repositories (i.e. they must not contain tags/digests), and match as prefixes of the fully expanded form.
	// For example, docker.io/library/busybox (not busybox) to specify that single repository, or docker.io/library (not an empty string) to specify the parent namespace of docker.io/library/busybox.
	// +kubebuilder:validation:Required
	Prefix IdentityRepositoryPrefix `json:"prefix"`
	// signedPrefix is the prefix of the image identity to be matched in the signature. The format is the same as "prefix". The values can be either host[:port] values (matching exactly the same host[:port], string), repository namespaces,
	// or repositories (i.e. they must not contain tags/digests), and match as prefixes of the fully expanded form.
	// For example, docker.io/library/busybox (not busybox) to specify that single repository, or docker.io/library (not an empty string) to specify the parent namespace of docker.io/library/busybox.
	// +kubebuilder:validation:Required
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
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ImagePolicyList is a list of ImagePolicy resources
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +openshift:compatibility-gen:level=4
type ImagePolicyList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata"`

	Items []ImagePolicy `json:"items"`
}

const (
	// ImagePolicyPending indicates that the customer resource contains a policy that cannot take effect. It is either overwritten by a global policy or the image scope is not valid.
	ImagePolicyPending = "Pending"
	// ImagePolicyApplied indicates that the policy has been applied
	ImagePolicyApplied = "Applied"
)
