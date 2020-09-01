package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// APIServer holds configuration (like serving certificates, client CA and CORS domains)
// shared by all API servers in the system, among them especially kube-apiserver
// and openshift-apiserver. The canonical name of an instance is 'cluster'.
type APIServer struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`
	// spec holds user settable values for configuration
	// +kubebuilder:validation:Required
	// +required
	Spec APIServerSpec `json:"spec"`
	// status holds observed values from the cluster. They may not be overridden.
	// +optional
	Status APIServerStatus `json:"status"`
}

type APIServerSpec struct {
	// servingCert is the TLS cert info for serving secure traffic. If not specified, operator managed certificates
	// will be used for serving secure traffic.
	// +optional
	ServingCerts APIServerServingCerts `json:"servingCerts"`
	// clientCA references a ConfigMap containing a certificate bundle for the signers that will be recognized for
	// incoming client certificates in addition to the operator managed signers. If this is empty, then only operator managed signers are valid.
	// You usually only have to set this if you have your own PKI you wish to honor client certificates from.
	// The ConfigMap must exist in the openshift-config namespace and contain the following required fields:
	// - ConfigMap.Data["ca-bundle.crt"] - CA bundle.
	// +optional
	ClientCA ConfigMapNameReference `json:"clientCA"`
	// additionalCORSAllowedOrigins lists additional, user-defined regular expressions describing hosts for which the
	// API server allows access using the CORS headers. This may be needed to access the API and the integrated OAuth
	// server from JavaScript applications.
	// The values are regular expressions that correspond to the Golang regular expression language.
	// +optional
	AdditionalCORSAllowedOrigins []string `json:"additionalCORSAllowedOrigins,omitempty"`
	// encryption allows the configuration of encryption of resources at the datastore layer.
	// +optional
	Encryption APIServerEncryption `json:"encryption"`
	// tlsSecurityProfile specifies settings for TLS connections for externally exposed servers.
	//
	// If unset, a default (which may change between releases) is chosen. Note that only Old and
	// Intermediate profiles are currently supported, and the maximum available MinTLSVersions
	// is VersionTLS12.
	// +optional
	TLSSecurityProfile *TLSSecurityProfile `json:"tlsSecurityProfile,omitempty"`
	// audit specifies the settings for audit configuration to be applied to all OpenShift-provided
	// API servers in the cluster.
	// +optional
	// +kubebuilder:default={profile: Default}
	Audit Audit `json:"audit"`
}

// AuditProfileType defines the audit policy profile type.
// +kubebuilder:validation:Enum=Default;WriteRequestBodies;AllRequestBodies
type AuditProfileType string

const (
	// "Default" is the existing default audit configuration policy.
	AuditProfileDefaultType AuditProfileType = "Default"

	// "WriteRequestBodies" is similar to Default but it logs request and response
	// HTTP payloads for write requests (create, update, patch)
	WriteRequestBodiesAuditProfileType AuditProfileType = "WriteRequestBodies"

	// "AllRequestBodies" is similar to WriteRequestBodies, but also logs request
	// and response HTTP payloads for read requests (get, list).
	AllRequestBodiesAuditProfileType AuditProfileType = "AllRequestBodies"
)

type Audit struct {
	// profile specifies the name of the desired audit policy configuration to be deployed to
	// all OpenShift-provided API servers in the cluster.
	//
	// The following profiles are provided:
	// - Default: the existing default policy.
	// - WriteRequestBodies: like 'Default', but logs request and response HTTP payloads for
	// write requests (create, update, patch).
	// - AllRequestBodies: like 'WriteRequestBodies', but also logs request and response
	// HTTP payloads for read requests (get, list).
	//
	// If unset, the 'Default' profile is used as the default.
	// +kubebuilder:default=Default
	Profile AuditProfileType `json:"profile,omitempty"`
}

type APIServerServingCerts struct {
	// namedCertificates references secrets containing the TLS cert info for serving secure traffic to specific hostnames.
	// If no named certificates are provided, or no named certificates match the server name as understood by a client,
	// the defaultServingCertificate will be used.
	// +optional
	NamedCertificates []APIServerNamedServingCert `json:"namedCertificates,omitempty"`
}

// APIServerNamedServingCert maps a server DNS name, as understood by a client, to a certificate.
type APIServerNamedServingCert struct {
	// names is a optional list of explicit DNS names (leading wildcards allowed) that should use this certificate to
	// serve secure traffic. If no names are provided, the implicit names will be extracted from the certificates.
	// Exact names trump over wildcard names. Explicit names defined here trump over extracted implicit names.
	// +optional
	Names []string `json:"names,omitempty"`
	// servingCertificate references a kubernetes.io/tls type secret containing the TLS cert info for serving secure traffic.
	// The secret must exist in the openshift-config namespace and contain the following required fields:
	// - Secret.Data["tls.key"] - TLS private key.
	// - Secret.Data["tls.crt"] - TLS certificate.
	ServingCertificate SecretNameReference `json:"servingCertificate"`
}

type APIServerEncryption struct {
	// type defines what encryption type should be used to encrypt resources at the datastore layer.
	// When this field is unset (i.e. when it is set to the empty string), identity is implied.
	// The behavior of unset can and will change over time.  Even if encryption is enabled by default,
	// the meaning of unset may change to a different encryption type based on changes in best practices.
	//
	// When encryption is enabled, all sensitive resources shipped with the platform are encrypted.
	// This list of sensitive resources can and will change over time.  The current authoritative list is:
	//
	//   1. secrets
	//   2. configmaps
	//   3. routes.route.openshift.io
	//   4. oauthaccesstokens.oauth.openshift.io
	//   5. oauthauthorizetokens.oauth.openshift.io
	//
	// +unionDiscriminator
	// +optional
	Type EncryptionType `json:"type,omitempty"`
}

// +kubebuilder:validation:Enum="";identity;aescbc
type EncryptionType string

const (
	// identity refers to a type where no encryption is performed at the datastore layer.
	// Resources are written as-is without encryption.
	EncryptionTypeIdentity EncryptionType = "identity"

	// aescbc refers to a type where AES-CBC with PKCS#7 padding and a 32-byte key
	// is used to perform encryption at the datastore layer.
	EncryptionTypeAESCBC EncryptionType = "aescbc"
)

type APIServerStatus struct {
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

type APIServerList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`
	Items           []APIServer `json:"items"`
}
