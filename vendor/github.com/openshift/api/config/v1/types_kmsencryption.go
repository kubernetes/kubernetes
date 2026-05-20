package v1

// KMSPluginConfig defines the configuration for the KMS instance
// that will be used with KMS encryption
// +kubebuilder:validation:XValidation:rule="self.type == 'Vault' ? has(self.vault) : !has(self.vault)",message="vault config is required when kms provider type is Vault, and forbidden otherwise"
// +union
type KMSPluginConfig struct {
	// type defines the kind of platform for the KMS provider.
	// Allowed values are Vault.
	// When set to Vault, the plugin connects to a HashiCorp Vault server for key management.
	//
	// +unionDiscriminator
	// +required
	Type KMSProviderType `json:"type"`

	// vault defines the configuration for the Vault KMS plugin.
	// The plugin connects to a Vault Enterprise server that is managed
	// by the user outside the purview of the control plane.
	// This field must be set when type is Vault, and must be unset otherwise.
	//
	// +unionMember
	// +optional
	Vault VaultKMSPluginConfig `json:"vault,omitempty,omitzero"`

	// --- TOMBSTONE ---
	// aws was a field that allowed configuring AWS KMS.
	// It was never implemented and has been removed.
	// The field name is reserved to prevent reuse.
	//
	// +optional
	// AWS *AWSKMSConfig `json:"aws,omitempty"`
}

// --- TOMBSTONE ---
// AWSKMSConfig was a type for AWS KMS configuration that was never implemented.
// The type name is reserved to prevent reuse.
//
// type AWSKMSConfig struct {
// 	KeyARN string `json:"keyARN"`
// 	Region string `json:"region"`
// }

// KMSProviderType is a specific supported KMS provider
// +kubebuilder:validation:Enum=Vault
type KMSProviderType string

const (
	// VaultKMSProvider represents a supported KMS provider for use with HashiCorp Vault
	VaultKMSProvider KMSProviderType = "Vault"

	// --- TOMBSTONE ---
	// AWSKMSProvider was a constant for AWS KMS support that was never implemented.
	// The constant name is reserved to prevent reuse.
	//
	// AWSKMSProvider KMSProviderType = "AWS"
)

// VaultSecretReference references a secret in the openshift-config namespace.
type VaultSecretReference struct {
	// name is the metadata.name of the referenced secret in the openshift-config namespace.
	// The name must be a valid DNS subdomain name: it must contain no more than 253 characters,
	// contain only lowercase alphanumeric characters, '-' or '.', and start and end with an alphanumeric character.
	//
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=253
	// +kubebuilder:validation:XValidation:rule="!format.dns1123Subdomain().validate(self).hasValue()",message="name must be a valid DNS subdomain name: contain no more than 253 characters, contain only lowercase alphanumeric characters, '-' or '.', and start and end with an alphanumeric character"
	// +required
	Name string `json:"name,omitempty"`
}

// VaultConfigMapReference references a ConfigMap in the openshift-config namespace.
type VaultConfigMapReference struct {
	// name is the metadata.name of the referenced ConfigMap in the openshift-config namespace.
	// The name must be a valid DNS subdomain name: it must contain no more than 253 characters,
	// contain only lowercase alphanumeric characters, '-' or '.', and start and end with an alphanumeric character.
	//
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=253
	// +kubebuilder:validation:XValidation:rule="!format.dns1123Subdomain().validate(self).hasValue()",message="name must be a valid DNS subdomain name: contain no more than 253 characters, contain only lowercase alphanumeric characters, '-' or '.', and start and end with an alphanumeric character"
	// +required
	Name string `json:"name,omitempty"`
}

// VaultAuthentication defines the authentication method used to authenticate with Vault.
// +kubebuilder:validation:XValidation:rule="self.type == 'AppRole' ? has(self.appRole) : !has(self.appRole)",message="appRole config is required when authentication type is AppRole, and forbidden otherwise"
// +union
type VaultAuthentication struct {
	// type defines the authentication method used to authenticate with Vault.
	// Allowed values are AppRole.
	// When set to AppRole, the plugin uses AppRole credentials to authenticate with Vault.
	//
	// +unionDiscriminator
	// +required
	Type VaultAuthenticationType `json:"type,omitempty"`

	// appRole defines the configuration for AppRole authentication.
	// This field must be set when type is AppRole, and must be unset otherwise.
	//
	// +unionMember
	// +optional
	AppRole VaultAppRoleAuthentication `json:"appRole,omitzero"`
}

// VaultAuthenticationType defines the authentication method type for Vault.
// +kubebuilder:validation:Enum=AppRole
type VaultAuthenticationType string

const (
	// VaultAuthenticationTypeAppRole represents AppRole authentication method.
	VaultAuthenticationTypeAppRole VaultAuthenticationType = "AppRole"
)

// VaultAppRoleAuthentication defines the configuration for AppRole authentication with Vault.
type VaultAppRoleAuthentication struct {
	// secret references a secret in the openshift-config namespace containing
	// the AppRole credentials used to authenticate with Vault.
	// The secret must contain two keys: "role-id" for the AppRole Role ID and "secret-id" for the AppRole Secret ID.
	//
	// +required
	Secret VaultSecretReference `json:"secret,omitzero"`
}

// VaultKMSPluginConfig defines the KMS plugin configuration specific to Vault KMS
type VaultKMSPluginConfig struct {
	// kmsPluginImage specifies the container image for the HashiCorp Vault KMS plugin.
	//
	// The image must be a fully qualified OCI image pull spec with a SHA256 digest.
	// The format is: host[:port][/namespace]/name@sha256:<digest>
	// where the digest must be 64 characters long and consist only of lowercase hexadecimal characters, a-f and 0-9.
	// The total length must be between 75 and 447 characters.
	//
	// Short names (e.g., "vault-plugin" or "hashicorp/vault-plugin") are not allowed.
	// The registry hostname must be included and must contain at least one dot.
	// Image tags (e.g., ":latest", ":v1.0.0") are not allowed.
	//
	// Consult the OpenShift documentation for compatible plugin versions with your cluster version,
	// then obtain the image digest for that version from HashiCorp's container registry.
	//
	// For disconnected environments, mirror the plugin image to an accessible registry
	// and reference the mirrored location with its digest.
	//
	// +kubebuilder:validation:MinLength=75
	// +kubebuilder:validation:MaxLength=447
	// +kubebuilder:validation:XValidation:rule=`(self.split('@').size() == 2 && self.split('@')[1].matches('^sha256:[a-f0-9]{64}$'))`,message="the OCI Image reference must end with a valid '@sha256:<digest>' suffix, where '<digest>' is 64 characters long"
	// +kubebuilder:validation:XValidation:rule=`(self.split('@')[0].matches('^([a-zA-Z0-9-]+\\.)+[a-zA-Z0-9-]+(:[0-9]{2,5})?(/[a-zA-Z0-9-_.]+)+$'))`,message="the OCI Image name should follow the host[:port][/namespace]/name format, resembling a valid URL without the scheme. Short names are not allowed, the registry hostname must be included."
	// +required
	KMSPluginImage string `json:"kmsPluginImage,omitempty"`

	// vaultAddress specifies the address of the HashiCorp Vault instance.
	// The value must be a valid HTTPS URL containing only scheme, host, and optional port.
	// Paths, user info, query parameters, and fragments are not allowed.
	//
	// Format: https://hostname[:port]
	// Example: https://vault.example.com:8200
	//
	// The value must be between 1 and 512 characters.
	//
	// +kubebuilder:validation:XValidation:rule="isURL(self)",message="must be a valid URL"
	// +kubebuilder:validation:XValidation:rule="isURL(self) && url(self).getScheme() == 'https'",message="must use the 'https' scheme"
	// +kubebuilder:validation:XValidation:rule="isURL(self) && (url(self).getEscapedPath() == '' || url(self).getEscapedPath() == '/')",message="must not contain a path"
	// +kubebuilder:validation:XValidation:rule="isURL(self) && url(self).getQuery() == {}",message="must not have a query"
	// +kubebuilder:validation:XValidation:rule="self.find('#(.+)$') == ''",message="must not have a fragment"
	// +kubebuilder:validation:XValidation:rule="self.find('@') == ''",message="must not have user info"
	// +kubebuilder:validation:MaxLength=512
	// +kubebuilder:validation:MinLength=1
	// +required
	VaultAddress string `json:"vaultAddress,omitempty"`

	// vaultNamespace specifies the Vault namespace where the Transit secrets engine is mounted.
	// This is only applicable for Vault Enterprise installations.
	// When this field is not set, no namespace is used.
	//
	// The value must be between 1 and 4096 characters.
	// The namespace cannot end with a forward slash, cannot contain spaces, and cannot be one of the reserved strings: root, sys, audit, auth, cubbyhole, or identity.
	//
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=4096
	// +kubebuilder:validation:XValidation:rule="!self.endsWith('/')",message="vaultNamespace cannot end with a forward slash"
	// +kubebuilder:validation:XValidation:rule="!self.contains(' ')",message="vaultNamespace cannot contain spaces"
	// +kubebuilder:validation:XValidation:rule="!(self in ['root', 'sys', 'audit', 'auth', 'cubbyhole', 'identity'])",message="vaultNamespace cannot be a reserved string (root, sys, audit, auth, cubbyhole, identity)"
	// +optional
	VaultNamespace string `json:"vaultNamespace,omitempty"`

	// tls contains the TLS configuration for connecting to the Vault server.
	// When this field is not set, system default TLS settings are used.
	// +optional
	TLS VaultTLSConfig `json:"tls,omitzero"`

	// authentication defines the authentication method used to authenticate with Vault.
	//
	// +required
	Authentication VaultAuthentication `json:"authentication,omitzero"`

	// transitMount specifies the mount path of the Vault Transit engine.
	//
	// When omitted, this means the user has no opinion and the platform is left
	// to choose a reasonable default. These defaults are subject to change over time.
	// The current default is "transit".
	//
	// The transit mount must be between 1 and 1024 characters when specified, cannot start or
	// end with a forward slash, cannot contain consecutive forward slashes, and must only contain
	// RFC 3986 unreserved characters (alphanumeric, hyphen, period, underscore, tilde) and forward
	// slashes as path separators.
	//
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=1024
	// +kubebuilder:validation:XValidation:rule="!self.startsWith('/')",message="transitMount cannot start with a forward slash"
	// +kubebuilder:validation:XValidation:rule="!self.endsWith('/')",message="transitMount cannot end with a forward slash"
	// +kubebuilder:validation:XValidation:rule="!self.contains('//')",message="transitMount cannot contain consecutive forward slashes"
	// +kubebuilder:validation:XValidation:rule="self.matches('^[a-zA-Z0-9._~/-]+$')",message="transitMount must only contain RFC 3986 unreserved characters (alphanumeric, hyphen, period, underscore, tilde) and forward slashes"
	// +optional
	TransitMount string `json:"transitMount,omitempty"`

	// transitKey specifies the name of the encryption key in Vault's Transit engine.
	// This key is used to encrypt and decrypt data.
	//
	// The transit key must be between 1 and 512 characters, cannot contain forward slashes,
	// and must only contain alphanumeric characters, hyphens, periods, and underscores.
	//
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=512
	// +kubebuilder:validation:XValidation:rule="!self.contains('/')",message="transitKey cannot contain forward slashes"
	// +kubebuilder:validation:XValidation:rule="self.matches('^[a-zA-Z0-9._-]+$')",message="transitKey must only contain alphanumeric characters, hyphens, periods, and underscores"
	// +required
	TransitKey string `json:"transitKey,omitempty"`
}

// VaultTLSConfig contains TLS configuration for connecting to Vault.
// +kubebuilder:validation:MinProperties=1
type VaultTLSConfig struct {
	// caBundle references a ConfigMap in the openshift-config namespace containing
	// the CA certificate bundle used to verify the TLS connection to the Vault server.
	// The ConfigMap must contain the CA bundle in the key "ca-bundle.crt".
	// When this field is not set, the system's trusted CA certificates are used.
	//
	// The namespace for the ConfigMap is openshift-config.
	//
	// Example ConfigMap:
	//   apiVersion: v1
	//   kind: ConfigMap
	//   metadata:
	//     name: vault-ca-bundle
	//     namespace: openshift-config
	//   data:
	//     ca-bundle.crt: |
	//       -----BEGIN CERTIFICATE-----
	//       ...
	//       -----END CERTIFICATE-----
	//
	// +optional
	CABundle VaultConfigMapReference `json:"caBundle,omitzero"`

	// serverName specifies the Server Name Indication (SNI) to use when connecting to Vault via TLS.
	// This is useful when the Vault server's hostname doesn't match its TLS certificate.
	// When this field is not set, the hostname from vaultAddress is used for SNI.
	//
	// The value must be a valid DNS hostname: it must contain no more than 253 characters,
	// contain only lowercase alphanumeric characters, '-' or '.', and start and end with an alphanumeric character.
	//
	// +kubebuilder:validation:MaxLength=253
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:XValidation:rule="!format.dns1123Subdomain().validate(self).hasValue()",message="serverName must be a valid DNS hostname: contain no more than 253 characters, contain only lowercase alphanumeric characters, '-' or '.', and start and end with an alphanumeric character"
	// +optional
	ServerName string `json:"serverName,omitempty"`
}
