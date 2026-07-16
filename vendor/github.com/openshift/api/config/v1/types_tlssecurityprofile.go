package v1

// TLSSecurityProfile defines the schema for a TLS security profile. This object
// is used by operators to apply TLS security settings to operands.
// +union
type TLSSecurityProfile struct {
	// type is one of Old, Intermediate, Modern or Custom. Custom provides the
	// ability to specify individual TLS security profile parameters.
	//
	// The cipher and groups lists in these profiles are based on version 5.8 of the
	// Mozilla Server Side TLS configuration guidelines.
	// See: https://ssl-config.mozilla.org/guidelines/5.8.json
	//
	// The groups are listed in suggested preference order, with the most preferred group first.
	// Note that not all platform components honor the ordering: Go-based components use Go's
	// internal preference order and treat this list as a filter of allowed groups rather than
	// an ordered preference.
	// Note that X25519MLKEM768 is a post-quantum hybrid group that is not
	// FIPS-approved and should be ignored by components running in FIPS mode.
	//
	// The profiles are intent based, so they may change over time as new ciphers are
	// developed and existing ciphers are found to be insecure. Depending on
	// precisely which ciphers are available to a process, the list may be reduced.
	//
	// +unionDiscriminator
	// +optional
	Type TLSProfileType `json:"type"`

	// old is a TLS profile for use when services need to be accessed by very old
	// clients or libraries and should be used only as a last resort.
	//
	// The supported groups list includes by default the following groups
	// in suggested preference order (ordering may not be honored by all implementations):
	// X25519MLKEM768, X25519, secp256r1, secp384r1.
	//
	// This profile is equivalent to a Custom profile specified as:
	//   minTLSVersion: VersionTLS10
	//   ciphers:
	//     - TLS_AES_128_GCM_SHA256
	//     - TLS_AES_256_GCM_SHA384
	//     - TLS_CHACHA20_POLY1305_SHA256
	//     - ECDHE-ECDSA-AES128-GCM-SHA256
	//     - ECDHE-RSA-AES128-GCM-SHA256
	//     - ECDHE-ECDSA-AES256-GCM-SHA384
	//     - ECDHE-RSA-AES256-GCM-SHA384
	//     - ECDHE-ECDSA-CHACHA20-POLY1305
	//     - ECDHE-RSA-CHACHA20-POLY1305
	//     - ECDHE-ECDSA-AES128-SHA256
	//     - ECDHE-RSA-AES128-SHA256
	//     - ECDHE-ECDSA-AES128-SHA
	//     - ECDHE-RSA-AES128-SHA
	//     - ECDHE-ECDSA-AES256-SHA384
	//     - ECDHE-RSA-AES256-SHA384
	//     - ECDHE-ECDSA-AES256-SHA
	//     - ECDHE-RSA-AES256-SHA
	//     - AES128-GCM-SHA256
	//     - AES256-GCM-SHA384
	//     - AES128-SHA256
	//     - AES256-SHA256
	//     - AES128-SHA
	//     - AES256-SHA
	//     - DES-CBC3-SHA
	//
	// +optional
	// +nullable
	Old *OldTLSProfile `json:"old,omitempty"`

	// intermediate is a TLS profile for use when you do not need compatibility with
	// legacy clients and want to remain highly secure while being compatible with
	// most clients currently in use.
	//
	// The supported groups list includes by default the following groups
	// in suggested preference order (ordering may not be honored by all implementations):
	// X25519MLKEM768, X25519, secp256r1, secp384r1.
	//
	// This profile is equivalent to a Custom profile specified as:
	//   minTLSVersion: VersionTLS12
	//   ciphers:
	//     - TLS_AES_128_GCM_SHA256
	//     - TLS_AES_256_GCM_SHA384
	//     - TLS_CHACHA20_POLY1305_SHA256
	//     - ECDHE-ECDSA-AES128-GCM-SHA256
	//     - ECDHE-RSA-AES128-GCM-SHA256
	//     - ECDHE-ECDSA-AES256-GCM-SHA384
	//     - ECDHE-RSA-AES256-GCM-SHA384
	//     - ECDHE-ECDSA-CHACHA20-POLY1305
	//     - ECDHE-RSA-CHACHA20-POLY1305
	//
	// +optional
	// +nullable
	Intermediate *IntermediateTLSProfile `json:"intermediate,omitempty"`

	// modern is a TLS security profile for use with clients that support TLS 1.3 and
	// do not need backward compatibility for older clients.
	// The supported groups list includes by default the following groups
	// in suggested preference order (ordering may not be honored by all implementations):
	// X25519MLKEM768, X25519, secp256r1, secp384r1.
	// This profile is equivalent to a Custom profile specified as:
	//   minTLSVersion: VersionTLS13
	//   ciphers:
	//     - TLS_AES_128_GCM_SHA256
	//     - TLS_AES_256_GCM_SHA384
	//     - TLS_CHACHA20_POLY1305_SHA256
	//
	// +optional
	// +nullable
	Modern *ModernTLSProfile `json:"modern,omitempty"`

	// custom is a user-defined TLS security profile. Be extremely careful using a custom
	// profile as invalid configurations can be catastrophic.
	//
	// The supported groups list for this profile is empty by default.
	//
	// An example custom profile looks like this:
	//
	//   minTLSVersion: VersionTLS11
	//   ciphers:
	//     - ECDHE-ECDSA-CHACHA20-POLY1305
	//     - ECDHE-RSA-CHACHA20-POLY1305
	//     - ECDHE-RSA-AES128-GCM-SHA256
	//     - ECDHE-ECDSA-AES128-GCM-SHA256
	//
	// +optional
	// +nullable
	Custom *CustomTLSProfile `json:"custom,omitempty"`
}

// OldTLSProfile is a TLS security profile based on the "old" configuration of
// the Mozilla Server Side TLS configuration guidelines.
type OldTLSProfile struct{}

// IntermediateTLSProfile is a TLS security profile based on the "intermediate"
// configuration of the Mozilla Server Side TLS configuration guidelines.
type IntermediateTLSProfile struct{}

// ModernTLSProfile is a TLS security profile based on the "modern" configuration
// of the Mozilla Server Side TLS configuration guidelines.
type ModernTLSProfile struct{}

// CustomTLSProfile is a user-defined TLS security profile. Be extremely careful
// using a custom TLS profile as invalid configurations can be catastrophic.
type CustomTLSProfile struct {
	TLSProfileSpec `json:",inline"`
}

// TLSProfileType defines a TLS security profile type.
// +kubebuilder:validation:Enum=Old;Intermediate;Modern;Custom
type TLSProfileType string

const (
	// TLSProfileOldType sets parameters based on the "old" configuration of
	// the Mozilla Server Side TLS configuration guidelines.
	TLSProfileOldType TLSProfileType = "Old"

	// TLSProfileIntermediateType sets parameters based on the "intermediate"
	// configuration of the Mozilla Server Side TLS configuration guidelines.
	TLSProfileIntermediateType TLSProfileType = "Intermediate"

	// TLSProfileModernType sets parameters based on the "modern" configuration
	// of the Mozilla Server Side TLS configuration guidelines.
	TLSProfileModernType TLSProfileType = "Modern"

	// TLSProfileCustomType is a TLS security profile that allows for user-defined parameters.
	TLSProfileCustomType TLSProfileType = "Custom"
)

// TLSGroup is a supported group identifier that can be used in TLSProfile.Groups.
// There is a one-to-one mapping between these names and the group IDs defined
// in Go's crypto/tls package based on IANA's "TLS Supported Groups" registry:
// https://www.iana.org/assignments/tls-parameters/tls-parameters.xhtml#tls-parameters-8
// Note that X25519MLKEM768 is a post-quantum hybrid group that is not
// FIPS-approved and should be ignored by components running in FIPS mode.
//
// +kubebuilder:validation:Enum=X25519;secp256r1;secp384r1;secp521r1;X25519MLKEM768;SecP256r1MLKEM768;SecP384r1MLKEM1024
type TLSGroup string

const (
	// TLSGroupX25519 represents X25519.
	TLSGroupX25519 TLSGroup = "X25519"
	// TLSGroupSecP256r1 represents P-256 (secp256r1).
	TLSGroupSecP256r1 TLSGroup = "secp256r1"
	// TLSGroupSecP384r1 represents P-384 (secp384r1).
	TLSGroupSecP384r1 TLSGroup = "secp384r1"
	// TLSGroupSecP521r1 represents P-521 (secp521r1).
	TLSGroupSecP521r1 TLSGroup = "secp521r1"
	// TLSGroupX25519MLKEM768 represents X25519MLKEM768.
	TLSGroupX25519MLKEM768 TLSGroup = "X25519MLKEM768"
	// TLSGroupSecP256r1MLKEM768 represents SecP256r1MLKEM768.
	TLSGroupSecP256r1MLKEM768 TLSGroup = "SecP256r1MLKEM768"
	// TLSGroupSecP384r1MLKEM1024 represents SecP384r1MLKEM1024.
	TLSGroupSecP384r1MLKEM1024 TLSGroup = "SecP384r1MLKEM1024"
)

// TLSProfileSpec is the desired behavior of a TLSSecurityProfile.
type TLSProfileSpec struct {
	// ciphers is used to specify the cipher algorithms that are negotiated
	// during the TLS handshake. Operators may remove entries that their operands
	// do not support. For example, to use only ECDHE-RSA-AES128-GCM-SHA256 (yaml):
	//
	//   ciphers:
	//     - ECDHE-RSA-AES128-GCM-SHA256
	//
	// TLS 1.3 cipher suites (e.g. TLS_AES_128_GCM_SHA256) are not configurable
	// and are always enabled when TLS 1.3 is negotiated.
	// +listType=atomic
	Ciphers []string `json:"ciphers"`
	// groups is an optional, ordered field used to specify the supported groups (formerly known as
	// elliptic curves) that are used during the TLS handshake.  The order of the groups represents
	// a suggested preference, with the most preferred group first. Note that not all platform
	// components honor the ordering: Go-based components use Go's internal preference order and
	// treat this list as a filter of allowed groups rather than an ordered preference.
	// Operators may remove entries their operands do not support.
	//
	// When omitted, this means no opinion and the platform is left to choose reasonable defaults which are
	// subject to change over time and may be different per platform component depending on the underlying TLS
	// libraries they use. If specified, the list must contain at least one and at most 7 groups,
	// and each group must be unique.
	//
	// For example, to use X25519 and secp256r1 (yaml):
	//
	//   groups:
	//     - X25519
	//     - secp256r1
	//
	// +optional
	// +listType=set
	// +kubebuilder:validation:MaxItems=7
	// +kubebuilder:validation:MinItems=1
	// +openshift:enable:FeatureGate=TLSGroupPreferences
	Groups []TLSGroup `json:"groups,omitempty"`
	// minTLSVersion is used to specify the minimal version of the TLS protocol
	// that is negotiated during the TLS handshake. For example, to use TLS
	// versions 1.1, 1.2 and 1.3 (yaml):
	//
	//   minTLSVersion: VersionTLS11
	//
	MinTLSVersion TLSProtocolVersion `json:"minTLSVersion"`
}

// TLSProtocolVersion is a way to specify the protocol version used for TLS connections.
// Protocol versions are based on the following most common TLS configurations:
//
//	https://ssl-config.mozilla.org/
//
// Note that SSLv3.0 is not a supported protocol version due to well known
// vulnerabilities such as POODLE: https://en.wikipedia.org/wiki/POODLE
// +kubebuilder:validation:Enum=VersionTLS10;VersionTLS11;VersionTLS12;VersionTLS13
type TLSProtocolVersion string

const (
	// VersionTLSv10 is version 1.0 of the TLS security protocol.
	VersionTLS10 TLSProtocolVersion = "VersionTLS10"
	// VersionTLSv11 is version 1.1 of the TLS security protocol.
	VersionTLS11 TLSProtocolVersion = "VersionTLS11"
	// VersionTLSv12 is version 1.2 of the TLS security protocol.
	VersionTLS12 TLSProtocolVersion = "VersionTLS12"
	// VersionTLSv13 is version 1.3 of the TLS security protocol.
	VersionTLS13 TLSProtocolVersion = "VersionTLS13"
)

// TLSProfiles contains a map of TLSProfileType names to TLSProfileSpec.
//
// The cipher and groups lists in these profiles are based on version 5.8 of the
// Mozilla Server Side TLS configuration guidelines.
// See: https://ssl-config.mozilla.org/guidelines/5.8.json
//
// Each Ciphers slice is the configuration's "ciphersuites" followed by the
// "ciphers" from the guidelines JSON.
//
// Groups are listed in suggested preference order, though Go-based components may use
// their own internal ordering. TLSProfiles Old, Intermediate, Modern include by default
// the following groups: X25519MLKEM768, X25519, secp256r1, secp384r1
//
// NOTE: The caller needs to make sure to check that these constants are valid
// for their binary. Not all entries map to values for all binaries. In the case
// of ties, the kube-apiserver wins. Do not fail, just be sure to include only
// valid entries and everything will be ok. In particular, X25519MLKEM768 is
// not FIPS-approved and must be omitted by components running in FIPS mode.
var TLSProfiles = map[TLSProfileType]*TLSProfileSpec{
	TLSProfileOldType: {
		Ciphers: []string{
			"TLS_AES_128_GCM_SHA256",
			"TLS_AES_256_GCM_SHA384",
			"TLS_CHACHA20_POLY1305_SHA256",
			"ECDHE-ECDSA-AES128-GCM-SHA256",
			"ECDHE-RSA-AES128-GCM-SHA256",
			"ECDHE-ECDSA-AES256-GCM-SHA384",
			"ECDHE-RSA-AES256-GCM-SHA384",
			"ECDHE-ECDSA-CHACHA20-POLY1305",
			"ECDHE-RSA-CHACHA20-POLY1305",
			"ECDHE-ECDSA-AES128-SHA256",
			"ECDHE-RSA-AES128-SHA256",
			"ECDHE-ECDSA-AES128-SHA",
			"ECDHE-RSA-AES128-SHA",
			"ECDHE-ECDSA-AES256-SHA384",
			"ECDHE-RSA-AES256-SHA384",
			"ECDHE-ECDSA-AES256-SHA",
			"ECDHE-RSA-AES256-SHA",
			"AES128-GCM-SHA256",
			"AES256-GCM-SHA384",
			"AES128-SHA256",
			"AES256-SHA256",
			"AES128-SHA",
			"AES256-SHA",
			"DES-CBC3-SHA",
		},
		Groups: []TLSGroup{
			TLSGroupX25519MLKEM768,
			TLSGroupX25519,
			TLSGroupSecP256r1,
			TLSGroupSecP384r1,
		},
		MinTLSVersion: VersionTLS10,
	},
	TLSProfileIntermediateType: {
		Ciphers: []string{
			"TLS_AES_128_GCM_SHA256",
			"TLS_AES_256_GCM_SHA384",
			"TLS_CHACHA20_POLY1305_SHA256",
			"ECDHE-ECDSA-AES128-GCM-SHA256",
			"ECDHE-RSA-AES128-GCM-SHA256",
			"ECDHE-ECDSA-AES256-GCM-SHA384",
			"ECDHE-RSA-AES256-GCM-SHA384",
			"ECDHE-ECDSA-CHACHA20-POLY1305",
			"ECDHE-RSA-CHACHA20-POLY1305",
		},
		Groups: []TLSGroup{
			TLSGroupX25519MLKEM768,
			TLSGroupX25519,
			TLSGroupSecP256r1,
			TLSGroupSecP384r1,
		},
		MinTLSVersion: VersionTLS12,
	},
	TLSProfileModernType: {
		Ciphers: []string{
			"TLS_AES_128_GCM_SHA256",
			"TLS_AES_256_GCM_SHA384",
			"TLS_CHACHA20_POLY1305_SHA256",
		},
		Groups: []TLSGroup{
			TLSGroupX25519MLKEM768,
			TLSGroupX25519,
			TLSGroupSecP256r1,
			TLSGroupSecP384r1,
		},
		MinTLSVersion: VersionTLS13,
	},
}
