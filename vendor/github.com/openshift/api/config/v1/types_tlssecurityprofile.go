package v1

// TLSSecurityProfile defines the schema for a TLS security profile. This object
// is used by operators to apply TLS security settings to operands.
// +union
type TLSSecurityProfile struct {
	// type is one of Old, Intermediate, Modern or Custom. Custom provides
	// the ability to specify individual TLS security profile parameters.
	// Old, Intermediate and Modern are TLS security profiles based on:
	//
	// https://wiki.mozilla.org/Security/Server_Side_TLS#Recommended_configurations
	//
	// The profiles are intent based, so they may change over time as new ciphers are developed and existing ciphers
	// are found to be insecure.  Depending on precisely which ciphers are available to a process, the list may be
	// reduced.
	//
	// Note that the Modern profile is currently not supported because it is not
	// yet well adopted by common software libraries.
	//
	// +unionDiscriminator
	// +optional
	Type TLSProfileType `json:"type"`
	// old is a TLS security profile based on:
	//
	// https://wiki.mozilla.org/Security/Server_Side_TLS#Old_backward_compatibility
	//
	// and looks like this (yaml):
	//
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
	//     - DHE-RSA-AES128-GCM-SHA256
	//     - DHE-RSA-AES256-GCM-SHA384
	//     - DHE-RSA-CHACHA20-POLY1305
	//     - ECDHE-ECDSA-AES128-SHA256
	//     - ECDHE-RSA-AES128-SHA256
	//     - ECDHE-ECDSA-AES128-SHA
	//     - ECDHE-RSA-AES128-SHA
	//     - ECDHE-ECDSA-AES256-SHA384
	//     - ECDHE-RSA-AES256-SHA384
	//     - ECDHE-ECDSA-AES256-SHA
	//     - ECDHE-RSA-AES256-SHA
	//     - DHE-RSA-AES128-SHA256
	//     - DHE-RSA-AES256-SHA256
	//     - AES128-GCM-SHA256
	//     - AES256-GCM-SHA384
	//     - AES128-SHA256
	//     - AES256-SHA256
	//     - AES128-SHA
	//     - AES256-SHA
	//     - DES-CBC3-SHA
	//   minTLSVersion: VersionTLS10
	//
	// +optional
	// +nullable
	Old *OldTLSProfile `json:"old,omitempty"`
	// intermediate is a TLS security profile based on:
	//
	// https://wiki.mozilla.org/Security/Server_Side_TLS#Intermediate_compatibility_.28recommended.29
	//
	// and looks like this (yaml):
	//
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
	//     - DHE-RSA-AES128-GCM-SHA256
	//     - DHE-RSA-AES256-GCM-SHA384
	//   minTLSVersion: VersionTLS12
	//
	// +optional
	// +nullable
	Intermediate *IntermediateTLSProfile `json:"intermediate,omitempty"`
	// modern is a TLS security profile based on:
	//
	// https://wiki.mozilla.org/Security/Server_Side_TLS#Modern_compatibility
	//
	// and looks like this (yaml):
	//
	//   ciphers:
	//     - TLS_AES_128_GCM_SHA256
	//     - TLS_AES_256_GCM_SHA384
	//     - TLS_CHACHA20_POLY1305_SHA256
	//   minTLSVersion: VersionTLS13
	//
	// NOTE: Currently unsupported.
	//
	// +optional
	// +nullable
	Modern *ModernTLSProfile `json:"modern,omitempty"`
	// custom is a user-defined TLS security profile. Be extremely careful using a custom
	// profile as invalid configurations can be catastrophic. An example custom profile
	// looks like this:
	//
	//   ciphers:
	//     - ECDHE-ECDSA-CHACHA20-POLY1305
	//     - ECDHE-RSA-CHACHA20-POLY1305
	//     - ECDHE-RSA-AES128-GCM-SHA256
	//     - ECDHE-ECDSA-AES128-GCM-SHA256
	//   minTLSVersion: VersionTLS11
	//
	// +optional
	// +nullable
	Custom *CustomTLSProfile `json:"custom,omitempty"`
}

// OldTLSProfile is a TLS security profile based on:
// https://wiki.mozilla.org/Security/Server_Side_TLS#Old_backward_compatibility
type OldTLSProfile struct{}

// IntermediateTLSProfile is a TLS security profile based on:
// https://wiki.mozilla.org/Security/Server_Side_TLS#Intermediate_compatibility_.28default.29
type IntermediateTLSProfile struct{}

// ModernTLSProfile is a TLS security profile based on:
// https://wiki.mozilla.org/Security/Server_Side_TLS#Modern_compatibility
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
	// Old is a TLS security profile based on:
	// https://wiki.mozilla.org/Security/Server_Side_TLS#Old_backward_compatibility
	TLSProfileOldType TLSProfileType = "Old"
	// Intermediate is a TLS security profile based on:
	// https://wiki.mozilla.org/Security/Server_Side_TLS#Intermediate_compatibility_.28default.29
	TLSProfileIntermediateType TLSProfileType = "Intermediate"
	// Modern is a TLS security profile based on:
	// https://wiki.mozilla.org/Security/Server_Side_TLS#Modern_compatibility
	TLSProfileModernType TLSProfileType = "Modern"
	// Custom is a TLS security profile that allows for user-defined parameters.
	TLSProfileCustomType TLSProfileType = "Custom"
)

// TLSProfileSpec is the desired behavior of a TLSSecurityProfile.
type TLSProfileSpec struct {
	// ciphers is used to specify the cipher algorithms that are negotiated
	// during the TLS handshake.  Operators may remove entries their operands
	// do not support.  For example, to use DES-CBC3-SHA  (yaml):
	//
	//   ciphers:
	//     - DES-CBC3-SHA
	//
	Ciphers []string `json:"ciphers"`
	// minTLSVersion is used to specify the minimal version of the TLS protocol
	// that is negotiated during the TLS handshake. For example, to use TLS
	// versions 1.1, 1.2 and 1.3 (yaml):
	//
	//   minTLSVersion: VersionTLS11
	//
	// NOTE: currently the highest minTLSVersion allowed is VersionTLS12
	//
	MinTLSVersion TLSProtocolVersion `json:"minTLSVersion"`
}

// TLSProtocolVersion is a way to specify the protocol version used for TLS connections.
// Protocol versions are based on the following most common TLS configurations:
//
//   https://ssl-config.mozilla.org/
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

// TLSProfiles Contains a map of TLSProfileType names to TLSProfileSpec.
//
// NOTE: The caller needs to make sure to check that these constants are valid for their binary. Not all
// entries map to values for all binaries.  In the case of ties, the kube-apiserver wins.  Do not fail,
// just be sure to whitelist only and everything will be ok.
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
			"DHE-RSA-AES128-GCM-SHA256",
			"DHE-RSA-AES256-GCM-SHA384",
			"DHE-RSA-CHACHA20-POLY1305",
			"ECDHE-ECDSA-AES128-SHA256",
			"ECDHE-RSA-AES128-SHA256",
			"ECDHE-ECDSA-AES128-SHA",
			"ECDHE-RSA-AES128-SHA",
			"ECDHE-ECDSA-AES256-SHA384",
			"ECDHE-RSA-AES256-SHA384",
			"ECDHE-ECDSA-AES256-SHA",
			"ECDHE-RSA-AES256-SHA",
			"DHE-RSA-AES128-SHA256",
			"DHE-RSA-AES256-SHA256",
			"AES128-GCM-SHA256",
			"AES256-GCM-SHA384",
			"AES128-SHA256",
			"AES256-SHA256",
			"AES128-SHA",
			"AES256-SHA",
			"DES-CBC3-SHA",
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
			"DHE-RSA-AES128-GCM-SHA256",
			"DHE-RSA-AES256-GCM-SHA384",
		},
		MinTLSVersion: VersionTLS12,
	},
	TLSProfileModernType: {
		Ciphers: []string{
			"TLS_AES_128_GCM_SHA256",
			"TLS_AES_256_GCM_SHA384",
			"TLS_CHACHA20_POLY1305_SHA256",
		},
		MinTLSVersion: VersionTLS13,
	},
}
