package tls

import (
	"fmt"
)

type hashAlgID uint8

const (
	HashNone hashAlgID = iota
	HashMD5
	HashSHA1
	HashSHA224
	HashSHA256
	HashSHA384
	HashSHA512
)

func (h hashAlgID) String() string {
	switch h {
	case HashNone:
		return "None"
	case HashMD5:
		return "MD5"
	case HashSHA1:
		return "SHA1"
	case HashSHA224:
		return "SHA224"
	case HashSHA256:
		return "SHA256"
	case HashSHA384:
		return "SHA384"
	case HashSHA512:
		return "SHA512"
	default:
		return "Unknown"
	}
}

type sigAlgID uint8

// Signature algorithms for TLS 1.2 (See RFC 5246, section A.4.1)
const (
	SigAnon sigAlgID = iota
	SigRSA
	SigDSA
	SigECDSA
)

func (sig sigAlgID) String() string {
	switch sig {
	case SigAnon:
		return "Anon"
	case SigRSA:
		return "RSA"
	case SigDSA:
		return "DSA"
	case SigECDSA:
		return "ECDSA"
	default:
		return "Unknown"
	}
}

// SignatureAndHash mirrors the TLS 1.2, SignatureAndHashAlgorithm struct. See
// RFC 5246, section A.4.1.
type SignatureAndHash struct {
	h hashAlgID
	s sigAlgID
}

func (sigAlg SignatureAndHash) String() string {
	return fmt.Sprintf("{%s,%s}", sigAlg.s, sigAlg.h)
}

func (sigAlg SignatureAndHash) MarshalJSON() ([]byte, error) {
	return []byte(fmt.Sprintf(`{"signature":"%s","hash":"%s"}`, sigAlg.s, sigAlg.h)), nil
}

func (sigAlg SignatureAndHash) internal() signatureAndHash {
	return signatureAndHash{uint8(sigAlg.h), uint8(sigAlg.s)}
}

// defaultSignatureAndHashAlgorithms contains the default signature and hash
// algorithm paris supported by `crypto/tls`
var defaultSignatureAndHashAlgorithms []signatureAndHash

// AllSignatureAndHashAlgorithms contains all possible signature and
// hash algorithm pairs that the can be advertised in a TLS 1.2 ClientHello.
var AllSignatureAndHashAlgorithms []SignatureAndHash

func init() {
	defaultSignatureAndHashAlgorithms = supportedSignatureAlgorithms
	for _, sighash := range supportedSignatureAlgorithms {
		AllSignatureAndHashAlgorithms = append(AllSignatureAndHashAlgorithms,
			SignatureAndHash{hashAlgID(sighash.hash), sigAlgID(sighash.signature)})
	}
}

// TLSVersions is a list of the current SSL/TLS Versions implemented by Go
var Versions = map[uint16]string{
	VersionSSL30: "SSL 3.0",
	VersionTLS10: "TLS 1.0",
	VersionTLS11: "TLS 1.1",
	VersionTLS12: "TLS 1.2",
}

// CipherSuite describes an individual cipher suite, with long and short names
// and security properties.
type CipherSuite struct {
	Name, ShortName string
	// ForwardSecret cipher suites negotiate ephemeral keys, allowing forward secrecy.
	ForwardSecret bool
	EllipticCurve bool
}

// Returns the (short) name of the cipher suite.
func (c CipherSuite) String() string {
	if c.ShortName != "" {
		return c.ShortName
	}
	return c.Name
}

// CipherSuites contains all values in the TLS Cipher Suite Registry
// https://www.iana.org/assignments/tls-parameters/tls-parameters.xhtml
var CipherSuites = map[uint16]CipherSuite{
	0X0000: {Name: "TLS_NULL_WITH_NULL_NULL"},
	0X0001: {Name: "TLS_RSA_WITH_NULL_MD5"},
	0X0002: {Name: "TLS_RSA_WITH_NULL_SHA"},
	0X0003: {Name: "TLS_RSA_EXPORT_WITH_RC4_40_MD5", ShortName: "EXP-RC4-MD5"},
	0X0004: {Name: "TLS_RSA_WITH_RC4_128_MD5", ShortName: "RC4-MD5"},
	0X0005: {Name: "TLS_RSA_WITH_RC4_128_SHA", ShortName: "RC4-SHA"},
	0X0006: {Name: "TLS_RSA_EXPORT_WITH_RC2_CBC_40_MD5", ShortName: "EXP-RC2-CBC-MD5"},
	0X0007: {Name: "TLS_RSA_WITH_IDEA_CBC_SHA", ShortName: "IDEA-CBC-SHA"},
	0X0008: {Name: "TLS_RSA_EXPORT_WITH_DES40_CBC_SHA", ShortName: "EXP-DES-CBC-SHA"},
	0X0009: {Name: "TLS_RSA_WITH_DES_CBC_SHA", ShortName: "DES-CBC-SHA"},
	0X000A: {Name: "TLS_RSA_WITH_3DES_EDE_CBC_SHA", ShortName: "DES-CBC3-SHA"},
	0X000B: {Name: "TLS_DH_DSS_EXPORT_WITH_DES40_CBC_SHA", ShortName: "EXP-DH-DSS-DES-CBC-SHA"},
	0X000C: {Name: "TLS_DH_DSS_WITH_DES_CBC_SHA", ShortName: "DH-DSS-DES-CBC-SHA"},
	0X000D: {Name: "TLS_DH_DSS_WITH_3DES_EDE_CBC_SHA", ShortName: "DH-DSS-DES-CBC3-SHA"},
	0X000E: {Name: "TLS_DH_RSA_EXPORT_WITH_DES40_CBC_SHA", ShortName: "EXP-DH-RSA-DES-CBC-SHA"},
	0X000F: {Name: "TLS_DH_RSA_WITH_DES_CBC_SHA", ShortName: "DH-RSA-DES-CBC-SHA"},
	0X0010: {Name: "TLS_DH_RSA_WITH_3DES_EDE_CBC_SHA", ShortName: "DH-RSA-DES-CBC3-SHA"},
	0X0011: {Name: "TLS_DHE_DSS_EXPORT_WITH_DES40_CBC_SHA", ShortName: "EXP-EDH-DSS-DES-CBC-SHA", ForwardSecret: true},
	0X0012: {Name: "TLS_DHE_DSS_WITH_DES_CBC_SHA", ShortName: "EDH-DSS-DES-CBC-SHA", ForwardSecret: true},
	0X0013: {Name: "TLS_DHE_DSS_WITH_3DES_EDE_CBC_SHA", ShortName: "EDH-DSS-DES-CBC3-SHA", ForwardSecret: true},
	0X0014: {Name: "TLS_DHE_RSA_EXPORT_WITH_DES40_CBC_SHA", ShortName: "EXP-EDH-RSA-DES-CBC-SHA", ForwardSecret: true},
	0X0015: {Name: "TLS_DHE_RSA_WITH_DES_CBC_SHA", ShortName: "EDH-RSA-DES-CBC-SHA", ForwardSecret: true},
	0X0016: {Name: "TLS_DHE_RSA_WITH_3DES_EDE_CBC_SHA", ShortName: "EDH-RSA-DES-CBC3-SHA", ForwardSecret: true},
	0X0017: {Name: "TLS_DH_anon_EXPORT_WITH_RC4_40_MD5"},
	0X0018: {Name: "TLS_DH_anon_WITH_RC4_128_MD5"},
	0X0019: {Name: "TLS_DH_anon_EXPORT_WITH_DES40_CBC_SHA"},
	0X001A: {Name: "TLS_DH_anon_WITH_DES_CBC_SHA"},
	0X001B: {Name: "TLS_DH_anon_WITH_3DES_EDE_CBC_SHA"},
	0X001E: {Name: "TLS_KRB5_WITH_DES_CBC_SHA"},
	0X001F: {Name: "TLS_KRB5_WITH_3DES_EDE_CBC_SHA"},
	0X0020: {Name: "TLS_KRB5_WITH_RC4_128_SHA"},
	0X0021: {Name: "TLS_KRB5_WITH_IDEA_CBC_SHA"},
	0X0022: {Name: "TLS_KRB5_WITH_DES_CBC_MD5"},
	0X0023: {Name: "TLS_KRB5_WITH_3DES_EDE_CBC_MD5"},
	0X0024: {Name: "TLS_KRB5_WITH_RC4_128_MD5"},
	0X0025: {Name: "TLS_KRB5_WITH_IDEA_CBC_MD5"},
	0X0026: {Name: "TLS_KRB5_EXPORT_WITH_DES_CBC_40_SHA"},
	0X0027: {Name: "TLS_KRB5_EXPORT_WITH_RC2_CBC_40_SHA"},
	0X0028: {Name: "TLS_KRB5_EXPORT_WITH_RC4_40_SHA"},
	0X0029: {Name: "TLS_KRB5_EXPORT_WITH_DES_CBC_40_MD5"},
	0X002A: {Name: "TLS_KRB5_EXPORT_WITH_RC2_CBC_40_MD5"},
	0X002B: {Name: "TLS_KRB5_EXPORT_WITH_RC4_40_MD5"},
	0X002C: {Name: "TLS_PSK_WITH_NULL_SHA"},
	0X002D: {Name: "TLS_DHE_PSK_WITH_NULL_SHA", ForwardSecret: true},
	0X002E: {Name: "TLS_RSA_PSK_WITH_NULL_SHA"},
	0X002F: {Name: "TLS_RSA_WITH_AES_128_CBC_SHA", ShortName: "AES128-SHA"},
	0X0030: {Name: "TLS_DH_DSS_WITH_AES_128_CBC_SHA", ShortName: "DH-DSS-AES128-SHA"},
	0X0031: {Name: "TLS_DH_RSA_WITH_AES_128_CBC_SHA", ShortName: "DH-RSA-AES128-SHA"},
	0X0032: {Name: "TLS_DHE_DSS_WITH_AES_128_CBC_SHA", ShortName: "DHE-DSS-AES128-SHA", ForwardSecret: true},
	0X0033: {Name: "TLS_DHE_RSA_WITH_AES_128_CBC_SHA", ShortName: "DHE-RSA-AES128-SHA", ForwardSecret: true},
	0X0034: {Name: "TLS_DH_anon_WITH_AES_128_CBC_SHA"},
	0X0035: {Name: "TLS_RSA_WITH_AES_256_CBC_SHA", ShortName: "AES256-SHA"},
	0X0036: {Name: "TLS_DH_DSS_WITH_AES_256_CBC_SHA", ShortName: "DH-DSS-AES256-SHA"},
	0X0037: {Name: "TLS_DH_RSA_WITH_AES_256_CBC_SHA", ShortName: "DH-RSA-AES256-SHA"},
	0X0038: {Name: "TLS_DHE_DSS_WITH_AES_256_CBC_SHA", ShortName: "DHE-DSS-AES256-SHA", ForwardSecret: true},
	0X0039: {Name: "TLS_DHE_RSA_WITH_AES_256_CBC_SHA", ShortName: "DHE-RSA-AES256-SHA", ForwardSecret: true},
	0X003A: {Name: "TLS_DH_anon_WITH_AES_256_CBC_SHA"},
	0X003B: {Name: "TLS_RSA_WITH_NULL_SHA256"},
	0X003C: {Name: "TLS_RSA_WITH_AES_128_CBC_SHA256", ShortName: "AES128-SHA256"},
	0X003D: {Name: "TLS_RSA_WITH_AES_256_CBC_SHA256", ShortName: "AES256-SHA256"},
	0X003E: {Name: "TLS_DH_DSS_WITH_AES_128_CBC_SHA256", ShortName: "DH-DSS-AES128-SHA256"},
	0X003F: {Name: "TLS_DH_RSA_WITH_AES_128_CBC_SHA256", ShortName: "DH-RSA-AES128-SHA256"},
	0X0040: {Name: "TLS_DHE_DSS_WITH_AES_128_CBC_SHA256", ShortName: "DHE-DSS-AES128-SHA256", ForwardSecret: true},
	0X0041: {Name: "TLS_RSA_WITH_CAMELLIA_128_CBC_SHA", ShortName: "CAMELLIA128-SHA"},
	0X0042: {Name: "TLS_DH_DSS_WITH_CAMELLIA_128_CBC_SHA", ShortName: "DH-DSS-CAMELLIA128-SHA"},
	0X0043: {Name: "TLS_DH_RSA_WITH_CAMELLIA_128_CBC_SHA", ShortName: "DH-RSA-CAMELLIA128-SHA"},
	0X0044: {Name: "TLS_DHE_DSS_WITH_CAMELLIA_128_CBC_SHA", ShortName: "DHE-DSS-CAMELLIA128-SHA", ForwardSecret: true},
	0X0045: {Name: "TLS_DHE_RSA_WITH_CAMELLIA_128_CBC_SHA", ShortName: "DHE-RSA-CAMELLIA128-SHA", ForwardSecret: true},
	0X0046: {Name: "TLS_DH_anon_WITH_CAMELLIA_128_CBC_SHA"},
	0X0067: {Name: "TLS_DHE_RSA_WITH_AES_128_CBC_SHA256", ShortName: "DHE-RSA-AES128-SHA256", ForwardSecret: true},
	0X0068: {Name: "TLS_DH_DSS_WITH_AES_256_CBC_SHA256", ShortName: "DH-DSS-AES256-SHA256"},
	0X0069: {Name: "TLS_DH_RSA_WITH_AES_256_CBC_SHA256", ShortName: "DH-RSA-AES256-SHA256"},
	0X006A: {Name: "TLS_DHE_DSS_WITH_AES_256_CBC_SHA256", ShortName: "DHE-DSS-AES256-SHA256", ForwardSecret: true},
	0X006B: {Name: "TLS_DHE_RSA_WITH_AES_256_CBC_SHA256", ShortName: "DHE-RSA-AES256-SHA256", ForwardSecret: true},
	0X006C: {Name: "TLS_DH_anon_WITH_AES_128_CBC_SHA256"},
	0X006D: {Name: "TLS_DH_anon_WITH_AES_256_CBC_SHA256"},
	0X0084: {Name: "TLS_RSA_WITH_CAMELLIA_256_CBC_SHA", ShortName: "CAMELLIA256-SHA"},
	0X0085: {Name: "TLS_DH_DSS_WITH_CAMELLIA_256_CBC_SHA", ShortName: "DH-DSS-CAMELLIA256-SHA"},
	0X0086: {Name: "TLS_DH_RSA_WITH_CAMELLIA_256_CBC_SHA", ShortName: "DH-RSA-CAMELLIA256-SHA"},
	0X0087: {Name: "TLS_DHE_DSS_WITH_CAMELLIA_256_CBC_SHA", ShortName: "DHE-DSS-CAMELLIA256-SHA", ForwardSecret: true},
	0X0088: {Name: "TLS_DHE_RSA_WITH_CAMELLIA_256_CBC_SHA", ShortName: "DHE-RSA-CAMELLIA256-SHA", ForwardSecret: true},
	0X0089: {Name: "TLS_DH_anon_WITH_CAMELLIA_256_CBC_SHA"},
	0X008A: {Name: "TLS_PSK_WITH_RC4_128_SHA", ShortName: "PSK-RC4-SHA"},
	0X008B: {Name: "TLS_PSK_WITH_3DES_EDE_CBC_SHA", ShortName: "PSK-3DES-EDE-CBC-SHA"},
	0X008C: {Name: "TLS_PSK_WITH_AES_128_CBC_SHA", ShortName: "PSK-AES128-CBC-SHA"},
	0X008D: {Name: "TLS_PSK_WITH_AES_256_CBC_SHA", ShortName: "PSK-AES256-CBC-SHA"},
	0X008E: {Name: "TLS_DHE_PSK_WITH_RC4_128_SHA", ForwardSecret: true},
	0X008F: {Name: "TLS_DHE_PSK_WITH_3DES_EDE_CBC_SHA", ForwardSecret: true},
	0X0090: {Name: "TLS_DHE_PSK_WITH_AES_128_CBC_SHA", ForwardSecret: true},
	0X0091: {Name: "TLS_DHE_PSK_WITH_AES_256_CBC_SHA", ForwardSecret: true},
	0X0092: {Name: "TLS_RSA_PSK_WITH_RC4_128_SHA"},
	0X0093: {Name: "TLS_RSA_PSK_WITH_3DES_EDE_CBC_SHA"},
	0X0094: {Name: "TLS_RSA_PSK_WITH_AES_128_CBC_SHA"},
	0X0095: {Name: "TLS_RSA_PSK_WITH_AES_256_CBC_SHA"},
	0X0096: {Name: "TLS_RSA_WITH_SEED_CBC_SHA", ShortName: "SEED-SHA"},
	0X0097: {Name: "TLS_DH_DSS_WITH_SEED_CBC_SHA", ShortName: "DH-DSS-SEED-SHA"},
	0X0098: {Name: "TLS_DH_RSA_WITH_SEED_CBC_SHA", ShortName: "DH-RSA-SEED-SHA"},
	0X0099: {Name: "TLS_DHE_DSS_WITH_SEED_CBC_SHA", ShortName: "DHE-DSS-SEED-SHA", ForwardSecret: true},
	0X009A: {Name: "TLS_DHE_RSA_WITH_SEED_CBC_SHA", ShortName: "DHE-RSA-SEED-SHA", ForwardSecret: true},
	0X009B: {Name: "TLS_DH_anon_WITH_SEED_CBC_SHA"},
	0X009C: {Name: "TLS_RSA_WITH_AES_128_GCM_SHA256", ShortName: "AES128-GCM-SHA256"},
	0X009D: {Name: "TLS_RSA_WITH_AES_256_GCM_SHA384", ShortName: "AES256-GCM-SHA384"},
	0X009E: {Name: "TLS_DHE_RSA_WITH_AES_128_GCM_SHA256", ShortName: "DHE-RSA-AES128-GCM-SHA256", ForwardSecret: true},
	0X009F: {Name: "TLS_DHE_RSA_WITH_AES_256_GCM_SHA384", ShortName: "DHE-RSA-AES256-GCM-SHA384", ForwardSecret: true},
	0X00A0: {Name: "TLS_DH_RSA_WITH_AES_128_GCM_SHA256", ShortName: "DH-RSA-AES128-GCM-SHA256"},
	0X00A1: {Name: "TLS_DH_RSA_WITH_AES_256_GCM_SHA384", ShortName: "DH-RSA-AES256-GCM-SHA384"},
	0X00A2: {Name: "TLS_DHE_DSS_WITH_AES_128_GCM_SHA256", ShortName: "DHE-DSS-AES128-GCM-SHA256", ForwardSecret: true},
	0X00A3: {Name: "TLS_DHE_DSS_WITH_AES_256_GCM_SHA384", ShortName: "DHE-DSS-AES256-GCM-SHA384", ForwardSecret: true},
	0X00A4: {Name: "TLS_DH_DSS_WITH_AES_128_GCM_SHA256", ShortName: "DH-DSS-AES128-GCM-SHA256"},
	0X00A5: {Name: "TLS_DH_DSS_WITH_AES_256_GCM_SHA384", ShortName: "DH-DSS-AES256-GCM-SHA384"},
	0X00A6: {Name: "TLS_DH_anon_WITH_AES_128_GCM_SHA256"},
	0X00A7: {Name: "TLS_DH_anon_WITH_AES_256_GCM_SHA384"},
	0X00A8: {Name: "TLS_PSK_WITH_AES_128_GCM_SHA256"},
	0X00A9: {Name: "TLS_PSK_WITH_AES_256_GCM_SHA384"},
	0X00AA: {Name: "TLS_DHE_PSK_WITH_AES_128_GCM_SHA256", ForwardSecret: true},
	0X00AB: {Name: "TLS_DHE_PSK_WITH_AES_256_GCM_SHA384", ForwardSecret: true},
	0X00AC: {Name: "TLS_RSA_PSK_WITH_AES_128_GCM_SHA256"},
	0X00AD: {Name: "TLS_RSA_PSK_WITH_AES_256_GCM_SHA384"},
	0X00AE: {Name: "TLS_PSK_WITH_AES_128_CBC_SHA256"},
	0X00AF: {Name: "TLS_PSK_WITH_AES_256_CBC_SHA384"},
	0X00B0: {Name: "TLS_PSK_WITH_NULL_SHA256"},
	0X00B1: {Name: "TLS_PSK_WITH_NULL_SHA384"},
	0X00B2: {Name: "TLS_DHE_PSK_WITH_AES_128_CBC_SHA256", ForwardSecret: true},
	0X00B3: {Name: "TLS_DHE_PSK_WITH_AES_256_CBC_SHA384", ForwardSecret: true},
	0X00B4: {Name: "TLS_DHE_PSK_WITH_NULL_SHA256", ForwardSecret: true},
	0X00B5: {Name: "TLS_DHE_PSK_WITH_NULL_SHA384", ForwardSecret: true},
	0X00B6: {Name: "TLS_RSA_PSK_WITH_AES_128_CBC_SHA256"},
	0X00B7: {Name: "TLS_RSA_PSK_WITH_AES_256_CBC_SHA384"},
	0X00B8: {Name: "TLS_RSA_PSK_WITH_NULL_SHA256"},
	0X00B9: {Name: "TLS_RSA_PSK_WITH_NULL_SHA384"},
	0X00BA: {Name: "TLS_RSA_WITH_CAMELLIA_128_CBC_SHA256"},
	0X00BB: {Name: "TLS_DH_DSS_WITH_CAMELLIA_128_CBC_SHA256"},
	0X00BC: {Name: "TLS_DH_RSA_WITH_CAMELLIA_128_CBC_SHA256"},
	0X00BD: {Name: "TLS_DHE_DSS_WITH_CAMELLIA_128_CBC_SHA256", ForwardSecret: true},
	0X00BE: {Name: "TLS_DHE_RSA_WITH_CAMELLIA_128_CBC_SHA256", ForwardSecret: true},
	0X00BF: {Name: "TLS_DH_anon_WITH_CAMELLIA_128_CBC_SHA256"},
	0X00C0: {Name: "TLS_RSA_WITH_CAMELLIA_256_CBC_SHA256"},
	0X00C1: {Name: "TLS_DH_DSS_WITH_CAMELLIA_256_CBC_SHA256"},
	0X00C2: {Name: "TLS_DH_RSA_WITH_CAMELLIA_256_CBC_SHA256"},
	0X00C3: {Name: "TLS_DHE_DSS_WITH_CAMELLIA_256_CBC_SHA256", ForwardSecret: true},
	0X00C4: {Name: "TLS_DHE_RSA_WITH_CAMELLIA_256_CBC_SHA256", ForwardSecret: true},
	0X00C5: {Name: "TLS_DH_anon_WITH_CAMELLIA_256_CBC_SHA256"},
	0X00FF: {Name: "TLS_EMPTY_RENEGOTIATION_INFO_SCSV"},
	0XC001: {Name: "TLS_ECDH_ECDSA_WITH_NULL_SHA", EllipticCurve: true},
	0XC002: {Name: "TLS_ECDH_ECDSA_WITH_RC4_128_SHA", ShortName: "ECDH-ECDSA-RC4-SHA", EllipticCurve: true},
	0XC003: {Name: "TLS_ECDH_ECDSA_WITH_3DES_EDE_CBC_SHA", ShortName: "ECDH-ECDSA-DES-CBC3-SHA", EllipticCurve: true},
	0XC004: {Name: "TLS_ECDH_ECDSA_WITH_AES_128_CBC_SHA", ShortName: "ECDH-ECDSA-AES128-SHA", EllipticCurve: true},
	0XC005: {Name: "TLS_ECDH_ECDSA_WITH_AES_256_CBC_SHA", ShortName: "ECDH-ECDSA-AES256-SHA", EllipticCurve: true},
	0XC006: {Name: "TLS_ECDHE_ECDSA_WITH_NULL_SHA", ForwardSecret: true, EllipticCurve: true},
	0XC007: {Name: "TLS_ECDHE_ECDSA_WITH_RC4_128_SHA", ShortName: "ECDHE-ECDSA-RC4-SHA", ForwardSecret: true, EllipticCurve: true},
	0XC008: {Name: "TLS_ECDHE_ECDSA_WITH_3DES_EDE_CBC_SHA", ShortName: "ECDHE-ECDSA-DES-CBC3-SHA", ForwardSecret: true, EllipticCurve: true},
	0XC009: {Name: "TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA", ShortName: "ECDHE-ECDSA-AES128-SHA", ForwardSecret: true, EllipticCurve: true},
	0XC00A: {Name: "TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA", ShortName: "ECDHE-ECDSA-AES256-SHA", ForwardSecret: true, EllipticCurve: true},
	0XC00B: {Name: "TLS_ECDH_RSA_WITH_NULL_SHA", EllipticCurve: true},
	0XC00C: {Name: "TLS_ECDH_RSA_WITH_RC4_128_SHA", ShortName: "ECDH-RSA-RC4-SHA", EllipticCurve: true},
	0XC00D: {Name: "TLS_ECDH_RSA_WITH_3DES_EDE_CBC_SHA", ShortName: "ECDH-RSA-DES-CBC3-SHA", EllipticCurve: true},
	0XC00E: {Name: "TLS_ECDH_RSA_WITH_AES_128_CBC_SHA", ShortName: "ECDH-RSA-AES128-SHA", EllipticCurve: true},
	0XC00F: {Name: "TLS_ECDH_RSA_WITH_AES_256_CBC_SHA", ShortName: "ECDH-RSA-AES256-SHA", EllipticCurve: true},
	0XC010: {Name: "TLS_ECDHE_RSA_WITH_NULL_SHA", ForwardSecret: true, EllipticCurve: true},
	0XC011: {Name: "TLS_ECDHE_RSA_WITH_RC4_128_SHA", ShortName: "ECDHE-RSA-RC4-SHA", ForwardSecret: true, EllipticCurve: true},
	0XC012: {Name: "TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA", ShortName: "ECDHE-RSA-DES-CBC3-SHA", ForwardSecret: true, EllipticCurve: true},
	0XC013: {Name: "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA", ShortName: "ECDHE-RSA-AES128-SHA", ForwardSecret: true, EllipticCurve: true},
	0XC014: {Name: "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA", ShortName: "ECDHE-RSA-AES256-SHA", ForwardSecret: true, EllipticCurve: true},
	0XC015: {Name: "TLS_ECDH_anon_WITH_NULL_SHA", EllipticCurve: true},
	0XC016: {Name: "TLS_ECDH_anon_WITH_RC4_128_SHA", EllipticCurve: true},
	0XC017: {Name: "TLS_ECDH_anon_WITH_3DES_EDE_CBC_SHA", EllipticCurve: true},
	0XC018: {Name: "TLS_ECDH_anon_WITH_AES_128_CBC_SHA", EllipticCurve: true},
	0XC019: {Name: "TLS_ECDH_anon_WITH_AES_256_CBC_SHA", EllipticCurve: true},
	0XC01A: {Name: "TLS_SRP_SHA_WITH_3DES_EDE_CBC_SHA", ShortName: "SRP-3DES-EDE-CBC-SHA"},
	0XC01B: {Name: "TLS_SRP_SHA_RSA_WITH_3DES_EDE_CBC_SHA", ShortName: "SRP-RSA-3DES-EDE-CBC-SHA"},
	0XC01C: {Name: "TLS_SRP_SHA_DSS_WITH_3DES_EDE_CBC_SHA", ShortName: "SRP-DSS-3DES-EDE-CBC-SHA"},
	0XC01D: {Name: "TLS_SRP_SHA_WITH_AES_128_CBC_SHA", ShortName: "SRP-AES-128-CBC-SHA"},
	0XC01E: {Name: "TLS_SRP_SHA_RSA_WITH_AES_128_CBC_SHA", ShortName: "SRP-RSA-AES-128-CBC-SHA"},
	0XC01F: {Name: "TLS_SRP_SHA_DSS_WITH_AES_128_CBC_SHA", ShortName: "SRP-DSS-AES-128-CBC-SHA"},
	0XC020: {Name: "TLS_SRP_SHA_WITH_AES_256_CBC_SHA", ShortName: "SRP-AES-256-CBC-SHA"},
	0XC021: {Name: "TLS_SRP_SHA_RSA_WITH_AES_256_CBC_SHA", ShortName: "SRP-RSA-AES-256-CBC-SHA"},
	0XC022: {Name: "TLS_SRP_SHA_DSS_WITH_AES_256_CBC_SHA", ShortName: "SRP-DSS-AES-256-CBC-SHA"},
	0XC023: {Name: "TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256", ShortName: "ECDHE-ECDSA-AES128-SHA256", ForwardSecret: true, EllipticCurve: true},
	0XC024: {Name: "TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA384", ShortName: "ECDHE-ECDSA-AES256-SHA384", ForwardSecret: true, EllipticCurve: true},
	0XC025: {Name: "TLS_ECDH_ECDSA_WITH_AES_128_CBC_SHA256", ShortName: "ECDH-ECDSA-AES128-SHA256", EllipticCurve: true},
	0XC026: {Name: "TLS_ECDH_ECDSA_WITH_AES_256_CBC_SHA384", ShortName: "ECDH-ECDSA-AES256-SHA384", EllipticCurve: true},
	0XC027: {Name: "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256", ShortName: "ECDHE-RSA-AES128-SHA256", ForwardSecret: true, EllipticCurve: true},
	0XC028: {Name: "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384", ShortName: "ECDHE-RSA-AES256-SHA384", ForwardSecret: true, EllipticCurve: true},
	0XC029: {Name: "TLS_ECDH_RSA_WITH_AES_128_CBC_SHA256", ShortName: "ECDH-RSA-AES128-SHA256", EllipticCurve: true},
	0XC02A: {Name: "TLS_ECDH_RSA_WITH_AES_256_CBC_SHA384", ShortName: "ECDH-RSA-AES256-SHA384", EllipticCurve: true},
	0XC02B: {Name: "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256", ShortName: "ECDHE-ECDSA-AES128-GCM-SHA256", ForwardSecret: true, EllipticCurve: true},
	0XC02C: {Name: "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384", ShortName: "ECDHE-ECDSA-AES256-GCM-SHA384", ForwardSecret: true, EllipticCurve: true},
	0XC02D: {Name: "TLS_ECDH_ECDSA_WITH_AES_128_GCM_SHA256", ShortName: "ECDH-ECDSA-AES128-GCM-SHA256", EllipticCurve: true},
	0XC02E: {Name: "TLS_ECDH_ECDSA_WITH_AES_256_GCM_SHA384", ShortName: "ECDH-ECDSA-AES256-GCM-SHA384", EllipticCurve: true},
	0XC02F: {Name: "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256", ShortName: "ECDHE-RSA-AES128-GCM-SHA256", ForwardSecret: true, EllipticCurve: true},
	0XC030: {Name: "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384", ShortName: "ECDHE-RSA-AES256-GCM-SHA384", ForwardSecret: true, EllipticCurve: true},
	0XC031: {Name: "TLS_ECDH_RSA_WITH_AES_128_GCM_SHA256", ShortName: "ECDH-RSA-AES128-GCM-SHA256", EllipticCurve: true},
	0XC032: {Name: "TLS_ECDH_RSA_WITH_AES_256_GCM_SHA384", ShortName: "ECDH-RSA-AES256-GCM-SHA384", EllipticCurve: true},
	0XC033: {Name: "TLS_ECDHE_PSK_WITH_RC4_128_SHA", ForwardSecret: true, EllipticCurve: true},
	0XC034: {Name: "TLS_ECDHE_PSK_WITH_3DES_EDE_CBC_SHA", ForwardSecret: true, EllipticCurve: true},
	0XC035: {Name: "TLS_ECDHE_PSK_WITH_AES_128_CBC_SHA", ForwardSecret: true, EllipticCurve: true},
	0XC036: {Name: "TLS_ECDHE_PSK_WITH_AES_256_CBC_SHA", ForwardSecret: true, EllipticCurve: true},
	0XC037: {Name: "TLS_ECDHE_PSK_WITH_AES_128_CBC_SHA256", ForwardSecret: true, EllipticCurve: true},
	0XC038: {Name: "TLS_ECDHE_PSK_WITH_AES_256_CBC_SHA384", ForwardSecret: true, EllipticCurve: true},
	0XC039: {Name: "TLS_ECDHE_PSK_WITH_NULL_SHA", ForwardSecret: true, EllipticCurve: true},
	0XC03A: {Name: "TLS_ECDHE_PSK_WITH_NULL_SHA256", ForwardSecret: true, EllipticCurve: true},
	0XC03B: {Name: "TLS_ECDHE_PSK_WITH_NULL_SHA384", ForwardSecret: true, EllipticCurve: true},
	0XC03C: {Name: "TLS_RSA_WITH_ARIA_128_CBC_SHA256"},
	0XC03D: {Name: "TLS_RSA_WITH_ARIA_256_CBC_SHA384"},
	0XC03E: {Name: "TLS_DH_DSS_WITH_ARIA_128_CBC_SHA256"},
	0XC03F: {Name: "TLS_DH_DSS_WITH_ARIA_256_CBC_SHA384"},
	0XC040: {Name: "TLS_DH_RSA_WITH_ARIA_128_CBC_SHA256"},
	0XC041: {Name: "TLS_DH_RSA_WITH_ARIA_256_CBC_SHA384"},
	0XC042: {Name: "TLS_DHE_DSS_WITH_ARIA_128_CBC_SHA256", ForwardSecret: true},
	0XC043: {Name: "TLS_DHE_DSS_WITH_ARIA_256_CBC_SHA384", ForwardSecret: true},
	0XC044: {Name: "TLS_DHE_RSA_WITH_ARIA_128_CBC_SHA256", ForwardSecret: true},
	0XC045: {Name: "TLS_DHE_RSA_WITH_ARIA_256_CBC_SHA384", ForwardSecret: true},
	0XC046: {Name: "TLS_DH_anon_WITH_ARIA_128_CBC_SHA256"},
	0XC047: {Name: "TLS_DH_anon_WITH_ARIA_256_CBC_SHA384"},
	0XC048: {Name: "TLS_ECDHE_ECDSA_WITH_ARIA_128_CBC_SHA256", ForwardSecret: true, EllipticCurve: true},
	0XC049: {Name: "TLS_ECDHE_ECDSA_WITH_ARIA_256_CBC_SHA384", ForwardSecret: true, EllipticCurve: true},
	0XC04A: {Name: "TLS_ECDH_ECDSA_WITH_ARIA_128_CBC_SHA256", EllipticCurve: true},
	0XC04B: {Name: "TLS_ECDH_ECDSA_WITH_ARIA_256_CBC_SHA384", EllipticCurve: true},
	0XC04C: {Name: "TLS_ECDHE_RSA_WITH_ARIA_128_CBC_SHA256", ForwardSecret: true, EllipticCurve: true},
	0XC04D: {Name: "TLS_ECDHE_RSA_WITH_ARIA_256_CBC_SHA384", ForwardSecret: true, EllipticCurve: true},
	0XC04E: {Name: "TLS_ECDH_RSA_WITH_ARIA_128_CBC_SHA256", EllipticCurve: true},
	0XC04F: {Name: "TLS_ECDH_RSA_WITH_ARIA_256_CBC_SHA384", EllipticCurve: true},
	0XC050: {Name: "TLS_RSA_WITH_ARIA_128_GCM_SHA256"},
	0XC051: {Name: "TLS_RSA_WITH_ARIA_256_GCM_SHA384"},
	0XC052: {Name: "TLS_DHE_RSA_WITH_ARIA_128_GCM_SHA256", ForwardSecret: true},
	0XC053: {Name: "TLS_DHE_RSA_WITH_ARIA_256_GCM_SHA384", ForwardSecret: true},
	0XC054: {Name: "TLS_DH_RSA_WITH_ARIA_128_GCM_SHA256"},
	0XC055: {Name: "TLS_DH_RSA_WITH_ARIA_256_GCM_SHA384"},
	0XC056: {Name: "TLS_DHE_DSS_WITH_ARIA_128_GCM_SHA256", ForwardSecret: true},
	0XC057: {Name: "TLS_DHE_DSS_WITH_ARIA_256_GCM_SHA384", ForwardSecret: true},
	0XC058: {Name: "TLS_DH_DSS_WITH_ARIA_128_GCM_SHA256"},
	0XC059: {Name: "TLS_DH_DSS_WITH_ARIA_256_GCM_SHA384"},
	0XC05A: {Name: "TLS_DH_anon_WITH_ARIA_128_GCM_SHA256"},
	0XC05B: {Name: "TLS_DH_anon_WITH_ARIA_256_GCM_SHA384"},
	0XC05C: {Name: "TLS_ECDHE_ECDSA_WITH_ARIA_128_GCM_SHA256", ForwardSecret: true, EllipticCurve: true},
	0XC05D: {Name: "TLS_ECDHE_ECDSA_WITH_ARIA_256_GCM_SHA384", ForwardSecret: true, EllipticCurve: true},
	0XC05E: {Name: "TLS_ECDH_ECDSA_WITH_ARIA_128_GCM_SHA256", EllipticCurve: true},
	0XC05F: {Name: "TLS_ECDH_ECDSA_WITH_ARIA_256_GCM_SHA384", EllipticCurve: true},
	0XC060: {Name: "TLS_ECDHE_RSA_WITH_ARIA_128_GCM_SHA256", ForwardSecret: true, EllipticCurve: true},
	0XC061: {Name: "TLS_ECDHE_RSA_WITH_ARIA_256_GCM_SHA384", ForwardSecret: true, EllipticCurve: true},
	0XC062: {Name: "TLS_ECDH_RSA_WITH_ARIA_128_GCM_SHA256", EllipticCurve: true},
	0XC063: {Name: "TLS_ECDH_RSA_WITH_ARIA_256_GCM_SHA384", EllipticCurve: true},
	0XC064: {Name: "TLS_PSK_WITH_ARIA_128_CBC_SHA256"},
	0XC065: {Name: "TLS_PSK_WITH_ARIA_256_CBC_SHA384"},
	0XC066: {Name: "TLS_DHE_PSK_WITH_ARIA_128_CBC_SHA256", ForwardSecret: true},
	0XC067: {Name: "TLS_DHE_PSK_WITH_ARIA_256_CBC_SHA384", ForwardSecret: true},
	0XC068: {Name: "TLS_RSA_PSK_WITH_ARIA_128_CBC_SHA256"},
	0XC069: {Name: "TLS_RSA_PSK_WITH_ARIA_256_CBC_SHA384"},
	0XC06A: {Name: "TLS_PSK_WITH_ARIA_128_GCM_SHA256"},
	0XC06B: {Name: "TLS_PSK_WITH_ARIA_256_GCM_SHA384"},
	0XC06C: {Name: "TLS_DHE_PSK_WITH_ARIA_128_GCM_SHA256", ForwardSecret: true},
	0XC06D: {Name: "TLS_DHE_PSK_WITH_ARIA_256_GCM_SHA384", ForwardSecret: true},
	0XC06E: {Name: "TLS_RSA_PSK_WITH_ARIA_128_GCM_SHA256"},
	0XC06F: {Name: "TLS_RSA_PSK_WITH_ARIA_256_GCM_SHA384"},
	0XC070: {Name: "TLS_ECDHE_PSK_WITH_ARIA_128_CBC_SHA256", ForwardSecret: true, EllipticCurve: true},
	0XC071: {Name: "TLS_ECDHE_PSK_WITH_ARIA_256_CBC_SHA384", ForwardSecret: true, EllipticCurve: true},
	0XC072: {Name: "TLS_ECDHE_ECDSA_WITH_CAMELLIA_128_CBC_SHA256", ForwardSecret: true, EllipticCurve: true},
	0XC073: {Name: "TLS_ECDHE_ECDSA_WITH_CAMELLIA_256_CBC_SHA384", ForwardSecret: true, EllipticCurve: true},
	0XC074: {Name: "TLS_ECDH_ECDSA_WITH_CAMELLIA_128_CBC_SHA256", EllipticCurve: true},
	0XC075: {Name: "TLS_ECDH_ECDSA_WITH_CAMELLIA_256_CBC_SHA384", EllipticCurve: true},
	0XC076: {Name: "TLS_ECDHE_RSA_WITH_CAMELLIA_128_CBC_SHA256", ForwardSecret: true, EllipticCurve: true},
	0XC077: {Name: "TLS_ECDHE_RSA_WITH_CAMELLIA_256_CBC_SHA384", ForwardSecret: true, EllipticCurve: true},
	0XC078: {Name: "TLS_ECDH_RSA_WITH_CAMELLIA_128_CBC_SHA256", EllipticCurve: true},
	0XC079: {Name: "TLS_ECDH_RSA_WITH_CAMELLIA_256_CBC_SHA384", EllipticCurve: true},
	0XC07A: {Name: "TLS_RSA_WITH_CAMELLIA_128_GCM_SHA256"},
	0XC07B: {Name: "TLS_RSA_WITH_CAMELLIA_256_GCM_SHA384"},
	0XC07C: {Name: "TLS_DHE_RSA_WITH_CAMELLIA_128_GCM_SHA256", ForwardSecret: true},
	0XC07D: {Name: "TLS_DHE_RSA_WITH_CAMELLIA_256_GCM_SHA384", ForwardSecret: true},
	0XC07E: {Name: "TLS_DH_RSA_WITH_CAMELLIA_128_GCM_SHA256"},
	0XC07F: {Name: "TLS_DH_RSA_WITH_CAMELLIA_256_GCM_SHA384"},
	0XC080: {Name: "TLS_DHE_DSS_WITH_CAMELLIA_128_GCM_SHA256", ForwardSecret: true},
	0XC081: {Name: "TLS_DHE_DSS_WITH_CAMELLIA_256_GCM_SHA384", ForwardSecret: true},
	0XC082: {Name: "TLS_DH_DSS_WITH_CAMELLIA_128_GCM_SHA256"},
	0XC083: {Name: "TLS_DH_DSS_WITH_CAMELLIA_256_GCM_SHA384"},
	0XC084: {Name: "TLS_DH_anon_WITH_CAMELLIA_128_GCM_SHA256"},
	0XC085: {Name: "TLS_DH_anon_WITH_CAMELLIA_256_GCM_SHA384"},
	0XC086: {Name: "TLS_ECDHE_ECDSA_WITH_CAMELLIA_128_GCM_SHA256", ForwardSecret: true, EllipticCurve: true},
	0XC087: {Name: "TLS_ECDHE_ECDSA_WITH_CAMELLIA_256_GCM_SHA384", ForwardSecret: true, EllipticCurve: true},
	0XC088: {Name: "TLS_ECDH_ECDSA_WITH_CAMELLIA_128_GCM_SHA256", EllipticCurve: true},
	0XC089: {Name: "TLS_ECDH_ECDSA_WITH_CAMELLIA_256_GCM_SHA384", EllipticCurve: true},
	0XC08A: {Name: "TLS_ECDHE_RSA_WITH_CAMELLIA_128_GCM_SHA256", ForwardSecret: true, EllipticCurve: true},
	0XC08B: {Name: "TLS_ECDHE_RSA_WITH_CAMELLIA_256_GCM_SHA384", ForwardSecret: true, EllipticCurve: true},
	0XC08C: {Name: "TLS_ECDH_RSA_WITH_CAMELLIA_128_GCM_SHA256", EllipticCurve: true},
	0XC08D: {Name: "TLS_ECDH_RSA_WITH_CAMELLIA_256_GCM_SHA384", EllipticCurve: true},
	0XC08E: {Name: "TLS_PSK_WITH_CAMELLIA_128_GCM_SHA256"},
	0XC08F: {Name: "TLS_PSK_WITH_CAMELLIA_256_GCM_SHA384"},
	0XC090: {Name: "TLS_DHE_PSK_WITH_CAMELLIA_128_GCM_SHA256", ForwardSecret: true},
	0XC091: {Name: "TLS_DHE_PSK_WITH_CAMELLIA_256_GCM_SHA384", ForwardSecret: true},
	0XC092: {Name: "TLS_RSA_PSK_WITH_CAMELLIA_128_GCM_SHA256"},
	0XC093: {Name: "TLS_RSA_PSK_WITH_CAMELLIA_256_GCM_SHA384"},
	0XC094: {Name: "TLS_PSK_WITH_CAMELLIA_128_CBC_SHA256"},
	0XC095: {Name: "TLS_PSK_WITH_CAMELLIA_256_CBC_SHA384"},
	0XC096: {Name: "TLS_DHE_PSK_WITH_CAMELLIA_128_CBC_SHA256", ForwardSecret: true},
	0XC097: {Name: "TLS_DHE_PSK_WITH_CAMELLIA_256_CBC_SHA384", ForwardSecret: true},
	0XC098: {Name: "TLS_RSA_PSK_WITH_CAMELLIA_128_CBC_SHA256"},
	0XC099: {Name: "TLS_RSA_PSK_WITH_CAMELLIA_256_CBC_SHA384"},
	0XC09A: {Name: "TLS_ECDHE_PSK_WITH_CAMELLIA_128_CBC_SHA256", ForwardSecret: true, EllipticCurve: true},
	0XC09B: {Name: "TLS_ECDHE_PSK_WITH_CAMELLIA_256_CBC_SHA384", ForwardSecret: true, EllipticCurve: true},
	0XC09C: {Name: "TLS_RSA_WITH_AES_128_CCM"},
	0XC09D: {Name: "TLS_RSA_WITH_AES_256_CCM"},
	0XC09E: {Name: "TLS_DHE_RSA_WITH_AES_128_CCM", ForwardSecret: true},
	0XC09F: {Name: "TLS_DHE_RSA_WITH_AES_256_CCM", ForwardSecret: true},
	0XC0A0: {Name: "TLS_RSA_WITH_AES_128_CCM_8"},
	0XC0A1: {Name: "TLS_RSA_WITH_AES_256_CCM_8"},
	0XC0A2: {Name: "TLS_DHE_RSA_WITH_AES_128_CCM_8", ForwardSecret: true},
	0XC0A3: {Name: "TLS_DHE_RSA_WITH_AES_256_CCM_8", ForwardSecret: true},
	0XC0A4: {Name: "TLS_PSK_WITH_AES_128_CCM"},
	0XC0A5: {Name: "TLS_PSK_WITH_AES_256_CCM"},
	0XC0A6: {Name: "TLS_DHE_PSK_WITH_AES_128_CCM", ForwardSecret: true},
	0XC0A7: {Name: "TLS_DHE_PSK_WITH_AES_256_CCM", ForwardSecret: true},
	0XC0A8: {Name: "TLS_PSK_WITH_AES_128_CCM_8"},
	0XC0A9: {Name: "TLS_PSK_WITH_AES_256_CCM_8"},
	0XC0AA: {Name: "TLS_PSK_DHE_WITH_AES_128_CCM_8", ForwardSecret: true},
	0XC0AB: {Name: "TLS_PSK_DHE_WITH_AES_256_CCM_8", ForwardSecret: true},
	0XC0AC: {Name: "TLS_ECDHE_ECDSA_WITH_AES_128_CCM", ForwardSecret: true, EllipticCurve: true},
	0XC0AD: {Name: "TLS_ECDHE_ECDSA_WITH_AES_256_CCM", ForwardSecret: true, EllipticCurve: true},
	0XC0AE: {Name: "TLS_ECDHE_ECDSA_WITH_AES_128_CCM_8", ForwardSecret: true, EllipticCurve: true},
	0XC0AF: {Name: "TLS_ECDHE_ECDSA_WITH_AES_256_CCM_8", ForwardSecret: true, EllipticCurve: true},
	// Non-IANA standardized cipher suites:
	// ChaCha20, Poly1305 cipher suites are defined in
	// https://tools.ietf.org/html/draft-agl-tls-chacha20poly1305-04
	0XCC13: {Name: "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256", ForwardSecret: true, EllipticCurve: true},
	0XCC14: {Name: "TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256", ForwardSecret: true, EllipticCurve: true},
	0XCC15: {Name: "TLS_DHE_RSA_WITH_CHACHA20_POLY1305_SHA256", ForwardSecret: true, EllipticCurve: true},
}

var Curves = map[CurveID]string{
	0:     "Unassigned",
	1:     "sect163k1",
	2:     "sect163r1",
	3:     "sect163r2",
	4:     "sect193r1",
	5:     "sect193r2",
	6:     "sect233k1",
	7:     "sect233r1",
	8:     "sect239k1",
	9:     "sect283k1",
	10:    "sect283r1",
	11:    "sect409k1",
	12:    "sect409r1",
	13:    "sect571k1",
	14:    "sect571r1",
	15:    "secp160k1",
	16:    "secp160r1",
	17:    "secp160r2",
	18:    "secp192k1",
	19:    "secp192r1",
	20:    "secp224k1",
	21:    "secp224r1",
	22:    "secp256k1",
	23:    "secp256r1",
	24:    "secp384r1",
	25:    "secp521r1",
	26:    "brainpoolP256r1",
	27:    "brainpoolP384r1",
	28:    "brainpoolP512r1",
	65281: "arbitrary_explicit_prime_curves",
	65282: "arbitrary_explicit_char2_curves",
}
