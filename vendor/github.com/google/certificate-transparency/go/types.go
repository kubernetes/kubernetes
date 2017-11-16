package ct

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"

	"github.com/google/certificate-transparency/go/x509"
)

const (
	issuerKeyHashLength = 32
)

///////////////////////////////////////////////////////////////////////////////
// The following structures represent those outlined in the RFC6962 document:
///////////////////////////////////////////////////////////////////////////////

// LogEntryType represents the LogEntryType enum from section 3.1 of the RFC:
//   enum { x509_entry(0), precert_entry(1), (65535) } LogEntryType;
type LogEntryType uint16

func (e LogEntryType) String() string {
	switch e {
	case X509LogEntryType:
		return "X509LogEntryType"
	case PrecertLogEntryType:
		return "PrecertLogEntryType"
	}
	panic(fmt.Sprintf("No string defined for LogEntryType constant value %d", e))
}

// LogEntryType constants, see section 3.1 of RFC6962.
const (
	X509LogEntryType    LogEntryType = 0
	PrecertLogEntryType LogEntryType = 1
)

// MerkleLeafType represents the MerkleLeafType enum from section 3.4 of the
// RFC: enum { timestamped_entry(0), (255) } MerkleLeafType;
type MerkleLeafType uint8

func (m MerkleLeafType) String() string {
	switch m {
	case TimestampedEntryLeafType:
		return "TimestampedEntryLeafType"
	default:
		return fmt.Sprintf("UnknownLeafType(%d)", m)
	}
}

// MerkleLeafType constants, see section 3.4 of the RFC.
const (
	TimestampedEntryLeafType MerkleLeafType = 0 // Entry type for an SCT
)

// Version represents the Version enum from section 3.2 of the RFC:
// enum { v1(0), (255) } Version;
type Version uint8

func (v Version) String() string {
	switch v {
	case V1:
		return "V1"
	default:
		return fmt.Sprintf("UnknownVersion(%d)", v)
	}
}

// CT Version constants, see section 3.2 of the RFC.
const (
	V1 Version = 0
)

// SignatureType differentiates STH signatures from SCT signatures, see RFC
// section 3.2
type SignatureType uint8

func (st SignatureType) String() string {
	switch st {
	case CertificateTimestampSignatureType:
		return "CertificateTimestamp"
	case TreeHashSignatureType:
		return "TreeHash"
	default:
		return fmt.Sprintf("UnknownSignatureType(%d)", st)
	}
}

// SignatureType constants, see RFC section 3.2
const (
	CertificateTimestampSignatureType SignatureType = 0
	TreeHashSignatureType             SignatureType = 1
)

// ASN1Cert type for holding the raw DER bytes of an ASN.1 Certificate
// (section 3.1)
type ASN1Cert []byte

// PreCert represents a Precertificate (section 3.2)
type PreCert struct {
	IssuerKeyHash  [issuerKeyHashLength]byte
	TBSCertificate []byte
}

// CTExtensions is a representation of the raw bytes of any CtExtension
// structure (see section 3.2)
type CTExtensions []byte

// MerkleTreeNode represents an internal node in the CT tree
type MerkleTreeNode []byte

// ConsistencyProof represents a CT consistency proof (see sections 2.1.2 and
// 4.4)
type ConsistencyProof []MerkleTreeNode

// AuditPath represents a CT inclusion proof (see sections 2.1.1 and 4.5)
type AuditPath []MerkleTreeNode

// LeafInput represents a serialized MerkleTreeLeaf structure
type LeafInput []byte

// HashAlgorithm from the DigitallySigned struct
type HashAlgorithm byte

// HashAlgorithm constants
const (
	None   HashAlgorithm = 0
	MD5    HashAlgorithm = 1
	SHA1   HashAlgorithm = 2
	SHA224 HashAlgorithm = 3
	SHA256 HashAlgorithm = 4
	SHA384 HashAlgorithm = 5
	SHA512 HashAlgorithm = 6
)

func (h HashAlgorithm) String() string {
	switch h {
	case None:
		return "None"
	case MD5:
		return "MD5"
	case SHA1:
		return "SHA1"
	case SHA224:
		return "SHA224"
	case SHA256:
		return "SHA256"
	case SHA384:
		return "SHA384"
	case SHA512:
		return "SHA512"
	default:
		return fmt.Sprintf("UNKNOWN(%d)", h)
	}
}

// SignatureAlgorithm from the the DigitallySigned struct
type SignatureAlgorithm byte

// SignatureAlgorithm constants
const (
	Anonymous SignatureAlgorithm = 0
	RSA       SignatureAlgorithm = 1
	DSA       SignatureAlgorithm = 2
	ECDSA     SignatureAlgorithm = 3
)

func (s SignatureAlgorithm) String() string {
	switch s {
	case Anonymous:
		return "Anonymous"
	case RSA:
		return "RSA"
	case DSA:
		return "DSA"
	case ECDSA:
		return "ECDSA"
	default:
		return fmt.Sprintf("UNKNOWN(%d)", s)
	}
}

// DigitallySigned represents an RFC5246 DigitallySigned structure
type DigitallySigned struct {
	HashAlgorithm      HashAlgorithm
	SignatureAlgorithm SignatureAlgorithm
	Signature          []byte
}

// FromBase64String populates the DigitallySigned structure from the base64 data passed in.
// Returns an error if the base64 data is invalid.
func (d *DigitallySigned) FromBase64String(b64 string) error {
	raw, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return fmt.Errorf("failed to unbase64 DigitallySigned: %v", err)
	}
	ds, err := UnmarshalDigitallySigned(bytes.NewReader(raw))
	if err != nil {
		return fmt.Errorf("failed to unmarshal DigitallySigned: %v", err)
	}
	*d = *ds
	return nil
}

// Base64String returns the base64 representation of the DigitallySigned struct.
func (d DigitallySigned) Base64String() (string, error) {
	b, err := MarshalDigitallySigned(d)
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(b), nil
}

// MarshalJSON implements the json.Marshaller interface.
func (d DigitallySigned) MarshalJSON() ([]byte, error) {
	b64, err := d.Base64String()
	if err != nil {
		return []byte{}, err
	}
	return []byte(`"` + b64 + `"`), nil
}

// UnmarshalJSON implements the json.Unmarshaler interface.
func (d *DigitallySigned) UnmarshalJSON(b []byte) error {
	var content string
	if err := json.Unmarshal(b, &content); err != nil {
		return fmt.Errorf("failed to unmarshal DigitallySigned: %v", err)
	}
	return d.FromBase64String(content)
}

// LogEntry represents the contents of an entry in a CT log, see section 3.1.
type LogEntry struct {
	Index    int64
	Leaf     MerkleTreeLeaf
	X509Cert *x509.Certificate
	Precert  *Precertificate
	Chain    []ASN1Cert
}

// SHA256Hash represents the output from the SHA256 hash function.
type SHA256Hash [sha256.Size]byte

// FromBase64String populates the SHA256 struct with the contents of the base64 data passed in.
func (s *SHA256Hash) FromBase64String(b64 string) error {
	bs, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return fmt.Errorf("failed to unbase64 LogID: %v", err)
	}
	if len(bs) != sha256.Size {
		return fmt.Errorf("invalid SHA256 length, expected 32 but got %d", len(bs))
	}
	copy(s[:], bs)
	return nil
}

// Base64String returns the base64 representation of this SHA256Hash.
func (s SHA256Hash) Base64String() string {
	return base64.StdEncoding.EncodeToString(s[:])
}

// MarshalJSON implements the json.Marshaller interface for SHA256Hash.
func (s SHA256Hash) MarshalJSON() ([]byte, error) {
	return []byte(`"` + s.Base64String() + `"`), nil
}

// UnmarshalJSON implements the json.Unmarshaller interface.
func (s *SHA256Hash) UnmarshalJSON(b []byte) error {
	var content string
	if err := json.Unmarshal(b, &content); err != nil {
		return fmt.Errorf("failed to unmarshal SHA256Hash: %v", err)
	}
	return s.FromBase64String(content)
}

// SignedTreeHead represents the structure returned by the get-sth CT method
// after base64 decoding. See sections 3.5 and 4.3 in the RFC)
type SignedTreeHead struct {
	Version           Version         `json:"sth_version"`         // The version of the protocol to which the STH conforms
	TreeSize          uint64          `json:"tree_size"`           // The number of entries in the new tree
	Timestamp         uint64          `json:"timestamp"`           // The time at which the STH was created
	SHA256RootHash    SHA256Hash      `json:"sha256_root_hash"`    // The root hash of the log's Merkle tree
	TreeHeadSignature DigitallySigned `json:"tree_head_signature"` // The Log's signature for this STH (see RFC section 3.5)
	LogID             SHA256Hash      `json:"log_id"`              // The SHA256 hash of the log's public key
}

// SignedCertificateTimestamp represents the structure returned by the
// add-chain and add-pre-chain methods after base64 decoding. (see RFC sections
// 3.2 ,4.1 and 4.2)
type SignedCertificateTimestamp struct {
	SCTVersion Version    // The version of the protocol to which the SCT conforms
	LogID      SHA256Hash // the SHA-256 hash of the log's public key, calculated over
	// the DER encoding of the key represented as SubjectPublicKeyInfo.
	Timestamp  uint64          // Timestamp (in ms since unix epoc) at which the SCT was issued
	Extensions CTExtensions    // For future extensions to the protocol
	Signature  DigitallySigned // The Log's signature for this SCT
}

func (s SignedCertificateTimestamp) String() string {
	return fmt.Sprintf("{Version:%d LogId:%s Timestamp:%d Extensions:'%s' Signature:%v}", s.SCTVersion,
		base64.StdEncoding.EncodeToString(s.LogID[:]),
		s.Timestamp,
		s.Extensions,
		s.Signature)
}

// TimestampedEntry is part of the MerkleTreeLeaf structure.
// See RFC section 3.4
type TimestampedEntry struct {
	Timestamp    uint64
	EntryType    LogEntryType
	X509Entry    ASN1Cert
	PrecertEntry PreCert
	Extensions   CTExtensions
}

// MerkleTreeLeaf represents the deserialized sructure of the hash input for the
// leaves of a log's Merkle tree. See RFC section 3.4
type MerkleTreeLeaf struct {
	Version          Version          // the version of the protocol to which the MerkleTreeLeaf corresponds
	LeafType         MerkleLeafType   // The type of the leaf input, currently only TimestampedEntry can exist
	TimestampedEntry TimestampedEntry // The entry data itself
}

// Precertificate represents the parsed CT Precertificate structure.
type Precertificate struct {
	// Raw DER bytes of the precert
	Raw []byte
	// SHA256 hash of the issuing key
	IssuerKeyHash [issuerKeyHashLength]byte
	// Parsed TBSCertificate structure (held in an x509.Certificate for ease of
	// access.
	TBSCertificate x509.Certificate
}

// X509Certificate returns the X.509 Certificate contained within the
// MerkleTreeLeaf.
// Returns a pointer to an x509.Certificate or a non-nil error.
func (m *MerkleTreeLeaf) X509Certificate() (*x509.Certificate, error) {
	return x509.ParseCertificate(m.TimestampedEntry.X509Entry)
}

type sctError int

// Preallocate errors for performance
var (
	ErrInvalidVersion  error = sctError(1)
	ErrNotEnoughBuffer error = sctError(2)
)

func (e sctError) Error() string {
	switch e {
	case ErrInvalidVersion:
		return "invalid SCT version detected"
	case ErrNotEnoughBuffer:
		return "provided buffer was too small"
	default:
		return "unknown error"
	}
}
