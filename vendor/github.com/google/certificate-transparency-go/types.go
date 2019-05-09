// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package ct holds core types and utilities for Certificate Transparency.
package ct

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"

	"github.com/google/certificate-transparency-go/tls"
	"github.com/google/certificate-transparency-go/x509"
)

///////////////////////////////////////////////////////////////////////////////
// The following structures represent those outlined in RFC6962; any section
// numbers mentioned refer to that RFC.
///////////////////////////////////////////////////////////////////////////////

// LogEntryType represents the LogEntryType enum from section 3.1:
//   enum { x509_entry(0), precert_entry(1), (65535) } LogEntryType;
type LogEntryType tls.Enum // tls:"maxval:65535"

// LogEntryType constants from section 3.1.
const (
	X509LogEntryType    LogEntryType = 0
	PrecertLogEntryType LogEntryType = 1
	XJSONLogEntryType   LogEntryType = 0x8000 // Experimental.  Don't rely on this!
)

func (e LogEntryType) String() string {
	switch e {
	case X509LogEntryType:
		return "X509LogEntryType"
	case PrecertLogEntryType:
		return "PrecertLogEntryType"
	case XJSONLogEntryType:
		return "XJSONLogEntryType"
	default:
		return fmt.Sprintf("UnknownEntryType(%d)", e)
	}
}

// RFC6962 section 2.1 requires a prefix byte on hash inputs for second preimage resistance.
const (
	TreeLeafPrefix = byte(0x00)
	TreeNodePrefix = byte(0x01)
)

// MerkleLeafType represents the MerkleLeafType enum from section 3.4:
//   enum { timestamped_entry(0), (255) } MerkleLeafType;
type MerkleLeafType tls.Enum // tls:"maxval:255"

// TimestampedEntryLeafType is the only defined MerkleLeafType constant from section 3.4.
const TimestampedEntryLeafType MerkleLeafType = 0 // Entry type for an SCT

func (m MerkleLeafType) String() string {
	switch m {
	case TimestampedEntryLeafType:
		return "TimestampedEntryLeafType"
	default:
		return fmt.Sprintf("UnknownLeafType(%d)", m)
	}
}

// Version represents the Version enum from section 3.2:
//   enum { v1(0), (255) } Version;
type Version tls.Enum // tls:"maxval:255"

// CT Version constants from section 3.2.
const (
	V1 Version = 0
)

func (v Version) String() string {
	switch v {
	case V1:
		return "V1"
	default:
		return fmt.Sprintf("UnknownVersion(%d)", v)
	}
}

// SignatureType differentiates STH signatures from SCT signatures, see section 3.2.
//   enum { certificate_timestamp(0), tree_hash(1), (255) } SignatureType;
type SignatureType tls.Enum // tls:"maxval:255"

// SignatureType constants from section 3.2.
const (
	CertificateTimestampSignatureType SignatureType = 0
	TreeHashSignatureType             SignatureType = 1
)

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

// ASN1Cert type for holding the raw DER bytes of an ASN.1 Certificate
// (section 3.1).
type ASN1Cert struct {
	Data []byte `tls:"minlen:1,maxlen:16777215"`
}

// LogID holds the hash of the Log's public key (section 3.2).
// TODO(pphaneuf): Users should be migrated to the one in the logid package.
type LogID struct {
	KeyID [sha256.Size]byte
}

// PreCert represents a Precertificate (section 3.2).
type PreCert struct {
	IssuerKeyHash  [sha256.Size]byte
	TBSCertificate []byte `tls:"minlen:1,maxlen:16777215"` // DER-encoded TBSCertificate
}

// CTExtensions is a representation of the raw bytes of any CtExtension
// structure (see section 3.2).
// nolint: golint
type CTExtensions []byte // tls:"minlen:0,maxlen:65535"`

// MerkleTreeNode represents an internal node in the CT tree.
type MerkleTreeNode []byte

// ConsistencyProof represents a CT consistency proof (see sections 2.1.2 and
// 4.4).
type ConsistencyProof []MerkleTreeNode

// AuditPath represents a CT inclusion proof (see sections 2.1.1 and 4.5).
type AuditPath []MerkleTreeNode

// LeafInput represents a serialized MerkleTreeLeaf structure.
type LeafInput []byte

// DigitallySigned is a local alias for tls.DigitallySigned so that we can
// attach a MarshalJSON method.
type DigitallySigned tls.DigitallySigned

// FromBase64String populates the DigitallySigned structure from the base64 data passed in.
// Returns an error if the base64 data is invalid.
func (d *DigitallySigned) FromBase64String(b64 string) error {
	raw, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return fmt.Errorf("failed to unbase64 DigitallySigned: %v", err)
	}
	var ds tls.DigitallySigned
	if rest, err := tls.Unmarshal(raw, &ds); err != nil {
		return fmt.Errorf("failed to unmarshal DigitallySigned: %v", err)
	} else if len(rest) > 0 {
		return fmt.Errorf("trailing data (%d bytes) after DigitallySigned", len(rest))
	}
	*d = DigitallySigned(ds)
	return nil
}

// Base64String returns the base64 representation of the DigitallySigned struct.
func (d DigitallySigned) Base64String() (string, error) {
	b, err := tls.Marshal(d)
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

// RawLogEntry represents the (TLS-parsed) contents of an entry in a CT log.
type RawLogEntry struct {
	// Index is a position of the entry in the log.
	Index int64
	// Leaf is a parsed Merkle leaf hash input.
	Leaf MerkleTreeLeaf
	// Cert is:
	// - A certificate if Leaf.TimestampedEntry.EntryType is X509LogEntryType.
	// - A precertificate if Leaf.TimestampedEntry.EntryType is
	//   PrecertLogEntryType, in the form of a DER-encoded Certificate as
	//   originally added (which includes the poison extension and a signature
	//   generated over the pre-cert by the pre-cert issuer).
	// - Empty otherwise.
	Cert ASN1Cert
	// Chain is the issuing certificate chain starting with the issuer of Cert,
	// or an empty slice if Cert is empty.
	Chain []ASN1Cert
}

// LogEntry represents the (parsed) contents of an entry in a CT log.  This is described
// in section 3.1, but note that this structure does *not* match the TLS structure
// defined there (the TLS structure is never used directly in RFC6962).
type LogEntry struct {
	Index int64
	Leaf  MerkleTreeLeaf
	// Exactly one of the following three fields should be non-empty.
	X509Cert *x509.Certificate // Parsed X.509 certificate
	Precert  *Precertificate   // Extracted precertificate
	JSONData []byte

	// Chain holds the issuing certificate chain, starting with the
	// issuer of the leaf certificate / pre-certificate.
	Chain []ASN1Cert
}

// PrecertChainEntry holds an precertificate together with a validation chain
// for it; see section 3.1.
type PrecertChainEntry struct {
	PreCertificate   ASN1Cert   `tls:"minlen:1,maxlen:16777215"`
	CertificateChain []ASN1Cert `tls:"minlen:0,maxlen:16777215"`
}

// CertificateChain holds a chain of certificates, as returned as extra data
// for get-entries (section 4.6).
type CertificateChain struct {
	Entries []ASN1Cert `tls:"minlen:0,maxlen:16777215"`
}

// JSONDataEntry holds arbitrary data.
type JSONDataEntry struct {
	Data []byte `tls:"minlen:0,maxlen:1677215"`
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
// after base64 decoding; see sections 3.5 and 4.3.
type SignedTreeHead struct {
	Version           Version         `json:"sth_version"`         // The version of the protocol to which the STH conforms
	TreeSize          uint64          `json:"tree_size"`           // The number of entries in the new tree
	Timestamp         uint64          `json:"timestamp"`           // The time at which the STH was created
	SHA256RootHash    SHA256Hash      `json:"sha256_root_hash"`    // The root hash of the log's Merkle tree
	TreeHeadSignature DigitallySigned `json:"tree_head_signature"` // Log's signature over a TLS-encoded TreeHeadSignature
	LogID             SHA256Hash      `json:"log_id"`              // The SHA256 hash of the log's public key
}

// TreeHeadSignature holds the data over which the signature in an STH is
// generated; see section 3.5
type TreeHeadSignature struct {
	Version        Version       `tls:"maxval:255"`
	SignatureType  SignatureType `tls:"maxval:255"` // == TreeHashSignatureType
	Timestamp      uint64
	TreeSize       uint64
	SHA256RootHash SHA256Hash
}

// SignedCertificateTimestamp represents the structure returned by the
// add-chain and add-pre-chain methods after base64 decoding; see sections
// 3.2, 4.1 and 4.2.
type SignedCertificateTimestamp struct {
	SCTVersion Version `tls:"maxval:255"`
	LogID      LogID
	Timestamp  uint64
	Extensions CTExtensions    `tls:"minlen:0,maxlen:65535"`
	Signature  DigitallySigned // Signature over TLS-encoded CertificateTimestamp
}

// CertificateTimestamp is the collection of data that the signature in an
// SCT is over; see section 3.2.
type CertificateTimestamp struct {
	SCTVersion    Version       `tls:"maxval:255"`
	SignatureType SignatureType `tls:"maxval:255"`
	Timestamp     uint64
	EntryType     LogEntryType   `tls:"maxval:65535"`
	X509Entry     *ASN1Cert      `tls:"selector:EntryType,val:0"`
	PrecertEntry  *PreCert       `tls:"selector:EntryType,val:1"`
	JSONEntry     *JSONDataEntry `tls:"selector:EntryType,val:32768"`
	Extensions    CTExtensions   `tls:"minlen:0,maxlen:65535"`
}

func (s SignedCertificateTimestamp) String() string {
	return fmt.Sprintf("{Version:%d LogId:%s Timestamp:%d Extensions:'%s' Signature:%v}", s.SCTVersion,
		base64.StdEncoding.EncodeToString(s.LogID.KeyID[:]),
		s.Timestamp,
		s.Extensions,
		s.Signature)
}

// TimestampedEntry is part of the MerkleTreeLeaf structure; see section 3.4.
type TimestampedEntry struct {
	Timestamp    uint64
	EntryType    LogEntryType   `tls:"maxval:65535"`
	X509Entry    *ASN1Cert      `tls:"selector:EntryType,val:0"`
	PrecertEntry *PreCert       `tls:"selector:EntryType,val:1"`
	JSONEntry    *JSONDataEntry `tls:"selector:EntryType,val:32768"`
	Extensions   CTExtensions   `tls:"minlen:0,maxlen:65535"`
}

// MerkleTreeLeaf represents the deserialized structure of the hash input for the
// leaves of a log's Merkle tree; see section 3.4.
type MerkleTreeLeaf struct {
	Version          Version           `tls:"maxval:255"`
	LeafType         MerkleLeafType    `tls:"maxval:255"`
	TimestampedEntry *TimestampedEntry `tls:"selector:LeafType,val:0"`
}

// Precertificate represents the parsed CT Precertificate structure.
type Precertificate struct {
	// DER-encoded pre-certificate as originally added, which includes a
	// poison extension and a signature generated over the pre-cert by
	// the pre-cert issuer (which might differ from the issuer of the final
	// cert, see RFC6962 s3.1).
	Submitted ASN1Cert
	// SHA256 hash of the issuing key
	IssuerKeyHash [sha256.Size]byte
	// Parsed TBSCertificate structure, held in an x509.Certificate for convenience.
	TBSCertificate *x509.Certificate
}

// X509Certificate returns the X.509 Certificate contained within the
// MerkleTreeLeaf.
func (m *MerkleTreeLeaf) X509Certificate() (*x509.Certificate, error) {
	if m.TimestampedEntry.EntryType != X509LogEntryType {
		return nil, fmt.Errorf("cannot call X509Certificate on a MerkleTreeLeaf that is not an X509 entry")
	}
	return x509.ParseCertificate(m.TimestampedEntry.X509Entry.Data)
}

// Precertificate returns the X.509 Precertificate contained within the MerkleTreeLeaf.
//
// The returned precertificate is embedded in an x509.Certificate, but is in the
// form stored internally in the log rather than the original submitted form
// (i.e. it does not include the poison extension and any changes to reflect the
// final certificate's issuer have been made; see x509.BuildPrecertTBS).
func (m *MerkleTreeLeaf) Precertificate() (*x509.Certificate, error) {
	if m.TimestampedEntry.EntryType != PrecertLogEntryType {
		return nil, fmt.Errorf("cannot call Precertificate on a MerkleTreeLeaf that is not a precert entry")
	}
	return x509.ParseTBSCertificate(m.TimestampedEntry.PrecertEntry.TBSCertificate)
}

// APIEndpoint is a string that represents one of the Certificate Transparency
// Log API endpoints.
type APIEndpoint string

// Certificate Transparency Log API endpoints; see section 4.
// WARNING: Should match the URI paths without the "/ct/v1/" prefix.  If
// changing these constants, may need to change those too.
const (
	AddChainStr          APIEndpoint = "add-chain"
	AddPreChainStr       APIEndpoint = "add-pre-chain"
	GetSTHStr            APIEndpoint = "get-sth"
	GetEntriesStr        APIEndpoint = "get-entries"
	GetProofByHashStr    APIEndpoint = "get-proof-by-hash"
	GetSTHConsistencyStr APIEndpoint = "get-sth-consistency"
	GetRootsStr          APIEndpoint = "get-roots"
	GetEntryAndProofStr  APIEndpoint = "get-entry-and-proof"
)

// URI paths for Log requests; see section 4.
// WARNING: Should match the API endpoints, with the "/ct/v1/" prefix.  If
// changing these constants, may need to change those too.
const (
	AddChainPath          = "/ct/v1/add-chain"
	AddPreChainPath       = "/ct/v1/add-pre-chain"
	GetSTHPath            = "/ct/v1/get-sth"
	GetEntriesPath        = "/ct/v1/get-entries"
	GetProofByHashPath    = "/ct/v1/get-proof-by-hash"
	GetSTHConsistencyPath = "/ct/v1/get-sth-consistency"
	GetRootsPath          = "/ct/v1/get-roots"
	GetEntryAndProofPath  = "/ct/v1/get-entry-and-proof"

	AddJSONPath = "/ct/v1/add-json" // Experimental addition
)

// AddChainRequest represents the JSON request body sent to the add-chain and
// add-pre-chain POST methods from sections 4.1 and 4.2.
type AddChainRequest struct {
	Chain [][]byte `json:"chain"`
}

// AddChainResponse represents the JSON response to the add-chain and
// add-pre-chain POST methods.
// An SCT represents a Log's promise to integrate a [pre-]certificate into the
// log within a defined period of time.
type AddChainResponse struct {
	SCTVersion Version `json:"sct_version"` // SCT structure version
	ID         []byte  `json:"id"`          // Log ID
	Timestamp  uint64  `json:"timestamp"`   // Timestamp of issuance
	Extensions string  `json:"extensions"`  // Holder for any CT extensions
	Signature  []byte  `json:"signature"`   // Log signature for this SCT
}

// AddJSONRequest represents the JSON request body sent to the add-json POST method.
// The corresponding response re-uses AddChainResponse.
// This is an experimental addition not covered by RFC6962.
type AddJSONRequest struct {
	Data interface{} `json:"data"`
}

// GetSTHResponse respresents the JSON response to the get-sth GET method from section 4.3.
type GetSTHResponse struct {
	TreeSize          uint64 `json:"tree_size"`           // Number of certs in the current tree
	Timestamp         uint64 `json:"timestamp"`           // Time that the tree was created
	SHA256RootHash    []byte `json:"sha256_root_hash"`    // Root hash of the tree
	TreeHeadSignature []byte `json:"tree_head_signature"` // Log signature for this STH
}

// ToSignedTreeHead creates a SignedTreeHead from the GetSTHResponse.
func (r *GetSTHResponse) ToSignedTreeHead() (*SignedTreeHead, error) {
	sth := SignedTreeHead{
		TreeSize:  r.TreeSize,
		Timestamp: r.Timestamp,
	}

	if len(r.SHA256RootHash) != sha256.Size {
		return nil, fmt.Errorf("sha256_root_hash is invalid length, expected %d got %d", sha256.Size, len(r.SHA256RootHash))
	}
	copy(sth.SHA256RootHash[:], r.SHA256RootHash)

	var ds DigitallySigned
	if rest, err := tls.Unmarshal(r.TreeHeadSignature, &ds); err != nil {
		return nil, fmt.Errorf("tls.Unmarshal(): %s", err)
	} else if len(rest) > 0 {
		return nil, fmt.Errorf("trailing data (%d bytes) after DigitallySigned", len(rest))
	}
	sth.TreeHeadSignature = ds

	return &sth, nil
}

// GetSTHConsistencyResponse represents the JSON response to the get-sth-consistency
// GET method from section 4.4.  (The corresponding GET request has parameters 'first' and
// 'second'.)
type GetSTHConsistencyResponse struct {
	Consistency [][]byte `json:"consistency"`
}

// GetProofByHashResponse represents the JSON response to the get-proof-by-hash GET
// method from section 4.5.  (The corresponding GET request has parameters 'hash'
// and 'tree_size'.)
type GetProofByHashResponse struct {
	LeafIndex int64    `json:"leaf_index"` // The 0-based index of the end entity corresponding to the "hash" parameter.
	AuditPath [][]byte `json:"audit_path"` // An array of base64-encoded Merkle Tree nodes proving the inclusion of the chosen certificate.
}

// LeafEntry represents a leaf in the Log's Merkle tree, as returned by the get-entries
// GET method from section 4.6.
type LeafEntry struct {
	// LeafInput is a TLS-encoded MerkleTreeLeaf
	LeafInput []byte `json:"leaf_input"`
	// ExtraData holds (unsigned) extra data, normally the cert validation chain.
	ExtraData []byte `json:"extra_data"`
}

// GetEntriesResponse respresents the JSON response to the get-entries GET method
// from section 4.6.
type GetEntriesResponse struct {
	Entries []LeafEntry `json:"entries"` // the list of returned entries
}

// GetRootsResponse represents the JSON response to the get-roots GET method from section 4.7.
type GetRootsResponse struct {
	Certificates []string `json:"certificates"`
}

// GetEntryAndProofResponse represents the JSON response to the get-entry-and-proof
// GET method from section 4.8. (The corresponding GET request has parameters 'leaf_index'
// and 'tree_size'.)
type GetEntryAndProofResponse struct {
	LeafInput []byte   `json:"leaf_input"` // the entry itself
	ExtraData []byte   `json:"extra_data"` // any chain provided when the entry was added to the log
	AuditPath [][]byte `json:"audit_path"` // the corresponding proof
}
