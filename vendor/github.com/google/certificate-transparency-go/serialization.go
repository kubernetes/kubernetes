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

package ct

import (
	"crypto"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/google/certificate-transparency-go/tls"
	"github.com/google/certificate-transparency-go/x509"
)

// SerializeSCTSignatureInput serializes the passed in sct and log entry into
// the correct format for signing.
func SerializeSCTSignatureInput(sct SignedCertificateTimestamp, entry LogEntry) ([]byte, error) {
	switch sct.SCTVersion {
	case V1:
		input := CertificateTimestamp{
			SCTVersion:    sct.SCTVersion,
			SignatureType: CertificateTimestampSignatureType,
			Timestamp:     sct.Timestamp,
			EntryType:     entry.Leaf.TimestampedEntry.EntryType,
			Extensions:    sct.Extensions,
		}
		switch entry.Leaf.TimestampedEntry.EntryType {
		case X509LogEntryType:
			input.X509Entry = entry.Leaf.TimestampedEntry.X509Entry
		case PrecertLogEntryType:
			input.PrecertEntry = &PreCert{
				IssuerKeyHash:  entry.Leaf.TimestampedEntry.PrecertEntry.IssuerKeyHash,
				TBSCertificate: entry.Leaf.TimestampedEntry.PrecertEntry.TBSCertificate,
			}
		case XJSONLogEntryType:
			input.JSONEntry = entry.Leaf.TimestampedEntry.JSONEntry
		default:
			return nil, fmt.Errorf("unsupported entry type %s", entry.Leaf.TimestampedEntry.EntryType)
		}
		return tls.Marshal(input)
	default:
		return nil, fmt.Errorf("unknown SCT version %d", sct.SCTVersion)
	}
}

// SerializeSTHSignatureInput serializes the passed in STH into the correct
// format for signing.
func SerializeSTHSignatureInput(sth SignedTreeHead) ([]byte, error) {
	switch sth.Version {
	case V1:
		if len(sth.SHA256RootHash) != crypto.SHA256.Size() {
			return nil, fmt.Errorf("invalid TreeHash length, got %d expected %d", len(sth.SHA256RootHash), crypto.SHA256.Size())
		}

		input := TreeHeadSignature{
			Version:        sth.Version,
			SignatureType:  TreeHashSignatureType,
			Timestamp:      sth.Timestamp,
			TreeSize:       sth.TreeSize,
			SHA256RootHash: sth.SHA256RootHash,
		}
		return tls.Marshal(input)
	default:
		return nil, fmt.Errorf("unsupported STH version %d", sth.Version)
	}
}

// CreateX509MerkleTreeLeaf generates a MerkleTreeLeaf for an X509 cert
func CreateX509MerkleTreeLeaf(cert ASN1Cert, timestamp uint64) *MerkleTreeLeaf {
	return &MerkleTreeLeaf{
		Version:  V1,
		LeafType: TimestampedEntryLeafType,
		TimestampedEntry: &TimestampedEntry{
			Timestamp: timestamp,
			EntryType: X509LogEntryType,
			X509Entry: &cert,
		},
	}
}

// CreateJSONMerkleTreeLeaf creates the merkle tree leaf for json data.
func CreateJSONMerkleTreeLeaf(data interface{}, timestamp uint64) *MerkleTreeLeaf {
	jsonData, err := json.Marshal(AddJSONRequest{Data: data})
	if err != nil {
		return nil
	}
	// Match the JSON serialization implemented by json-c
	jsonStr := strings.Replace(string(jsonData), ":", ": ", -1)
	jsonStr = strings.Replace(jsonStr, ",", ", ", -1)
	jsonStr = strings.Replace(jsonStr, "{", "{ ", -1)
	jsonStr = strings.Replace(jsonStr, "}", " }", -1)
	jsonStr = strings.Replace(jsonStr, "/", `\/`, -1)
	// TODO: Pending google/certificate-transparency#1243, replace with
	// ObjectHash once supported by CT server.

	return &MerkleTreeLeaf{
		Version:  V1,
		LeafType: TimestampedEntryLeafType,
		TimestampedEntry: &TimestampedEntry{
			Timestamp: timestamp,
			EntryType: XJSONLogEntryType,
			JSONEntry: &JSONDataEntry{Data: []byte(jsonStr)},
		},
	}
}

// MerkleTreeLeafFromRawChain generates a MerkleTreeLeaf from a chain (in DER-encoded form) and timestamp.
func MerkleTreeLeafFromRawChain(rawChain []ASN1Cert, etype LogEntryType, timestamp uint64) (*MerkleTreeLeaf, error) {
	// Need at most 3 of the chain
	count := 3
	if count > len(rawChain) {
		count = len(rawChain)
	}
	chain := make([]*x509.Certificate, count)
	for i := range chain {
		cert, err := x509.ParseCertificate(rawChain[i].Data)
		if err != nil {
			return nil, fmt.Errorf("failed to parse chain[%d] cert: %v", i, err)
		}
		chain[i] = cert
	}
	return MerkleTreeLeafFromChain(chain, etype, timestamp)
}

// MerkleTreeLeafFromChain generates a MerkleTreeLeaf from a chain and timestamp.
func MerkleTreeLeafFromChain(chain []*x509.Certificate, etype LogEntryType, timestamp uint64) (*MerkleTreeLeaf, error) {
	leaf := MerkleTreeLeaf{
		Version:  V1,
		LeafType: TimestampedEntryLeafType,
		TimestampedEntry: &TimestampedEntry{
			EntryType: etype,
			Timestamp: timestamp,
		},
	}
	if etype == X509LogEntryType {
		leaf.TimestampedEntry.X509Entry = &ASN1Cert{Data: chain[0].Raw}
		return &leaf, nil
	}
	if etype != PrecertLogEntryType {
		return nil, fmt.Errorf("unknown LogEntryType %d", etype)
	}

	// Pre-certs are more complicated. First, parse the leaf pre-cert and its
	// putative issuer.
	if len(chain) < 2 {
		return nil, fmt.Errorf("no issuer cert available for precert leaf building")
	}
	issuer := chain[1]
	cert := chain[0]

	var preIssuer *x509.Certificate
	if IsPreIssuer(issuer) {
		// Replace the cert's issuance information with details from the pre-issuer.
		preIssuer = issuer

		// The issuer of the pre-cert is not going to be the issuer of the final
		// cert.  Change to use the final issuer's key hash.
		if len(chain) < 3 {
			return nil, fmt.Errorf("no issuer cert available for pre-issuer")
		}
		issuer = chain[2]
	}

	// Next, post-process the DER-encoded TBSCertificate, to remove the CT poison
	// extension and possibly update the issuer field.
	defangedTBS, err := x509.BuildPrecertTBS(cert.RawTBSCertificate, preIssuer)
	if err != nil {
		return nil, fmt.Errorf("failed to remove poison extension: %v", err)
	}

	leaf.TimestampedEntry.EntryType = PrecertLogEntryType
	leaf.TimestampedEntry.PrecertEntry = &PreCert{
		IssuerKeyHash:  sha256.Sum256(issuer.RawSubjectPublicKeyInfo),
		TBSCertificate: defangedTBS,
	}
	return &leaf, nil
}

// IsPreIssuer indicates whether a certificate is a pre-cert issuer with the specific
// certificate transparency extended key usage.
func IsPreIssuer(issuer *x509.Certificate) bool {
	for _, eku := range issuer.ExtKeyUsage {
		if eku == x509.ExtKeyUsageCertificateTransparency {
			return true
		}
	}
	return false
}

// LogEntryFromLeaf converts a LeafEntry object (which has the raw leaf data after JSON parsing)
// into a LogEntry object (which includes x509.Certificate objects, after TLS and ASN.1 parsing).
// Note that this function may return a valid LogEntry object and a non-nil error value, when
// the error indicates a non-fatal parsing error (of type x509.NonFatalErrors).
func LogEntryFromLeaf(index int64, leafEntry *LeafEntry) (*LogEntry, error) {
	var leaf MerkleTreeLeaf
	if rest, err := tls.Unmarshal(leafEntry.LeafInput, &leaf); err != nil {
		return nil, fmt.Errorf("failed to unmarshal MerkleTreeLeaf for index %d: %v", index, err)
	} else if len(rest) > 0 {
		return nil, fmt.Errorf("trailing data (%d bytes) after MerkleTreeLeaf for index %d", len(rest), index)
	}

	var err error
	entry := LogEntry{Index: index, Leaf: leaf}
	switch leaf.TimestampedEntry.EntryType {
	case X509LogEntryType:
		var certChain CertificateChain
		if rest, err := tls.Unmarshal(leafEntry.ExtraData, &certChain); err != nil {
			return nil, fmt.Errorf("failed to unmarshal ExtraData for index %d: %v", index, err)
		} else if len(rest) > 0 {
			return nil, fmt.Errorf("trailing data (%d bytes) after CertificateChain for index %d", len(rest), index)
		}
		entry.Chain = certChain.Entries
		entry.X509Cert, err = leaf.X509Certificate()
		if _, ok := err.(x509.NonFatalErrors); !ok && err != nil {
			return nil, fmt.Errorf("failed to parse certificate in MerkleTreeLeaf for index %d: %v", index, err)
		}

	case PrecertLogEntryType:
		var precertChain PrecertChainEntry
		if rest, err := tls.Unmarshal(leafEntry.ExtraData, &precertChain); err != nil {
			return nil, fmt.Errorf("failed to unmarshal PrecertChainEntry for index %d: %v", index, err)
		} else if len(rest) > 0 {
			return nil, fmt.Errorf("trailing data (%d bytes) after PrecertChainEntry for index %d", len(rest), index)
		}
		entry.Chain = precertChain.CertificateChain
		var tbsCert *x509.Certificate
		tbsCert, err = leaf.Precertificate()
		if _, ok := err.(x509.NonFatalErrors); !ok && err != nil {
			return nil, fmt.Errorf("failed to parse precertificate in MerkleTreeLeaf for index %d: %v", index, err)
		}
		entry.Precert = &Precertificate{
			Submitted:      precertChain.PreCertificate,
			IssuerKeyHash:  leaf.TimestampedEntry.PrecertEntry.IssuerKeyHash,
			TBSCertificate: tbsCert,
		}

	default:
		return nil, fmt.Errorf("saw unknown entry type at index %d: %v", index, leaf.TimestampedEntry.EntryType)
	}
	// err may hold a x509.NonFatalErrors object.
	return &entry, err
}
