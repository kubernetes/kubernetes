package ct

import (
	"bytes"
	"container/list"
	"crypto"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
)

// Variable size structure prefix-header byte lengths
const (
	CertificateLengthBytes      = 3
	PreCertificateLengthBytes   = 3
	ExtensionsLengthBytes       = 2
	CertificateChainLengthBytes = 3
	SignatureLengthBytes        = 2
)

// Max lengths
const (
	MaxCertificateLength = (1 << 24) - 1
	MaxExtensionsLength  = (1 << 16) - 1
)

func writeUint(w io.Writer, value uint64, numBytes int) error {
	buf := make([]uint8, numBytes)
	for i := 0; i < numBytes; i++ {
		buf[numBytes-i-1] = uint8(value & 0xff)
		value >>= 8
	}
	if value != 0 {
		return errors.New("numBytes was insufficiently large to represent value")
	}
	if _, err := w.Write(buf); err != nil {
		return err
	}
	return nil
}

func writeVarBytes(w io.Writer, value []byte, numLenBytes int) error {
	if err := writeUint(w, uint64(len(value)), numLenBytes); err != nil {
		return err
	}
	if _, err := w.Write(value); err != nil {
		return err
	}
	return nil
}

func readUint(r io.Reader, numBytes int) (uint64, error) {
	var l uint64
	for i := 0; i < numBytes; i++ {
		l <<= 8
		var t uint8
		if err := binary.Read(r, binary.BigEndian, &t); err != nil {
			return 0, err
		}
		l |= uint64(t)
	}
	return l, nil
}

// Reads a variable length array of bytes from |r|. |numLenBytes| specifies the
// number of (BigEndian) prefix-bytes which contain the length of the actual
// array data bytes that follow.
// Allocates an array to hold the contents and returns a slice view into it if
// the read was successful, or an error otherwise.
func readVarBytes(r io.Reader, numLenBytes int) ([]byte, error) {
	switch {
	case numLenBytes > 8:
		return nil, fmt.Errorf("numLenBytes too large (%d)", numLenBytes)
	case numLenBytes == 0:
		return nil, errors.New("numLenBytes should be > 0")
	}
	l, err := readUint(r, numLenBytes)
	if err != nil {
		return nil, err
	}
	data := make([]byte, l)
	n, err := r.Read(data)
	if err != nil {
		return nil, err
	}
	if n != int(l) {
		return nil, fmt.Errorf("short read: expected %d but got %d", l, n)
	}
	return data, nil
}

// Reads a list of ASN1Cert types from |r|
func readASN1CertList(r io.Reader, totalLenBytes int, elementLenBytes int) ([]ASN1Cert, error) {
	listBytes, err := readVarBytes(r, totalLenBytes)
	if err != nil {
		return []ASN1Cert{}, err
	}
	list := list.New()
	listReader := bytes.NewReader(listBytes)
	var entry []byte
	for err == nil {
		entry, err = readVarBytes(listReader, elementLenBytes)
		if err != nil {
			if err != io.EOF {
				return []ASN1Cert{}, err
			}
		} else {
			list.PushBack(entry)
		}
	}
	ret := make([]ASN1Cert, list.Len())
	i := 0
	for e := list.Front(); e != nil; e = e.Next() {
		ret[i] = e.Value.([]byte)
		i++
	}
	return ret, nil
}

// ReadTimestampedEntryInto parses the byte-stream representation of a
// TimestampedEntry from |r| and populates the struct |t| with the data.  See
// RFC section 3.4 for details on the format.
// Returns a non-nil error if there was a problem.
func ReadTimestampedEntryInto(r io.Reader, t *TimestampedEntry) error {
	var err error
	if err = binary.Read(r, binary.BigEndian, &t.Timestamp); err != nil {
		return err
	}
	if err = binary.Read(r, binary.BigEndian, &t.EntryType); err != nil {
		return err
	}
	switch t.EntryType {
	case X509LogEntryType:
		if t.X509Entry, err = readVarBytes(r, CertificateLengthBytes); err != nil {
			return err
		}
	case PrecertLogEntryType:
		if err := binary.Read(r, binary.BigEndian, &t.PrecertEntry.IssuerKeyHash); err != nil {
			return err
		}
		if t.PrecertEntry.TBSCertificate, err = readVarBytes(r, PreCertificateLengthBytes); err != nil {
			return err
		}
	default:
		return fmt.Errorf("unknown EntryType: %d", t.EntryType)
	}
	t.Extensions, err = readVarBytes(r, ExtensionsLengthBytes)
	return nil
}

// ReadMerkleTreeLeaf parses the byte-stream representation of a MerkleTreeLeaf
// and returns a pointer to a new MerkleTreeLeaf structure containing the
// parsed data.
// See RFC section 3.4 for details on the format.
// Returns a pointer to a new MerkleTreeLeaf or non-nil error if there was a
// problem
func ReadMerkleTreeLeaf(r io.Reader) (*MerkleTreeLeaf, error) {
	var m MerkleTreeLeaf
	if err := binary.Read(r, binary.BigEndian, &m.Version); err != nil {
		return nil, err
	}
	if m.Version != V1 {
		return nil, fmt.Errorf("unknown Version %d", m.Version)
	}
	if err := binary.Read(r, binary.BigEndian, &m.LeafType); err != nil {
		return nil, err
	}
	if m.LeafType != TimestampedEntryLeafType {
		return nil, fmt.Errorf("unknown LeafType %d", m.LeafType)
	}
	if err := ReadTimestampedEntryInto(r, &m.TimestampedEntry); err != nil {
		return nil, err
	}
	return &m, nil
}

// UnmarshalX509ChainArray unmarshalls the contents of the "chain:" entry in a
// GetEntries response in the case where the entry refers to an X509 leaf.
func UnmarshalX509ChainArray(b []byte) ([]ASN1Cert, error) {
	return readASN1CertList(bytes.NewReader(b), CertificateChainLengthBytes, CertificateLengthBytes)
}

// UnmarshalPrecertChainArray unmarshalls the contents of the "chain:" entry in
// a GetEntries response in the case where the entry refers to a Precertificate
// leaf.
func UnmarshalPrecertChainArray(b []byte) ([]ASN1Cert, error) {
	var chain []ASN1Cert

	reader := bytes.NewReader(b)
	// read the pre-cert entry:
	precert, err := readVarBytes(reader, CertificateLengthBytes)
	if err != nil {
		return chain, err
	}
	chain = append(chain, precert)
	// and then read and return the chain up to the root:
	remainingChain, err := readASN1CertList(reader, CertificateChainLengthBytes, CertificateLengthBytes)
	if err != nil {
		return chain, err
	}
	chain = append(chain, remainingChain...)
	return chain, nil
}

// UnmarshalDigitallySigned reconstructs a DigitallySigned structure from a Reader
func UnmarshalDigitallySigned(r io.Reader) (*DigitallySigned, error) {
	var h byte
	if err := binary.Read(r, binary.BigEndian, &h); err != nil {
		return nil, fmt.Errorf("failed to read HashAlgorithm: %v", err)
	}

	var s byte
	if err := binary.Read(r, binary.BigEndian, &s); err != nil {
		return nil, fmt.Errorf("failed to read SignatureAlgorithm: %v", err)
	}

	sig, err := readVarBytes(r, SignatureLengthBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to read Signature bytes: %v", err)
	}

	return &DigitallySigned{
		HashAlgorithm:      HashAlgorithm(h),
		SignatureAlgorithm: SignatureAlgorithm(s),
		Signature:          sig,
	}, nil
}

func marshalDigitallySignedHere(ds DigitallySigned, here []byte) ([]byte, error) {
	sigLen := len(ds.Signature)
	dsOutLen := 2 + SignatureLengthBytes + sigLen
	if here == nil {
		here = make([]byte, dsOutLen)
	}
	if len(here) < dsOutLen {
		return nil, ErrNotEnoughBuffer
	}
	here = here[0:dsOutLen]

	here[0] = byte(ds.HashAlgorithm)
	here[1] = byte(ds.SignatureAlgorithm)
	binary.BigEndian.PutUint16(here[2:4], uint16(sigLen))
	copy(here[4:], ds.Signature)

	return here, nil
}

// MarshalDigitallySigned marshalls a DigitallySigned structure into a byte array
func MarshalDigitallySigned(ds DigitallySigned) ([]byte, error) {
	return marshalDigitallySignedHere(ds, nil)
}

func checkCertificateFormat(cert ASN1Cert) error {
	if len(cert) == 0 {
		return errors.New("certificate is zero length")
	}
	if len(cert) > MaxCertificateLength {
		return errors.New("certificate too large")
	}
	return nil
}

func checkExtensionsFormat(ext CTExtensions) error {
	if len(ext) > MaxExtensionsLength {
		return errors.New("extensions too large")
	}
	return nil
}

func serializeV1CertSCTSignatureInput(timestamp uint64, cert ASN1Cert, ext CTExtensions) ([]byte, error) {
	if err := checkCertificateFormat(cert); err != nil {
		return nil, err
	}
	if err := checkExtensionsFormat(ext); err != nil {
		return nil, err
	}
	var buf bytes.Buffer
	if err := binary.Write(&buf, binary.BigEndian, V1); err != nil {
		return nil, err
	}
	if err := binary.Write(&buf, binary.BigEndian, CertificateTimestampSignatureType); err != nil {
		return nil, err
	}
	if err := binary.Write(&buf, binary.BigEndian, timestamp); err != nil {
		return nil, err
	}
	if err := binary.Write(&buf, binary.BigEndian, X509LogEntryType); err != nil {
		return nil, err
	}
	if err := writeVarBytes(&buf, cert, CertificateLengthBytes); err != nil {
		return nil, err
	}
	if err := writeVarBytes(&buf, ext, ExtensionsLengthBytes); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func serializeV1PrecertSCTSignatureInput(timestamp uint64, issuerKeyHash [issuerKeyHashLength]byte, tbs []byte, ext CTExtensions) ([]byte, error) {
	if err := checkCertificateFormat(tbs); err != nil {
		return nil, err
	}
	if err := checkExtensionsFormat(ext); err != nil {
		return nil, err
	}
	var buf bytes.Buffer
	if err := binary.Write(&buf, binary.BigEndian, V1); err != nil {
		return nil, err
	}
	if err := binary.Write(&buf, binary.BigEndian, CertificateTimestampSignatureType); err != nil {
		return nil, err
	}
	if err := binary.Write(&buf, binary.BigEndian, timestamp); err != nil {
		return nil, err
	}
	if err := binary.Write(&buf, binary.BigEndian, PrecertLogEntryType); err != nil {
		return nil, err
	}
	if _, err := buf.Write(issuerKeyHash[:]); err != nil {
		return nil, err
	}
	if err := writeVarBytes(&buf, tbs, CertificateLengthBytes); err != nil {
		return nil, err
	}
	if err := writeVarBytes(&buf, ext, ExtensionsLengthBytes); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func serializeV1SCTSignatureInput(sct SignedCertificateTimestamp, entry LogEntry) ([]byte, error) {
	if sct.SCTVersion != V1 {
		return nil, fmt.Errorf("unsupported SCT version, expected V1, but got %s", sct.SCTVersion)
	}
	if entry.Leaf.LeafType != TimestampedEntryLeafType {
		return nil, fmt.Errorf("Unsupported leaf type %s", entry.Leaf.LeafType)
	}
	switch entry.Leaf.TimestampedEntry.EntryType {
	case X509LogEntryType:
		return serializeV1CertSCTSignatureInput(sct.Timestamp, entry.Leaf.TimestampedEntry.X509Entry, entry.Leaf.TimestampedEntry.Extensions)
	case PrecertLogEntryType:
		return serializeV1PrecertSCTSignatureInput(sct.Timestamp, entry.Leaf.TimestampedEntry.PrecertEntry.IssuerKeyHash,
			entry.Leaf.TimestampedEntry.PrecertEntry.TBSCertificate,
			entry.Leaf.TimestampedEntry.Extensions)
	default:
		return nil, fmt.Errorf("unknown TimestampedEntryLeafType %s", entry.Leaf.TimestampedEntry.EntryType)
	}
}

// SerializeSCTSignatureInput serializes the passed in sct and log entry into
// the correct format for signing.
func SerializeSCTSignatureInput(sct SignedCertificateTimestamp, entry LogEntry) ([]byte, error) {
	switch sct.SCTVersion {
	case V1:
		return serializeV1SCTSignatureInput(sct, entry)
	default:
		return nil, fmt.Errorf("unknown SCT version %d", sct.SCTVersion)
	}
}

// SerializedLength will return the space (in bytes)
func (sct SignedCertificateTimestamp) SerializedLength() (int, error) {
	switch sct.SCTVersion {
	case V1:
		extLen := len(sct.Extensions)
		sigLen := len(sct.Signature.Signature)
		return 1 + 32 + 8 + 2 + extLen + 2 + 2 + sigLen, nil
	default:
		return 0, ErrInvalidVersion
	}
}

func serializeV1SCTHere(sct SignedCertificateTimestamp, here []byte) ([]byte, error) {
	if sct.SCTVersion != V1 {
		return nil, ErrInvalidVersion
	}
	sctLen, err := sct.SerializedLength()
	if err != nil {
		return nil, err
	}
	if here == nil {
		here = make([]byte, sctLen)
	}
	if len(here) < sctLen {
		return nil, ErrNotEnoughBuffer
	}
	if err := checkExtensionsFormat(sct.Extensions); err != nil {
		return nil, err
	}

	here = here[0:sctLen]

	// Write Version
	here[0] = byte(sct.SCTVersion)

	// Write LogID
	copy(here[1:33], sct.LogID[:])

	// Write Timestamp
	binary.BigEndian.PutUint64(here[33:41], sct.Timestamp)

	// Write Extensions
	extLen := len(sct.Extensions)
	binary.BigEndian.PutUint16(here[41:43], uint16(extLen))
	n := 43 + extLen
	copy(here[43:n], sct.Extensions)

	// Write Signature
	_, err = marshalDigitallySignedHere(sct.Signature, here[n:])
	if err != nil {
		return nil, err
	}
	return here, nil
}

// SerializeSCTHere serializes the passed in sct into the format specified
// by RFC6962 section 3.2.
// If a bytes slice here is provided then it will attempt to serialize into the
// provided byte slice, ErrNotEnoughBuffer will be returned if the buffer is
// too small.
// If a nil byte slice is provided, a buffer for will be allocated for you
// The returned slice will be sliced to the correct length.
func SerializeSCTHere(sct SignedCertificateTimestamp, here []byte) ([]byte, error) {
	switch sct.SCTVersion {
	case V1:
		return serializeV1SCTHere(sct, here)
	default:
		return nil, fmt.Errorf("unknown SCT version %d", sct.SCTVersion)
	}
}

// SerializeSCT serializes the passed in sct into the format specified
// by RFC6962 section 3.2
// Equivalent to SerializeSCTHere(sct, nil)
func SerializeSCT(sct SignedCertificateTimestamp) ([]byte, error) {
	return SerializeSCTHere(sct, nil)
}

func deserializeSCTV1(r io.Reader, sct *SignedCertificateTimestamp) error {
	if err := binary.Read(r, binary.BigEndian, &sct.LogID); err != nil {
		return err
	}
	if err := binary.Read(r, binary.BigEndian, &sct.Timestamp); err != nil {
		return err
	}
	ext, err := readVarBytes(r, ExtensionsLengthBytes)
	if err != nil {
		return err
	}
	sct.Extensions = ext
	ds, err := UnmarshalDigitallySigned(r)
	if err != nil {
		return err
	}
	sct.Signature = *ds
	return nil
}

func DeserializeSCT(r io.Reader) (*SignedCertificateTimestamp, error) {
	var sct SignedCertificateTimestamp
	if err := binary.Read(r, binary.BigEndian, &sct.SCTVersion); err != nil {
		return nil, err
	}
	switch sct.SCTVersion {
	case V1:
		return &sct, deserializeSCTV1(r, &sct)
	default:
		return nil, fmt.Errorf("unknown SCT version %d", sct.SCTVersion)
	}
}

func serializeV1STHSignatureInput(sth SignedTreeHead) ([]byte, error) {
	if sth.Version != V1 {
		return nil, fmt.Errorf("invalid STH version %d", sth.Version)
	}
	if sth.TreeSize < 0 {
		return nil, fmt.Errorf("invalid tree size %d", sth.TreeSize)
	}
	if len(sth.SHA256RootHash) != crypto.SHA256.Size() {
		return nil, fmt.Errorf("invalid TreeHash length, got %d expected %d", len(sth.SHA256RootHash), crypto.SHA256.Size())
	}

	var buf bytes.Buffer
	if err := binary.Write(&buf, binary.BigEndian, V1); err != nil {
		return nil, err
	}
	if err := binary.Write(&buf, binary.BigEndian, TreeHashSignatureType); err != nil {
		return nil, err
	}
	if err := binary.Write(&buf, binary.BigEndian, sth.Timestamp); err != nil {
		return nil, err
	}
	if err := binary.Write(&buf, binary.BigEndian, sth.TreeSize); err != nil {
		return nil, err
	}
	if err := binary.Write(&buf, binary.BigEndian, sth.SHA256RootHash); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// SerializeSTHSignatureInput serializes the passed in sth into the correct
// format for signing.
func SerializeSTHSignatureInput(sth SignedTreeHead) ([]byte, error) {
	switch sth.Version {
	case V1:
		return serializeV1STHSignatureInput(sth)
	default:
		return nil, fmt.Errorf("unsupported STH version %d", sth.Version)
	}
}
