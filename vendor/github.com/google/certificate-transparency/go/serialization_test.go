package ct

import (
	"bytes"
	"encoding/hex"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

// Returns a "variable-length" byte buffer containing |dataSize| data bytes
// along with an appropriate header.
// The buffer format is [header][data]
// where [header] is a bigendian representation of the size of [data].
// sizeof([header]) is the minimum number of bytes necessary to represent
// |dataSize|.
func createVarByteBuf(dataSize uint64) []byte {
	lenBytes := uint64(0)
	for x := dataSize; x > 0; x >>= 8 {
		lenBytes++
	}
	buf := make([]byte, dataSize+lenBytes)
	for t, x := dataSize, uint64(0); x < lenBytes; x++ {
		buf[lenBytes-x-1] = byte(t)
		t >>= 8
	}
	for x := uint64(0); x < dataSize; x++ {
		buf[lenBytes+x] = byte(x)
	}
	return buf
}

func TestCreateVarByteBuf(t *testing.T) {
	buf := createVarByteBuf(56)
	if len(buf) != 56+1 {
		t.Errorf("Wrong buffer size returned, expected %d", 56+1)
	}
	if buf[0] != 56 {
		t.Errorf("Buffer has incorrect size header %02x", buf[0])
	}
	buf = createVarByteBuf(256)
	if len(buf) != 256+2 {
		t.Errorf("Wrong buffer size returned, expected %d", 256+2)
	}
	if buf[0] != 0x01 || buf[1] != 0x00 {
		t.Errorf("Buffer has incorrect size header %02x,%02x", buf[0], buf[1])
	}
	buf = createVarByteBuf(65536)
	if len(buf) != 65536+3 {
		t.Errorf("Wrong buffer size returned, expected %d", 65536+3)
	}
	if buf[0] != 0x01 || buf[1] != 0x00 || buf[2] != 0x00 {
		t.Errorf("Buffer has incorrect size header %02x,%02x,%02x", buf[0], buf[1], buf[2])
	}
}

func TestWriteVarBytes(t *testing.T) {
	const dataSize = 453641
	data := make([]byte, dataSize)
	for x := uint64(0); x < dataSize; x++ {
		data[x] = byte(x)
	}

	var buf bytes.Buffer
	if err := writeVarBytes(&buf, data, 3); err != nil {
		t.Errorf("Failed to write data to buffer: %v", err)
	}
	if buf.Len() != dataSize+3 {
		t.Errorf("Wrong buffer size created, expected %d but got %d", dataSize+3, buf.Len())
	}
	b := buf.Bytes()
	if b[0] != 0x06 || b[1] != 0xec || b[2] != 0x09 {
		t.Errorf("Buffer has incorrect size header %02x,%02x,%02x", b[0], b[1], b[2])
	}
	if bytes.Compare(data, b[3:]) != 0 {
		t.Errorf("Buffer data corrupt")
	}
}

func TestReadVarBytes(t *testing.T) {
	const BufSize = 453641
	r := createVarByteBuf(BufSize)
	buf, err := readVarBytes(bytes.NewReader(r), 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(buf) != BufSize {
		t.Fatalf("Incorrect size buffer returned, expected %d, got %d", BufSize, len(buf))
	}
	for i := range buf {
		if buf[i] != byte(i) {
			t.Fatalf("Buffer contents incorrect, expected %02x, got %02x.", byte(i), buf[i])
		}
	}
}

func TestReadVarBytesTooLarge(t *testing.T) {
	_, err := readVarBytes(nil, 9)
	if err == nil || !strings.Contains(err.Error(), "too large") {
		t.Fatal("readVarBytes didn't fail when trying to read too large a data size: ", err)
	}
}

func TestReadVarBytesZero(t *testing.T) {
	_, err := readVarBytes(nil, 0)
	if err == nil || !strings.Contains(err.Error(), "should be > 0") {
		t.Fatal("readVarBytes didn't fail when trying to read zero length data")
	}
}

func TestReadVarBytesShortRead(t *testing.T) {
	r := make([]byte, 2)
	r[0] = 2 // but only 1 byte available...
	_, err := readVarBytes(bytes.NewReader(r), 1)
	if err == nil || !strings.Contains(err.Error(), "short read") {
		t.Fatal("readVarBytes didn't fail with a short read")
	}
}

func TestReadTimestampedEntryIntoChecksEntryType(t *testing.T) {
	buffer := []byte{0, 1, 2, 3, 4, 5, 6, 7, 0x45, 0x45}
	var tse TimestampedEntry
	err := ReadTimestampedEntryInto(bytes.NewReader(buffer), &tse)
	if err == nil || !strings.Contains(err.Error(), "unknown EntryType") {
		t.Fatal("Failed to check EntryType - accepted 0x4545")
	}
}

func TestCheckCertificateFormatOk(t *testing.T) {
	if err := checkCertificateFormat([]byte("I'm a cert, honest.")); err != nil {
		t.Fatalf("checkCertificateFormat objected to valid format: %v", err)
	}
}

func TestCheckCertificateFormatZeroSize(t *testing.T) {
	if checkCertificateFormat([]byte("")) == nil {
		t.Fatalf("checkCertificateFormat failed to object to zero length cert")
	}
}

func TestCheckCertificateFormatTooBig(t *testing.T) {
	big := make([]byte, MaxCertificateLength+1)
	if checkCertificateFormat(big) == nil {
		t.Fatalf("checkCertificateFormat failed to object to cert of length %d (max %d)", len(big), MaxCertificateLength)
	}
}

func TestCheckExtensionsFormatOk(t *testing.T) {
	if err := checkExtensionsFormat([]byte("I'm an extension, honest.")); err != nil {
		t.Fatalf("checkExtensionsFormat objected to valid format: %v", err)
	}
}

func TestCheckExtensionsFormatTooBig(t *testing.T) {
	big := make([]byte, MaxExtensionsLength+1)
	if checkExtensionsFormat(big) == nil {
		t.Fatalf("checkExtensionsFormat failed to object to extension of length %d (max %d)", len(big), MaxExtensionsLength)
	}
}

const (
	defaultSCTLogIDString          string = "iamapublickeyshatwofivesixdigest"
	defaultSCTTimestamp            uint64 = 1234
	defaultSCTSignatureString      string = "\x04\x03\x00\x09signature"
	defaultCertifictateString      string = "certificate"
	defaultPrecertString           string = "precert"
	defaultPrecertIssuerHashString string = "iamapublickeyshatwofivesixdigest"
	defaultPrecertTBSString        string = "tbs"

	defaultCertificateSCTSignatureInputHexString string =
	// version, 1 byte
	"00" +
		// signature type, 1 byte
		"00" +
		// timestamp, 8 bytes
		"00000000000004d2" +
		// entry type, 2 bytes
		"0000" +
		// leaf certificate length, 3 bytes
		"00000b" +
		// leaf certificate, 11 bytes
		"6365727469666963617465" +
		// extensions length, 2 bytes
		"0000" +
		// extensions, 0 bytes
		""

	defaultPrecertSCTSignatureInputHexString string =
	// version, 1 byte
	"00" +
		// signature type, 1 byte
		"00" +
		// timestamp, 8 bytes
		"00000000000004d2" +
		// entry type, 2 bytes
		"0001" +
		// issuer key hash, 32 bytes
		"69616d617075626c69636b657973686174776f66697665736978646967657374" +
		// tbs certificate length, 3 bytes
		"000003" +
		// tbs certificate, 3 bytes
		"746273" +
		// extensions length, 2 bytes
		"0000" +
		// extensions, 0 bytes
		""

	defaultSTHSignedHexString string =
	// version, 1 byte
	"00" +
		// signature type, 1 byte
		"01" +
		// timestamp, 8 bytes
		"0000000000000929" +
		// tree size, 8 bytes
		"0000000000000006" +
		// root hash, 32 bytes
		"696d757374626565786163746c7974686972747974776f62797465736c6f6e67"

	defaultSCTHexString string =
	// version, 1 byte
	"00" +
		// keyid, 32 bytes
		"69616d617075626c69636b657973686174776f66697665736978646967657374" +
		// timestamp, 8 bytes
		"00000000000004d2" +
		// extensions length, 2 bytes
		"0000" +
		// extensions, 0 bytes
		// hash algo, sig algo, 2 bytes
		"0403" +
		// signature length, 2 bytes
		"0009" +
		// signature, 9 bytes
		"7369676e6174757265"
)

func defaultSCTLogID() SHA256Hash {
	var id SHA256Hash
	copy(id[:], defaultSCTLogIDString)
	return id
}

func defaultSCTSignature() DigitallySigned {
	ds, err := UnmarshalDigitallySigned(bytes.NewReader([]byte(defaultSCTSignatureString)))
	if err != nil {
		panic(err)
	}
	return *ds
}

func defaultSCT() SignedCertificateTimestamp {
	return SignedCertificateTimestamp{
		SCTVersion: V1,
		LogID:      defaultSCTLogID(),
		Timestamp:  defaultSCTTimestamp,
		Extensions: []byte{},
		Signature:  defaultSCTSignature()}
}

func defaultCertificate() []byte {
	return []byte(defaultCertifictateString)
}

func defaultExtensions() []byte {
	return []byte{}
}

func defaultCertificateSCTSignatureInput(t *testing.T) []byte {
	r, err := hex.DecodeString(defaultCertificateSCTSignatureInputHexString)
	if err != nil {
		t.Fatalf("failed to decode defaultCertificateSCTSignatureInputHexString: %v", err)
	}
	return r
}

func defaultCertificateLogEntry() LogEntry {
	return LogEntry{
		Index: 1,
		Leaf: MerkleTreeLeaf{
			Version:  V1,
			LeafType: TimestampedEntryLeafType,
			TimestampedEntry: TimestampedEntry{
				Timestamp: defaultSCTTimestamp,
				EntryType: X509LogEntryType,
				X509Entry: defaultCertificate(),
			},
		},
	}
}

func defaultPrecertSCTSignatureInput(t *testing.T) []byte {
	r, err := hex.DecodeString(defaultPrecertSCTSignatureInputHexString)
	if err != nil {
		t.Fatalf("failed to decode defaultPrecertSCTSignatureInputHexString: %v", err)
	}
	return r
}

func defaultPrecertTBS() []byte {
	return []byte(defaultPrecertTBSString)
}

func defaultPrecertIssuerHash() [issuerKeyHashLength]byte {
	var b [issuerKeyHashLength]byte
	copy(b[:], []byte(defaultPrecertIssuerHashString))
	return b
}

func defaultPrecertLogEntry() LogEntry {
	return LogEntry{
		Index: 1,
		Leaf: MerkleTreeLeaf{
			Version:  V1,
			LeafType: TimestampedEntryLeafType,
			TimestampedEntry: TimestampedEntry{
				Timestamp: defaultSCTTimestamp,
				EntryType: PrecertLogEntryType,
				PrecertEntry: PreCert{
					IssuerKeyHash:  defaultPrecertIssuerHash(),
					TBSCertificate: defaultPrecertTBS(),
				},
			},
		},
	}
}

func defaultSTH() SignedTreeHead {
	var root SHA256Hash
	copy(root[:], "imustbeexactlythirtytwobyteslong")
	return SignedTreeHead{
		TreeSize:       6,
		Timestamp:      2345,
		SHA256RootHash: root,
		TreeHeadSignature: DigitallySigned{
			HashAlgorithm:      SHA256,
			SignatureAlgorithm: ECDSA,
			Signature:          []byte("tree_signature"),
		},
	}
}

//////////////////////////////////////////////////////////////////////////////////
// Tests start here:
//////////////////////////////////////////////////////////////////////////////////

func TestSerializeV1SCTSignatureInputForCertificateKAT(t *testing.T) {
	serialized, err := SerializeSCTSignatureInput(defaultSCT(), defaultCertificateLogEntry())
	if err != nil {
		t.Fatalf("Failed to serialize SCT for signing: %v", err)
	}
	if bytes.Compare(serialized, defaultCertificateSCTSignatureInput(t)) != 0 {
		t.Fatalf("Serialized certificate signature input doesn't match expected answer:\n%v\n%v", serialized, defaultCertificateSCTSignatureInput(t))
	}
}

func TestSerializeV1SCTSignatureInputForPrecertKAT(t *testing.T) {
	serialized, err := SerializeSCTSignatureInput(defaultSCT(), defaultPrecertLogEntry())
	if err != nil {
		t.Fatalf("Failed to serialize SCT for signing: %v", err)
	}
	if bytes.Compare(serialized, defaultPrecertSCTSignatureInput(t)) != 0 {
		t.Fatalf("Serialized precertificate signature input doesn't match expected answer:\n%v\n%v", serialized, defaultPrecertSCTSignatureInput(t))
	}
}

func TestSerializeV1STHSignatureKAT(t *testing.T) {
	b, err := SerializeSTHSignatureInput(defaultSTH())
	if err != nil {
		t.Fatalf("Failed to serialize defaultSTH: %v", err)
	}
	if bytes.Compare(b, mustDehex(t, defaultSTHSignedHexString)) != 0 {
		t.Fatalf("defaultSTH incorrectly serialized, expected:\n%v\ngot:\n%v", mustDehex(t, defaultSTHSignedHexString), b)
	}
}

func TestMarshalDigitallySigned(t *testing.T) {
	b, err := MarshalDigitallySigned(
		DigitallySigned{
			HashAlgorithm:      SHA512,
			SignatureAlgorithm: ECDSA,
			Signature:          []byte("signature")})
	if err != nil {
		t.Fatalf("Failed to marshal DigitallySigned struct: %v", err)
	}
	if b[0] != byte(SHA512) {
		t.Fatalf("Expected b[0] == SHA512, but found %v", HashAlgorithm(b[0]))
	}
	if b[1] != byte(ECDSA) {
		t.Fatalf("Expected b[1] == ECDSA, but found %v", SignatureAlgorithm(b[1]))
	}
	if b[2] != 0x00 || b[3] != 0x09 {
		t.Fatalf("Found incorrect length bytes, expected (0x00, 0x09) found %v", b[2:3])
	}
	if string(b[4:]) != "signature" {
		t.Fatalf("Found incorrect signature bytes, expected %v, found %v", []byte("signature"), b[4:])
	}
}

func TestUnmarshalDigitallySigned(t *testing.T) {
	ds, err := UnmarshalDigitallySigned(bytes.NewReader([]byte("\x01\x02\x00\x0aSiGnAtUrE!")))
	if err != nil {
		t.Fatalf("Failed to unmarshal DigitallySigned: %v", err)
	}
	if ds.HashAlgorithm != MD5 {
		t.Fatalf("Expected HashAlgorithm %v, but got %v", MD5, ds.HashAlgorithm)
	}
	if ds.SignatureAlgorithm != DSA {
		t.Fatalf("Expected SignatureAlgorithm %v, but got %v", DSA, ds.SignatureAlgorithm)
	}
	if string(ds.Signature) != "SiGnAtUrE!" {
		t.Fatalf("Expected Signature %v, but got %v", []byte("SiGnAtUrE!"), ds.Signature)
	}
}

func TestSCTSerializationRoundTrip(t *testing.T) {
	b, err := SerializeSCT(defaultSCT())
	if err != nil {
		t.Fatalf("Failed to serialize SCT: %v", err)
	}
	sct, err := DeserializeSCT(bytes.NewReader(b))
	if err != nil {
		t.Fatalf("Failed to deserialize SCT: %v", err)
	}
	assert.Equal(t, defaultSCT(), *sct)
}

func TestSerializeSCT(t *testing.T) {
	b, err := SerializeSCT(defaultSCT())
	if err != nil {
		t.Fatalf("Failed to serialize SCT: %v", err)
	}
	if bytes.Compare(mustDehex(t, defaultSCTHexString), b) != 0 {
		t.Fatalf("Serialized SCT differs from expected KA. Expected:\n%v\nGot:\n%v", mustDehex(t, defaultSCTHexString), b)
	}
}

func TestDeserializeSCT(t *testing.T) {
	sct, err := DeserializeSCT(bytes.NewReader(mustDehex(t, defaultSCTHexString)))
	if err != nil {
		t.Fatalf("Failed to deserialize SCT: %v", err)
	}
	assert.Equal(t, defaultSCT(), *sct)
}
