package ct

import (
	"bytes"
	"encoding/base64"
	"strings"
	"testing"
)

const (
	PrecertEntryB64 = "AAAAAAFLSYHwyAABN2DieQ8zpJj5tsFJ/s/KOZOVS1NvvzatRdCoQVt5M30ABHowggR2oAMCAQICEAUyKYw5aj4l/KoZd+gntfMwDQYJKoZIhvcNAQELBQAwbTELMAkGA1UEBhMCVVMxFjAUBgNVBAoTDUdlb1RydXN0IEluYy4xHzAdBgNVBAsTFkZPUiBURVNUIFBVUlBPU0VTIE9OTFkxJTAjBgNVBAMTHEdlb1RydXN0IEVWIFNTTCBURVNUIENBIC0gRzQwHhcNMTUwMjAyMDAwMDAwWhcNMTYwMjI3MjM1OTU5WjCBwzETMBEGCysGAQQBgjc8AgEDEwJHQjEbMBkGCysGAQQBgjc8AgECFApDYWxpZm9ybmlhMR4wHAYLKwYBBAGCNzwCAQEMDU1vdW50YWluIFZpZXcxCzAJBgNVBAYTAkdCMRMwEQYDVQQIDApDYWxpZm9ybmlhMRYwFAYDVQQHDA1Nb3VudGFpbiBWaWV3MR0wGwYDVQQKDBRTeW1hbnRlYyBDb3Jwb3JhdGlvbjEWMBQGA1UEAwwNc2RmZWRzZi50cnVzdDCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBALGdl97zn/gpxl6gmaMlcpizP/Z1RR/cVkGiIjR67kpWIB9MGkBvLxmBXYbewaYRdo59VWyOM6fxtMeNsZzOrlQOl64fBmCy7k+M/yBFuEqdoig0l0RAbs6u0LCNRv2rNUOz2G6nCGJ6YaUpt5Onatxrd2vI1bPU/iHixKqSz9M7RedBIGjgaDor7/rR3y/DILjdvwL/tgPSz3R5gnf9lla1rNRWWbDl12HgLc+VxTCVVVqTGtW/qbSWfARdXxLeLWtTfNk68q2LReVUC9QyeYdtE+N2+2SXeOEN+lYWW5Ab036d7k5GAntMBzLKftZEkYYquvaiSkqu2PSaCSLKT7UCAwEAAaOCAdEwggHNMEcGA1UdEQRAMD6CDWtqYXNkaGYudHJ1c3SCC3NzZGZzLnRydXN0gg1zZGZlZHNmLnRydXN0ghF3d3cuc2RmZWRzZi50cnVzdDAJBgNVHRMEAjAAMA4GA1UdDwEB/wQEAwIFoDArBgNVHR8EJDAiMCCgHqAchhpodHRwOi8vZ20uc3ltY2IuY29tL2dtLmNybDCBoAYDVR0gBIGYMIGVMIGSBgkrBgEEAfAiAQYwgYQwPwYIKwYBBQUHAgEWM2h0dHBzOi8vd3d3Lmdlb3RydXN0LmNvbS9yZXNvdXJjZXMvcmVwb3NpdG9yeS9sZWdhbDBBBggrBgEFBQcCAjA1DDNodHRwczovL3d3dy5nZW90cnVzdC5jb20vcmVzb3VyY2VzL3JlcG9zaXRvcnkvbGVnYWwwHQYDVR0lBBYwFAYIKwYBBQUHAwEGCCsGAQUFBwMCMB8GA1UdIwQYMBaAFLFplGGr5ssMTOdZr1pJixgzweFHMFcGCCsGAQUFBwEBBEswSTAfBggrBgEFBQcwAYYTaHR0cDovL2dtLnN5bWNkLmNvbTAmBggrBgEFBQcwAoYaaHR0cDovL2dtLnN5bWNiLmNvbS9nbS5jcnQAAA=="
	CertEntryB64    = "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"
)

func TestReadMerkleTreeLeafForX509Cert(t *testing.T) {
	entry, err := base64.StdEncoding.DecodeString(CertEntryB64)
	if err != nil {
		t.Fatal(err)
	}

	m, err := ReadMerkleTreeLeaf(bytes.NewReader(entry))
	if err != nil {
		t.Fatal(err)
	}
	if m.Version != V1 {
		t.Fatal("Invalid version number")
	}
	if m.LeafType != TimestampedEntryLeafType {
		t.Fatal("Invalid LeafType")
	}
	if m.TimestampedEntry.EntryType != X509LogEntryType {
		t.Fatal("Incorrect EntryType")
	}
}

func TestReadMerkleTreeLeafForPrecert(t *testing.T) {
	entry, err := base64.StdEncoding.DecodeString(PrecertEntryB64)
	if err != nil {
		t.Fatal(err)
	}

	m, err := ReadMerkleTreeLeaf(bytes.NewReader(entry))
	if err != nil {
		t.Fatal(err)
	}
	if m.Version != V1 {
		t.Fatal("Invalid version number")
	}
	if m.LeafType != TimestampedEntryLeafType {
		t.Fatal("Invalid LeafType")
	}
	if m.TimestampedEntry.EntryType != PrecertLogEntryType {
		t.Fatal("Incorrect EntryType")
	}
}

func TestReadMerkleTreeLeafChecksVersion(t *testing.T) {
	buffer := []byte{1}
	_, err := ReadMerkleTreeLeaf(bytes.NewReader(buffer))
	if err == nil || !strings.Contains(err.Error(), "unknown Version") {
		t.Fatal("Failed to check Version - accepted 1")
	}
}

func TestReadMerkleTreeLeafChecksLeafType(t *testing.T) {
	buffer := []byte{0, 0x12, 0x34}
	_, err := ReadMerkleTreeLeaf(bytes.NewReader(buffer))
	if err == nil || !strings.Contains(err.Error(), "unknown LeafType") {
		t.Fatal("Failed to check LeafType - accepted 0x1234")
	}
}
