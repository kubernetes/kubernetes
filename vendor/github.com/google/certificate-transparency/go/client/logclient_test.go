package client

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strconv"
	"testing"
	"time"

	"github.com/google/certificate-transparency/go"
	"golang.org/x/net/context"
)

const (
	ValidSTHResponse = `{"tree_size":3721782,"timestamp":1396609800587,
        "sha256_root_hash":"SxKOxksguvHPyUaKYKXoZHzXl91Q257+JQ0AUMlFfeo=",
        "tree_head_signature":"BAMARjBEAiBUYO2tODlUUw4oWGiVPUHqZadRRyXs9T2rSXchA79VsQIgLASkQv3cu4XdPFCZbgFkIUefniNPCpO3LzzHX53l+wg="}`
	ValidSTHResponseTreeSize          = 3721782
	ValidSTHResponseTimestamp         = 1396609800587
	ValidSTHResponseSHA256RootHash    = "SxKOxksguvHPyUaKYKXoZHzXl91Q257+JQ0AUMlFfeo="
	ValidSTHResponseTreeHeadSignature = "BAMARjBEAiBUYO2tODlUUw4oWGiVPUHqZadRRyXs9T2rSXchA79VsQIgLASkQv3cu4XdPFCZbgFkIUefniNPCpO3LzzHX53l+wg="

	PrecertEntryB64          = "AAAAAAFLSYHwyAABN2DieQ8zpJj5tsFJ/s/KOZOVS1NvvzatRdCoQVt5M30ABHowggR2oAMCAQICEAUyKYw5aj4l/KoZd+gntfMwDQYJKoZIhvcNAQELBQAwbTELMAkGA1UEBhMCVVMxFjAUBgNVBAoTDUdlb1RydXN0IEluYy4xHzAdBgNVBAsTFkZPUiBURVNUIFBVUlBPU0VTIE9OTFkxJTAjBgNVBAMTHEdlb1RydXN0IEVWIFNTTCBURVNUIENBIC0gRzQwHhcNMTUwMjAyMDAwMDAwWhcNMTYwMjI3MjM1OTU5WjCBwzETMBEGCysGAQQBgjc8AgEDEwJHQjEbMBkGCysGAQQBgjc8AgECFApDYWxpZm9ybmlhMR4wHAYLKwYBBAGCNzwCAQEMDU1vdW50YWluIFZpZXcxCzAJBgNVBAYTAkdCMRMwEQYDVQQIDApDYWxpZm9ybmlhMRYwFAYDVQQHDA1Nb3VudGFpbiBWaWV3MR0wGwYDVQQKDBRTeW1hbnRlYyBDb3Jwb3JhdGlvbjEWMBQGA1UEAwwNc2RmZWRzZi50cnVzdDCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBALGdl97zn/gpxl6gmaMlcpizP/Z1RR/cVkGiIjR67kpWIB9MGkBvLxmBXYbewaYRdo59VWyOM6fxtMeNsZzOrlQOl64fBmCy7k+M/yBFuEqdoig0l0RAbs6u0LCNRv2rNUOz2G6nCGJ6YaUpt5Onatxrd2vI1bPU/iHixKqSz9M7RedBIGjgaDor7/rR3y/DILjdvwL/tgPSz3R5gnf9lla1rNRWWbDl12HgLc+VxTCVVVqTGtW/qbSWfARdXxLeLWtTfNk68q2LReVUC9QyeYdtE+N2+2SXeOEN+lYWW5Ab036d7k5GAntMBzLKftZEkYYquvaiSkqu2PSaCSLKT7UCAwEAAaOCAdEwggHNMEcGA1UdEQRAMD6CDWtqYXNkaGYudHJ1c3SCC3NzZGZzLnRydXN0gg1zZGZlZHNmLnRydXN0ghF3d3cuc2RmZWRzZi50cnVzdDAJBgNVHRMEAjAAMA4GA1UdDwEB/wQEAwIFoDArBgNVHR8EJDAiMCCgHqAchhpodHRwOi8vZ20uc3ltY2IuY29tL2dtLmNybDCBoAYDVR0gBIGYMIGVMIGSBgkrBgEEAfAiAQYwgYQwPwYIKwYBBQUHAgEWM2h0dHBzOi8vd3d3Lmdlb3RydXN0LmNvbS9yZXNvdXJjZXMvcmVwb3NpdG9yeS9sZWdhbDBBBggrBgEFBQcCAjA1DDNodHRwczovL3d3dy5nZW90cnVzdC5jb20vcmVzb3VyY2VzL3JlcG9zaXRvcnkvbGVnYWwwHQYDVR0lBBYwFAYIKwYBBQUHAwEGCCsGAQUFBwMCMB8GA1UdIwQYMBaAFLFplGGr5ssMTOdZr1pJixgzweFHMFcGCCsGAQUFBwEBBEswSTAfBggrBgEFBQcwAYYTaHR0cDovL2dtLnN5bWNkLmNvbTAmBggrBgEFBQcwAoYaaHR0cDovL2dtLnN5bWNiLmNvbS9nbS5jcnQAAA=="
	PrecertEntryExtraDataB64 = "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"

	CertEntryB64          = "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"
	CertEntryExtraDataB64 = "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"

	SubmissionCertB64 = "MIIEijCCA3KgAwIBAgICEk0wDQYJKoZIhvcNAQELBQAwKzEpMCcGA1UEAwwgY2Fja2xpbmcgY3J5cHRvZ3JhcGhlciBmYWtlIFJPT1QwHhcNMTUxMDIxMjAxMTUyWhcNMjAxMDE5MjAxMTUyWjAfMR0wGwYDVQQDExRoYXBweSBoYWNrZXIgZmFrZSBDQTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAMIKR3maBcUSsncXYzQT13D5Nr+Z3mLxMMh3TUdt6sACmqbJ0btRlgXfMtNLM2OU1I6a3Ju+tIZSdn2v21JBwvxUzpZQ4zy2cimIiMQDZCQHJwzC9GZn8HaW091iz9H0Go3A7WDXwYNmsdLNRi00o14UjoaVqaPsYrZWvRKaIRqaU0hHmS0AWwQSvN/93iMIXuyiwywmkwKbWnnxCQ/gsctKFUtcNrwEx9Wgj6KlhwDTyI1QWSBbxVYNyUgPFzKxrSmwMO0yNff7ho+QT9x5+Y/7XE59S4Mc4ZXxcXKew/gSlN9U5mvT+D2BhDtkCupdfsZNCQWp27A+b/DmrFI9NqsCAwEAAaOCAcIwggG+MBIGA1UdEwEB/wQIMAYBAf8CAQAwQwYDVR0eBDwwOqE4MAaCBC5taWwwCocIAAAAAAAAAAAwIocgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwDgYDVR0PAQH/BAQDAgGGMH8GCCsGAQUFBwEBBHMwcTAyBggrBgEFBQcwAYYmaHR0cDovL2lzcmcudHJ1c3RpZC5vY3NwLmlkZW50cnVzdC5jb20wOwYIKwYBBQUHMAKGL2h0dHA6Ly9hcHBzLmlkZW50cnVzdC5jb20vcm9vdHMvZHN0cm9vdGNheDMucDdjMB8GA1UdIwQYMBaAFOmkP+6epeby1dd5YDyTpi4kjpeqMFQGA1UdIARNMEswCAYGZ4EMAQIBMD8GCysGAQQBgt8TAQEBMDAwLgYIKwYBBQUHAgEWImh0dHA6Ly9jcHMucm9vdC14MS5sZXRzZW5jcnlwdC5vcmcwPAYDVR0fBDUwMzAxoC+gLYYraHR0cDovL2NybC5pZGVudHJ1c3QuY29tL0RTVFJPT1RDQVgzQ1JMLmNybDAdBgNVHQ4EFgQU+3hPEvlgFYMsnxd/NBmzLjbqQYkwDQYJKoZIhvcNAQELBQADggEBAA0YAeLXOklx4hhCikUUl+BdnFfn1g0W5AiQLVNIOL6PnqXu0wjnhNyhqdwnfhYMnoy4idRh4lB6pz8Gf9pnlLd/DnWSV3gS+/I/mAl1dCkKby6H2V790e6IHmIK2KYm3jm+U++FIdGpBdsQTSdmiX/rAyuxMDM0adMkNBwTfQmZQCz6nGHw1QcSPZMvZpsC8SkvekzxsjF1otOrMUPNPQvtTWrVx8GlR2qfx/4xbQa1v2frNvFBCmO59goz+jnWvfTtj2NjwDZ7vlMBsPm16dbKYC840uvRoZjxqsdc3ChCZjqimFqlNG/xoPA8+dTicZzCXE9ijPIcvW6y1aa3bGw="
)

func TestGetEntriesWorks(t *testing.T) {
	positiveDecimalNumber := regexp.MustCompile("[0-9]+")
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/ct/v1/get-entries" {
			t.Fatalf("Incorrect URL path: %s", r.URL.Path)
		}
		q := r.URL.Query()
		if q["start"] == nil {
			t.Fatal("Missing 'start' parameter")
		}
		if !positiveDecimalNumber.MatchString(q["start"][0]) {
			t.Fatal("Invalid 'start' parameter: " + q["start"][0])
		}
		if q["end"] == nil {
			t.Fatal("Missing 'end' parameter")
		}
		if !positiveDecimalNumber.MatchString(q["end"][0]) {
			t.Fatal("Invalid 'end' parameter: " + q["end"][0])
		}
		fmt.Fprintf(w, `{"entries":[{"leaf_input": "%s","extra_data": "%s"},{"leaf_input": "%s","extra_data": "%s"}]}`, PrecertEntryB64, PrecertEntryExtraDataB64, CertEntryB64, CertEntryExtraDataB64)
	}))
	defer ts.Close()

	client := New(ts.URL)
	leaves, err := client.GetEntries(0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(leaves) != 2 {
		t.Fatal("Incorrect number of leaves returned")
	}
}

func TestGetSTHWorks(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/ct/v1/get-sth" {
			t.Fatalf("Incorrect URL path: %s", r.URL.Path)
		}
		fmt.Fprintf(w, `{"tree_size": %d, "timestamp": %d, "sha256_root_hash": "%s", "tree_head_signature": "%s"}`,
			ValidSTHResponseTreeSize, int64(ValidSTHResponseTimestamp), ValidSTHResponseSHA256RootHash,
			ValidSTHResponseTreeHeadSignature)
	}))
	defer ts.Close()

	client := New(ts.URL)
	sth, err := client.GetSTH()
	if err != nil {
		t.Fatal(err)
	}
	if sth.TreeSize != ValidSTHResponseTreeSize {
		t.Fatal("Invalid tree size")
	}
	if sth.Timestamp != ValidSTHResponseTimestamp {
		t.Fatal("Invalid Timestamp")
	}
	if sth.SHA256RootHash.Base64String() != ValidSTHResponseSHA256RootHash {
		t.Fatal("Invalid SHA256RootHash")
	}
	expectedRawSignature, err := base64.StdEncoding.DecodeString(ValidSTHResponseTreeHeadSignature)
	if err != nil {
		t.Fatal("Couldn't b64 decode 'correct' STH signature!")
	}
	expectedDS, err := ct.UnmarshalDigitallySigned(bytes.NewReader(expectedRawSignature))
	if err != nil {
		t.Fatalf("Couldn't unmarshal DigitallySigned: %v", err)
	}
	if sth.TreeHeadSignature.HashAlgorithm != expectedDS.HashAlgorithm {
		t.Fatalf("Invalid TreeHeadSignature.HashAlgorithm: expected %v, got %v", sth.TreeHeadSignature.HashAlgorithm, expectedDS.HashAlgorithm)
	}
	if sth.TreeHeadSignature.SignatureAlgorithm != expectedDS.SignatureAlgorithm {
		t.Fatalf("Invalid TreeHeadSignature.SignatureAlgorithm: expected %v, got %v", sth.TreeHeadSignature.SignatureAlgorithm, expectedDS.SignatureAlgorithm)
	}
	if bytes.Compare(sth.TreeHeadSignature.Signature, expectedDS.Signature) != 0 {
		t.Fatalf("Invalid TreeHeadSignature.Signature: expected %v, got %v", sth.TreeHeadSignature.Signature, expectedDS.Signature)
	}
}

func TestAddChainWithContext(t *testing.T) {
	retryAfter := 0
	currentFailures := 0
	failuresBeforeSuccess := 0
	hs := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if failuresBeforeSuccess > 0 && currentFailures < failuresBeforeSuccess {
			currentFailures++
			if retryAfter > 0 {
				w.Header().Add("Retry-After", strconv.Itoa(retryAfter))
				w.WriteHeader(503)
				return
			}
			w.WriteHeader(408)
			return
		}
		_, err := w.Write([]byte(`{"sct_version":0,"id":"KHYaGJAn++880NYaAY12sFBXKcenQRvMvfYE9F1CYVM=","timestamp":1337,"extensions":"","signature":"BAMARjBEAiAIc21J5ZbdKZHw5wLxCP+MhBEsV5+nfvGyakOIv6FOvAIgWYMZb6Pw///uiNM7QTg2Of1OqmK1GbeGuEl9VJN8v8c="}`))
		if err != nil {
			return
		}
	}))
	defer hs.Close()

	certBytes, err := base64.StdEncoding.DecodeString(SubmissionCertB64)
	if err != nil {
		t.Fatalf("Failed to decode chain array B64: %s", err)
	}
	chain := []ct.ASN1Cert{certBytes}

	c := New(hs.URL)
	leeway := time.Millisecond * 100
	instant := time.Millisecond
	fiveSeconds := time.Second * 5

	testCases := []struct {
		deadlineLength        int
		expected              time.Duration
		retryAfter            int
		failuresBeforeSuccess int
		success               bool
	}{
		{-1, instant, 0, 0, true},
		{6, fiveSeconds, 5, 1, true},
		{5, fiveSeconds, 10, 1, false},
		{10, fiveSeconds, 1, 5, true},
		{1, instant * 10, 0, 10, true},
	}

	for _, tc := range testCases {
		var deadline context.Context
		if tc.deadlineLength >= 0 {
			deadline, _ = context.WithDeadline(context.Background(), time.Now().Add(time.Duration(tc.deadlineLength)*time.Second))
		}
		retryAfter = tc.retryAfter
		failuresBeforeSuccess = tc.failuresBeforeSuccess
		currentFailures = 0

		started := time.Now()
		sct, err := c.AddChainWithContext(deadline, chain)
		took := time.Since(started)
		if math.Abs(float64(took-tc.expected)) > float64(leeway) {
			t.Fatalf("Submission took an unexpected length of time: %s, expected ~%s", took, tc.expected)
		}
		if tc.success && err != nil {
			t.Fatalf("Failed to submit chain: %s", err)
		} else if !tc.success && err == nil {
			t.Fatal("Expected AddChainWithContext to fail")
		}
		if tc.success && sct == nil {
			t.Fatal("Nil SCT returned")
		}
	}
}
