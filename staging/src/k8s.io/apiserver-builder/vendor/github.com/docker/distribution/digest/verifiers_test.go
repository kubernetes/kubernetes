package digest

import (
	"bytes"
	"crypto/rand"
	"io"
	"testing"
)

func TestDigestVerifier(t *testing.T) {
	p := make([]byte, 1<<20)
	rand.Read(p)
	digest := FromBytes(p)

	verifier, err := NewDigestVerifier(digest)
	if err != nil {
		t.Fatalf("unexpected error getting digest verifier: %s", err)
	}

	io.Copy(verifier, bytes.NewReader(p))

	if !verifier.Verified() {
		t.Fatalf("bytes not verified")
	}
}

// TestVerifierUnsupportedDigest ensures that unsupported digest validation is
// flowing through verifier creation.
func TestVerifierUnsupportedDigest(t *testing.T) {
	unsupported := Digest("bean:0123456789abcdef")

	_, err := NewDigestVerifier(unsupported)
	if err == nil {
		t.Fatalf("expected error when creating verifier")
	}

	if err != ErrDigestUnsupported {
		t.Fatalf("incorrect error for unsupported digest: %v", err)
	}
}

// TODO(stevvooe): Add benchmarks to measure bytes/second throughput for
// DigestVerifier.
//
// The relevant benchmark for comparison can be run with the following
// commands:
//
// 	go test -bench . crypto/sha1
//
