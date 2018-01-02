package digest

import (
	"bytes"
	"crypto/rand"
	"io"
	"reflect"
	"testing"
)

func TestDigestVerifier(t *testing.T) {
	p := make([]byte, 1<<20)
	rand.Read(p)
	digest := FromBytes(p)

	verifier := digest.Verifier()

	io.Copy(verifier, bytes.NewReader(p))

	if !verifier.Verified() {
		t.Fatalf("bytes not verified")
	}
}

// TestVerifierUnsupportedDigest ensures that unsupported digest validation is
// flowing through verifier creation.
func TestVerifierUnsupportedDigest(t *testing.T) {
	for _, testcase := range []struct {
		Name     string
		Digest   Digest
		Expected interface{} // expected panic target
	}{
		{
			Name:     "Empty",
			Digest:   "",
			Expected: "no ':' separator in digest \"\"",
		},
		{
			Name:     "EmptyAlg",
			Digest:   ":",
			Expected: "empty digest algorithm, validate before calling Algorithm.Hash()",
		},
		{
			Name:     "Unsupported",
			Digest:   Digest("bean:0123456789abcdef"),
			Expected: "bean not available (make sure it is imported)",
		},
		{
			Name:     "Garbage",
			Digest:   Digest("sha256-garbage:pure"),
			Expected: "sha256-garbage not available (make sure it is imported)",
		},
	} {
		t.Run(testcase.Name, func(t *testing.T) {
			expected := testcase.Expected
			defer func() {
				recovered := recover()
				if !reflect.DeepEqual(recovered, expected) {
					t.Fatalf("unexpected recover: %v != %v", recovered, expected)
				}
			}()

			_ = testcase.Digest.Verifier()
		})
	}
}
