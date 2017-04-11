package digest

import (
	"hash"
	"io"
)

// Verifier presents a general verification interface to be used with message
// digests and other byte stream verifications. Users instantiate a Verifier
// from one of the various methods, write the data under test to it then check
// the result with the Verified method.
type Verifier interface {
	io.Writer

	// Verified will return true if the content written to Verifier matches
	// the digest.
	Verified() bool
}

// NewDigestVerifier returns a verifier that compares the written bytes
// against a passed in digest.
func NewDigestVerifier(d Digest) (Verifier, error) {
	if err := d.Validate(); err != nil {
		return nil, err
	}

	return hashVerifier{
		hash:   d.Algorithm().Hash(),
		digest: d,
	}, nil
}

type hashVerifier struct {
	digest Digest
	hash   hash.Hash
}

func (hv hashVerifier) Write(p []byte) (n int, err error) {
	return hv.hash.Write(p)
}

func (hv hashVerifier) Verified() bool {
	return hv.digest == NewDigest(hv.digest.Algorithm(), hv.hash)
}
