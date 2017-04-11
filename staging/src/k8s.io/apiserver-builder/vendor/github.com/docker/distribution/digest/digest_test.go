package digest

import (
	"testing"
)

func TestParseDigest(t *testing.T) {
	for _, testcase := range []struct {
		input     string
		err       error
		algorithm Algorithm
		hex       string
	}{
		{
			input:     "sha256:e58fcf7418d4390dec8e8fb69d88c06ec07039d651fedd3aa72af9972e7d046b",
			algorithm: "sha256",
			hex:       "e58fcf7418d4390dec8e8fb69d88c06ec07039d651fedd3aa72af9972e7d046b",
		},
		{
			input:     "sha384:d3fc7881460b7e22e3d172954463dddd7866d17597e7248453c48b3e9d26d9596bf9c4a9cf8072c9d5bad76e19af801d",
			algorithm: "sha384",
			hex:       "d3fc7881460b7e22e3d172954463dddd7866d17597e7248453c48b3e9d26d9596bf9c4a9cf8072c9d5bad76e19af801d",
		},
		{
			// empty hex
			input: "sha256:",
			err:   ErrDigestInvalidFormat,
		},
		{
			// just hex
			input: "d41d8cd98f00b204e9800998ecf8427e",
			err:   ErrDigestInvalidFormat,
		},
		{
			// not hex
			input: "sha256:d41d8cd98f00b204e9800m98ecf8427e",
			err:   ErrDigestInvalidFormat,
		},
		{
			// too short
			input: "sha256:abcdef0123456789",
			err:   ErrDigestInvalidLength,
		},
		{
			// too short (from different algorithm)
			input: "sha512:abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
			err:   ErrDigestInvalidLength,
		},
		{
			input: "foo:d41d8cd98f00b204e9800998ecf8427e",
			err:   ErrDigestUnsupported,
		},
	} {
		digest, err := ParseDigest(testcase.input)
		if err != testcase.err {
			t.Fatalf("error differed from expected while parsing %q: %v != %v", testcase.input, err, testcase.err)
		}

		if testcase.err != nil {
			continue
		}

		if digest.Algorithm() != testcase.algorithm {
			t.Fatalf("incorrect algorithm for parsed digest: %q != %q", digest.Algorithm(), testcase.algorithm)
		}

		if digest.Hex() != testcase.hex {
			t.Fatalf("incorrect hex for parsed digest: %q != %q", digest.Hex(), testcase.hex)
		}

		// Parse string return value and check equality
		newParsed, err := ParseDigest(digest.String())

		if err != nil {
			t.Fatalf("unexpected error parsing input %q: %v", testcase.input, err)
		}

		if newParsed != digest {
			t.Fatalf("expected equal: %q != %q", newParsed, digest)
		}
	}
}
