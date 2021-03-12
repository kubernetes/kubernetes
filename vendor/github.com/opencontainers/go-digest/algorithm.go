// Copyright 2019, 2020 OCI Contributors
// Copyright 2017 Docker, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package digest

import (
	"crypto"
	"fmt"
	"hash"
	"io"
	"regexp"
)

// Algorithm identifies and implementation of a digester by an identifier.
// Note the that this defines both the hash algorithm used and the string
// encoding.
type Algorithm string

// supported digest types
const (
	SHA256 Algorithm = "sha256" // sha256 with hex encoding (lower case only)
	SHA384 Algorithm = "sha384" // sha384 with hex encoding (lower case only)
	SHA512 Algorithm = "sha512" // sha512 with hex encoding (lower case only)

	// Canonical is the primary digest algorithm used with the distribution
	// project. Other digests may be used but this one is the primary storage
	// digest.
	Canonical = SHA256
)

var (
	// TODO(stevvooe): Follow the pattern of the standard crypto package for
	// registration of digests. Effectively, we are a registerable set and
	// common symbol access.

	// algorithms maps values to hash.Hash implementations. Other algorithms
	// may be available but they cannot be calculated by the digest package.
	algorithms = map[Algorithm]crypto.Hash{
		SHA256: crypto.SHA256,
		SHA384: crypto.SHA384,
		SHA512: crypto.SHA512,
	}

	// anchoredEncodedRegexps contains anchored regular expressions for hex-encoded digests.
	// Note that /A-F/ disallowed.
	anchoredEncodedRegexps = map[Algorithm]*regexp.Regexp{
		SHA256: regexp.MustCompile(`^[a-f0-9]{64}$`),
		SHA384: regexp.MustCompile(`^[a-f0-9]{96}$`),
		SHA512: regexp.MustCompile(`^[a-f0-9]{128}$`),
	}
)

// Available returns true if the digest type is available for use. If this
// returns false, Digester and Hash will return nil.
func (a Algorithm) Available() bool {
	h, ok := algorithms[a]
	if !ok {
		return false
	}

	// check availability of the hash, as well
	return h.Available()
}

func (a Algorithm) String() string {
	return string(a)
}

// Size returns number of bytes returned by the hash.
func (a Algorithm) Size() int {
	h, ok := algorithms[a]
	if !ok {
		return 0
	}
	return h.Size()
}

// Set implemented to allow use of Algorithm as a command line flag.
func (a *Algorithm) Set(value string) error {
	if value == "" {
		*a = Canonical
	} else {
		// just do a type conversion, support is queried with Available.
		*a = Algorithm(value)
	}

	if !a.Available() {
		return ErrDigestUnsupported
	}

	return nil
}

// Digester returns a new digester for the specified algorithm. If the algorithm
// does not have a digester implementation, nil will be returned. This can be
// checked by calling Available before calling Digester.
func (a Algorithm) Digester() Digester {
	return &digester{
		alg:  a,
		hash: a.Hash(),
	}
}

// Hash returns a new hash as used by the algorithm. If not available, the
// method will panic. Check Algorithm.Available() before calling.
func (a Algorithm) Hash() hash.Hash {
	if !a.Available() {
		// Empty algorithm string is invalid
		if a == "" {
			panic(fmt.Sprintf("empty digest algorithm, validate before calling Algorithm.Hash()"))
		}

		// NOTE(stevvooe): A missing hash is usually a programming error that
		// must be resolved at compile time. We don't import in the digest
		// package to allow users to choose their hash implementation (such as
		// when using stevvooe/resumable or a hardware accelerated package).
		//
		// Applications that may want to resolve the hash at runtime should
		// call Algorithm.Available before call Algorithm.Hash().
		panic(fmt.Sprintf("%v not available (make sure it is imported)", a))
	}

	return algorithms[a].New()
}

// Encode encodes the raw bytes of a digest, typically from a hash.Hash, into
// the encoded portion of the digest.
func (a Algorithm) Encode(d []byte) string {
	// TODO(stevvooe): Currently, all algorithms use a hex encoding. When we
	// add support for back registration, we can modify this accordingly.
	return fmt.Sprintf("%x", d)
}

// FromReader returns the digest of the reader using the algorithm.
func (a Algorithm) FromReader(rd io.Reader) (Digest, error) {
	digester := a.Digester()

	if _, err := io.Copy(digester.Hash(), rd); err != nil {
		return "", err
	}

	return digester.Digest(), nil
}

// FromBytes digests the input and returns a Digest.
func (a Algorithm) FromBytes(p []byte) Digest {
	digester := a.Digester()

	if _, err := digester.Hash().Write(p); err != nil {
		// Writes to a Hash should never fail. None of the existing
		// hash implementations in the stdlib or hashes vendored
		// here can return errors from Write. Having a panic in this
		// condition instead of having FromBytes return an error value
		// avoids unnecessary error handling paths in all callers.
		panic("write to hash function returned error: " + err.Error())
	}

	return digester.Digest()
}

// FromString digests the string input and returns a Digest.
func (a Algorithm) FromString(s string) Digest {
	return a.FromBytes([]byte(s))
}

// Validate validates the encoded portion string
func (a Algorithm) Validate(encoded string) error {
	r, ok := anchoredEncodedRegexps[a]
	if !ok {
		return ErrDigestUnsupported
	}
	// Digests much always be hex-encoded, ensuring that their hex portion will
	// always be size*2
	if a.Size()*2 != len(encoded) {
		return ErrDigestInvalidLength
	}
	if r.MatchString(encoded) {
		return nil
	}
	return ErrDigestInvalidFormat
}
