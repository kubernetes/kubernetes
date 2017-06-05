package digest

import (
	"crypto"
	"fmt"
	"hash"
	"io"
)

// Algorithm identifies and implementation of a digester by an identifier.
// Note the that this defines both the hash algorithm used and the string
// encoding.
type Algorithm string

// supported digest types
const (
	SHA256 Algorithm = "sha256" // sha256 with hex encoding
	SHA384 Algorithm = "sha384" // sha384 with hex encoding
	SHA512 Algorithm = "sha512" // sha512 with hex encoding

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
)

// Available returns true if the digest type is available for use. If this
// returns false, New and Hash will return nil.
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

	return nil
}

// New returns a new digester for the specified algorithm. If the algorithm
// does not have a digester implementation, nil will be returned. This can be
// checked by calling Available before calling New.
func (a Algorithm) New() Digester {
	return &digester{
		alg:  a,
		hash: a.Hash(),
	}
}

// Hash returns a new hash as used by the algorithm. If not available, the
// method will panic. Check Algorithm.Available() before calling.
func (a Algorithm) Hash() hash.Hash {
	if !a.Available() {
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

// FromReader returns the digest of the reader using the algorithm.
func (a Algorithm) FromReader(rd io.Reader) (Digest, error) {
	digester := a.New()

	if _, err := io.Copy(digester.Hash(), rd); err != nil {
		return "", err
	}

	return digester.Digest(), nil
}

// FromBytes digests the input and returns a Digest.
func (a Algorithm) FromBytes(p []byte) Digest {
	digester := a.New()

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

// TODO(stevvooe): Allow resolution of verifiers using the digest type and
// this registration system.

// Digester calculates the digest of written data. Writes should go directly
// to the return value of Hash, while calling Digest will return the current
// value of the digest.
type Digester interface {
	Hash() hash.Hash // provides direct access to underlying hash instance.
	Digest() Digest
}

// digester provides a simple digester definition that embeds a hasher.
type digester struct {
	alg  Algorithm
	hash hash.Hash
}

func (d *digester) Hash() hash.Hash {
	return d.hash
}

func (d *digester) Digest() Digest {
	return NewDigest(d.alg, d.hash)
}
