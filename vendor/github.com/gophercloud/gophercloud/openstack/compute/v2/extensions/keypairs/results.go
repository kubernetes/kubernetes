package keypairs

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// KeyPair is an SSH key known to the OpenStack cluster that is available to be injected into
// servers.
type KeyPair struct {
	// Name is used to refer to this keypair from other services within this region.
	Name string `json:"name"`

	// Fingerprint is a short sequence of bytes that can be used to authenticate or validate a longer
	// public key.
	Fingerprint string `json:"fingerprint"`

	// PublicKey is the public key from this pair, in OpenSSH format. "ssh-rsa AAAAB3Nz..."
	PublicKey string `json:"public_key"`

	// PrivateKey is the private key from this pair, in PEM format.
	// "-----BEGIN RSA PRIVATE KEY-----\nMIICXA..." It is only present if this keypair was just
	// returned from a Create call
	PrivateKey string `json:"private_key"`

	// UserID is the user who owns this keypair.
	UserID string `json:"user_id"`
}

// KeyPairPage stores a single, only page of KeyPair results from a List call.
type KeyPairPage struct {
	pagination.SinglePageBase
}

// IsEmpty determines whether or not a KeyPairPage is empty.
func (page KeyPairPage) IsEmpty() (bool, error) {
	ks, err := ExtractKeyPairs(page)
	return len(ks) == 0, err
}

// ExtractKeyPairs interprets a page of results as a slice of KeyPairs.
func ExtractKeyPairs(r pagination.Page) ([]KeyPair, error) {
	type pair struct {
		KeyPair KeyPair `json:"keypair"`
	}
	var s struct {
		KeyPairs []pair `json:"keypairs"`
	}
	err := (r.(KeyPairPage)).ExtractInto(&s)
	results := make([]KeyPair, len(s.KeyPairs))
	for i, pair := range s.KeyPairs {
		results[i] = pair.KeyPair
	}
	return results, err
}

type keyPairResult struct {
	gophercloud.Result
}

// Extract is a method that attempts to interpret any KeyPair resource response as a KeyPair struct.
func (r keyPairResult) Extract() (*KeyPair, error) {
	var s struct {
		KeyPair *KeyPair `json:"keypair"`
	}
	err := r.ExtractInto(&s)
	return s.KeyPair, err
}

// CreateResult is the response from a Create operation. Call its Extract method to interpret it
// as a KeyPair.
type CreateResult struct {
	keyPairResult
}

// GetResult is the response from a Get operation. Call its Extract method to interpret it
// as a KeyPair.
type GetResult struct {
	keyPairResult
}

// DeleteResult is the response from a Delete operation. Call its Extract method to determine if
// the call succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}
