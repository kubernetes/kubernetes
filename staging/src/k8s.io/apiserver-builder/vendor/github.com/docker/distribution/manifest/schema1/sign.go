package schema1

import (
	"crypto/x509"
	"encoding/json"

	"github.com/docker/libtrust"
)

// Sign signs the manifest with the provided private key, returning a
// SignedManifest. This typically won't be used within the registry, except
// for testing.
func Sign(m *Manifest, pk libtrust.PrivateKey) (*SignedManifest, error) {
	p, err := json.MarshalIndent(m, "", "   ")
	if err != nil {
		return nil, err
	}

	js, err := libtrust.NewJSONSignature(p)
	if err != nil {
		return nil, err
	}

	if err := js.Sign(pk); err != nil {
		return nil, err
	}

	pretty, err := js.PrettySignature("signatures")
	if err != nil {
		return nil, err
	}

	return &SignedManifest{
		Manifest:  *m,
		all:       pretty,
		Canonical: p,
	}, nil
}

// SignWithChain signs the manifest with the given private key and x509 chain.
// The public key of the first element in the chain must be the public key
// corresponding with the sign key.
func SignWithChain(m *Manifest, key libtrust.PrivateKey, chain []*x509.Certificate) (*SignedManifest, error) {
	p, err := json.MarshalIndent(m, "", "   ")
	if err != nil {
		return nil, err
	}

	js, err := libtrust.NewJSONSignature(p)
	if err != nil {
		return nil, err
	}

	if err := js.SignWithChain(key, chain); err != nil {
		return nil, err
	}

	pretty, err := js.PrettySignature("signatures")
	if err != nil {
		return nil, err
	}

	return &SignedManifest{
		Manifest:  *m,
		all:       pretty,
		Canonical: p,
	}, nil
}
