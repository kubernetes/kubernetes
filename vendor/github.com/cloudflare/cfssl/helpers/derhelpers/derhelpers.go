// Package derhelpers implements common functionality
// on DER encoded data
package derhelpers

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/rsa"
	"crypto/x509"

	cferr "github.com/cloudflare/cfssl/errors"
	"golang.org/x/crypto/ed25519"
)

// ParsePrivateKeyDER parses a PKCS #1, PKCS #8, ECDSA, or Ed25519 DER-encoded
// private key. The key must not be in PEM format.
func ParsePrivateKeyDER(keyDER []byte) (key crypto.Signer, err error) {
	generalKey, err := x509.ParsePKCS8PrivateKey(keyDER)
	if err != nil {
		generalKey, err = x509.ParsePKCS1PrivateKey(keyDER)
		if err != nil {
			generalKey, err = x509.ParseECPrivateKey(keyDER)
			if err != nil {
				generalKey, err = ParseEd25519PrivateKey(keyDER)
				if err != nil {
					// We don't include the actual error into
					// the final error. The reason might be
					// we don't want to leak any info about
					// the private key.
					return nil, cferr.New(cferr.PrivateKeyError,
						cferr.ParseFailed)
				}
			}
		}
	}

	switch generalKey.(type) {
	case *rsa.PrivateKey:
		return generalKey.(*rsa.PrivateKey), nil
	case *ecdsa.PrivateKey:
		return generalKey.(*ecdsa.PrivateKey), nil
	case ed25519.PrivateKey:
		return generalKey.(ed25519.PrivateKey), nil
	}

	// should never reach here
	return nil, cferr.New(cferr.PrivateKeyError, cferr.ParseFailed)
}
