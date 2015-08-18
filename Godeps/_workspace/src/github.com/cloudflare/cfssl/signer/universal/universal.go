// Package universal implements a signer that can do remote or local
package universal

import (
	"github.com/cloudflare/cfssl/config"
	cferr "github.com/cloudflare/cfssl/errors"
	"github.com/cloudflare/cfssl/signer"
	"github.com/cloudflare/cfssl/signer/local"
	"github.com/cloudflare/cfssl/signer/pkcs11"
	"github.com/cloudflare/cfssl/signer/remote"
)

// Root is used to define where the universal signer gets its public
// certificate and private keys for signing.
type Root struct {
	Config      map[string]string
	ForceRemote bool
}

// a localSignerCheck looks at the Config keys in the Root, and
// decides whether it has enough information to produce a signer.
type localSignerCheck func(root *Root, policy *config.Signing) (signer.Signer, bool, error)

// fileBackedSigner determines whether a file-backed local signer is supported.
func fileBackedSigner(root *Root, policy *config.Signing) (signer.Signer, bool, error) {
	keyFile := root.Config["key-file"]
	certFile := root.Config["cert-file"]

	if keyFile == "" {
		return nil, false, nil
	}

	signer, err := local.NewSignerFromFile(certFile, keyFile, policy)
	return signer, true, err
}

// pkcs11Signer looks for token, module, slot, and PIN configuration
// options in the root.
func pkcs11Signer(root *Root, policy *config.Signing) (signer.Signer, bool, error) {
	module := root.Config["pkcs11-module"]
	token := root.Config["pkcs11-token"]
	label := root.Config["pkcs11-label"]
	userPIN := root.Config["pkcs11-user-pin"]
	certFile := root.Config["cert-file"]

	if module == "" && token == "" && label == "" && userPIN == "" {
		return nil, false, nil
	}

	if !pkcs11.Enabled {
		return nil, true, cferr.New(cferr.PrivateKeyError, cferr.Unavailable)
	}

	conf := pkcs11.Config{
		Module: module,
		Token:  token,
		Label:  label,
		PIN:    userPIN,
	}

	s, err := pkcs11.New(certFile, policy, &conf)
	return s, true, err
}

var localSignerList = []localSignerCheck{
	pkcs11Signer,
	fileBackedSigner,
}

// NewSigner generates a new certificate signer from a Root structure.
// This is one of two standard signers: local or remote. If the root
// structure specifies a force remote, then a remote signer is created,
// otherwise either a remote or local signer is generated based on the
// policy. For a local signer, the CertFile and KeyFile need to be
// defined in Root.
func NewSigner(root Root, policy *config.Signing) (signer.Signer, error) {
	if policy == nil {
		policy = &config.Signing{
			Profiles: map[string]*config.SigningProfile{},
			Default:  config.DefaultConfig(),
		}
	}

	if !policy.Valid() {
		return nil, cferr.New(cferr.PolicyError, cferr.InvalidPolicy)
	}

	var s signer.Signer
	var err error
	if root.ForceRemote {
		s, err = remote.NewSigner(policy)
	} else {
		if policy.NeedsLocalSigner() && policy.NeedsRemoteSigner() {
			// Currently we don't support a hybrid signer
			return nil, cferr.New(cferr.PolicyError, cferr.InvalidPolicy)
		}

		if policy.NeedsLocalSigner() {
			// shouldProvide indicates whether the
			// function *should* have produced a key. If
			// it's true, we should use the signer and
			// error returned. Otherwise, keep looking for
			// signers.
			var shouldProvide bool
			// localSignerList is defined in the
			// universal_signers*.go files. These activate
			// and deactivate signers based on build
			// flags; for example,
			// universal_signers_pkcs11.go contains a list
			// of valid signers when PKCS #11 is turned
			// on.
			for _, possibleSigner := range localSignerList {
				s, shouldProvide, err = possibleSigner(&root, policy)
				if shouldProvide {
					break
				}
			}

			if s == nil {
				err = cferr.New(cferr.PrivateKeyError, cferr.Unknown)
			}
		}

		if policy.NeedsRemoteSigner() {
			s, err = remote.NewSigner(policy)
		}
	}

	return s, err
}
