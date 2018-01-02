package universal

import (
	"github.com/cloudflare/cfssl/ocsp"
	ocspConfig "github.com/cloudflare/cfssl/ocsp/config"
)

// NewSignerFromConfig generates a new OCSP signer from a config object.
func NewSignerFromConfig(cfg ocspConfig.Config) (ocsp.Signer, error) {
	return ocsp.NewSignerFromFile(cfg.CACertFile, cfg.ResponderCertFile,
		cfg.KeyFile, cfg.Interval)
}
