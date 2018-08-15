// Package bundle implements the HTTP handler for the bundle command.
package bundle

import (
	"net/http"

	"github.com/cloudflare/cfssl/api"
	"github.com/cloudflare/cfssl/bundler"
	"github.com/cloudflare/cfssl/errors"
	"github.com/cloudflare/cfssl/log"
)

// Handler accepts requests for either remote or uploaded
// certificates to be bundled, and returns a certificate bundle (or
// error).
type Handler struct {
	bundler *bundler.Bundler
}

// NewHandler creates a new bundler that uses the root bundle and
// intermediate bundle in the trust chain.
func NewHandler(caBundleFile, intBundleFile string) (http.Handler, error) {
	var err error

	b := new(Handler)
	if b.bundler, err = bundler.NewBundler(caBundleFile, intBundleFile); err != nil {
		return nil, err
	}

	log.Info("bundler API ready")
	return api.HTTPHandler{Handler: b, Methods: []string{"POST"}}, nil
}

// Handle implements an http.Handler interface for the bundle handler.
func (h *Handler) Handle(w http.ResponseWriter, r *http.Request) error {
	blob, matched, err := api.ProcessRequestFirstMatchOf(r,
		[][]string{
			{"certificate"},
			{"domain"},
		})
	if err != nil {
		log.Warningf("invalid request: %v", err)
		return err
	}

	flavor := blob["flavor"]
	bf := bundler.Ubiquitous
	if flavor != "" {
		bf = bundler.BundleFlavor(flavor)
	}
	log.Infof("request for flavor %v", bf)

	var result *bundler.Bundle
	switch matched[0] {
	case "domain":
		bundle, err := h.bundler.BundleFromRemote(blob["domain"], blob["ip"], bf)
		if err != nil {
			log.Warningf("couldn't bundle from remote: %v", err)
			return err
		}
		result = bundle
	case "certificate":
		bundle, err := h.bundler.BundleFromPEMorDER([]byte(blob["certificate"]), []byte(blob["private_key"]), bf, "")
		if err != nil {
			log.Warning("bad PEM certifcate or private key")
			return err
		}

		serverName := blob["domain"]
		ip := blob["ip"]

		if serverName != "" {
			err := bundle.Cert.VerifyHostname(serverName)
			if err != nil {
				return errors.Wrap(errors.CertificateError, errors.VerifyFailed, err)
			}

		}

		if ip != "" {
			err := bundle.Cert.VerifyHostname(ip)
			if err != nil {
				return errors.Wrap(errors.CertificateError, errors.VerifyFailed, err)
			}
		}

		result = bundle
	}
	log.Info("wrote response")
	return api.SendResponse(w, result)
}
