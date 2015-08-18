// Package genkey implements the genkey command.
package genkey

import (
	"encoding/json"
	"errors"

	"github.com/cloudflare/cfssl/cli"
	"github.com/cloudflare/cfssl/csr"
	cferr "github.com/cloudflare/cfssl/errors"
	"github.com/cloudflare/cfssl/initca"
)

var genkeyUsageText = `cfssl genkey -- generate a new key and CSR

Usage of genkey:
        cfssl genkey CSRJSON

Arguments:
        CSRJSON:    JSON file containing the request, use '-' for reading JSON from stdin

Flags:
`

var genkeyFlags = []string{"initca", "config"}

func genkeyMain(args []string, c cli.Config) (err error) {
	csrFile, args, err := cli.PopFirstArgument(args)
	if err != nil {
		return
	}

	csrFileBytes, err := cli.ReadStdin(csrFile)
	if err != nil {
		return
	}

	var req csr.CertificateRequest
	err = json.Unmarshal(csrFileBytes, &req)
	if err != nil {
		return
	}

	if c.IsCA {
		var key, csrPEM, cert []byte
		cert, csrPEM, key, err = initca.New(&req)
		if err != nil {
			return
		}

		cli.PrintCert(key, csrPEM, cert)
	} else {
		if req.CA != nil {
			err = errors.New("ca section only permitted in initca")
			return
		}

		var key, csrPEM []byte
		g := &csr.Generator{Validator: Validator}
		csrPEM, key, err = g.ProcessRequest(&req)
		if err != nil {
			key = nil
			return
		}

		cli.PrintCert(key, csrPEM, nil)
	}
	return nil
}

// Validator returns true if the csr has at least one host
func Validator(req *csr.CertificateRequest) error {
	if len(req.Hosts) == 0 {
		return cferr.Wrap(cferr.PolicyError, cferr.InvalidRequest, errors.New("missing hosts field"))
	}
	return nil
}

// CLIGenKey is a subcommand for generating a new key and CSR from a
// JSON CSR request file.
var Command = &cli.Command{UsageText: genkeyUsageText, Flags: genkeyFlags, Main: genkeyMain}
