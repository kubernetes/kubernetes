// Package gencert implements the gencert command.
package gencert

import (
	"encoding/json"
	"errors"

	"github.com/cloudflare/cfssl/cli"
	"github.com/cloudflare/cfssl/cli/genkey"
	"github.com/cloudflare/cfssl/cli/sign"
	"github.com/cloudflare/cfssl/csr"
	"github.com/cloudflare/cfssl/initca"
	"github.com/cloudflare/cfssl/log"
	"github.com/cloudflare/cfssl/signer"
)

var gencertUsageText = `cfssl gencert -- generate a new key and signed certificate

Usage of gencert:
        cfssl gencert -initca CSRJSON
        cfssl gencert -ca cert -ca-key key [-config config] [-profile profile] [-hostname hostname] CSRJSON
        cfssl gencert -remote remote_host [-config config] [-profile profile] [-label label] [-hostname hostname] CSRJSON

Arguments:
        CSRJSON:    JSON file containing the request, use '-' for reading JSON from stdin

Flags:
`

var gencertFlags = []string{"initca", "remote", "ca", "ca-key", "config", "hostname", "profile", "label"}

func gencertMain(args []string, c cli.Config) (err error) {

	csrJSONFile, args, err := cli.PopFirstArgument(args)
	if err != nil {
		return
	}

	csrJSONFileBytes, err := cli.ReadStdin(csrJSONFile)
	if err != nil {
		return
	}

	var req csr.CertificateRequest
	err = json.Unmarshal(csrJSONFileBytes, &req)
	if err != nil {
		return
	}

	if c.IsCA {
		var key, csrPEM, cert []byte
		cert, csrPEM, err = initca.NewFromPEM(&req, c.CAKeyFile)
		if err != nil {
			log.Errorf("%v\n", err)
			log.Infof("generating a new CA key and certificate from CSR")
			cert, csrPEM, key, err = initca.New(&req)
			if err != nil {
				return
			}

		}
		cli.PrintCert(key, csrPEM, cert)
	} else {
		if req.CA != nil {
			err = errors.New("ca section only permitted in initca")
			return
		}

		// Remote can be forced on the command line or in the config
		if c.Remote == "" && c.CFG == nil {
			if c.CAFile == "" {
				log.Error("need a CA certificate (provide one with -ca)")
				return
			}

			if c.CAKeyFile == "" {
				log.Error("need a CA key (provide one with -ca-key)")
				return
			}
		}

		var key, csrBytes []byte
		g := &csr.Generator{Validator: genkey.Validator}
		csrBytes, key, err = g.ProcessRequest(&req)
		if err != nil {
			key = nil
			return
		}

		s, err := sign.SignerFromConfig(c)
		if err != nil {
			return err
		}

		var cert []byte
		req := signer.SignRequest{
			Request: string(csrBytes),
			Hosts:   signer.SplitHosts(c.Hostname),
			Profile: c.Profile,
			Label:   c.Label,
		}

		cert, err = s.Sign(req)
		if err != nil {
			return err
		}

		cli.PrintCert(key, csrBytes, cert)
	}
	return nil
}

// CLIGenCert is a subcommand that generates a new certificate from a
// JSON CSR request file.
var Command = &cli.Command{UsageText: gencertUsageText, Flags: gencertFlags, Main: gencertMain}
