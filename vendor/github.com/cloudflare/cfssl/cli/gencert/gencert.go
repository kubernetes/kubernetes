// Package gencert implements the gencert command.
package gencert

import (
	"encoding/json"
	"errors"

	"github.com/cloudflare/cfssl/api/generator"
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
    Generate a new key and cert from CSR:
        cfssl gencert -initca CSRJSON
        cfssl gencert -ca cert -ca-key key [-config config] [-profile profile] [-hostname hostname] CSRJSON
        cfssl gencert -remote remote_host [-config config] [-profile profile] [-label label] [-hostname hostname] CSRJSON

    Re-generate a CA cert with the CA key and CSR:
        cfssl gencert -initca -ca-key key CSRJSON

    Re-generate a CA cert with the CA key and certificate:
        cfssl gencert -renewca -ca cert -ca-key key

Arguments:
        CSRJSON:    JSON file containing the request, use '-' for reading JSON from stdin

Flags:
`

var gencertFlags = []string{"initca", "remote", "ca", "ca-key", "config", "cn", "hostname", "profile", "label"}

func gencertMain(args []string, c cli.Config) error {
	if c.RenewCA {
		log.Infof("re-generate a CA certificate from CA cert and key")
		cert, err := initca.RenewFromPEM(c.CAFile, c.CAKeyFile)
		if err != nil {
			log.Errorf("%v\n", err)
			return err
		}
		cli.PrintCert(nil, nil, cert)
		return nil
	}

	csrJSONFile, args, err := cli.PopFirstArgument(args)
	if err != nil {
		return err
	}

	if len(args) > 0 {
		return errors.New("only one argument is accepted, please check with usage")
	}

	csrJSONFileBytes, err := cli.ReadStdin(csrJSONFile)
	if err != nil {
		return err
	}

	req := csr.CertificateRequest{
		KeyRequest: csr.NewBasicKeyRequest(),
	}
	err = json.Unmarshal(csrJSONFileBytes, &req)
	if err != nil {
		return err
	}
	if c.CNOverride != "" {
		req.CN = c.CNOverride
	}
	switch {
	case c.IsCA:
		var key, csrPEM, cert []byte
		if c.CAKeyFile != "" {
			log.Infof("re-generate a CA certificate from CSR and CA key")
			cert, csrPEM, err = initca.NewFromPEM(&req, c.CAKeyFile)
			if err != nil {
				log.Errorf("%v\n", err)
				return err
			}
		} else {
			log.Infof("generating a new CA key and certificate from CSR")
			cert, csrPEM, key, err = initca.New(&req)
			if err != nil {
				return err
			}

		}
		cli.PrintCert(key, csrPEM, cert)

	default:
		if req.CA != nil {
			err = errors.New("ca section only permitted in initca")
			return err
		}

		if c.Hostname != "" {
			req.Hosts = signer.SplitHosts(c.Hostname)
		}
		// Remote can be forced on the command line or in the config
		if c.Remote == "" && c.CFG == nil {
			if c.CAFile == "" {
				log.Error("need a CA certificate (provide one with -ca)")
				return nil
			}

			if c.CAKeyFile == "" {
				log.Error("need a CA key (provide one with -ca-key)")
				return nil
			}
		}

		var key, csrBytes []byte
		g := &csr.Generator{Validator: genkey.Validator}
		csrBytes, key, err = g.ProcessRequest(&req)
		if err != nil {
			key = nil
			return err
		}

		s, err := sign.SignerFromConfig(c)
		if err != nil {
			return err
		}

		var cert []byte
		signReq := signer.SignRequest{
			Request: string(csrBytes),
			Hosts:   signer.SplitHosts(c.Hostname),
			Profile: c.Profile,
			Label:   c.Label,
		}

		if c.CRL != "" {
			signReq.CRLOverride = c.CRL
		}
		cert, err = s.Sign(signReq)
		if err != nil {
			return err
		}

		// This follows the Baseline Requirements for the Issuance and
		// Management of Publicly-Trusted Certificates, v.1.1.6, from the CA/Browser
		// Forum (https://cabforum.org). Specifically, section 10.2.3 ("Information
		// Requirements"), states:
		//
		// "Applicant information MUST include, but not be limited to, at least one
		// Fully-Qualified Domain Name or IP address to be included in the Certificate’s
		// SubjectAltName extension."
		if len(signReq.Hosts) == 0 && len(req.Hosts) == 0 {
			log.Warning(generator.CSRNoHostMessage)
		}

		cli.PrintCert(key, csrBytes, cert)
	}
	return nil
}

// Command assembles the definition of Command 'gencert'
var Command = &cli.Command{UsageText: gencertUsageText, Flags: gencertFlags, Main: gencertMain}
