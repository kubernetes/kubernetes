// Package selfsign implements the selfsign command.
package selfsign

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/cloudflare/cfssl/cli"
	"github.com/cloudflare/cfssl/cli/genkey"
	"github.com/cloudflare/cfssl/config"
	"github.com/cloudflare/cfssl/csr"
	"github.com/cloudflare/cfssl/helpers"
	"github.com/cloudflare/cfssl/selfsign"
)

var selfSignUsageText = `cfssl selfsign -- generate a new self-signed key and signed certificate

Usage of gencert:
        cfssl selfsign HOSTNAME CSRJSON

WARNING: this should ONLY be used for testing. This should never be
used in production.

WARNING: self-signed certificates are insecure; they do not provide
the authentication required for secure systems. Use these at your own
risk.

Arguments:
        HOSTNAME:   Hostname for the cert
        CSRJSON:    JSON file containing the request, use '-' for reading JSON from stdin

Flags:
`

var selfSignFlags = []string{"config"}

func selfSignMain(args []string, c cli.Config) (err error) {
	if c.Hostname == "" && !c.IsCA {
		c.Hostname, args, err = cli.PopFirstArgument(args)
		if err != nil {
			return
		}
	}

	csrFile, args, err := cli.PopFirstArgument(args)
	if err != nil {
		return
	}

	if len(args) > 0 {
		return errors.New("too many arguments are provided, please check with usage")
	}

	csrFileBytes, err := cli.ReadStdin(csrFile)
	if err != nil {
		return
	}

	var req = csr.New()
	err = json.Unmarshal(csrFileBytes, req)
	if err != nil {
		return
	}

	var key, csrPEM []byte
	g := &csr.Generator{Validator: genkey.Validator}
	csrPEM, key, err = g.ProcessRequest(req)
	if err != nil {
		key = nil
		return
	}

	priv, err := helpers.ParsePrivateKeyPEM(key)
	if err != nil {
		key = nil
		return
	}

	var profile *config.SigningProfile

	// If there is a config, use its signing policy. Otherwise, leave policy == nil
	// and NewSigner will use DefaultConfig().
	if c.CFG != nil {
		if c.Profile != "" && c.CFG.Signing.Profiles != nil {
			profile = c.CFG.Signing.Profiles[c.Profile]
		}
	}

	if profile == nil {
		profile = config.DefaultConfig()
		profile.Expiry = 2190 * time.Hour
	}

	cert, err := selfsign.Sign(priv, csrPEM, profile)
	if err != nil {
		key = nil
		priv = nil
		return
	}

	fmt.Fprintf(os.Stderr, `*** WARNING ***

Self-signed certificates are dangerous. Use this self-signed
certificate at your own risk.

It is strongly recommended that these certificates NOT be used
in production.

*** WARNING ***

`)
	cli.PrintCert(key, csrPEM, cert)
	return
}

// Command assembles the definition of Command 'selfsign'
var Command = &cli.Command{UsageText: selfSignUsageText, Flags: selfSignFlags, Main: selfSignMain}
