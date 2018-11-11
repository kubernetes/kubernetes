//Package gencrl implements the gencrl command
package gencrl

import (
	"github.com/cloudflare/cfssl/cli"
	"github.com/cloudflare/cfssl/crl"
	"strings"
)

var gencrlUsageText = `cfssl gencrl -- generate a new Certificate Revocation List

Usage of gencrl:
        cfssl gencrl INPUTFILE CERT KEY TIME

Arguments:
        INPUTFILE:               Text file with one serial number per line, use '-' for reading text from stdin
        CERT:                    The certificate that is signing this CRL, use '-' for reading text from stdin
        KEY:                     The private key of the certificate that is signing the CRL, use '-' for reading text from stdin
        TIME (OPTIONAL):         The desired expiration from now, in seconds

Flags:
`
var gencrlFlags = []string{}

func gencrlMain(args []string, c cli.Config) (err error) {
	serialList, args, err := cli.PopFirstArgument(args)
	if err != nil {
		return
	}

	serialListBytes, err := cli.ReadStdin(serialList)
	if err != nil {
		return
	}

	certFile, args, err := cli.PopFirstArgument(args)
	if err != nil {
		return
	}

	certFileBytes, err := cli.ReadStdin(certFile)
	if err != nil {
		return
	}

	keyFile, args, err := cli.PopFirstArgument(args)
	if err != nil {
		return
	}

	keyBytes, err := cli.ReadStdin(keyFile)
	if err != nil {
		return
	}

	// Default value if no expiry time is given
	timeString := string("0")

	if len(args) > 0 {
		timeArg, _, err := cli.PopFirstArgument(args)
		if err != nil {
			return err
		}

		timeString = string(timeArg)

		// This is used to get rid of newlines
		timeString = strings.TrimSpace(timeString)

	}

	req, err := crl.NewCRLFromFile(serialListBytes, certFileBytes, keyBytes, timeString)
	if err != nil {
		return
	}

	cli.PrintCRL(req)
	return nil
}

// Command assembles the definition of Command 'gencrl'
var Command = &cli.Command{UsageText: gencrlUsageText, Flags: gencrlFlags, Main: gencrlMain}
