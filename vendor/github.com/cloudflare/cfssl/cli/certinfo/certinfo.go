// Package certinfo implements the certinfo command
package certinfo

import (
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/cloudflare/cfssl/certinfo"
	"github.com/cloudflare/cfssl/cli"
)

// Usage text of 'cfssl certinfo'
var dataUsageText = `cfssl certinfo -- output certinfo about the given cert

Usage of certinfo:
	- Data from local certificate files
        cfssl certinfo -cert file
	- Data from local CSR file
        cfssl certinfo -csr file
	- Data from certificate from remote server.
        cfssl certinfo -domain domain_name

Flags:
`

// flags used by 'cfssl certinfo'
var certinfoFlags = []string{"cert", "csr", "domain"}

// certinfoMain is the main CLI of certinfo functionality
func certinfoMain(args []string, c cli.Config) (err error) {
	var cert *certinfo.Certificate
	var csr *x509.CertificateRequest

	if c.CertFile != "" {
		if c.CertFile == "-" {
			var certPEM []byte
			if certPEM, err = cli.ReadStdin(c.CertFile); err != nil {
				return
			}

			if cert, err = certinfo.ParseCertificatePEM(certPEM); err != nil {
				return
			}
		} else {
			if cert, err = certinfo.ParseCertificateFile(c.CertFile); err != nil {
				return
			}
		}
	} else if c.CSRFile != "" {
		if c.CSRFile == "-" {
			var csrPEM []byte
			if csrPEM, err = cli.ReadStdin(c.CSRFile); err != nil {
				return
			}
			if csr, err = certinfo.ParseCSRPEM(csrPEM); err != nil {
				return
			}
		} else {
			if csr, err = certinfo.ParseCSRFile(c.CSRFile); err != nil {
				return
			}
		}
	} else if c.Domain != "" {
		if cert, err = certinfo.ParseCertificateDomain(c.Domain); err != nil {
			return
		}
	} else {
		return errors.New("Must specify certinfo target through -cert, -csr, or -domain")
	}

	var b []byte
	if cert != nil {
		b, err = json.MarshalIndent(cert, "", "  ")
	} else if csr != nil {
		b, err = json.MarshalIndent(csr, "", "  ")
	}

	if err != nil {
		return
	}

	fmt.Println(string(b))
	return
}

// Command assembles the definition of Command 'certinfo'
var Command = &cli.Command{UsageText: dataUsageText, Flags: certinfoFlags, Main: certinfoMain}
