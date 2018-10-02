package scan

import (
	"bytes"
	"crypto/x509"
	"fmt"
	"time"

	"github.com/cloudflare/cfssl/helpers"
	"github.com/cloudflare/cfssl/revoke"
	"github.com/cloudflare/cfssl/scan/crypto/tls"
)

// PKI contains scanners for the Public Key Infrastructure.
var PKI = &Family{
	Description: "Scans for the Public Key Infrastructure",
	Scanners: map[string]*Scanner{
		"ChainExpiration": {
			"Host's chain hasn't expired and won't expire in the next 30 days",
			chainExpiration,
		},
		"ChainValidation": {
			"All certificates in host's chain are valid",
			chainValidation,
		},
		"MultipleCerts": {
			"Host serves same certificate chain across all IPs",
			multipleCerts,
		},
	},
}

// getChain is a helper function that retreives the host's certificate chain.
func getChain(addr string, config *tls.Config) (chain []*x509.Certificate, err error) {
	var conn *tls.Conn
	conn, err = tls.DialWithDialer(Dialer, Network, addr, config)
	if err != nil {
		return
	}

	err = conn.Close()
	if err != nil {
		return
	}

	chain = conn.ConnectionState().PeerCertificates
	if len(chain) == 0 {
		err = fmt.Errorf("%s returned empty certificate chain", addr)
	}
	return
}

type expiration time.Time

func (e expiration) String() string {
	return time.Time(e).Format("Jan 2 15:04:05 2006 MST")
}

func chainExpiration(addr, hostname string) (grade Grade, output Output, err error) {
	chain, err := getChain(addr, defaultTLSConfig(hostname))
	if err != nil {
		return
	}

	expirationTime := helpers.ExpiryTime(chain)
	output = expirationTime

	if time.Now().After(expirationTime) {
		return
	}

	// Warn if cert will expire in the next 30 days
	if time.Now().Add(time.Hour * 24 * 30).After(expirationTime) {
		grade = Warning
		return
	}

	grade = Good
	return
}

func chainValidation(addr, hostname string) (grade Grade, output Output, err error) {
	chain, err := getChain(addr, defaultTLSConfig(hostname))
	if err != nil {
		return
	}

	var warnings []string

	for i := 0; i < len(chain)-1; i++ {
		cert, parent := chain[i], chain[i+1]

		valid := helpers.ValidExpiry(cert)
		if !valid {
			warnings = append(warnings, fmt.Sprintf("Certificate for %s is valid for too long", cert.Subject.CommonName))
		}

		revoked, ok := revoke.VerifyCertificate(cert)
		if !ok {
			warnings = append(warnings, fmt.Sprintf("couldn't check if %s is revoked", cert.Subject.CommonName))
		}
		if revoked {
			err = fmt.Errorf("%s is revoked", cert.Subject.CommonName)
			return
		}

		if !parent.IsCA {
			err = fmt.Errorf("%s is not a CA", parent.Subject.CommonName)
			return
		}

		if !bytes.Equal(cert.AuthorityKeyId, parent.SubjectKeyId) {
			err = fmt.Errorf("%s AuthorityKeyId differs from %s SubjectKeyId", cert.Subject.CommonName, parent.Subject.CommonName)
			return
		}

		if err = cert.CheckSignatureFrom(parent); err != nil {
			return
		}

		switch cert.SignatureAlgorithm {
		case x509.ECDSAWithSHA1:
			warnings = append(warnings, fmt.Sprintf("%s is signed by ECDSAWithSHA1", cert.Subject.CommonName))
		case x509.SHA1WithRSA:
			warnings = append(warnings, fmt.Sprintf("%s is signed by RSAWithSHA1", cert.Subject.CommonName))
		}
	}

	if len(warnings) == 0 {
		grade = Good
	} else {
		grade = Warning
		output = warnings
	}
	return
}

func multipleCerts(addr, hostname string) (grade Grade, output Output, err error) {
	config := defaultTLSConfig(hostname)

	firstChain, err := getChain(addr, config)
	if err != nil {
		return
	}

	grade, _, err = multiscan(addr, func(addrport string) (g Grade, o Output, e error) {
		g = Good
		chain, e1 := getChain(addrport, config)
		if e1 != nil {
			return
		}

		if !chain[0].Equal(firstChain[0]) {
			e = fmt.Errorf("%s not equal to %s", chain[0].Subject.CommonName, firstChain[0].Subject.CommonName)
			g = Bad
			return
		}
		return
	})
	return
}
