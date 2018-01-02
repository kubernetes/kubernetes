package certinfo

import (
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"errors"
	"fmt"
	"io/ioutil"
	"net"
	"strings"
	"time"

	"github.com/cloudflare/cfssl/helpers"
)

// Certificate represents a JSON description of an X.509 certificate.
type Certificate struct {
	Subject            Name      `json:"subject,omitempty"`
	Issuer             Name      `json:"issuer,omitempty"`
	SerialNumber       string    `json:"serial_number,omitempty"`
	SANs               []string  `json:"sans,omitempty"`
	NotBefore          time.Time `json:"not_before"`
	NotAfter           time.Time `json:"not_after"`
	SignatureAlgorithm string    `json:"sigalg"`
	AKI                string    `json:"authority_key_id"`
	SKI                string    `json:"subject_key_id"`
	RawPEM             string    `json:"pem"`
}

// Name represents a JSON description of a PKIX Name
type Name struct {
	CommonName         string        `json:"common_name,omitempty"`
	SerialNumber       string        `json:"serial_number,omitempty"`
	Country            string        `json:"country,omitempty"`
	Organization       string        `json:"organization,omitempty"`
	OrganizationalUnit string        `json:"organizational_unit,omitempty"`
	Locality           string        `json:"locality,omitempty"`
	Province           string        `json:"province,omitempty"`
	StreetAddress      string        `json:"street_address,omitempty"`
	PostalCode         string        `json:"postal_code,omitempty"`
	Names              []interface{} `json:"names,omitempty"`
	//ExtraNames         []interface{} `json:"extra_names,omitempty"`
}

// ParseName parses a new name from a *pkix.Name
func ParseName(name pkix.Name) Name {
	n := Name{
		CommonName:         name.CommonName,
		SerialNumber:       name.SerialNumber,
		Country:            strings.Join(name.Country, ","),
		Organization:       strings.Join(name.Organization, ","),
		OrganizationalUnit: strings.Join(name.OrganizationalUnit, ","),
		Locality:           strings.Join(name.Locality, ","),
		Province:           strings.Join(name.Province, ","),
		StreetAddress:      strings.Join(name.StreetAddress, ","),
		PostalCode:         strings.Join(name.PostalCode, ","),
	}

	for i := range name.Names {
		n.Names = append(n.Names, name.Names[i].Value)
	}

	// ExtraNames aren't supported in Go 1.4
	// for i := range name.ExtraNames {
	// 	n.ExtraNames = append(n.ExtraNames, name.ExtraNames[i].Value)
	// }

	return n
}

func formatKeyID(id []byte) string {
	var s string

	for i, c := range id {
		if i > 0 {
			s += ":"
		}
		s += fmt.Sprintf("%X", c)
	}

	return s
}

// ParseCertificate parses an x509 certificate.
func ParseCertificate(cert *x509.Certificate) *Certificate {
	c := &Certificate{
		RawPEM:             string(helpers.EncodeCertificatePEM(cert)),
		SignatureAlgorithm: helpers.SignatureString(cert.SignatureAlgorithm),
		NotBefore:          cert.NotBefore,
		NotAfter:           cert.NotAfter,
		Subject:            ParseName(cert.Subject),
		Issuer:             ParseName(cert.Issuer),
		SANs:               cert.DNSNames,
		AKI:                formatKeyID(cert.AuthorityKeyId),
		SKI:                formatKeyID(cert.SubjectKeyId),
		SerialNumber:       cert.SerialNumber.String(),
	}
	for _, ip := range cert.IPAddresses {
		c.SANs = append(c.SANs, ip.String())
	}
	return c
}

// ParseCertificateFile parses x509 certificate file.
func ParseCertificateFile(certFile string) (*Certificate, error) {
	certPEM, err := ioutil.ReadFile(certFile)
	if err != nil {
		return nil, err
	}

	return ParseCertificatePEM(certPEM)
}

// ParseCertificatePEM parses an x509 certificate PEM.
func ParseCertificatePEM(certPEM []byte) (*Certificate, error) {
	cert, err := helpers.ParseCertificatePEM(certPEM)
	if err != nil {
		return nil, err
	}

	return ParseCertificate(cert), nil
}

// ParseCSRPEM uses the helper to parse an x509 CSR PEM.
func ParseCSRPEM(csrPEM []byte) (*x509.CertificateRequest, error) {
	csrObject, err := helpers.ParseCSRPEM(csrPEM)
	if err != nil {
		return nil, err
	}

	return csrObject, nil
}

// ParseCSRFile uses the helper to parse an x509 CSR PEM file.
func ParseCSRFile(csrFile string) (*x509.CertificateRequest, error) {
	csrPEM, err := ioutil.ReadFile(csrFile)
	if err != nil {
		return nil, err
	}

	return ParseCSRPEM(csrPEM)
}

// ParseCertificateDomain parses the certificate served by the given domain.
func ParseCertificateDomain(domain string) (cert *Certificate, err error) {
	var host, port string
	if host, port, err = net.SplitHostPort(domain); err != nil {
		host = domain
		port = "443"
	}

	var conn *tls.Conn
	conn, err = tls.DialWithDialer(&net.Dialer{Timeout: 10 * time.Second}, "tcp", net.JoinHostPort(host, port), &tls.Config{InsecureSkipVerify: true})
	if err != nil {
		return
	}
	defer conn.Close()

	if len(conn.ConnectionState().PeerCertificates) == 0 {
		return nil, errors.New("received no server certificates")
	}

	cert = ParseCertificate(conn.ConnectionState().PeerCertificates[0])
	return
}
