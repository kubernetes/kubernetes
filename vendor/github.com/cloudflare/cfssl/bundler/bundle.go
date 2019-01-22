package bundler

import (
	"bytes"
	"crypto/ecdsa"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"time"

	"github.com/cloudflare/cfssl/helpers"
)

// A Bundle contains a certificate and its trust chain. It is intended
// to store the most widely applicable chain, with shortness an
// explicit goal.
type Bundle struct {
	Chain       []*x509.Certificate
	Cert        *x509.Certificate
	Root        *x509.Certificate
	Key         interface{}
	Issuer      *pkix.Name
	Subject     *pkix.Name
	Expires     time.Time
	LeafExpires time.Time
	Hostnames   []string
	Status      *BundleStatus
}

// BundleStatus is designated for various status reporting.
type BundleStatus struct {
	// A flag on whether a new bundle is generated
	IsRebundled bool `json:"rebundled"`
	// A list of SKIs of expiring certificates
	ExpiringSKIs []string `json:"expiring_SKIs"`
	// A list of untrusted root store names
	Untrusted []string `json:"untrusted_root_stores"`
	// A list of human readable warning messages based on the bundle status.
	Messages []string `json:"messages"`
	// A status code consists of binary flags
	Code int `json:"code"`
}

type chain []*x509.Certificate

func (c chain) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer

	for _, cert := range c {
		buf.Write(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: cert.Raw}))
	}
	ret := bytes.TrimSpace(buf.Bytes())
	return json.Marshal(string(ret))
}

// PemBlockToString turns a pem.Block into the string encoded form.
func PemBlockToString(block *pem.Block) string {
	if block.Bytes == nil || block.Type == "" {
		return ""
	}
	return string(bytes.TrimSpace(pem.EncodeToMemory(block)))
}

var typeToName = map[int]string{
	3:  "CommonName",
	5:  "SerialNumber",
	6:  "Country",
	7:  "Locality",
	8:  "Province",
	9:  "StreetAddress",
	10: "Organization",
	11: "OrganizationalUnit",
	17: "PostalCode",
}

type names []pkix.AttributeTypeAndValue

func (n names) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer

	for _, name := range n {
		buf.WriteString(fmt.Sprintf("/%s=%s", typeToName[name.Type[3]], name.Value))
	}
	return json.Marshal(buf.String())
}

// MarshalJSON serialises the bundle to JSON. The resulting JSON
// structure contains the bundle (as a sequence of PEM-encoded
// certificates), the certificate, the private key, the size of they
// key, the issuer(s), the subject name(s), the expiration, the
// hostname(s), the OCSP server, and the signature on the certificate.
func (b *Bundle) MarshalJSON() ([]byte, error) {
	if b == nil || b.Cert == nil {
		return nil, errors.New("no certificate in bundle")
	}
	var keyBytes, rootBytes []byte
	var keyLength int
	var keyType, keyString string
	keyLength = helpers.KeyLength(b.Cert.PublicKey)
	switch b.Cert.PublicKeyAlgorithm {
	case x509.ECDSA:
		keyType = fmt.Sprintf("%d-bit ECDSA", keyLength)
	case x509.RSA:
		keyType = fmt.Sprintf("%d-bit RSA", keyLength)
	case x509.DSA:
		keyType = "DSA"
	default:
		keyType = "Unknown"
	}

	switch key := b.Key.(type) {
	case *rsa.PrivateKey:
		keyBytes = x509.MarshalPKCS1PrivateKey(key)
		keyString = PemBlockToString(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: keyBytes})
	case *ecdsa.PrivateKey:
		keyBytes, _ = x509.MarshalECPrivateKey(key)
		keyString = PemBlockToString(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyBytes})
	case fmt.Stringer:
		keyString = key.String()
	}

	if len(b.Hostnames) == 0 {
		b.buildHostnames()
	}
	var ocspSupport = false
	if b.Cert.OCSPServer != nil {
		ocspSupport = true
	}
	var crlSupport = false
	if b.Cert.CRLDistributionPoints != nil {
		crlSupport = true
	}
	if b.Root != nil {
		rootBytes = b.Root.Raw
	}

	return json.Marshal(map[string]interface{}{
		"bundle":       chain(b.Chain),
		"root":         PemBlockToString(&pem.Block{Type: "CERTIFICATE", Bytes: rootBytes}),
		"crt":          PemBlockToString(&pem.Block{Type: "CERTIFICATE", Bytes: b.Cert.Raw}),
		"key":          keyString,
		"key_type":     keyType,
		"key_size":     keyLength,
		"issuer":       names(b.Issuer.Names),
		"subject":      names(b.Subject.Names),
		"expires":      b.Expires,
		"leaf_expires": b.LeafExpires,
		"hostnames":    b.Hostnames,
		"ocsp_support": ocspSupport,
		"crl_support":  crlSupport,
		"ocsp":         b.Cert.OCSPServer,
		"signature":    helpers.SignatureString(b.Cert.SignatureAlgorithm),
		"status":       b.Status,
	})
}

// buildHostnames sets bundle.Hostnames by the x509 cert's subject CN and DNS names
// Since the subject CN may overlap with one of the DNS names, it needs to handle
// the duplication by a set.
func (b *Bundle) buildHostnames() {
	if b.Cert == nil {
		return
	}
	// hset keeps a set of unique hostnames.
	hset := make(map[string]bool)
	// insert CN into hset
	if b.Cert.Subject.CommonName != "" {
		hset[b.Cert.Subject.CommonName] = true
	}
	// insert all DNS names into hset
	for _, h := range b.Cert.DNSNames {
		hset[h] = true
	}

	// convert hset to an array of hostnames
	b.Hostnames = make([]string, len(hset))
	i := 0
	for h := range hset {
		b.Hostnames[i] = h
		i++
	}
}
