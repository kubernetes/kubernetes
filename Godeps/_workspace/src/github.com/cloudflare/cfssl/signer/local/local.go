// Package local implements certificate signature functionality for CFSSL.
package local

import (
	"bytes"
	"crypto"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"io/ioutil"
	"net"

	"github.com/cloudflare/cfssl/config"
	cferr "github.com/cloudflare/cfssl/errors"
	"github.com/cloudflare/cfssl/helpers"
	"github.com/cloudflare/cfssl/info"
	"github.com/cloudflare/cfssl/log"
	"github.com/cloudflare/cfssl/signer"
)

// Signer contains a signer that uses the standard library to
// support both ECDSA and RSA CA keys.
type Signer struct {
	ca      *x509.Certificate
	priv    crypto.Signer
	policy  *config.Signing
	sigAlgo x509.SignatureAlgorithm
}

// NewSigner creates a new Signer directly from a
// private key and certificate, with optional policy.
func NewSigner(priv crypto.Signer, cert *x509.Certificate, sigAlgo x509.SignatureAlgorithm, policy *config.Signing) (*Signer, error) {
	if policy == nil {
		policy = &config.Signing{
			Profiles: map[string]*config.SigningProfile{},
			Default:  config.DefaultConfig()}
	}

	if !policy.Valid() {
		return nil, cferr.New(cferr.PolicyError, cferr.InvalidPolicy)
	}

	return &Signer{
		ca:      cert,
		priv:    priv,
		sigAlgo: sigAlgo,
		policy:  policy,
	}, nil
}

// NewSignerFromFile generates a new local signer from a caFile
// and a caKey file, both PEM encoded.
func NewSignerFromFile(caFile, caKeyFile string, policy *config.Signing) (*Signer, error) {
	log.Debug("Loading CA: ", caFile)
	ca, err := ioutil.ReadFile(caFile)
	if err != nil {
		return nil, err
	}
	log.Debug("Loading CA key: ", caKeyFile)
	cakey, err := ioutil.ReadFile(caKeyFile)
	if err != nil {
		return nil, cferr.Wrap(cferr.CertificateError, cferr.ReadFailed, err)
	}

	parsedCa, err := helpers.ParseCertificatePEM(ca)
	if err != nil {
		return nil, err
	}

	priv, err := helpers.ParsePrivateKeyPEM(cakey)
	if err != nil {
		log.Debug("Malformed private key %v", err)
		return nil, err
	}

	return NewSigner(priv, parsedCa, signer.DefaultSigAlgo(priv), policy)
}

func (s *Signer) sign(template *x509.Certificate, profile *config.SigningProfile, serialSeq string) (cert []byte, err error) {
	err = signer.FillTemplate(template, s.policy.Default, profile, serialSeq)
	if err != nil {
		return
	}

	serialNumber := template.SerialNumber
	var initRoot bool
	if s.ca == nil {
		if !template.IsCA {
			err = cferr.New(cferr.PolicyError, cferr.InvalidRequest)
			return
		}
		template.DNSNames = nil
		s.ca = template
		initRoot = true
		template.MaxPathLen = signer.MaxPathLen
	} else if template.IsCA {
		template.MaxPathLen = 1
		template.DNSNames = nil
	}

	derBytes, err := x509.CreateCertificate(rand.Reader, template, s.ca, template.PublicKey, s.priv)
	if err != nil {
		return nil, cferr.Wrap(cferr.CertificateError, cferr.Unknown, err)
	}
	if initRoot {
		s.ca, err = x509.ParseCertificate(derBytes)
		if err != nil {
			return nil, cferr.Wrap(cferr.CertificateError, cferr.ParseFailed, err)
		}
	}

	cert = pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: derBytes})
	log.Infof("signed certificate with serial number %s", serialNumber)
	return
}

// replaceSliceIfEmpty replaces the contents of replaced with newContents if
// the slice referenced by replaced is empty
func replaceSliceIfEmpty(replaced, newContents *[]string) {
	if len(*replaced) == 0 {
		*replaced = *newContents
	}
}

// PopulateSubjectFromCSR has functionality similar to Name, except
// it fills the fields of the resulting pkix.Name with req's if the
// subject's corresponding fields are empty
func PopulateSubjectFromCSR(s *signer.Subject, req pkix.Name) pkix.Name {
	// if no subject, use req
	if s == nil {
		return req
	}

	name := s.Name()

	if name.CommonName == "" {
		name.CommonName = req.CommonName
	}

	replaceSliceIfEmpty(&name.Country, &req.Country)
	replaceSliceIfEmpty(&name.Province, &req.Province)
	replaceSliceIfEmpty(&name.Locality, &req.Locality)
	replaceSliceIfEmpty(&name.Organization, &req.Organization)
	replaceSliceIfEmpty(&name.OrganizationalUnit, &req.OrganizationalUnit)

	return name
}

// OverrideHosts fills template's IPAddresses and DNSNames with the
// content of hosts, if it is not nil.
func OverrideHosts(template *x509.Certificate, hosts []string) {
	if hosts != nil {
		template.IPAddresses = []net.IP{}
		template.DNSNames = []string{}
	}

	for i := range hosts {
		if ip := net.ParseIP(hosts[i]); ip != nil {
			template.IPAddresses = append(template.IPAddresses, ip)
		} else {
			template.DNSNames = append(template.DNSNames, hosts[i])
		}
	}

}

// Sign signs a new certificate based on the PEM-encoded client
// certificate or certificate request with the signing profile,
// specified by profileName.
func (s *Signer) Sign(req signer.SignRequest) (cert []byte, err error) {
	profile, err := signer.Profile(s, req.Profile)
	if err != nil {
		return
	}

	serialSeq := ""
	if profile.UseSerialSeq {
		serialSeq = req.SerialSeq
	}

	block, _ := pem.Decode([]byte(req.Request))
	if block == nil {
		return nil, cferr.New(cferr.CSRError, cferr.DecodeFailed)
	}

	if block.Type != "CERTIFICATE REQUEST" {
		return nil, cferr.Wrap(cferr.CSRError,
			cferr.BadRequest, errors.New("not a certificate or csr"))
	}

	csrTemplate, err := signer.ParseCertificateRequest(s, block.Bytes)
	if err != nil {
		return nil, err
	}

	// Copy out only the fields from the CSR authorized by policy.
	safeTemplate := x509.Certificate{}
	// If the profile contains no explicit whitelist, assume that all fields
	// should be copied from the CSR.
	if profile.CSRWhitelist == nil {
		safeTemplate = *csrTemplate
	} else {
		if profile.CSRWhitelist.Subject {
			safeTemplate.Subject = csrTemplate.Subject
		}
		if profile.CSRWhitelist.PublicKeyAlgorithm {
			safeTemplate.PublicKeyAlgorithm = csrTemplate.PublicKeyAlgorithm
		}
		if profile.CSRWhitelist.PublicKey {
			safeTemplate.PublicKey = csrTemplate.PublicKey
		}
		if profile.CSRWhitelist.SignatureAlgorithm {
			safeTemplate.SignatureAlgorithm = csrTemplate.SignatureAlgorithm
		}
		if profile.CSRWhitelist.DNSNames {
			safeTemplate.DNSNames = csrTemplate.DNSNames
		}
		if profile.CSRWhitelist.IPAddresses {
			safeTemplate.IPAddresses = csrTemplate.IPAddresses
		}
	}

	OverrideHosts(&safeTemplate, req.Hosts)
	safeTemplate.Subject = PopulateSubjectFromCSR(req.Subject, safeTemplate.Subject)

	// If there is a whitelist, ensure that both the Common Name and SAN DNSNames match
	if profile.NameWhitelist != nil {
		if safeTemplate.Subject.CommonName != "" {
			if profile.NameWhitelist.Find([]byte(safeTemplate.Subject.CommonName)) == nil {
				return nil, cferr.New(cferr.PolicyError, cferr.InvalidPolicy)
			}
		}
		for _, name := range safeTemplate.DNSNames {
			if profile.NameWhitelist.Find([]byte(name)) == nil {
				return nil, cferr.New(cferr.PolicyError, cferr.InvalidPolicy)
			}
		}
	}

	return s.sign(&safeTemplate, profile, serialSeq)
}

// Info return a populated info.Resp struct or an error.
func (s *Signer) Info(req info.Req) (resp *info.Resp, err error) {
	cert, err := s.Certificate(req.Label, req.Profile)
	if err != nil {
		return
	}

	profile, err := signer.Profile(s, req.Profile)
	if err != nil {
		return
	}

	resp = new(info.Resp)
	if cert.Raw != nil {
		resp.Certificate = string(bytes.TrimSpace(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: cert.Raw})))
	}
	resp.Usage = profile.Usage
	resp.ExpiryString = profile.ExpiryString

	return
}

// SigAlgo returns the RSA signer's signature algorithm.
func (s *Signer) SigAlgo() x509.SignatureAlgorithm {
	return s.sigAlgo
}

// Certificate returns the signer's certificate.
func (s *Signer) Certificate(label, profile string) (*x509.Certificate, error) {
	cert := *s.ca
	return &cert, nil
}

// SetPolicy sets the signer's signature policy.
func (s *Signer) SetPolicy(policy *config.Signing) {
	s.policy = policy
}

// Policy returns the signer's policy.
func (s *Signer) Policy() *config.Signing {
	return s.policy
}
