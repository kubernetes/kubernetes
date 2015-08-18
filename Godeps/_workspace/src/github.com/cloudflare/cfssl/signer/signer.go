// Package signer implements certificate signature functionality for CFSSL.
package signer

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha1"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/asn1"
	"errors"
	"fmt"
	"math"
	"math/big"
	"strings"
	"time"

	"github.com/cloudflare/cfssl/config"
	"github.com/cloudflare/cfssl/csr"
	cferr "github.com/cloudflare/cfssl/errors"
	"github.com/cloudflare/cfssl/info"
)

// MaxPathLen is the default path length for a new CA certificate.
var MaxPathLen = 2

// Subject contains the information that should be used to override the
// subject information when signing a certificate.
type Subject struct {
	CN    string
	Names []csr.Name `json:"names"`
}

// SignRequest stores a signature request, which contains the hostname,
// the CSR, optional subject information, and the signature profile.
type SignRequest struct {
	Hosts     []string `json:"hosts"`
	Request   string   `json:"certificate_request"`
	Subject   *Subject `json:"subject,omitempty"`
	Profile   string   `json:"profile"`
	Label     string   `json:"label"`
	SerialSeq string   `json:"serial_sequence,omitempty"`
}

// appendIf appends to a if s is not an empty string.
func appendIf(s string, a *[]string) {
	if s != "" {
		*a = append(*a, s)
	}
}

// Name returns the PKIX name for the subject.
func (s *Subject) Name() pkix.Name {
	var name pkix.Name
	name.CommonName = s.CN

	for _, n := range s.Names {
		appendIf(n.C, &name.Country)
		appendIf(n.ST, &name.Province)
		appendIf(n.L, &name.Locality)
		appendIf(n.O, &name.Organization)
		appendIf(n.OU, &name.OrganizationalUnit)
	}
	return name
}

// SplitHosts takes a comma-spearated list of hosts and returns a slice
// with the hosts split
func SplitHosts(hostList string) []string {
	if hostList == "" {
		return nil
	}

	return strings.Split(hostList, ",")
}

// A Signer contains a CA's certificate and private key for signing
// certificates, a Signing policy to refer to and a SignatureAlgorithm.
type Signer interface {
	Info(info.Req) (*info.Resp, error)
	Policy() *config.Signing
	SetPolicy(*config.Signing)
	SigAlgo() x509.SignatureAlgorithm
	Sign(req SignRequest) (cert []byte, err error)
}

// Profile gets the specific profile from the signer
func Profile(s Signer, profile string) (*config.SigningProfile, error) {
	var p *config.SigningProfile
	policy := s.Policy()
	if policy != nil && policy.Profiles != nil && profile != "" {
		p = policy.Profiles[profile]
	}

	if p == nil && policy != nil {
		p = policy.Default
	}

	if p == nil {
		return nil, cferr.Wrap(cferr.APIClientError, cferr.ClientHTTPError, errors.New("profile must not be nil"))
	}
	return p, nil
}

// DefaultSigAlgo returns an appropriate X.509 signature algorithm given
// the CA's private key.
func DefaultSigAlgo(priv crypto.Signer) x509.SignatureAlgorithm {
	pub := priv.Public()
	switch pub := pub.(type) {
	case *rsa.PublicKey:
		keySize := pub.N.BitLen()
		switch {
		case keySize >= 4096:
			return x509.SHA512WithRSA
		case keySize >= 3072:
			return x509.SHA384WithRSA
		case keySize >= 2048:
			return x509.SHA256WithRSA
		default:
			return x509.SHA1WithRSA
		}
	case *ecdsa.PublicKey:
		switch pub.Curve {
		case elliptic.P256():
			return x509.ECDSAWithSHA256
		case elliptic.P384():
			return x509.ECDSAWithSHA384
		case elliptic.P521():
			return x509.ECDSAWithSHA512
		default:
			return x509.ECDSAWithSHA1
		}
	default:
		return x509.UnknownSignatureAlgorithm
	}
}

// ParseCertificateRequest takes an incoming certificate request and
// builds a certificate template from it.
func ParseCertificateRequest(s Signer, csrBytes []byte) (template *x509.Certificate, err error) {
	csr, err := x509.ParseCertificateRequest(csrBytes)
	if err != nil {
		err = cferr.Wrap(cferr.CSRError, cferr.ParseFailed, err)
		return
	}

	err = CheckSignature(csr, csr.SignatureAlgorithm, csr.RawTBSCertificateRequest, csr.Signature)
	if err != nil {
		err = cferr.Wrap(cferr.CSRError, cferr.KeyMismatch, err)
		return
	}

	template = &x509.Certificate{
		Subject:            csr.Subject,
		PublicKeyAlgorithm: csr.PublicKeyAlgorithm,
		PublicKey:          csr.PublicKey,
		SignatureAlgorithm: s.SigAlgo(),
		DNSNames:           csr.DNSNames,
		IPAddresses:        csr.IPAddresses,
	}

	return
}

// CheckSignature verifies a signature made by the key on a CSR, such
// as on the CSR itself.
func CheckSignature(csr *x509.CertificateRequest, algo x509.SignatureAlgorithm, signed, signature []byte) error {
	var hashType crypto.Hash

	switch algo {
	case x509.SHA1WithRSA, x509.ECDSAWithSHA1:
		hashType = crypto.SHA1
	case x509.SHA256WithRSA, x509.ECDSAWithSHA256:
		hashType = crypto.SHA256
	case x509.SHA384WithRSA, x509.ECDSAWithSHA384:
		hashType = crypto.SHA384
	case x509.SHA512WithRSA, x509.ECDSAWithSHA512:
		hashType = crypto.SHA512
	default:
		return x509.ErrUnsupportedAlgorithm
	}

	if !hashType.Available() {
		return x509.ErrUnsupportedAlgorithm
	}
	h := hashType.New()

	h.Write(signed)
	digest := h.Sum(nil)

	switch pub := csr.PublicKey.(type) {
	case *rsa.PublicKey:
		return rsa.VerifyPKCS1v15(pub, hashType, digest, signature)
	case *ecdsa.PublicKey:
		ecdsaSig := new(struct{ R, S *big.Int })
		if _, err := asn1.Unmarshal(signature, ecdsaSig); err != nil {
			return err
		}
		if ecdsaSig.R.Sign() <= 0 || ecdsaSig.S.Sign() <= 0 {
			return errors.New("x509: ECDSA signature contained zero or negative values")
		}
		if !ecdsa.Verify(pub, digest, ecdsaSig.R, ecdsaSig.S) {
			return errors.New("x509: ECDSA verification failure")
		}
		return nil
	}
	return x509.ErrUnsupportedAlgorithm
}

type subjectPublicKeyInfo struct {
	Algorithm        pkix.AlgorithmIdentifier
	SubjectPublicKey asn1.BitString
}

// ComputeSKI derives an SKI from the certificate's public key in a
// standard manner. This is done by computing the SHA-1 digest of the
// SubjectPublicKeyInfo component of the certificate.
func ComputeSKI(template *x509.Certificate) ([]byte, error) {
	pub := template.PublicKey
	encodedPub, err := x509.MarshalPKIXPublicKey(pub)
	if err != nil {
		return nil, err
	}

	var subPKI subjectPublicKeyInfo
	_, err = asn1.Unmarshal(encodedPub, &subPKI)
	if err != nil {
		return nil, err
	}

	pubHash := sha1.Sum(subPKI.SubjectPublicKey.Bytes)
	return pubHash[:], nil
}

// FillTemplate is a utility function that tries to load as much of
// the certificate template as possible from the profiles and current
// template. It fills in the key uses, expiration, revocation URLs,
// serial number, and SKI.
func FillTemplate(template *x509.Certificate, defaultProfile, profile *config.SigningProfile, serialSeq string) error {
	ski, err := ComputeSKI(template)

	var (
		eku             []x509.ExtKeyUsage
		ku              x509.KeyUsage
		backdate        time.Duration
		expiry          time.Duration
		notBefore       time.Time
		notAfter        time.Time
		crlURL, ocspURL string
	)

	// The third value returned from Usages is a list of unknown key usages.
	// This should be used when validating the profile at load, and isn't used
	// here.
	ku, eku, _ = profile.Usages()
	if profile.IssuerURL == nil {
		profile.IssuerURL = defaultProfile.IssuerURL
	}

	if ku == 0 && len(eku) == 0 {
		return cferr.New(cferr.PolicyError, cferr.NoKeyUsages)
	}

	if expiry = profile.Expiry; expiry == 0 {
		expiry = defaultProfile.Expiry
	}

	if crlURL = profile.CRL; crlURL == "" {
		crlURL = defaultProfile.CRL
	}
	if ocspURL = profile.OCSP; ocspURL == "" {
		ocspURL = defaultProfile.OCSP
	}
	if backdate = profile.Backdate; backdate == 0 {
		backdate = -5 * time.Minute
	} else {
		backdate = -1 * profile.Backdate
	}

	if !profile.NotBefore.IsZero() {
		notBefore = profile.NotBefore.UTC()
	} else {
		notBefore = time.Now().Round(time.Minute).Add(backdate).UTC()
	}

	if !profile.NotAfter.IsZero() {
		notAfter = profile.NotAfter.UTC()
	} else {
		notAfter = notBefore.Add(expiry).UTC()
	}

	serialNumber, err := rand.Int(rand.Reader, new(big.Int).SetInt64(math.MaxInt64))

	if err != nil {
		return cferr.Wrap(cferr.CertificateError, cferr.Unknown, err)
	}

	if serialSeq != "" {
		randomPart := fmt.Sprintf("%016X", serialNumber) // 016 ensures we're left-0-padded
		_, ok := serialNumber.SetString(serialSeq+randomPart, 16)
		if !ok {
			return cferr.Wrap(cferr.CertificateError, cferr.SerialSeqParseError, err)
		}
	}

	template.SerialNumber = serialNumber
	template.NotBefore = notBefore
	template.NotAfter = notAfter
	template.KeyUsage = ku
	template.ExtKeyUsage = eku
	template.BasicConstraintsValid = true
	template.IsCA = profile.CA
	template.SubjectKeyId = ski

	if ocspURL != "" {
		template.OCSPServer = []string{ocspURL}
	}
	if crlURL != "" {
		template.CRLDistributionPoints = []string{crlURL}
	}

	if len(profile.IssuerURL) != 0 {
		template.IssuingCertificateURL = profile.IssuerURL
	}
	if len(profile.Policies) != 0 {
		err = addPolicies(template, profile.Policies)
		if err != nil {
			return cferr.Wrap(cferr.PolicyError, cferr.InvalidPolicy, err)
		}
	}
	if profile.OCSPNoCheck {
		ocspNoCheckExtension := pkix.Extension{
			Id:       asn1.ObjectIdentifier{1, 3, 6, 1, 5, 5, 7, 48, 1, 5},
			Critical: false,
			Value:    []byte{0x05, 0x00},
		}
		template.ExtraExtensions = append(template.ExtraExtensions, ocspNoCheckExtension)
	}

	return nil
}

type policyInformation struct {
	PolicyIdentifier    asn1.ObjectIdentifier
	Qualifiers          []interface{}
	CPSPolicyQualifiers []cpsPolicyQualifier `asn1:"omitempty"`
	// User Notice policy qualifiers have a slightly different ASN.1 structure
	// from that used for CPS policy qualifiers.
	UserNoticePolicyQualifiers []userNoticePolicyQualifier `asn1:"omitempty"`
}

type cpsPolicyQualifier struct {
	PolicyQualifierID asn1.ObjectIdentifier
	Qualifier         string `asn1:"tag:optional,ia5"`
}

type userNotice struct {
	ExplicitText string `asn1:"tag:optional,utf8"`
}
type userNoticePolicyQualifier struct {
	PolicyQualifierID asn1.ObjectIdentifier
	Qualifier         userNotice
}

var (
	// Per https://tools.ietf.org/html/rfc3280.html#page-106, this represents:
	// iso(1) identified-organization(3) dod(6) internet(1) security(5)
	//   mechanisms(5) pkix(7) id-qt(2) id-qt-cps(1)
	iDQTCertificationPracticeStatement = asn1.ObjectIdentifier{1, 3, 6, 1, 5, 5, 7, 2, 1}
	// iso(1) identified-organization(3) dod(6) internet(1) security(5)
	//   mechanisms(5) pkix(7) id-qt(2) id-qt-unotice(2)
	iDQTUserNotice = asn1.ObjectIdentifier{1, 3, 6, 1, 5, 5, 7, 2, 2}
)

// addPolicies adds Certificate Policies and optional Policy Qualifiers to a
// certificate, based on the input config. Go's x509 library allows setting
// Certificate Policies easily, but does not support nested Policy Qualifiers
// under those policies. So we need to construct the ASN.1 structure ourselves.
func addPolicies(template *x509.Certificate, policies []config.CertificatePolicy) error {
	asn1PolicyList := []policyInformation{}

	for _, policy := range policies {
		pi := policyInformation{
			// The PolicyIdentifier is an OID assigned to a given issuer.
			PolicyIdentifier: asn1.ObjectIdentifier(policy.ID),
		}
		for _, qualifier := range policy.Qualifiers {
			switch qualifier.Type {
			case "id-qt-unotice":
				pi.Qualifiers = append(pi.Qualifiers,
					userNoticePolicyQualifier{
						PolicyQualifierID: iDQTUserNotice,
						Qualifier: userNotice{
							ExplicitText: qualifier.Value,
						},
					})
			case "id-qt-cps":
				pi.Qualifiers = append(pi.Qualifiers,
					cpsPolicyQualifier{
						PolicyQualifierID: iDQTCertificationPracticeStatement,
						Qualifier:         qualifier.Value,
					})
			default:
				return errors.New("Invalid qualifier type in Policies " + qualifier.Type)
			}
		}
		asn1PolicyList = append(asn1PolicyList, pi)
	}

	asn1Bytes, err := asn1.Marshal(asn1PolicyList)
	if err != nil {
		return err
	}

	template.ExtraExtensions = append(template.ExtraExtensions, pkix.Extension{
		Id:       asn1.ObjectIdentifier{2, 5, 29, 32},
		Critical: false,
		Value:    asn1Bytes,
	})
	return nil
}
