// Package revoke provides functionality for checking the validity of
// a cert. Specifically, the temporal validity of the certificate is
// checked first, then any CRL and OCSP url in the cert is checked.
package revoke

import (
	"bytes"
	"crypto"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/pem"
	"errors"
	"io"
	"io/ioutil"
	"net/http"
	neturl "net/url"
	"sync"
	"time"

	"golang.org/x/crypto/ocsp"

	"github.com/cloudflare/cfssl/helpers"
	"github.com/cloudflare/cfssl/log"
)

// HardFail determines whether the failure to check the revocation
// status of a certificate (i.e. due to network failure) causes
// verification to fail (a hard failure).
var HardFail = false

// CRLSet associates a PKIX certificate list with the URL the CRL is
// fetched from.
var CRLSet = map[string]*pkix.CertificateList{}
var crlLock = new(sync.Mutex)

// We can't handle LDAP certificates, so this checks to see if the
// URL string points to an LDAP resource so that we can ignore it.
func ldapURL(url string) bool {
	u, err := neturl.Parse(url)
	if err != nil {
		log.Warningf("error parsing url %s: %v", url, err)
		return false
	}
	if u.Scheme == "ldap" {
		return true
	}
	return false
}

// revCheck should check the certificate for any revocations. It
// returns a pair of booleans: the first indicates whether the certificate
// is revoked, the second indicates whether the revocations were
// successfully checked.. This leads to the following combinations:
//
//  false, false: an error was encountered while checking revocations.
//
//  false, true:  the certificate was checked successfully and
//                  it is not revoked.
//
//  true, true:   the certificate was checked successfully and
//                  it is revoked.
//
//  true, false:  failure to check revocation status causes
//                  verification to fail
func revCheck(cert *x509.Certificate) (revoked, ok bool) {
	for _, url := range cert.CRLDistributionPoints {
		if ldapURL(url) {
			log.Infof("skipping LDAP CRL: %s", url)
			continue
		}

		if revoked, ok := certIsRevokedCRL(cert, url); !ok {
			log.Warning("error checking revocation via CRL")
			if HardFail {
				return true, false
			}
			return false, false
		} else if revoked {
			log.Info("certificate is revoked via CRL")
			return true, true
		}
	}

	if revoked, ok := certIsRevokedOCSP(cert, HardFail); !ok {
		log.Warning("error checking revocation via OCSP")
		if HardFail {
			return true, false
		}
		return false, false
	} else if revoked {
		log.Info("certificate is revoked via OCSP")
		return true, true
	}

	return false, true
}

// fetchCRL fetches and parses a CRL.
func fetchCRL(url string) (*pkix.CertificateList, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	} else if resp.StatusCode >= 300 {
		return nil, errors.New("failed to retrieve CRL")
	}

	body, err := crlRead(resp.Body)
	if err != nil {
		return nil, err
	}
	resp.Body.Close()

	return x509.ParseCRL(body)
}

func getIssuer(cert *x509.Certificate) *x509.Certificate {
	var issuer *x509.Certificate
	var err error
	for _, issuingCert := range cert.IssuingCertificateURL {
		issuer, err = fetchRemote(issuingCert)
		if err != nil {
			continue
		}
		break
	}

	return issuer

}

// check a cert against a specific CRL. Returns the same bool pair
// as revCheck.
func certIsRevokedCRL(cert *x509.Certificate, url string) (revoked, ok bool) {
	crl, ok := CRLSet[url]
	if ok && crl == nil {
		ok = false
		crlLock.Lock()
		delete(CRLSet, url)
		crlLock.Unlock()
	}

	var shouldFetchCRL = true
	if ok {
		if !crl.HasExpired(time.Now()) {
			shouldFetchCRL = false
		}
	}

	issuer := getIssuer(cert)

	if shouldFetchCRL {
		var err error
		crl, err = fetchCRL(url)
		if err != nil {
			log.Warningf("failed to fetch CRL: %v", err)
			return false, false
		}

		// check CRL signature
		if issuer != nil {
			err = issuer.CheckCRLSignature(crl)
			if err != nil {
				log.Warningf("failed to verify CRL: %v", err)
				return false, false
			}
		}

		crlLock.Lock()
		CRLSet[url] = crl
		crlLock.Unlock()
	}

	for _, revoked := range crl.TBSCertList.RevokedCertificates {
		if cert.SerialNumber.Cmp(revoked.SerialNumber) == 0 {
			log.Info("Serial number match: intermediate is revoked.")
			return true, true
		}
	}

	return false, true
}

// VerifyCertificate ensures that the certificate passed in hasn't
// expired and checks the CRL for the server.
func VerifyCertificate(cert *x509.Certificate) (revoked, ok bool) {
	if !time.Now().Before(cert.NotAfter) {
		log.Infof("Certificate expired %s\n", cert.NotAfter)
		return true, true
	} else if !time.Now().After(cert.NotBefore) {
		log.Infof("Certificate isn't valid until %s\n", cert.NotBefore)
		return true, true
	}

	return revCheck(cert)
}

func fetchRemote(url string) (*x509.Certificate, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}

	in, err := remoteRead(resp.Body)
	if err != nil {
		return nil, err
	}
	resp.Body.Close()

	p, _ := pem.Decode(in)
	if p != nil {
		return helpers.ParseCertificatePEM(in)
	}

	return x509.ParseCertificate(in)
}

var ocspOpts = ocsp.RequestOptions{
	Hash: crypto.SHA1,
}

func certIsRevokedOCSP(leaf *x509.Certificate, strict bool) (revoked, ok bool) {
	var err error

	ocspURLs := leaf.OCSPServer
	if len(ocspURLs) == 0 {
		// OCSP not enabled for this certificate.
		return false, true
	}

	issuer := getIssuer(leaf)

	if issuer == nil {
		return false, false
	}

	ocspRequest, err := ocsp.CreateRequest(leaf, issuer, &ocspOpts)
	if err != nil {
		return
	}

	for _, server := range ocspURLs {
		resp, err := sendOCSPRequest(server, ocspRequest, leaf, issuer)
		if err != nil {
			if strict {
				return
			}
			continue
		}

		// There wasn't an error fetching the OCSP status.
		ok = true

		if resp.Status != ocsp.Good {
			// The certificate was revoked.
			revoked = true
		}

		return
	}
	return
}

// sendOCSPRequest attempts to request an OCSP response from the
// server. The error only indicates a failure to *fetch* the
// certificate, and *does not* mean the certificate is valid.
func sendOCSPRequest(server string, req []byte, leaf, issuer *x509.Certificate) (*ocsp.Response, error) {
	var resp *http.Response
	var err error
	if len(req) > 256 {
		buf := bytes.NewBuffer(req)
		resp, err = http.Post(server, "application/ocsp-request", buf)
	} else {
		reqURL := server + "/" + base64.StdEncoding.EncodeToString(req)
		resp, err = http.Get(reqURL)
	}

	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, errors.New("failed to retrieve OSCP")
	}

	body, err := ocspRead(resp.Body)
	if err != nil {
		return nil, err
	}
	resp.Body.Close()

	switch {
	case bytes.Equal(body, ocsp.UnauthorizedErrorResponse):
		return nil, errors.New("OSCP unauthorized")
	case bytes.Equal(body, ocsp.MalformedRequestErrorResponse):
		return nil, errors.New("OSCP malformed")
	case bytes.Equal(body, ocsp.InternalErrorErrorResponse):
		return nil, errors.New("OSCP internal error")
	case bytes.Equal(body, ocsp.TryLaterErrorResponse):
		return nil, errors.New("OSCP try later")
	case bytes.Equal(body, ocsp.SigRequredErrorResponse):
		return nil, errors.New("OSCP signature required")
	}

	return ocsp.ParseResponseForCert(body, leaf, issuer)
}

var crlRead = ioutil.ReadAll

// SetCRLFetcher sets the function to use to read from the http response body
func SetCRLFetcher(fn func(io.Reader) ([]byte, error)) {
	crlRead = fn
}

var remoteRead = ioutil.ReadAll

// SetRemoteFetcher sets the function to use to read from the http response body
func SetRemoteFetcher(fn func(io.Reader) ([]byte, error)) {
	remoteRead = fn
}

var ocspRead = ioutil.ReadAll

// SetOCSPFetcher sets the function to use to read from the http response body
func SetOCSPFetcher(fn func(io.Reader) ([]byte, error)) {
	ocspRead = fn
}
