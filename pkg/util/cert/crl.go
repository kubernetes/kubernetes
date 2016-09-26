/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package cert

import (
	"bytes"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"io/ioutil"
	"sort"
	"time"
)

// RevocationPolicy is a strategy for querying CRLs and evaluating if a certificate
// has been revoked.
type RevocationPolicy interface {
	VerifyCertificate(cert *x509.Certificate) (valid bool, err error)
}

// revocationPolicies unions a set of revocation policies.
type revocationPolicies []RevocationPolicy

func (r revocationPolicies) VerifyCertificate(cert *x509.Certificate) (valid bool, err error) {
	for _, rp := range r {
		valid, err = rp.VerifyCertificate(cert)
		if !valid || err != nil {
			// errors should indicate a
			return
		}
	}
	return true, nil
}

// LoadCRLFile parses a set of PEM encoded revocation lists from a file and uses them
// to statically mark certificates as valid or revoked. The signatures of the CRLs and
// the CRL distribution points of passed certificates are not evaluated.
//
// Each CRL only applies to certificates that were issued by cert that issued the CRL.
func LoadCRLFile(file string) (RevocationPolicy, error) {
	data, err := ioutil.ReadFile(file)
	if err != nil {
		return nil, fmt.Errorf("read file: %v", err)
	}
	var r []RevocationPolicy

	for {
		var block *pem.Block
		block, data = pem.Decode(data)
		if block == nil {
			break
		}

		if block.Type != "X509 CRL" {
			return nil, fmt.Errorf("expected PEM block of type 'X509 CRL', got '%s'", block.Type)
		}

		crl, err := x509.ParseDERCRL(block.Bytes)
		if err != nil {
			return nil, fmt.Errorf("parse crl: %v", err)
		}

		l := localCRL{crl: crl}
		l.issuer.FillFromRDNSequence(&crl.TBSCertList.Issuer)

		l.revokedSerialNums = make(map[string]struct{})
		for _, cert := range crl.TBSCertList.RevokedCertificates {
			l.revokedSerialNums[cert.SerialNumber.String()] = struct{}{}
		}

		r = append(r, l)
	}
	return revocationPolicies(r), nil
}

// localCRL is a static, in memory revocation list.
type localCRL struct {
	// The CA which issued this cert.
	issuer pkix.Name

	revokedSerialNums map[string]struct{}

	crl *pkix.CertificateList
}

func (c localCRL) VerifyCertificate(cert *x509.Certificate) (valid bool, err error) {
	if !pkixNameEqual(c.issuer, cert.Issuer) {
		return true, nil
	}

	// Only return errors for certificates this issuer should be able to judge.
	if c.crl.HasExpired(time.Now()) {
		return false, fmt.Errorf("cert: crl for ca %q is expired", formatDN(c.issuer))
	}

	if _, found := c.revokedSerialNums[cert.SerialNumber.String()]; found {
		return false, nil
	}
	return true, nil
}

func setEq(s1, s2 []string) bool {
	if len(s1) != len(s2) {
		return false
	}

	sort.Strings(s1)
	sort.Strings(s2)
	for i := range s1 {
		if s1[i] != s2[i] {
			return false
		}
	}
	return true
}

// NOTE(ericchiang): Issuer DNs are not guarenteed to be in any particular order so we
// can't compare the raw bytes. If we care about extra fields beyond the ones listed
// here we should compare signatures instead.
func pkixNameEqual(n1, n2 pkix.Name) bool {
	return n1.SerialNumber == n2.SerialNumber &&
		n1.CommonName == n2.CommonName &&
		setEq(n1.Organization, n2.Organization) &&
		setEq(n1.OrganizationalUnit, n2.OrganizationalUnit) &&
		setEq(n1.Country, n2.Country) &&
		setEq(n1.Locality, n2.Locality) &&
		setEq(n1.Province, n2.Province) &&
		setEq(n1.StreetAddress, n2.StreetAddress)
}

// formatDN formats a distinguished name in a style similar to OpenSSL.
func formatDN(name pkix.Name) string {
	b := new(bytes.Buffer)
	first := true
	w := func(typ string, vals ...string) {
		for _, val := range vals {
			if len(val) == 0 {
				continue
			}
			if !first {
				b.WriteString(", ")
			}
			first = false
			b.WriteString(typ)
			b.WriteString("=")
			b.WriteString(val)
		}
	}
	w("C", name.Country...)
	w("ST", name.Province...)
	w("L", name.Locality...)
	w("O", name.Organization...)
	w("OU", name.OrganizationalUnit...)
	w("CN", name.CommonName)
	w("SN", name.SerialNumber)
	return b.String()
}
