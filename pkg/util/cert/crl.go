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
	"crypto/x509"
	"crypto/x509/pkix"
	"fmt"
	"io/ioutil"
	"sort"
)

// RevocationPolicy is a strategy for querying CRLs and evaluating if a certificate
// has been revoked.
type RevocationPolicy interface {
	VerifyCertificate(cert *x509.Certificate) (valid bool, err error)
}

// LoadCRLFile parses a revocation list from a file and uses it to statically mark
// certificates as valid or revoked. The signatures of the CRL and the CRL distribution
// points of passed certificates are not evaluated.
func LoadCRLFile(file string) (RevocationPolicy, error) {
	data, err := ioutil.ReadFile(file)
	if err != nil {
		return nil, fmt.Errorf("read file: %v", err)
	}
	certList, err := x509.ParseCRL(data)
	if err != nil {
		return nil, fmt.Errorf("parse crl: %v", err)
	}

	l := localCRL{certList: certList}
	l.issuer.FillFromRDNSequence(&certList.TBSCertList.Issuer)
	return l, nil
}

// localCRL is a static, in memory revocation list.
type localCRL struct {
	issuer   pkix.Name
	certList *pkix.CertificateList
}

func setEq(s1, s2 []string) bool {
	sort.Strings(s1)
	sort.Strings(s2)
	if len(s1) != len(s2) {
		return false
	}
	for i := range s1 {
		if s1[i] != s2[i] {
			return false
		}
	}
	return true
}

// NOTE(ericchiang): Issuer DNs are not guarenteed to be in any particular order so we
// can't just compare the raw bytes. If we care about extra fields beyond the ones listed
// here we should just compare signatures instead.
func pkixNameEqual(n1, n2 pkix.Name) bool {
	return setEq(n1.Country, n2.Country) &&
		setEq(n1.Organization, n2.Organization) &&
		setEq(n1.OrganizationalUnit, n2.OrganizationalUnit) &&
		setEq(n1.Locality, n2.Locality) &&
		setEq(n1.Province, n2.Province) &&
		setEq(n1.StreetAddress, n2.StreetAddress) &&
		n1.SerialNumber == n2.SerialNumber &&
		n1.CommonName == n2.CommonName
}

func (c localCRL) VerifyCertificate(cert *x509.Certificate) (valid bool, err error) {
	if !pkixNameEqual(c.issuer, cert.Issuer) {
		return true, nil
	}
	for _, r := range c.certList.TBSCertList.RevokedCertificates {
		if cert.SerialNumber.Cmp(r.SerialNumber) == 0 {
			return false, nil
		}
	}
	return true, nil
}
