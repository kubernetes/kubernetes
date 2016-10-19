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

package genericapiserver

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
)

// getNamedCertificateMap returns a map of strings to *tls.Certificate, suitable for use in
// tls.Config#NamedCertificates. Returns an error if any of the certs cannot be loaded.
// Returns nil if len(namedKeyCerts) == 0
func getNamedCertificateMap(namedCertKeys []NamedCertKey) (map[string]*tls.Certificate, error) {
	if len(namedCertKeys) == 0 {
		return nil, nil
	}

	// load keys
	tlsCerts := make([]tls.Certificate, len(namedCertKeys))
	for i := range namedCertKeys {
		var err error
		nkc := &namedCertKeys[i]
		tlsCerts[i], err = tls.LoadX509KeyPair(nkc.CertFile, nkc.KeyFile)
		if err != nil {
			return nil, err
		}
	}

	// register certs with implicit names first, reverse order such that earlier trump over the later
	tlsCertsByName := map[string]*tls.Certificate{}
	for i := len(namedCertKeys) - 1; i >= 0; i-- {
		nkc := &namedCertKeys[i]
		if len(nkc.Names) > 0 {
			continue
		}
		cert := &tlsCerts[i]

		// read names from certificate common names and DNS names
		if len(cert.Certificate) == 0 {
			return nil, fmt.Errorf("no certificate found in %q", nkc.CertFile)
		}
		x509Cert, err := x509.ParseCertificate(cert.Certificate[0])
		if err != nil {
			return nil, fmt.Errorf("parse error for certificate in %q: %v", nkc.CertFile, err)
		}
		if len(x509Cert.Subject.CommonName) > 0 {
			tlsCertsByName[x509Cert.Subject.CommonName] = cert
		}
		for _, san := range x509Cert.DNSNames {
			tlsCertsByName[san] = cert
		}
		// intentionally all IPs in the cert are ignored as SNI forbids passing IPs
		// to select a cert. Before go 1.6 the tls happily passed IPs as SNI values.
	}

	// register certs with explicit names last, overwriting every of the implicit ones,
	// again in reverse order.
	for i := len(namedCertKeys) - 1; i >= 0; i-- {
		nkc := &namedCertKeys[i]
		if len(nkc.Names) == 0 {
			continue
		}
		for _, name := range nkc.Names {
			tlsCertsByName[name] = &tlsCerts[i]
		}
	}

	return tlsCertsByName, nil
}
