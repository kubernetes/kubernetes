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
	"encoding/pem"
	"io/ioutil"
	"net"
	"testing"
)

func TestMakeCSR(t *testing.T) {
	keyFile := "testdata/dontUseThisKey.pem"
	subject := &pkix.Name{
		CommonName: "kube-worker",
	}
	dnsSANs := []string{"localhost"}
	ipSANs := []net.IP{net.ParseIP("127.0.0.1")}

	keyData, err := ioutil.ReadFile(keyFile)
	if err != nil {
		t.Fatal(err)
	}
	key, err := ParsePrivateKeyPEM(keyData)
	if err != nil {
		t.Fatal(err)
	}
	csrPEM, err := MakeCSR(key, subject, dnsSANs, ipSANs)
	if err != nil {
		t.Error(err)
	}
	csrBlock, rest := pem.Decode(csrPEM)
	if csrBlock == nil {
		t.Error("Unable to decode MakeCSR result.")
	}
	if len(rest) != 0 {
		t.Error("Found more than one PEM encoded block in the result.")
	}
	if csrBlock.Type != CertificateRequestBlockType {
		t.Errorf("Found block type %q, wanted 'CERTIFICATE REQUEST'", csrBlock.Type)
	}
	csr, err := x509.ParseCertificateRequest(csrBlock.Bytes)
	if err != nil {
		t.Errorf("Found %v parsing MakeCSR result as a CertificateRequest.", err)
	}
	if csr.Subject.CommonName != subject.CommonName {
		t.Errorf("Wanted %v, got %v", subject, csr.Subject)
	}
	if len(csr.DNSNames) != 1 {
		t.Errorf("Wanted 1 DNS name in the result, got %d", len(csr.DNSNames))
	} else if csr.DNSNames[0] != dnsSANs[0] {
		t.Errorf("Wanted %v, got %v", dnsSANs[0], csr.DNSNames[0])
	}
	if len(csr.IPAddresses) != 1 {
		t.Errorf("Wanted 1 IP address in the result, got %d", len(csr.IPAddresses))
	} else if csr.IPAddresses[0].String() != ipSANs[0].String() {
		t.Errorf("Wanted %v, got %v", ipSANs[0], csr.IPAddresses[0])
	}
}
