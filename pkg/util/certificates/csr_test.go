package certificates

import (
	"crypto/x509/pkix"
	"io/ioutil"
	"net"
	"testing"
)

func TestNewCertificateRequest(t *testing.T) {
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
	_, _, err := NewCertificateRequest(keyData, subject, dnsSANs, ipSANs)
	if err != nil {
		t.Error(err)
	}
}
