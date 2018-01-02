package crl

import (
	"crypto/x509"
	"io/ioutil"
	"testing"
)

const (
	serverCertFile = "testdata/ca.pem"
	serverKeyFile  = "testdata/ca-key.pem"
	tryTwoCert     = "testdata/caTwo.pem"
	tryTwoKey      = "testdata/ca-keyTwo.pem"
	serialList     = "testdata/serialList"
)

func TestNewCRLFromFile(t *testing.T) {

	tryTwoKeyBytes, err := ioutil.ReadFile(tryTwoKey)
	if err != nil {
		t.Fatal(err)
	}

	tryTwoCertBytes, err := ioutil.ReadFile(tryTwoCert)
	if err != nil {
		t.Fatal(err)
	}

	serialListBytes, err := ioutil.ReadFile(serialList)
	if err != nil {
		t.Fatal(err)
	}

	crl, err := NewCRLFromFile(serialListBytes, tryTwoCertBytes, tryTwoKeyBytes, "0")
	if err != nil {
		t.Fatal(err)
	}

	certList, err := x509.ParseDERCRL(crl)
	if err != nil {
		t.Fatal(err)
	}

	numCerts := len(certList.TBSCertList.RevokedCertificates)
	expectedNum := 4
	if expectedNum != numCerts {
		t.Fatal("Wrong number of expired certificates")
	}
}

func TestNewCRLFromFileWithoutRevocations(t *testing.T) {
	tryTwoKeyBytes, err := ioutil.ReadFile(tryTwoKey)
	if err != nil {
		t.Fatal(err)
	}

	tryTwoCertBytes, err := ioutil.ReadFile(tryTwoCert)
	if err != nil {
		t.Fatal(err)
	}

	crl, err := NewCRLFromFile([]byte("\n \n"), tryTwoCertBytes, tryTwoKeyBytes, "0")
	if err != nil {
		t.Fatal(err)
	}

	certList, err := x509.ParseDERCRL(crl)
	if err != nil {
		t.Fatal(err)
	}

	numCerts := len(certList.TBSCertList.RevokedCertificates)
	expectedNum := 0
	if expectedNum != numCerts {
		t.Fatal("Wrong number of expired certificates")
	}
}
