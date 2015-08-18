package signer

import (
	"bytes"
	"crypto/x509"
	"encoding/asn1"
	"encoding/hex"
	"fmt"
	"reflect"
	"testing"

	"github.com/cloudflare/cfssl/config"
)

func TestSplitHosts(t *testing.T) {
	list := SplitHosts("")
	if list != nil {
		t.Fatal("SplitHost should return nil with empty input")
	}

	list = SplitHosts("single.domain")
	if len(list) != 1 {
		t.Fatal("SplitHost fails to split single domain")
	}

	list = SplitHosts("comma,separated,values")
	if len(list) != 3 {
		t.Fatal("SplitHost fails to split multiple domains")
	}
	if list[0] != "comma" || list[1] != "separated" || list[2] != "values" {
		t.Fatal("SplitHost fails to split multiple domains")
	}
}

func TestAddPolicies(t *testing.T) {
	var cert x509.Certificate
	addPolicies(&cert, []config.CertificatePolicy{
		config.CertificatePolicy{
			ID: config.OID{1, 2, 3, 4},
		},
	})

	if len(cert.ExtraExtensions) != 1 {
		t.Fatal("No extension added")
	}
	ext := cert.ExtraExtensions[0]
	if !reflect.DeepEqual(ext.Id, asn1.ObjectIdentifier{2, 5, 29, 32}) {
		t.Fatal(fmt.Sprintf("Wrong OID for policy qualifier %v", ext.Id))
	}
	if ext.Critical {
		t.Fatal("Policy qualifier marked critical")
	}
	expectedBytes, _ := hex.DecodeString("3009300706032a03043000")
	if !bytes.Equal(ext.Value, expectedBytes) {
		t.Fatal(fmt.Sprintf("Value didn't match expected bytes: %s vs %s",
			hex.EncodeToString(ext.Value), hex.EncodeToString(expectedBytes)))
	}
}

func TestAddPoliciesWithQualifiers(t *testing.T) {
	var cert x509.Certificate
	addPolicies(&cert, []config.CertificatePolicy{
		config.CertificatePolicy{
			ID: config.OID{1, 2, 3, 4},
			Qualifiers: []config.CertificatePolicyQualifier{
				config.CertificatePolicyQualifier{
					Type:  "id-qt-cps",
					Value: "http://example.com/cps",
				},
				config.CertificatePolicyQualifier{
					Type:  "id-qt-unotice",
					Value: "Do What Thou Wilt",
				},
			},
		},
	})

	if len(cert.ExtraExtensions) != 1 {
		t.Fatal("No extension added")
	}
	ext := cert.ExtraExtensions[0]
	if !reflect.DeepEqual(ext.Id, asn1.ObjectIdentifier{2, 5, 29, 32}) {
		t.Fatal(fmt.Sprintf("Wrong OID for policy qualifier %v", ext.Id))
	}
	if ext.Critical {
		t.Fatal("Policy qualifier marked critical")
	}
	expectedBytes, _ := hex.DecodeString("304e304c06032a03043045302206082b060105050702011616687474703a2f2f6578616d706c652e636f6d2f637073301f06082b0601050507020230130c11446f20576861742054686f752057696c74")
	if !bytes.Equal(ext.Value, expectedBytes) {
		t.Fatal(fmt.Sprintf("Value didn't match expected bytes: %s vs %s",
			hex.EncodeToString(ext.Value), hex.EncodeToString(expectedBytes)))
	}
}
