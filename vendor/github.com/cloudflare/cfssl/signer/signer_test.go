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
	"github.com/cloudflare/cfssl/csr"
)

func TestAppendIf(t *testing.T) {
	s := ""
	a := make([]string, 0, 5)
	appendIf(s, &a)
	if len(a) != 0 {
		t.Fatal("appendIf should not append to a with an empty s")
	}
	s = "test"
	appendIf(s, &a)
	if len(a[0]) != 4 {
		t.Fatal("appendIf should append s to a")
	}
}

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
		{
			ID: config.OID([]int{1, 2, 3, 4}),
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
	expectedBytes, _ := hex.DecodeString("3007300506032a0304")
	if !bytes.Equal(ext.Value, expectedBytes) {
		t.Fatal(fmt.Sprintf("Value didn't match expected bytes: got %s, expected %s",
			hex.EncodeToString(ext.Value), hex.EncodeToString(expectedBytes)))
	}
}

func TestAddPoliciesWithQualifiers(t *testing.T) {
	var cert x509.Certificate
	addPolicies(&cert, []config.CertificatePolicy{
		{
			ID: config.OID([]int{1, 2, 3, 4}),
			Qualifiers: []config.CertificatePolicyQualifier{
				{
					Type:  "id-qt-cps",
					Value: "http://example.com/cps",
				},
				{
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

func TestName(t *testing.T) {
	sub := &Subject{
		CN: "foobar",
		Names: []csr.Name{
			{
				C:  "US",
				ST: "CA",
				L:  "Cool Locality",
				O:  "Cool Org",
				OU: "Really Cool Sub Org",
			},
			{
				L: "Another Cool Locality",
			},
		},
		SerialNumber: "deadbeef",
	}
	name := sub.Name()
	if name.CommonName != sub.CN {
		t.Errorf("CommonName: want %#v, got %#v", sub.CN, name.CommonName)
	}
	if name.SerialNumber != sub.SerialNumber {
		t.Errorf("SerialNumber: want %#v, got %#v", sub.SerialNumber, name.SerialNumber)
	}
	if !reflect.DeepEqual([]string{"US"}, name.Country) {
		t.Errorf("Country: want %s, got %s", []string{"US"}, name.Country)
	}
	if !reflect.DeepEqual([]string{"CA"}, name.Province) {
		t.Errorf("Province: want %s, got %s", []string{"CA"}, name.Province)
	}
	if !reflect.DeepEqual([]string{"Cool Org"}, name.Organization) {
		t.Errorf("Organization: want %s, got %s", []string{"Cool Org"}, name.Organization)
	}
	if !reflect.DeepEqual([]string{"Really Cool Sub Org"}, name.OrganizationalUnit) {
		t.Errorf("Province: want %s, got %s", []string{"Really Cool Sub Org"}, name.OrganizationalUnit)
	}
	if !reflect.DeepEqual([]string{"Cool Locality", "Another Cool Locality"}, name.Locality) {
		t.Errorf("Locality: want %s, got %s", []string{"CA"}, name.Locality)
	}

}
