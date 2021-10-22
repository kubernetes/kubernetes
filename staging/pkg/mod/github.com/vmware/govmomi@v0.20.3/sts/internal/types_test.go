/*
Copyright (c) 2018 VMware, Inc. All Rights Reserved.

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

package internal

import (
	"log"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/vmware/govmomi/vim25/xml"
)

func isC14N(s string) bool {
	p, err := exec.LookPath("xmlstarlet")
	if err != nil {
		log.Printf("cannot validate C14N: %s", err)
		return true
	}

	cmd := exec.Command(p, "c14n", "--exc-without-comments", "-")
	log.Printf("validating with %s", cmd.Args)
	cmd.Stdin = strings.NewReader(s)
	out, err := cmd.CombinedOutput()
	if err != nil {
		log.Fatal(err)
	}

	if s == string(out) {
		return true
	}

	log.Printf(" IN:%s", s)
	log.Printf(" OUT:%s", string(out))

	return false
}

func TestTimestamp(t *testing.T) {
	created := time.Now().UTC()
	timestamp := Timestamp{
		NS:      WSU,
		ID:      "_id",
		Created: created.Format(Time),
		Expires: created.Add(time.Hour).Format(Time),
	}

	if !isC14N(timestamp.C14N()) {
		t.Error("not c14n")
	}
}

func TestAssertion(t *testing.T) {
	token := `<saml2:Assertion xmlns:saml2="urn:oasis:names:tc:SAML:2.0:assertion" xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" ID="_1881a9ba-4a76-4baa-839b-36e2cba10743" IssueInstant="2018-03-04T00:27:56.409Z" Version="2.0"><saml2:Issuer Format="urn:oasis:names:tc:SAML:2.0:nameid-format:entity">https://office1-sfo2-dhcp221.eng.vmware.com/websso/SAML2/Metadata/vsphere.local</saml2:Issuer><ds:Signature xmlns:ds="http://www.w3.org/2000/09/xmldsig#"><ds:SignedInfo><ds:CanonicalizationMethod Algorithm="http://www.w3.org/2001/10/xml-exc-c14n#"/><ds:SignatureMethod Algorithm="http://www.w3.org/2001/04/xmldsig-more#rsa-sha256"/><ds:Reference URI="#_1881a9ba-4a76-4baa-839b-36e2cba10743"><ds:Transforms><ds:Transform Algorithm="http://www.w3.org/2000/09/xmldsig#enveloped-signature"/><ds:Transform Algorithm="http://www.w3.org/2001/10/xml-exc-c14n#"><ec:InclusiveNamespaces xmlns:ec="http://www.w3.org/2001/10/xml-exc-c14n#" PrefixList="xs xsi"/></ds:Transform></ds:Transforms><ds:DigestMethod Algorithm="http://www.w3.org/2001/04/xmlenc#sha256"/><ds:DigestValue>l/0AzCGiPB69oTstUdrCkihBIDtwb83A93zAe10tG3k=</ds:DigestValue></ds:Reference></ds:SignedInfo><ds:SignatureValue>EKHf14V0CHctwqXRlhYSYNyID5lNJLimbw57eUBm/QlAMLY7GJ1wth44oeQPSj3eMpJaXKHEYYtn
fqMngciTrq4ZP2SS7KizxuBjcHChWGmcp+t0zn7+fTbp5sL8HfF3AfOwcyZxwj8n2S7E6Eee7zeC
cjZpKKZ1QIEwASwpuMCs7vU9IuXsUguHAaN55Jpx3N5u7PlSo/NZE0TJZ+zNWP8m9H5shPDY272D
Vnp3MGfoD+Dj6T4H8OVF6bMp6czbHsEHTthwPh+pBTzR8ppkyxPKWLkC7OWiOtZBKqLSMTchQyqn
GNJdl72FBXHS8WXGtJjbwL+MKf+WujhqwdRbXw==</ds:SignatureValue><ds:KeyInfo><ds:X509Data><ds:X509Certificate>MIIDxTCCAq2gAwIBAgIJAMYXe1r3pfByMA0GCSqGSIb3DQEBCwUAMIGqMQswCQYDVQQDDAJDQTEX
MBUGCgmSJomT8ixkARkWB3ZzcGhlcmUxFTATBgoJkiaJk/IsZAEZFgVsb2NhbDELMAkGA1UEBhMC
VVMxEzARBgNVBAgMCkNhbGlmb3JuaWExLDAqBgNVBAoMI29mZmljZTEtc2ZvMi1kaGNwMjIxLmVu
Zy52bXdhcmUuY29tMRswGQYDVQQLDBJWTXdhcmUgRW5naW5lZXJpbmcwHhcNMTgwMTExMjE1MjQ3
WhcNMjgwMTA2MjIwMjMxWjAYMRYwFAYDVQQDDA1zc29zZXJ2ZXJTaWduMIIBIjANBgkqhkiG9w0B
AQEFAAOCAQ8AMIIBCgKCAQEAohfKdXEpiCB+EewJJKk98he/KeAK/1bZ2MjnLspwt3Nvv2uh2xoa
1asP/TMAhxcztPxhqEZmi0W+nihF/yffY/AhQrGx9XynaOMUNarCNGVI2qBovi8gohT2pXlbKxgZ
b8VZkVl41WYkDBfQrzoP0XU/sFeOoNIHcFQX/82NFAYtN/4aBZ9gDqhyPihv2RSNG4MnvxxgxtZI
FPb3eyDt8poKOMjt8zG2JkJRQYiEOCLo/sKJEKXLZeWiqYsbk391/vIk2vaX3L3pgu8yYx/dLfxv
X/mRYIOcVzpXWQCEPdCejQBwrmVeRaepW5cMhOVlMAAw+mEXYVVTaIi1pfN53wIDAQABo38wfTAL
BgNVHQ8EBAMCBeAwLgYDVR0RBCcwJYIjb2ZmaWNlMS1zZm8yLWRoY3AyMjEuZW5nLnZtd2FyZS5j
b20wHQYDVR0OBBYEFAtGcFg9jVO3aBjgd2K0iBFTAPNSMB8GA1UdIwQYMBaAFLpyqy2v1I7a3URK
ohtSLAtqve5qMA0GCSqGSIb3DQEBCwUAA4IBAQB91dZHRFunBs+YvuOYFRlwJTZOPXzlSYurxC7h
VeYv6LUGZnuTkp0KfVMsfHyaeDslM8+5F9Iug1jxmEmpeyoaY12zQmxQB6P8lN4jj1Aazj8qmDH6
ClaSY4Pp0lOSp9ROVlnLi6sRsRphOg+4MS4UeXGgSFlMN1BWJmXcwCazbii8l/EzGx2QhlVjWMAz
lPFQlWQ4FvV5vUCf8iE+UTin+6oJSXmFzip1NOBOGiIbClmpergZUchNiqTYTrpqblD/Qex5Bv9e
+xAwuw8e0Lm0XICOcFmKvpotLKKiqMMsRqPoeTqnoSyKqvCGRo2hUs4Y4O6SqEd80+E5lbXImrSt</ds:X509Certificate><ds:X509Certificate>MIIEPzCCAyegAwIBAgIJANS+QleTVJNbMA0GCSqGSIb3DQEBCwUAMIGqMQswCQYDVQQDDAJDQTEX
MBUGCgmSJomT8ixkARkWB3ZzcGhlcmUxFTATBgoJkiaJk/IsZAEZFgVsb2NhbDELMAkGA1UEBhMC
VVMxEzARBgNVBAgMCkNhbGlmb3JuaWExLDAqBgNVBAoMI29mZmljZTEtc2ZvMi1kaGNwMjIxLmVu
Zy52bXdhcmUuY29tMRswGQYDVQQLDBJWTXdhcmUgRW5naW5lZXJpbmcwHhcNMTgwMTA4MjIwMjMx
WhcNMjgwMTA2MjIwMjMxWjCBqjELMAkGA1UEAwwCQ0ExFzAVBgoJkiaJk/IsZAEZFgd2c3BoZXJl
MRUwEwYKCZImiZPyLGQBGRYFbG9jYWwxCzAJBgNVBAYTAlVTMRMwEQYDVQQIDApDYWxpZm9ybmlh
MSwwKgYDVQQKDCNvZmZpY2UxLXNmbzItZGhjcDIyMS5lbmcudm13YXJlLmNvbTEbMBkGA1UECwwS
Vk13YXJlIEVuZ2luZWVyaW5nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAxcN7rsoK
CIapsEOYejPK38Qk7CUSPFcOmT7iF15UFlZDogHe1G/ZkYvcP0IvLvpemRiYuRpVGVuUZ9XOgeW6
J5xpSuNRXMHSMDTUwLM9t/4NMAQxgWVlJjFmPVBIZiWaQgdCzEbCDcv/XaZeb6uJYlbmLKvopmwy
oDfncGXRUuQIZFsVIUhUgOtbbp9UmvXyjo9ukWdVcTkKlKK7NZGaVa4JYy7q4cc6g5eRmD9qp16o
vx8DageNAasTP6arnb5CyoGI4KPqJjaI7V4Z1KiOUs+Zj+VtC3XdpVthNtiJ+vgXccO8e7zYfP0y
d1PCQ/GEZAlRabus5Iplu4/xC23NywIDAQABo2YwZDAdBgNVHQ4EFgQUunKrLa/UjtrdREqiG1Is
C2q97mowHwYDVR0RBBgwFoEOZW1haWxAYWNtZS5jb22HBH8AAAEwDgYDVR0PAQH/BAQDAgEGMBIG
A1UdEwEB/wQIMAYBAf8CAQAwDQYJKoZIhvcNAQELBQADggEBAC8bMIhFtlXnCF2fUixTXJ5HZFNY
vbxa1eFjLFYuBsGBqhPEHkHkdKwgpfo1sd4t0L7JaGS9wsH6zyRUQs97subV5YUI6rvAPOBGDQTm
RmCeqz3ODZq6JwZEnTTqZjvUVckmt/L/QaRUHAW27MU+SuN8rP0Nghf/gkOabsaWfyT2ADquko4e
b7seYIlR5mJs+pxVBBsBB2nzxuaV5EjkgestxBqpGkxMnKEDhG6+VjqVxsZoEiNzdBNU7eM67Jc2
2KU85jHKAao9LfMbwbHOA//1RStXXElyzPQvecq17ATvpw8AxCRu2KeKRwp3Pm2RiquDQFx8aiCe
2Re4gkrEemA=</ds:X509Certificate></ds:X509Data></ds:KeyInfo></ds:Signature><saml2:Subject><saml2:NameID Format="http://schemas.xmlsoap.org/claims/UPN">Administrator@VSPHERE.LOCAL</saml2:NameID><saml2:SubjectConfirmation Method="urn:oasis:names:tc:SAML:2.0:cm:bearer"><saml2:SubjectConfirmationData NotOnOrAfter="2018-03-04T00:27:01.401Z"/></saml2:SubjectConfirmation></saml2:Subject><saml2:Conditions NotBefore="2018-03-04T00:22:01.401Z" NotOnOrAfter="2018-03-04T00:27:01.401Z"><saml2:ProxyRestriction Count="10"/></saml2:Conditions><saml2:AuthnStatement AuthnInstant="2018-03-04T00:27:56.402Z"><saml2:AuthnContext><saml2:AuthnContextClassRef>urn:oasis:names:tc:SAML:2.0:ac:classes:PasswordProtectedTransport</saml2:AuthnContextClassRef></saml2:AuthnContext></saml2:AuthnStatement><saml2:AttributeStatement><saml2:Attribute FriendlyName="Groups" Name="http://rsa.com/schemas/attr-names/2009/01/GroupIdentity" NameFormat="urn:oasis:names:tc:SAML:2.0:attrname-format:uri"><saml2:AttributeValue xsi:type="xs:string">vsphere.local\Users</saml2:AttributeValue><saml2:AttributeValue xsi:type="xs:string">vsphere.local\Administrators</saml2:AttributeValue><saml2:AttributeValue xsi:type="xs:string">vsphere.local\CAAdmins</saml2:AttributeValue><saml2:AttributeValue xsi:type="xs:string">vsphere.local\ComponentManager.Administrators</saml2:AttributeValue><saml2:AttributeValue xsi:type="xs:string">vsphere.local\SystemConfiguration.BashShellAdministrators</saml2:AttributeValue><saml2:AttributeValue xsi:type="xs:string">vsphere.local\SystemConfiguration.Administrators</saml2:AttributeValue><saml2:AttributeValue xsi:type="xs:string">vsphere.local\LicenseService.Administrators</saml2:AttributeValue><saml2:AttributeValue xsi:type="xs:string">vsphere.local\ActAsUsers</saml2:AttributeValue><saml2:AttributeValue xsi:type="xs:string">vsphere.local\Everyone</saml2:AttributeValue></saml2:Attribute><saml2:Attribute FriendlyName="givenName" Name="http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname" NameFormat="urn:oasis:names:tc:SAML:2.0:attrname-format:uri"><saml2:AttributeValue xsi:type="xs:string">Administrator</saml2:AttributeValue></saml2:Attribute><saml2:Attribute FriendlyName="surname" Name="http://schemas.xmlsoap.org/ws/2005/05/identity/claims/surname" NameFormat="urn:oasis:names:tc:SAML:2.0:attrname-format:uri"><saml2:AttributeValue xsi:type="xs:string">vsphere.local</saml2:AttributeValue></saml2:Attribute><saml2:Attribute FriendlyName="Subject Type" Name="http://vmware.com/schemas/attr-names/2011/07/isSolution" NameFormat="urn:oasis:names:tc:SAML:2.0:attrname-format:uri"><saml2:AttributeValue xsi:type="xs:string">false</saml2:AttributeValue></saml2:Attribute></saml2:AttributeStatement></saml2:Assertion>`

	var a Assertion
	err := xml.Unmarshal([]byte(token), &a)
	if err != nil {
		t.Fatal(err)
	}

	if !isC14N(a.C14N()) {
		t.Error("not c14n")
	}

	a.Signature.SignedInfo.NS = DSIG
	if !isC14N(a.Signature.SignedInfo.C14N()) {
		t.Error("not c14n")
	}
}
