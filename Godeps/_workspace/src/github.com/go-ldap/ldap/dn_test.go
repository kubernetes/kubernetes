package ldap

import (
	"reflect"
	"testing"
)

func TestSuccessfulDNParsing(t *testing.T) {
	testcases := map[string]DN {
		"": DN{[]*RelativeDN{}},
		"cn=Jim\\2C \\22Hasse Hö\\22 Hansson!,dc=dummy,dc=com": DN{[]*RelativeDN{
			&RelativeDN{[]*AttributeTypeAndValue{&AttributeTypeAndValue{"cn", "Jim, \"Hasse Hö\" Hansson!"},}},
			&RelativeDN{[]*AttributeTypeAndValue{&AttributeTypeAndValue{"dc", "dummy"},}},
			&RelativeDN{[]*AttributeTypeAndValue{&AttributeTypeAndValue{"dc", "com"}, }},}},
		"UID=jsmith,DC=example,DC=net": DN{[]*RelativeDN{
			&RelativeDN{[]*AttributeTypeAndValue{&AttributeTypeAndValue{"UID", "jsmith"},}},
			&RelativeDN{[]*AttributeTypeAndValue{&AttributeTypeAndValue{"DC", "example"},}},
			&RelativeDN{[]*AttributeTypeAndValue{&AttributeTypeAndValue{"DC", "net"}, }},}},
		"OU=Sales+CN=J. Smith,DC=example,DC=net": DN{[]*RelativeDN{
			&RelativeDN{[]*AttributeTypeAndValue{
				&AttributeTypeAndValue{"OU", "Sales"},
				&AttributeTypeAndValue{"CN", "J. Smith"},}},
			&RelativeDN{[]*AttributeTypeAndValue{&AttributeTypeAndValue{"DC", "example"},}},
			&RelativeDN{[]*AttributeTypeAndValue{&AttributeTypeAndValue{"DC", "net"}, }},}},
		"1.3.6.1.4.1.1466.0=#04024869": DN{[]*RelativeDN{
			&RelativeDN{[]*AttributeTypeAndValue{&AttributeTypeAndValue{"1.3.6.1.4.1.1466.0", "Hi"},}},}},
		"1.3.6.1.4.1.1466.0=#04024869,DC=net": DN{[]*RelativeDN{
			&RelativeDN{[]*AttributeTypeAndValue{&AttributeTypeAndValue{"1.3.6.1.4.1.1466.0", "Hi"},}},
			&RelativeDN{[]*AttributeTypeAndValue{&AttributeTypeAndValue{"DC", "net"}, }},}},
		"CN=Lu\\C4\\8Di\\C4\\87": DN{[]*RelativeDN{
			&RelativeDN{[]*AttributeTypeAndValue{&AttributeTypeAndValue{"CN", "Lučić"},}},}},
	}

	for test, answer := range testcases {
		dn, err := ParseDN(test)
		if err != nil {
			t.Errorf(err.Error())
			continue
		}
		if !reflect.DeepEqual(dn, &answer) {
			t.Errorf("Parsed DN %s is not equal to the expected structure", test)
			for _, rdn := range dn.RDNs {
				for _, attribs := range rdn.Attributes {
					t.Logf("#%v\n", attribs)
				}
			}
		}
	}
}

func TestErrorDNParsing(t *testing.T) {
	testcases := map[string]string {
		"*": "DN ended with incomplete type, value pair",
		"cn=Jim\\0Test": "Failed to decode escaped character: encoding/hex: invalid byte: U+0054 'T'",
		"cn=Jim\\0": "Got corrupted escaped character",
		"DC=example,=net": "DN ended with incomplete type, value pair",
		"1=#0402486": "Failed to decode BER encoding: encoding/hex: odd length hex string",
	}

	for test, answer := range testcases {
		_, err := ParseDN(test)
		if err == nil {
			t.Errorf("Expected %s to fail parsing but succeeded\n", test)
		} else if err.Error() != answer {
			t.Errorf("Unexpected error on %s:\n%s\nvs.\n%s\n", test, answer, err.Error())
		}
	}
}


