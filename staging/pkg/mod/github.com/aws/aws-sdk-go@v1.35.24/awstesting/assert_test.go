package awstesting_test

import (
	"encoding/xml"
	"testing"

	"github.com/aws/aws-sdk-go/awstesting"
)

func TestAssertJSON(t *testing.T) {
	cases := []struct {
		e, a    string
		asserts bool
	}{
		{
			e:       `{"RecursiveStruct":{"RecursiveMap":{"foo":{"NoRecurse":"foo"},"bar":{"NoRecurse":"bar"}}}}`,
			a:       `{"RecursiveStruct":{"RecursiveMap":{"bar":{"NoRecurse":"bar"},"foo":{"NoRecurse":"foo"}}}}`,
			asserts: true,
		},
	}

	for i, c := range cases {
		mockT := &testing.T{}
		if awstesting.AssertJSON(mockT, c.e, c.a) != c.asserts {
			t.Error("Assert JSON result was not expected.", i)
		}
	}
}

func TestAssertXML(t *testing.T) {
	cases := []struct {
		e, a      string
		asserts   bool
		container struct {
			XMLName         xml.Name `xml:"OperationRequest"`
			NS              string   `xml:"xmlns,attr"`
			RecursiveStruct struct {
				XMLName      xml.Name
				RecursiveMap struct {
					XMLName xml.Name
					Entries []struct {
						Key   string `xml:"key"`
						Value struct {
							XMLName   xml.Name `xml:"value"`
							NoRecurse string
						}
					} `xml:"entry"`
				}
			}
		}
	}{
		{
			e:       `<OperationRequest xmlns="https://foo/"><RecursiveStruct xmlns="https://foo/"><RecursiveMap xmlns="https://foo/"><entry xmlns="https://foo/"><key xmlns="https://foo/">foo</key><value xmlns="https://foo/"><NoRecurse xmlns="https://foo/">foo</NoRecurse></value></entry><entry xmlns="https://foo/"><key xmlns="https://foo/">bar</key><value xmlns="https://foo/"><NoRecurse xmlns="https://foo/">bar</NoRecurse></value></entry></RecursiveMap></RecursiveStruct></OperationRequest>`,
			a:       `<OperationRequest xmlns="https://foo/"><RecursiveStruct xmlns="https://foo/"><RecursiveMap xmlns="https://foo/"><entry xmlns="https://foo/"><key xmlns="https://foo/">foo</key><value xmlns="https://foo/"><NoRecurse xmlns="https://foo/">foo</NoRecurse></value></entry><entry xmlns="https://foo/"><key xmlns="https://foo/">bar</key><value xmlns="https://foo/"><NoRecurse xmlns="https://foo/">bar</NoRecurse></value></entry></RecursiveMap></RecursiveStruct></OperationRequest>`,
			asserts: true,
		},
		{
			e:       `<OperationRequest xmlns="https://foo/"><RecursiveStruct xmlns="https://foo/"><RecursiveMap xmlns="https://foo/"><entry xmlns="https://foo/"><key xmlns="https://foo/">foo</key><value xmlns="https://foo/"><NoRecurse xmlns="https://foo/">foo</NoRecurse></value></entry><entry xmlns="https://foo/"><key xmlns="https://foo/">bar</key><value xmlns="https://foo/"><NoRecurse xmlns="https://foo/">bar</NoRecurse></value></entry></RecursiveMap></RecursiveStruct></OperationRequest>`,
			a:       `<OperationRequest xmlns="https://foo/"><RecursiveStruct xmlns="https://foo/"><RecursiveMap xmlns="https://foo/"><entry xmlns="https://foo/"><key xmlns="https://foo/">baz</key><value xmlns="https://foo/"><NoRecurse xmlns="https://foo/">baz</NoRecurse></value></entry></RecursiveMap></RecursiveStruct></OperationRequest>`,
			asserts: false,
		},
	}

	for i, c := range cases {
		mockT := &testing.T{}
		if awstesting.AssertXML(mockT, c.e, c.a) != c.asserts {
			t.Error("Assert XML result was not expected.", i)
		}
	}
}

func TestAssertQuery(t *testing.T) {
	cases := []struct {
		e, a    string
		asserts bool
	}{
		{
			e:       `Action=OperationName&Version=2014-01-01&Foo=val1&Bar=val2`,
			a:       `Action=OperationName&Version=2014-01-01&Foo=val2&Bar=val3`,
			asserts: false,
		},
		{
			e:       `Action=OperationName&Version=2014-01-01&Foo=val1&Bar=val2`,
			a:       `Action=OperationName&Version=2014-01-01&Foo=val1&Bar=val2`,
			asserts: true,
		},
	}

	for i, c := range cases {
		mockT := &testing.T{}
		if awstesting.AssertQuery(mockT, c.e, c.a) != c.asserts {
			t.Error("Assert Query result was not expected.", i)
		}
	}
}
