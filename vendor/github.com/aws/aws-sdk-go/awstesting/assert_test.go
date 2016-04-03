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
				RecursiveMap struct {
					Entries []struct {
						XMLName xml.Name `xml:"entries"`
						Key     string   `xml:"key"`
						Value   struct {
							XMLName   xml.Name `xml:"value"`
							NoRecurse string
						}
					}
				}
			}
		}
	}{
		{
			e:       `<OperationRequest xmlns="https://foo/"><RecursiveStruct xmlns="https://foo/"><RecursiveMap xmlns="https://foo/"><entry xmlns="https://foo/"><key xmlns="https://foo/">foo</key><value xmlns="https://foo/"><NoRecurse xmlns="https://foo/">foo</NoRecurse></value></entry><entry xmlns="https://foo/"><key xmlns="https://foo/">bar</key><value xmlns="https://foo/"><NoRecurse xmlns="https://foo/">bar</NoRecurse></value></entry></RecursiveMap></RecursiveStruct></OperationRequest>`,
			a:       `<OperationRequest xmlns="https://foo/"><RecursiveStruct xmlns="https://foo/"><RecursiveMap xmlns="https://foo/"><entry xmlns="https://foo/"><key xmlns="https://foo/">bar</key><value xmlns="https://foo/"><NoRecurse xmlns="https://foo/">bar</NoRecurse></value></entry><entry xmlns="https://foo/"><key xmlns="https://foo/">foo</key><value xmlns="https://foo/"><NoRecurse xmlns="https://foo/">foo</NoRecurse></value></entry></RecursiveMap></RecursiveStruct></OperationRequest>`,
			asserts: true,
		},
	}

	for i, c := range cases {
		//		mockT := &testing.T{}
		if awstesting.AssertXML(t, c.e, c.a, c.container) != c.asserts {
			t.Error("Assert XML result was not expected.", i)
		}
	}
}
