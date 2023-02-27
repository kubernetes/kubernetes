// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

package defs

import "encoding/xml"

type Definitions struct {
	XMLName         xml.Name   `xml:"definitions"`
	Text            string     `xml:",chardata"`
	Name            string     `xml:"name,attr"`
	TargetNamespace string     `xml:"targetNamespace,attr"`
	WSDL            string     `xml:"wsdl,attr"`
	XSD             string     `xml:"xsd,attr"`
	T               string     `xml:"t,attr"`
	SOAPENC         string     `xml:"soapenc,attr"`
	SOAP            string     `xml:"soap,attr"`
	TNS             string     `xml:"tns,attr"`
	MSC             string     `xml:"msc,attr"`
	WSAM            string     `xml:"wsam,attr"`
	SOAP12          string     `xml:"soap12,attr"`
	WSA10           string     `xml:"wsa10,attr"`
	WSA             string     `xml:"wsa,attr"`
	WSAW            string     `xml:"wsaw,attr"`
	WSX             string     `xml:"wsx,attr"`
	WSAP            string     `xml:"wsap,attr"`
	WSU             string     `xml:"wsu,attr"`
	Trust           string     `xml:"trust,attr"`
	WSP             string     `xml:"wsp,attr"`
	Policy          []Policy   `xml:"Policy"`
	Types           Types      `xml:"types"`
	Message         []Message  `xml:"message"`
	PortType        []PortType `xml:"portType"`
	Binding         []Binding  `xml:"binding"`
	Service         Service    `xml:"service"`
}

type Policy struct {
	Text       string     `xml:",chardata"`
	ID         string     `xml:"Id,attr"`
	ExactlyOne ExactlyOne `xml:"ExactlyOne"`
}

type ExactlyOne struct {
	Text string `xml:",chardata"`
	All  All    `xml:"All"`
}

type All struct {
	Text                            string                          `xml:",chardata"`
	NegotiateAuthentication         NegotiateAuthentication         `xml:"NegotiateAuthentication"`
	TransportBinding                TransportBinding                `xml:"TransportBinding"`
	UsingAddressing                 Text                            `xml:"UsingAddressing"`
	EndorsingSupportingTokens       EndorsingSupportingTokens       `xml:"EndorsingSupportingTokens"`
	WSS11                           WSS11                           `xml:"Wss11"`
	Trust10                         Trust10                         `xml:"Trust10"`
	SignedSupportingTokens          SignedSupportingTokens          `xml:"SignedSupportingTokens"`
	Trust13                         WSTrust13                       `xml:"Trust13"`
	SignedEncryptedSupportingTokens SignedEncryptedSupportingTokens `xml:"SignedEncryptedSupportingTokens"`
}

type NegotiateAuthentication struct {
	Text    string `xml:",chardata"`
	HTTP    string `xml:"http,attr"`
	XMLName xml.Name
}

type TransportBinding struct {
	Text   string                 `xml:",chardata"`
	SP     string                 `xml:"sp,attr"`
	Policy TransportBindingPolicy `xml:"Policy"`
}

type TransportBindingPolicy struct {
	Text             string         `xml:",chardata"`
	TransportToken   TransportToken `xml:"TransportToken"`
	AlgorithmSuite   AlgorithmSuite `xml:"AlgorithmSuite"`
	Layout           Layout         `xml:"Layout"`
	IncludeTimestamp Text           `xml:"IncludeTimestamp"`
}

type TransportToken struct {
	Text   string               `xml:",chardata"`
	Policy TransportTokenPolicy `xml:"Policy"`
}

type TransportTokenPolicy struct {
	Text       string     `xml:",chardata"`
	HTTPSToken HTTPSToken `xml:"HttpsToken"`
}

type HTTPSToken struct {
	Text                     string `xml:",chardata"`
	RequireClientCertificate string `xml:"RequireClientCertificate,attr"`
}

type AlgorithmSuite struct {
	Text   string               `xml:",chardata"`
	Policy AlgorithmSuitePolicy `xml:"Policy"`
}

type AlgorithmSuitePolicy struct {
	Text     string `xml:",chardata"`
	Basic256 Text   `xml:"Basic256"`
	Basic128 Text   `xml:"Basic128"`
}

type Layout struct {
	Text   string       `xml:",chardata"`
	Policy LayoutPolicy `xml:"Policy"`
}

type LayoutPolicy struct {
	Text   string `xml:",chardata"`
	Strict Text   `xml:"Strict"`
}

type EndorsingSupportingTokens struct {
	Text   string                          `xml:",chardata"`
	SP     string                          `xml:"sp,attr"`
	Policy EndorsingSupportingTokensPolicy `xml:"Policy"`
}

type EndorsingSupportingTokensPolicy struct {
	Text          string        `xml:",chardata"`
	X509Token     X509Token     `xml:"X509Token"`
	RSAToken      RSAToken      `xml:"RsaToken"`
	SignedParts   SignedParts   `xml:"SignedParts"`
	KerberosToken KerberosToken `xml:"KerberosToken"`
	IssuedToken   IssuedToken   `xml:"IssuedToken"`
	KeyValueToken KeyValueToken `xml:"KeyValueToken"`
}

type X509Token struct {
	Text         string          `xml:",chardata"`
	IncludeToken string          `xml:"IncludeToken,attr"`
	Policy       X509TokenPolicy `xml:"Policy"`
}

type X509TokenPolicy struct {
	Text                       string `xml:",chardata"`
	RequireThumbprintReference Text   `xml:"RequireThumbprintReference"`
	WSSX509V3Token10           Text   `xml:"WssX509V3Token10"`
}

type RSAToken struct {
	Text         string `xml:",chardata"`
	IncludeToken string `xml:"IncludeToken,attr"`
	Optional     string `xml:"Optional,attr"`
	MSSP         string `xml:"mssp,attr"`
}

type SignedParts struct {
	Text   string            `xml:",chardata"`
	Header SignedPartsHeader `xml:"Header"`
}

type SignedPartsHeader struct {
	Text      string `xml:",chardata"`
	Name      string `xml:"Name,attr"`
	Namespace string `xml:"Namespace,attr"`
}

type KerberosToken struct {
	Text         string              `xml:",chardata"`
	IncludeToken string              `xml:"IncludeToken,attr"`
	Policy       KerberosTokenPolicy `xml:"Policy"`
}

type KerberosTokenPolicy struct {
	Text                         string `xml:",chardata"`
	WSSGSSKerberosV5ApReqToken11 Text   `xml:"WssGssKerberosV5ApReqToken11"`
}

type IssuedToken struct {
	Text                         string                       `xml:",chardata"`
	IncludeToken                 string                       `xml:"IncludeToken,attr"`
	RequestSecurityTokenTemplate RequestSecurityTokenTemplate `xml:"RequestSecurityTokenTemplate"`
	Policy                       IssuedTokenPolicy            `xml:"Policy"`
}

type RequestSecurityTokenTemplate struct {
	Text                      string `xml:",chardata"`
	KeyType                   Text   `xml:"KeyType"`
	EncryptWith               Text   `xml:"EncryptWith"`
	SignatureAlgorithm        Text   `xml:"SignatureAlgorithm"`
	CanonicalizationAlgorithm Text   `xml:"CanonicalizationAlgorithm"`
	EncryptionAlgorithm       Text   `xml:"EncryptionAlgorithm"`
	KeySize                   Text   `xml:"KeySize"`
	KeyWrapAlgorithm          Text   `xml:"KeyWrapAlgorithm"`
}

type IssuedTokenPolicy struct {
	Text                     string `xml:",chardata"`
	RequireInternalReference Text   `xml:"RequireInternalReference"`
}

type KeyValueToken struct {
	Text         string `xml:",chardata"`
	IncludeToken string `xml:"IncludeToken,attr"`
	Optional     string `xml:"Optional,attr"`
}

type WSS11 struct {
	Text   string      `xml:",chardata"`
	SP     string      `xml:"sp,attr"`
	Policy Wss11Policy `xml:"Policy"`
}

type Wss11Policy struct {
	Text                     string `xml:",chardata"`
	MustSupportRefThumbprint Text   `xml:"MustSupportRefThumbprint"`
}

type Trust10 struct {
	Text   string        `xml:",chardata"`
	SP     string        `xml:"sp,attr"`
	Policy Trust10Policy `xml:"Policy"`
}

type Trust10Policy struct {
	Text                    string `xml:",chardata"`
	MustSupportIssuedTokens Text   `xml:"MustSupportIssuedTokens"`
	RequireClientEntropy    Text   `xml:"RequireClientEntropy"`
	RequireServerEntropy    Text   `xml:"RequireServerEntropy"`
}

type SignedSupportingTokens struct {
	Text   string                 `xml:",chardata"`
	SP     string                 `xml:"sp,attr"`
	Policy SupportingTokensPolicy `xml:"Policy"`
}

type SupportingTokensPolicy struct {
	Text          string        `xml:",chardata"`
	UsernameToken UsernameToken `xml:"UsernameToken"`
}
type UsernameToken struct {
	Text         string              `xml:",chardata"`
	IncludeToken string              `xml:"IncludeToken,attr"`
	Policy       UsernameTokenPolicy `xml:"Policy"`
}

type UsernameTokenPolicy struct {
	Text               string             `xml:",chardata"`
	WSSUsernameToken10 WSSUsernameToken10 `xml:"WssUsernameToken10"`
}

type WSSUsernameToken10 struct {
	Text    string `xml:",chardata"`
	XMLName xml.Name
}

type WSTrust13 struct {
	Text   string          `xml:",chardata"`
	SP     string          `xml:"sp,attr"`
	Policy WSTrust13Policy `xml:"Policy"`
}

type WSTrust13Policy struct {
	Text                    string `xml:",chardata"`
	MustSupportIssuedTokens Text   `xml:"MustSupportIssuedTokens"`
	RequireClientEntropy    Text   `xml:"RequireClientEntropy"`
	RequireServerEntropy    Text   `xml:"RequireServerEntropy"`
}

type SignedEncryptedSupportingTokens struct {
	Text   string                 `xml:",chardata"`
	SP     string                 `xml:"sp,attr"`
	Policy SupportingTokensPolicy `xml:"Policy"`
}

type Types struct {
	Text   string `xml:",chardata"`
	Schema Schema `xml:"schema"`
}

type Schema struct {
	Text            string   `xml:",chardata"`
	TargetNamespace string   `xml:"targetNamespace,attr"`
	Import          []Import `xml:"import"`
}

type Import struct {
	Text           string `xml:",chardata"`
	SchemaLocation string `xml:"schemaLocation,attr"`
	Namespace      string `xml:"namespace,attr"`
}

type Message struct {
	Text string `xml:",chardata"`
	Name string `xml:"name,attr"`
	Part Part   `xml:"part"`
}

type Part struct {
	Text    string `xml:",chardata"`
	Name    string `xml:"name,attr"`
	Element string `xml:"element,attr"`
}

type PortType struct {
	Text      string    `xml:",chardata"`
	Name      string    `xml:"name,attr"`
	Operation Operation `xml:"operation"`
}

type Operation struct {
	Text   string      `xml:",chardata"`
	Name   string      `xml:"name,attr"`
	Input  OperationIO `xml:"input"`
	Output OperationIO `xml:"output"`
}

type OperationIO struct {
	Text    string          `xml:",chardata"`
	Action  string          `xml:"Action,attr"`
	Message string          `xml:"message,attr"`
	Body    OperationIOBody `xml:"body"`
}

type OperationIOBody struct {
	Text string `xml:",chardata"`
	Use  string `xml:"use,attr"`
}

type Binding struct {
	Text            string             `xml:",chardata"`
	Name            string             `xml:"name,attr"`
	Type            string             `xml:"type,attr"`
	PolicyReference PolicyReference    `xml:"PolicyReference"`
	Binding         DefinitionsBinding `xml:"binding"`
	Operation       BindingOperation   `xml:"operation"`
}

type PolicyReference struct {
	Text string `xml:",chardata"`
	URI  string `xml:"URI,attr"`
}

type DefinitionsBinding struct {
	Text      string `xml:",chardata"`
	Transport string `xml:"transport,attr"`
}

type BindingOperation struct {
	Text      string                    `xml:",chardata"`
	Name      string                    `xml:"name,attr"`
	Operation BindingOperationOperation `xml:"operation"`
	Input     BindingOperationIO        `xml:"input"`
	Output    BindingOperationIO        `xml:"output"`
}

type BindingOperationOperation struct {
	Text       string `xml:",chardata"`
	SoapAction string `xml:"soapAction,attr"`
	Style      string `xml:"style,attr"`
}

type BindingOperationIO struct {
	Text string          `xml:",chardata"`
	Body OperationIOBody `xml:"body"`
}

type Service struct {
	Text string `xml:",chardata"`
	Name string `xml:"name,attr"`
	Port []Port `xml:"port"`
}

type Port struct {
	Text              string                `xml:",chardata"`
	Name              string                `xml:"name,attr"`
	Binding           string                `xml:"binding,attr"`
	Address           Address               `xml:"address"`
	EndpointReference PortEndpointReference `xml:"EndpointReference"`
}

type Address struct {
	Text     string `xml:",chardata"`
	Location string `xml:"location,attr"`
}

type PortEndpointReference struct {
	Text     string   `xml:",chardata"`
	Address  Text     `xml:"Address"`
	Identity Identity `xml:"Identity"`
}

type Identity struct {
	Text  string `xml:",chardata"`
	XMLNS string `xml:"xmlns,attr"`
	SPN   Text   `xml:"Spn"`
}
