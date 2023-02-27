// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

package defs

import "encoding/xml"

// TODO(msal): Someone (and it ain't gonna be me) needs to document these attributes or
// at the least put a link to RFC.

type SAMLDefinitions struct {
	XMLName xml.Name `xml:"Envelope"`
	Text    string   `xml:",chardata"`
	S       string   `xml:"s,attr"`
	A       string   `xml:"a,attr"`
	U       string   `xml:"u,attr"`
	Header  Header   `xml:"Header"`
	Body    Body     `xml:"Body"`
}

type Header struct {
	Text     string   `xml:",chardata"`
	Action   Action   `xml:"Action"`
	Security Security `xml:"Security"`
}

type Action struct {
	Text           string `xml:",chardata"`
	MustUnderstand string `xml:"mustUnderstand,attr"`
}

type Security struct {
	Text           string    `xml:",chardata"`
	MustUnderstand string    `xml:"mustUnderstand,attr"`
	O              string    `xml:"o,attr"`
	Timestamp      Timestamp `xml:"Timestamp"`
}

type Timestamp struct {
	Text    string `xml:",chardata"`
	ID      string `xml:"Id,attr"`
	Created Text   `xml:"Created"`
	Expires Text   `xml:"Expires"`
}

type Text struct {
	Text string `xml:",chardata"`
}

type Body struct {
	Text                                   string                                 `xml:",chardata"`
	RequestSecurityTokenResponseCollection RequestSecurityTokenResponseCollection `xml:"RequestSecurityTokenResponseCollection"`
}

type RequestSecurityTokenResponseCollection struct {
	Text                         string                         `xml:",chardata"`
	Trust                        string                         `xml:"trust,attr"`
	RequestSecurityTokenResponse []RequestSecurityTokenResponse `xml:"RequestSecurityTokenResponse"`
}

type RequestSecurityTokenResponse struct {
	Text                         string                       `xml:",chardata"`
	Lifetime                     Lifetime                     `xml:"Lifetime"`
	AppliesTo                    AppliesTo                    `xml:"AppliesTo"`
	RequestedSecurityToken       RequestedSecurityToken       `xml:"RequestedSecurityToken"`
	RequestedAttachedReference   RequestedAttachedReference   `xml:"RequestedAttachedReference"`
	RequestedUnattachedReference RequestedUnattachedReference `xml:"RequestedUnattachedReference"`
	TokenType                    Text                         `xml:"TokenType"`
	RequestType                  Text                         `xml:"RequestType"`
	KeyType                      Text                         `xml:"KeyType"`
}

type Lifetime struct {
	Text    string       `xml:",chardata"`
	Created WSUTimestamp `xml:"Created"`
	Expires WSUTimestamp `xml:"Expires"`
}

type WSUTimestamp struct {
	Text string `xml:",chardata"`
	Wsu  string `xml:"wsu,attr"`
}

type AppliesTo struct {
	Text              string            `xml:",chardata"`
	Wsp               string            `xml:"wsp,attr"`
	EndpointReference EndpointReference `xml:"EndpointReference"`
}

type EndpointReference struct {
	Text    string `xml:",chardata"`
	Wsa     string `xml:"wsa,attr"`
	Address Text   `xml:"Address"`
}

type RequestedSecurityToken struct {
	Text            string    `xml:",chardata"`
	AssertionRawXML string    `xml:",innerxml"`
	Assertion       Assertion `xml:"Assertion"`
}

type Assertion struct {
	XMLName                 xml.Name                // Normally its `xml:"Assertion"`, but I think they want to capture the xmlns
	Text                    string                  `xml:",chardata"`
	MajorVersion            string                  `xml:"MajorVersion,attr"`
	MinorVersion            string                  `xml:"MinorVersion,attr"`
	AssertionID             string                  `xml:"AssertionID,attr"`
	Issuer                  string                  `xml:"Issuer,attr"`
	IssueInstant            string                  `xml:"IssueInstant,attr"`
	Saml                    string                  `xml:"saml,attr"`
	Conditions              Conditions              `xml:"Conditions"`
	AttributeStatement      AttributeStatement      `xml:"AttributeStatement"`
	AuthenticationStatement AuthenticationStatement `xml:"AuthenticationStatement"`
	Signature               Signature               `xml:"Signature"`
}

type Conditions struct {
	Text                         string                       `xml:",chardata"`
	NotBefore                    string                       `xml:"NotBefore,attr"`
	NotOnOrAfter                 string                       `xml:"NotOnOrAfter,attr"`
	AudienceRestrictionCondition AudienceRestrictionCondition `xml:"AudienceRestrictionCondition"`
}

type AudienceRestrictionCondition struct {
	Text     string `xml:",chardata"`
	Audience Text   `xml:"Audience"`
}

type AttributeStatement struct {
	Text      string      `xml:",chardata"`
	Subject   Subject     `xml:"Subject"`
	Attribute []Attribute `xml:"Attribute"`
}

type Subject struct {
	Text                string              `xml:",chardata"`
	NameIdentifier      NameIdentifier      `xml:"NameIdentifier"`
	SubjectConfirmation SubjectConfirmation `xml:"SubjectConfirmation"`
}

type NameIdentifier struct {
	Text   string `xml:",chardata"`
	Format string `xml:"Format,attr"`
}

type SubjectConfirmation struct {
	Text               string `xml:",chardata"`
	ConfirmationMethod Text   `xml:"ConfirmationMethod"`
}

type Attribute struct {
	Text               string `xml:",chardata"`
	AttributeName      string `xml:"AttributeName,attr"`
	AttributeNamespace string `xml:"AttributeNamespace,attr"`
	AttributeValue     Text   `xml:"AttributeValue"`
}

type AuthenticationStatement struct {
	Text                  string  `xml:",chardata"`
	AuthenticationMethod  string  `xml:"AuthenticationMethod,attr"`
	AuthenticationInstant string  `xml:"AuthenticationInstant,attr"`
	Subject               Subject `xml:"Subject"`
}

type Signature struct {
	Text           string     `xml:",chardata"`
	Ds             string     `xml:"ds,attr"`
	SignedInfo     SignedInfo `xml:"SignedInfo"`
	SignatureValue Text       `xml:"SignatureValue"`
	KeyInfo        KeyInfo    `xml:"KeyInfo"`
}

type SignedInfo struct {
	Text                   string    `xml:",chardata"`
	CanonicalizationMethod Method    `xml:"CanonicalizationMethod"`
	SignatureMethod        Method    `xml:"SignatureMethod"`
	Reference              Reference `xml:"Reference"`
}

type Method struct {
	Text      string `xml:",chardata"`
	Algorithm string `xml:"Algorithm,attr"`
}

type Reference struct {
	Text         string     `xml:",chardata"`
	URI          string     `xml:"URI,attr"`
	Transforms   Transforms `xml:"Transforms"`
	DigestMethod Method     `xml:"DigestMethod"`
	DigestValue  Text       `xml:"DigestValue"`
}

type Transforms struct {
	Text      string   `xml:",chardata"`
	Transform []Method `xml:"Transform"`
}

type KeyInfo struct {
	Text     string   `xml:",chardata"`
	Xmlns    string   `xml:"xmlns,attr"`
	X509Data X509Data `xml:"X509Data"`
}

type X509Data struct {
	Text            string `xml:",chardata"`
	X509Certificate Text   `xml:"X509Certificate"`
}

type RequestedAttachedReference struct {
	Text                   string                 `xml:",chardata"`
	SecurityTokenReference SecurityTokenReference `xml:"SecurityTokenReference"`
}

type SecurityTokenReference struct {
	Text          string        `xml:",chardata"`
	TokenType     string        `xml:"TokenType,attr"`
	O             string        `xml:"o,attr"`
	K             string        `xml:"k,attr"`
	KeyIdentifier KeyIdentifier `xml:"KeyIdentifier"`
}

type KeyIdentifier struct {
	Text      string `xml:",chardata"`
	ValueType string `xml:"ValueType,attr"`
}

type RequestedUnattachedReference struct {
	Text                   string                 `xml:",chardata"`
	SecurityTokenReference SecurityTokenReference `xml:"SecurityTokenReference"`
}
