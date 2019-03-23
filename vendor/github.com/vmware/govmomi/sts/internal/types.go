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

// The sts/internal package provides the types for invoking the sts.Issue method.
// The sts.Issue and SessionManager LoginByToken methods require an XML signature.
// Unlike the JRE and .NET runtimes, the Go stdlib does not support XML signing.
// We should considering contributing to the goxmldsig package and gosaml2 to meet
// the needs of sts.Issue rather than maintaining this package long term.
// The tricky part of xmldig is the XML canonicalization (C14N), which is responsible
// for most of the make-your-eyes bleed XML formatting in this package.
// C14N is also why some structures use xml.Name without a field tag and methods modify the xml.Name directly,
// though also working around Go's handling of XML namespace prefixes.
// Most of the types in this package were originally generated from the wsdl and hacked up gen/ scripts,
// but have since been modified by hand.

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"log"
	"path"
	"reflect"
	"strings"

	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
	"github.com/vmware/govmomi/vim25/xml"
)

const (
	XSI    = "http://www.w3.org/2001/XMLSchema-instance"
	WSU    = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd"
	DSIG   = "http://www.w3.org/2000/09/xmldsig#"
	SHA256 = "http://www.w3.org/2001/04/xmldsig-more#rsa-sha256"
	Time   = "2006-01-02T15:04:05.000Z"
)

// Security is used as soap.Envelope.Header.Security when signing requests.
type Security struct {
	XMLName             xml.Name `xml:"wsse:Security"`
	WSSE                string   `xml:"xmlns:wsse,attr"`
	WSU                 string   `xml:"xmlns:wsu,attr"`
	Timestamp           Timestamp
	BinarySecurityToken *BinarySecurityToken `xml:",omitempty"`
	UsernameToken       *UsernameToken       `xml:",omitempty"`
	Assertion           string               `xml:",innerxml"`
	Signature           *Signature           `xml:",omitempty"`
}

type Timestamp struct {
	XMLName xml.Name `xml:"wsu:Timestamp"`
	NS      string   `xml:"xmlns:wsu,attr"`
	ID      string   `xml:"wsu:Id,attr"`
	Created string   `xml:"wsu:Created"`
	Expires string   `xml:"wsu:Expires"`
}

func (t *Timestamp) C14N() string {
	return Marshal(t)
}

type BinarySecurityToken struct {
	XMLName      xml.Name `xml:"wsse:BinarySecurityToken"`
	EncodingType string   `xml:"EncodingType,attr"`
	ValueType    string   `xml:"ValueType,attr"`
	ID           string   `xml:"wsu:Id,attr"`
	Value        string   `xml:",chardata"`
}

type UsernameToken struct {
	XMLName  xml.Name `xml:"wsse:UsernameToken"`
	Username string   `xml:"wsse:Username"`
	Password string   `xml:"wsse:Password"`
}

type Signature struct {
	XMLName        xml.Name
	NS             string `xml:"xmlns:ds,attr"`
	ID             string `xml:"Id,attr"`
	SignedInfo     SignedInfo
	SignatureValue Value
	KeyInfo        KeyInfo
}

func (s *Signature) C14N() string {
	return fmt.Sprintf(`<ds:Signature xmlns:ds="%s">%s%s%s</ds:Signature>`,
		DSIG, s.SignedInfo.C14N(), s.SignatureValue.C14N(), s.KeyInfo.C14N())
}

type SignedInfo struct {
	XMLName                xml.Name
	NS                     string `xml:"xmlns:ds,attr,omitempty"`
	CanonicalizationMethod Method
	SignatureMethod        Method
	Reference              []Reference
}

func (s SignedInfo) C14N() string {
	ns := "" // empty in ActAs c14n form for example
	if s.NS != "" {
		ns = fmt.Sprintf(` xmlns:ds="%s"`, s.NS)
	}

	c14n := []string{fmt.Sprintf("<ds:SignedInfo%s>", ns)}
	c14n = append(c14n, s.CanonicalizationMethod.C14N(), s.SignatureMethod.C14N())
	for i := range s.Reference {
		c14n = append(c14n, s.Reference[i].C14N())
	}
	c14n = append(c14n, "</ds:SignedInfo>")

	return strings.Join(c14n, "")
}

type Method struct {
	XMLName   xml.Name
	Algorithm string `xml:",attr"`
}

func (m *Method) C14N() string {
	return mkns("ds", m, &m.XMLName)
}

type Value struct {
	XMLName xml.Name
	Value   string `xml:",innerxml"`
}

func (v *Value) C14N() string {
	return mkns("ds", v, &v.XMLName)
}

type Reference struct {
	XMLName      xml.Name
	URI          string `xml:",attr"`
	Transforms   Transforms
	DigestMethod Method
	DigestValue  Value
}

func (r Reference) C14N() string {
	for i := range r.Transforms.Transform {
		t := &r.Transforms.Transform[i]
		t.XMLName.Local = "ds:Transform"
		t.XMLName.Space = ""

		if t.InclusiveNamespaces != nil {
			name := &t.InclusiveNamespaces.XMLName
			if !strings.HasPrefix(name.Local, "ec:") {
				name.Local = "ec:" + name.Local
				name.Space = ""
			}
			t.InclusiveNamespaces.NS = t.Algorithm
		}
	}

	c14n := []string{
		fmt.Sprintf(`<ds:Reference URI="%s">`, r.URI),
		r.Transforms.C14N(),
		r.DigestMethod.C14N(),
		r.DigestValue.C14N(),
		"</ds:Reference>",
	}

	return strings.Join(c14n, "")
}

func NewReference(id string, val string) Reference {
	sum := sha256.Sum256([]byte(val))

	return Reference{
		XMLName: xml.Name{Local: "ds:Reference"},
		URI:     "#" + id,
		Transforms: Transforms{
			XMLName: xml.Name{Local: "ds:Transforms"},
			Transform: []Transform{
				Transform{
					XMLName:   xml.Name{Local: "ds:Transform"},
					Algorithm: "http://www.w3.org/2001/10/xml-exc-c14n#",
				},
			},
		},
		DigestMethod: Method{
			XMLName:   xml.Name{Local: "ds:DigestMethod"},
			Algorithm: "http://www.w3.org/2001/04/xmlenc#sha256",
		},
		DigestValue: Value{
			XMLName: xml.Name{Local: "ds:DigestValue"},
			Value:   base64.StdEncoding.EncodeToString(sum[:]),
		},
	}
}

type Transforms struct {
	XMLName   xml.Name
	Transform []Transform
}

func (t *Transforms) C14N() string {
	return mkns("ds", t, &t.XMLName)
}

type Transform struct {
	XMLName             xml.Name
	Algorithm           string               `xml:",attr"`
	InclusiveNamespaces *InclusiveNamespaces `xml:",omitempty"`
}

type InclusiveNamespaces struct {
	XMLName    xml.Name
	NS         string `xml:"xmlns:ec,attr,omitempty"`
	PrefixList string `xml:",attr"`
}

type X509Data struct {
	XMLName         xml.Name
	X509Certificate string `xml:",innerxml"`
}

type KeyInfo struct {
	XMLName                xml.Name
	NS                     string                  `xml:"xmlns:ds,attr,omitempty"`
	SecurityTokenReference *SecurityTokenReference `xml:",omitempty"`
	X509Data               *X509Data               `xml:",omitempty"`
}

func (o *KeyInfo) C14N() string {
	names := []*xml.Name{
		&o.XMLName,
	}

	if o.SecurityTokenReference != nil {
		names = append(names, &o.SecurityTokenReference.XMLName)
	}
	if o.X509Data != nil {
		names = append(names, &o.X509Data.XMLName)
	}

	return mkns("ds", o, names...)
}

type SecurityTokenReference struct {
	XMLName       xml.Name           `xml:"wsse:SecurityTokenReference"`
	WSSE11        string             `xml:"xmlns:wsse11,attr,omitempty"`
	TokenType     string             `xml:"wsse11:TokenType,attr,omitempty"`
	Reference     *SecurityReference `xml:",omitempty"`
	KeyIdentifier *KeyIdentifier     `xml:",omitempty"`
}

type SecurityReference struct {
	XMLName   xml.Name `xml:"wsse:Reference"`
	URI       string   `xml:",attr"`
	ValueType string   `xml:",attr"`
}

type KeyIdentifier struct {
	XMLName   xml.Name `xml:"wsse:KeyIdentifier"`
	ID        string   `xml:",innerxml"`
	ValueType string   `xml:",attr"`
}

type Issuer struct {
	XMLName xml.Name
	Format  string `xml:",attr"`
	Value   string `xml:",innerxml"`
}

func (i *Issuer) C14N() string {
	return mkns("saml2", i, &i.XMLName)
}

type Assertion struct {
	XMLName            xml.Name
	ID                 string `xml:",attr"`
	IssueInstant       string `xml:",attr"`
	Version            string `xml:",attr"`
	Issuer             Issuer
	Signature          Signature
	Subject            Subject
	Conditions         Conditions
	AuthnStatement     AuthnStatement
	AttributeStatement AttributeStatement
}

func (a *Assertion) C14N() string {
	start := `<saml2:Assertion xmlns:saml2="%s" ID="%s" IssueInstant="%s" Version="%s">`
	c14n := []string{
		fmt.Sprintf(start, a.XMLName.Space, a.ID, a.IssueInstant, a.Version),
		a.Issuer.C14N(),
		a.Signature.C14N(),
		a.Subject.C14N(),
		a.Conditions.C14N(),
		a.AuthnStatement.C14N(),
		a.AttributeStatement.C14N(),
		`</saml2:Assertion>`,
	}

	return strings.Join(c14n, "")
}

type NameID struct {
	XMLName xml.Name
	Format  string `xml:",attr"`
	ID      string `xml:",innerxml"`
}

type Subject struct {
	XMLName             xml.Name
	NameID              NameID
	SubjectConfirmation SubjectConfirmation
}

func (s *Subject) C14N() string {
	data := &s.SubjectConfirmation.SubjectConfirmationData
	names := []*xml.Name{
		&s.XMLName,
		&s.NameID.XMLName,
		&s.SubjectConfirmation.XMLName,
		&data.XMLName,
	}
	if s.SubjectConfirmation.NameID != nil {
		names = append(names, &s.SubjectConfirmation.NameID.XMLName)
	}
	if data.KeyInfo != nil {
		data.NS = XSI
		data.Type = "saml2:KeyInfoConfirmationDataType"
		data.KeyInfo.XMLName = xml.Name{Local: "ds:KeyInfo"}
		data.KeyInfo.X509Data.XMLName = xml.Name{Local: "ds:X509Data"}
		data.KeyInfo.NS = DSIG
	}
	return mkns("saml2", s, names...)
}

type SubjectConfirmationData struct {
	XMLName      xml.Name
	NS           string `xml:"xmlns:xsi,attr,omitempty"`
	Type         string `xml:"xsi:type,attr,omitempty"`
	NotOnOrAfter string `xml:",attr,omitempty"`
	KeyInfo      *KeyInfo
}

type SubjectConfirmation struct {
	XMLName                 xml.Name
	Method                  string `xml:",attr"`
	NameID                  *NameID
	SubjectConfirmationData SubjectConfirmationData
}

type Condition struct {
	Type string `xml:"xsi:type,attr,omitempty"`
}

func (c *Condition) GetCondition() *Condition {
	return c
}

type BaseCondition interface {
	GetCondition() *Condition
}

func init() {
	types.Add("BaseCondition", reflect.TypeOf((*Condition)(nil)).Elem())
	types.Add("del:DelegationRestrictionType", reflect.TypeOf((*DelegateRestriction)(nil)).Elem())
	types.Add("rsa:RenewRestrictionType", reflect.TypeOf((*RenewRestriction)(nil)).Elem())
}

type Conditions struct {
	XMLName          xml.Name
	NotBefore        string            `xml:",attr"`
	NotOnOrAfter     string            `xml:",attr"`
	ProxyRestriction *ProxyRestriction `xml:",omitempty"`
	Condition        []BaseCondition   `xml:",omitempty"`
}

func (c *Conditions) C14N() string {
	names := []*xml.Name{
		&c.XMLName,
	}

	if c.ProxyRestriction != nil {
		names = append(names, &c.ProxyRestriction.XMLName)
	}

	for i := range c.Condition {
		switch r := c.Condition[i].(type) {
		case *DelegateRestriction:
			names = append(names, &r.XMLName, &r.Delegate.NameID.XMLName)
			r.NS = XSI
			r.Type = "del:DelegationRestrictionType"
			r.Delegate.NS = r.Delegate.XMLName.Space
			r.Delegate.XMLName = xml.Name{Local: "del:Delegate"}
		case *RenewRestriction:
			names = append(names, &r.XMLName)
			r.NS = XSI
			r.Type = "rsa:RenewRestrictionType"
		}
	}

	return mkns("saml2", c, names...)
}

type ProxyRestriction struct {
	XMLName xml.Name
	Count   int32 `xml:",attr"`
}

type RenewRestriction struct {
	XMLName xml.Name
	NS      string `xml:"xmlns:xsi,attr,omitempty"`
	Count   int32  `xml:",attr"`
	Condition
}

type Delegate struct {
	XMLName           xml.Name
	NS                string `xml:"xmlns:del,attr,omitempty"`
	DelegationInstant string `xml:",attr"`
	NameID            NameID
}

type DelegateRestriction struct {
	XMLName xml.Name
	NS      string `xml:"xmlns:xsi,attr,omitempty"`
	Condition
	Delegate Delegate
}

type AuthnStatement struct {
	XMLName      xml.Name
	AuthnInstant string `xml:",attr"`
	AuthnContext struct {
		XMLName              xml.Name
		AuthnContextClassRef struct {
			XMLName xml.Name
			Value   string `xml:",innerxml"`
		}
	}
}

func (a *AuthnStatement) C14N() string {
	return mkns("saml2", a, &a.XMLName, &a.AuthnContext.XMLName, &a.AuthnContext.AuthnContextClassRef.XMLName)
}

type AttributeStatement struct {
	XMLName   xml.Name
	Attribute []Attribute
}

func (a *AttributeStatement) C14N() string {
	c14n := []string{"<saml2:AttributeStatement>"}
	for i := range a.Attribute {
		c14n = append(c14n, a.Attribute[i].C14N())
	}
	c14n = append(c14n, "</saml2:AttributeStatement>")
	return strings.Join(c14n, "")
}

type AttributeValue struct {
	XMLName xml.Name
	Type    string `xml:"type,attr"`
	Value   string `xml:",innerxml"`
}

func (a *AttributeValue) C14N() string {
	return fmt.Sprintf(`<saml2:AttributeValue xmlns:xsi="%s" xsi:type="xs:string">%s</saml2:AttributeValue>`, XSI, a.Value)
}

type Attribute struct {
	XMLName        xml.Name
	FriendlyName   string `xml:",attr"`
	Name           string `xml:",attr"`
	NameFormat     string `xml:",attr"`
	AttributeValue []AttributeValue
}

func (a *Attribute) C14N() string {
	c14n := []string{
		fmt.Sprintf(`<saml2:Attribute FriendlyName="%s" Name="%s" NameFormat="%s">`, a.FriendlyName, a.Name, a.NameFormat),
	}

	for i := range a.AttributeValue {
		c14n = append(c14n, a.AttributeValue[i].C14N())
	}

	c14n = append(c14n, `</saml2:Attribute>`)

	return strings.Join(c14n, "")
}

type Lifetime struct {
	Created string `xml:"http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd Created"`
	Expires string `xml:"http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd Expires"`
}

func (t *Lifetime) C14N() string {
	return fmt.Sprintf(`<Lifetime><wsu:Created>%s</wsu:Created><wsu:Expires>%s</wsu:Expires></Lifetime>`, t.Created, t.Expires)
}

type Renewing struct {
	Allow bool `xml:",attr"`
	OK    bool `xml:",attr"`
}

type UseKey struct {
	Sig string `xml:",attr"`
}

type Target struct {
	Token string `xml:",innerxml"`
}

type RequestSecurityToken struct {
	TokenType          string    `xml:",omitempty"`
	RequestType        string    `xml:",omitempty"`
	Lifetime           *Lifetime `xml:",omitempty"`
	Renewing           *Renewing `xml:",omitempty"`
	Delegatable        bool      `xml:",omitempty"`
	KeyType            string    `xml:",omitempty"`
	SignatureAlgorithm string    `xml:",omitempty"`
	UseKey             *UseKey   `xml:",omitempty"`
	ActAs              *Target   `xml:",omitempty"`
	ValidateTarget     *Target   `xml:",omitempty"`
	RenewTarget        *Target   `xml:",omitempty"`
}

func Unmarshal(data []byte, v interface{}) error {
	dec := xml.NewDecoder(bytes.NewReader(data))
	dec.TypeFunc = types.TypeFunc()
	return dec.Decode(v)
}

// toString returns an XML encoded RequestSecurityToken.
// When c14n is true, returns the canonicalized ActAs.Assertion which is required to sign the Issue request.
// When c14n is false, returns the original content of the ActAs.Assertion.
// The original content must be used within the request Body, as it has its own signature.
func (r *RequestSecurityToken) toString(c14n bool) string {
	actas := ""
	if r.ActAs != nil {
		token := r.ActAs.Token
		if c14n {
			var a Assertion
			err := Unmarshal([]byte(r.ActAs.Token), &a)
			if err != nil {
				log.Printf("decode ActAs: %s", err)
			}
			token = a.C14N()
		}

		actas = fmt.Sprintf(`<wst:ActAs xmlns:wst="http://docs.oasis-open.org/ws-sx/ws-trust/200802">%s</wst:ActAs>`, token)
	}

	body := []string{
		fmt.Sprintf(`<RequestSecurityToken xmlns="http://docs.oasis-open.org/ws-sx/ws-trust/200512">`),
		fmt.Sprintf(`<TokenType>%s</TokenType>`, r.TokenType),
		fmt.Sprintf(`<RequestType>%s</RequestType>`, r.RequestType),
		r.Lifetime.C14N(),
	}

	if r.RenewTarget == nil {
		body = append(body,
			fmt.Sprintf(`<Renewing Allow="%t" OK="%t"></Renewing>`, r.Renewing.Allow, r.Renewing.OK),
			fmt.Sprintf(`<Delegatable>%t</Delegatable>`, r.Delegatable),
			actas,
			fmt.Sprintf(`<KeyType>%s</KeyType>`, r.KeyType),
			fmt.Sprintf(`<SignatureAlgorithm>%s</SignatureAlgorithm>`, r.SignatureAlgorithm),
			fmt.Sprintf(`<UseKey Sig="%s"></UseKey>`, r.UseKey.Sig))
	} else {
		token := r.RenewTarget.Token
		if c14n {
			var a Assertion
			err := Unmarshal([]byte(r.RenewTarget.Token), &a)
			if err != nil {
				log.Printf("decode Renew: %s", err)
			}
			token = a.C14N()
		}

		body = append(body,
			fmt.Sprintf(`<UseKey Sig="%s"></UseKey>`, r.UseKey.Sig),
			fmt.Sprintf(`<RenewTarget>%s</RenewTarget>`, token))
	}

	return strings.Join(append(body, `</RequestSecurityToken>`), "")
}

func (r *RequestSecurityToken) C14N() string {
	return r.toString(true)
}

func (r *RequestSecurityToken) String() string {
	return r.toString(false)
}

type RequestSecurityTokenResponseCollection struct {
	RequestSecurityTokenResponse RequestSecurityTokenResponse
}

type RequestSecurityTokenResponse struct {
	RequestedSecurityToken RequestedSecurityToken
	Lifetime               *Lifetime `xml:"http://docs.oasis-open.org/ws-sx/ws-trust/200512 Lifetime"`
}

type RequestedSecurityToken struct {
	Assertion string `xml:",innerxml"`
}

type RequestSecurityTokenBody struct {
	Req    *RequestSecurityToken                   `xml:"http://docs.oasis-open.org/ws-sx/ws-trust/200512 RequestSecurityToken,omitempty"`
	Res    *RequestSecurityTokenResponseCollection `xml:"http://docs.oasis-open.org/ws-sx/ws-trust/200512 RequestSecurityTokenResponseCollection,omitempty"`
	Fault_ *soap.Fault                             `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *RequestSecurityTokenBody) Fault() *soap.Fault { return b.Fault_ }

func (b *RequestSecurityTokenBody) RequestSecurityToken() *RequestSecurityToken { return b.Req }

func (r *RequestSecurityToken) Action() string {
	kind := path.Base(r.RequestType)
	return "http://docs.oasis-open.org/ws-sx/ws-trust/200512/RST/" + kind
}

func Issue(ctx context.Context, r soap.RoundTripper, req *RequestSecurityToken) (*RequestSecurityTokenResponseCollection, error) {
	var reqBody, resBody RequestSecurityTokenBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type RenewSecurityTokenBody struct {
	Req    *RequestSecurityToken         `xml:"http://docs.oasis-open.org/ws-sx/ws-trust/200512 RequestSecurityToken,omitempty"`
	Res    *RequestSecurityTokenResponse `xml:"http://docs.oasis-open.org/ws-sx/ws-trust/200512 RequestSecurityTokenResponse,omitempty"`
	Fault_ *soap.Fault                   `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *RenewSecurityTokenBody) Fault() *soap.Fault { return b.Fault_ }

func (b *RenewSecurityTokenBody) RequestSecurityToken() *RequestSecurityToken { return b.Req }

func Renew(ctx context.Context, r soap.RoundTripper, req *RequestSecurityToken) (*RequestSecurityTokenResponse, error) {
	var reqBody, resBody RenewSecurityTokenBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

// Marshal panics if xml.Marshal returns an error
func Marshal(val interface{}) string {
	b, err := xml.Marshal(val)
	if err != nil {
		panic(err)
	}
	return string(b)
}

// mkns prepends the given namespace to xml.Name.Local and returns obj encoded as xml.
// Note that the namespace is required when encoding, but the namespace prefix must not be
// present when decoding as Go's decoding does not handle namespace prefix.
func mkns(ns string, obj interface{}, name ...*xml.Name) string {
	ns = ns + ":"
	for i := range name {
		name[i].Space = ""
		if !strings.HasPrefix(name[i].Local, ns) {
			name[i].Local = ns + name[i].Local
		}
	}

	return Marshal(obj)
}
