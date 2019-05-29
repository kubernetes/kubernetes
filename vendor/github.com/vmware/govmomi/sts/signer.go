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

package sts

import (
	"bytes"
	"compress/gzip"
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/tls"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	mrand "math/rand"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/vmware/govmomi/sts/internal"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/xml"
)

// Signer implements the soap.Signer interface.
type Signer struct {
	Token       string           // Token is a SAML token
	Certificate *tls.Certificate // Certificate is used to sign requests
	Lifetime    struct {
		Created time.Time
		Expires time.Time
	}
	user  *url.Userinfo // user contains the credentials for bearer token request
	keyID string        // keyID is the Signature UseKey ID, which is referenced in both the soap body and header
}

// signedEnvelope is similar to soap.Envelope, but with namespace and Body as innerxml
type signedEnvelope struct {
	XMLName xml.Name    `xml:"soap:Envelope"`
	NS      string      `xml:"xmlns:soap,attr"`
	Header  soap.Header `xml:"soap:Header"`
	Body    string      `xml:",innerxml"`
}

// newID returns a unique Reference ID, with a leading underscore as required by STS.
func newID() string {
	return "_" + uuid.New().String()
}

func (s *Signer) setTokenReference(info *internal.KeyInfo) error {
	var token struct {
		ID       string `xml:",attr"`     // parse saml2:Assertion ID attribute
		InnerXML string `xml:",innerxml"` // no need to parse the entire token
	}
	if err := xml.Unmarshal([]byte(s.Token), &token); err != nil {
		return err
	}

	info.SecurityTokenReference = &internal.SecurityTokenReference{
		WSSE11:    "http://docs.oasis-open.org/wss/oasis-wss-wssecurity-secext-1.1.xsd",
		TokenType: "http://docs.oasis-open.org/wss/oasis-wss-saml-token-profile-1.1#SAMLV2.0",
		KeyIdentifier: &internal.KeyIdentifier{
			ID:        token.ID,
			ValueType: "http://docs.oasis-open.org/wss/oasis-wss-saml-token-profile-1.1#SAMLID",
		},
	}

	return nil
}

// Sign is a soap.Signer implementation which can be used to sign RequestSecurityToken and LoginByTokenBody requests.
func (s *Signer) Sign(env soap.Envelope) ([]byte, error) {
	var key *rsa.PrivateKey
	hasKey := false
	if s.Certificate != nil {
		key, hasKey = s.Certificate.PrivateKey.(*rsa.PrivateKey)
		if !hasKey {
			return nil, errors.New("sts: rsa.PrivateKey is required")
		}
	}

	created := time.Now().UTC()
	header := &internal.Security{
		WSU:  internal.WSU,
		WSSE: "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd",
		Timestamp: internal.Timestamp{
			NS:      internal.WSU,
			ID:      newID(),
			Created: created.Format(internal.Time),
			Expires: created.Add(time.Minute).Format(internal.Time), // If STS receives this request after this, it is assumed to have expired.
		},
	}
	env.Header.Security = header

	info := internal.KeyInfo{XMLName: xml.Name{Local: "ds:KeyInfo"}}
	var c14n, body string
	type requestToken interface {
		RequestSecurityToken() *internal.RequestSecurityToken
	}

	switch x := env.Body.(type) {
	case requestToken:
		if hasKey {
			// We need c14n for all requests, as its digest is included in the signature and must match on the server side.
			// We need the body in original form when using an ActAs or RenewTarget token, where the token and its signature are embedded in the body.
			req := x.RequestSecurityToken()
			c14n = req.C14N()
			body = req.String()
			id := newID()

			info.SecurityTokenReference = &internal.SecurityTokenReference{
				Reference: &internal.SecurityReference{
					URI:       "#" + id,
					ValueType: "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-x509-token-profile-1.0#X509v3",
				},
			}

			header.BinarySecurityToken = &internal.BinarySecurityToken{
				EncodingType: "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary",
				ValueType:    "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-x509-token-profile-1.0#X509v3",
				ID:           id,
				Value:        base64.StdEncoding.EncodeToString(s.Certificate.Certificate[0]),
			}
		} else {
			header.UsernameToken = &internal.UsernameToken{
				Username: s.user.Username(),
			}
			header.UsernameToken.Password, _ = s.user.Password()
		}
	case *methods.LoginByTokenBody:
		header.Assertion = s.Token

		if hasKey {
			if err := s.setTokenReference(&info); err != nil {
				return nil, err
			}

			c14n = internal.Marshal(x.Req)
		}
	default:
		// We can end up here via ssoadmin.SessionManager.Login().
		// No other known cases where a signed request is needed.
		header.Assertion = s.Token
		if hasKey {
			if err := s.setTokenReference(&info); err != nil {
				return nil, err
			}
			type Req interface {
				C14N() string
			}
			c14n = env.Body.(Req).C14N()
		}
	}

	if !hasKey {
		return xml.Marshal(env) // Bearer token without key to sign
	}

	id := newID()
	tmpl := `<soap:Body xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/" xmlns:wsu="%s" wsu:Id="%s">%s</soap:Body>`
	c14n = fmt.Sprintf(tmpl, internal.WSU, id, c14n)
	if body == "" {
		body = c14n
	} else {
		body = fmt.Sprintf(tmpl, internal.WSU, id, body)
	}

	header.Signature = &internal.Signature{
		XMLName: xml.Name{Local: "ds:Signature"},
		NS:      internal.DSIG,
		ID:      s.keyID,
		KeyInfo: info,
		SignedInfo: internal.SignedInfo{
			XMLName: xml.Name{Local: "ds:SignedInfo"},
			NS:      internal.DSIG,
			CanonicalizationMethod: internal.Method{
				XMLName:   xml.Name{Local: "ds:CanonicalizationMethod"},
				Algorithm: "http://www.w3.org/2001/10/xml-exc-c14n#",
			},
			SignatureMethod: internal.Method{
				XMLName:   xml.Name{Local: "ds:SignatureMethod"},
				Algorithm: internal.SHA256,
			},
			Reference: []internal.Reference{
				internal.NewReference(header.Timestamp.ID, header.Timestamp.C14N()),
				internal.NewReference(id, c14n),
			},
		},
	}

	sum := sha256.Sum256([]byte(header.Signature.SignedInfo.C14N()))
	sig, err := rsa.SignPKCS1v15(rand.Reader, key, crypto.SHA256, sum[:])
	if err != nil {
		return nil, err
	}

	header.Signature.SignatureValue = internal.Value{
		XMLName: xml.Name{Local: "ds:SignatureValue"},
		Value:   base64.StdEncoding.EncodeToString(sig),
	}

	return xml.Marshal(signedEnvelope{
		NS:     "http://schemas.xmlsoap.org/soap/envelope/",
		Header: *env.Header,
		Body:   body,
	})
}

// SignRequest is a rest.Signer implementation which can be used to sign rest.Client.LoginByTokenBody requests.
func (s *Signer) SignRequest(req *http.Request) error {
	type param struct {
		key, val string
	}
	var params []string
	add := func(p param) {
		params = append(params, fmt.Sprintf(`%s="%s"`, p.key, p.val))
	}

	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	if _, err := io.WriteString(gz, s.Token); err != nil {
		return fmt.Errorf("zip token: %s", err)
	}
	if err := gz.Close(); err != nil {
		return fmt.Errorf("zip token: %s", err)
	}
	add(param{
		key: "token",
		val: base64.StdEncoding.EncodeToString(buf.Bytes()),
	})

	if s.Certificate != nil {
		nonce := fmt.Sprintf("%d:%d", time.Now().UnixNano()/1e6, mrand.Int())
		var body []byte
		if req.GetBody != nil {
			r, rerr := req.GetBody()
			if rerr != nil {
				return fmt.Errorf("sts: getting http.Request body: %s", rerr)
			}
			defer r.Close()
			body, rerr = ioutil.ReadAll(r)
			if rerr != nil {
				return fmt.Errorf("sts: reading http.Request body: %s", rerr)
			}
		}
		bhash := sha256.New().Sum(body)

		port := req.URL.Port()
		if port == "" {
			port = "80" // Default port for the "Host" header on the server side
		}

		var buf bytes.Buffer
		msg := []string{
			nonce,
			req.Method,
			req.URL.Path,
			strings.ToLower(req.URL.Hostname()),
			port,
		}
		for i := range msg {
			buf.WriteString(msg[i])
			buf.WriteByte('\n')
		}
		buf.Write(bhash)
		buf.WriteByte('\n')

		sum := sha256.Sum256(buf.Bytes())
		key, ok := s.Certificate.PrivateKey.(*rsa.PrivateKey)
		if !ok {
			return errors.New("sts: rsa.PrivateKey is required to sign http.Request")
		}
		sig, err := rsa.SignPKCS1v15(rand.Reader, key, crypto.SHA256, sum[:])
		if err != nil {
			return err
		}

		add(param{
			key: "signature_alg",
			val: "RSA-SHA256",
		})
		add(param{
			key: "signature",
			val: base64.StdEncoding.EncodeToString(sig),
		})
		add(param{
			key: "nonce",
			val: nonce,
		})
		add(param{
			key: "bodyhash",
			val: base64.StdEncoding.EncodeToString(bhash),
		})
	}

	req.Header.Set("Authorization", fmt.Sprintf("SIGN %s", strings.Join(params, ", ")))

	return nil
}

func (s *Signer) NewRequest() TokenRequest {
	return TokenRequest{
		Token:       s.Token,
		Certificate: s.Certificate,
		Userinfo:    s.user,
		KeyID:       s.keyID,
	}
}
