/*
Copyright 2016 The Kubernetes Authors All rights reserved.
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

package keystone

import (
	"bytes"
	"compress/zlib"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"

	"github.com/fullsailor/pkcs7"
)

const (
	PKI_ASN1_PREFIX = "MII"
	PKIZ_PREFIX     = "PKIZ_"
)

type localValidator struct {
	signingCert *x509.Certificate
}

// newValidator returns a token validator used for local verification
func newLocalValidator(keystoneURL string) (*localValidator, error) {
	resp, err := http.Get(keystoneURL + "/certificates/signing")
	defer resp.Body.Close()
	if err != nil {
		return nil, fmt.Errorf("download signing cert failed: %s", err)
	}
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("download signing cert failed, status code: %d", resp.StatusCode)
	}
	signPEM, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read signing cert failed: %s", err)
	}
	block, _ := pem.Decode(signPEM)
	if block == nil {
		return nil, fmt.Errorf("decode pem failed")
	}
	cert, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		return nil, fmt.Errorf("parse cert failed: %s", err)
	}
	return &localValidator{signingCert: cert}, nil
}

func (v *localValidator) support(token string) bool {
	return strings.HasPrefix(token, PKIZ_PREFIX) || strings.HasPrefix(token, PKI_ASN1_PREFIX)
}

func (v *localValidator) validate(token string) (*response, error) {
	switch {
	case strings.HasPrefix(token, PKIZ_PREFIX):
		token = strings.TrimPrefix(token, PKIZ_PREFIX)
		decompressedToken, err := decompressToken(token)
		if err != nil {
			return nil, fmt.Errorf("validate failed: %s", err)
		}
		token = trimCMSFormat(decompressedToken)
	case strings.HasPrefix(token, PKI_ASN1_PREFIX):
	default:
		return nil, errors.New("unsupported token type")
	}
	decodedToken, err := base64DecodeFromCms(token)
	if err != nil {
		return nil, fmt.Errorf("validate failed: %s", err)
	}
	p7, err := pkcs7.Parse(decodedToken)
	if err != nil {
		return nil, fmt.Errorf("parse token failed: %s", err)
	}
	if len(p7.Signers) != 1 {
		return nil, fmt.Errorf("only one signature expected, but got: %d", len(p7.Signers))
	}
	signer := p7.Signers[0]
	err = v.signingCert.CheckSignature(x509.SHA256WithRSA, p7.Content, signer.EncryptedDigest)
	if err != nil {
		return nil, fmt.Errorf("check signature failed: %s", err)
	}
	r := &response{}
	err = json.Unmarshal(p7.Content, r)
	if err != nil {
		return nil, fmt.Errorf("unmarshal to response failed: %s", err)
	}
	return r, nil
}

func decompressToken(token string) (string, error) {
	decToken, err := base64.URLEncoding.DecodeString(token)
	if err != nil {
		return "", fmt.Errorf("decode token failed: %s", err)
	}

	zr, err := zlib.NewReader(bytes.NewBuffer(decToken))
	if err != nil {
		return "", fmt.Errorf("read token failed: %s", err)
	}
	bb, err := ioutil.ReadAll(zr)
	if err != nil {
		return "", fmt.Errorf("read zlib failed: %s", err)
	}
	return string(bb), nil
}

func base64DecodeFromCms(token string) ([]byte, error) {
	t := strings.Replace(token, "-", "/", -1)
	decToken, err := base64.StdEncoding.DecodeString(t)
	if err != nil {
		return nil, fmt.Errorf("decode token failed: %s", err)
	}
	return decToken, nil
}

// remove the customerized header and footer in PEM token
// -----BEGIN CMS-----
// -----END CMS-----
func trimCMSFormat(token string) string {
	token = strings.Trim(token, "\n")
	l := strings.Index(token, "\n")
	r := strings.LastIndex(token, "\n")
	return token[l:r]
}
