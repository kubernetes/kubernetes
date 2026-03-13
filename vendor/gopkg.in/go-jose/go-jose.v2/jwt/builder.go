/*-
 * Copyright 2016 Zbigniew Mandziejewicz
 * Copyright 2016 Square, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package jwt

import (
	"bytes"
	"reflect"

	"gopkg.in/go-jose/go-jose.v2/json"

	"gopkg.in/go-jose/go-jose.v2"
)

// Builder is a utility for making JSON Web Tokens. Calls can be chained, and
// errors are accumulated until the final call to CompactSerialize/FullSerialize.
type Builder interface {
	// Claims encodes claims into JWE/JWS form. Multiple calls will merge claims
	// into single JSON object. If you are passing private claims, make sure to set
	// struct field tags to specify the name for the JSON key to be used when
	// serializing.
	Claims(i interface{}) Builder
	// Token builds a JSONWebToken from provided data.
	Token() (*JSONWebToken, error)
	// FullSerialize serializes a token using the full serialization format.
	FullSerialize() (string, error)
	// CompactSerialize serializes a token using the compact serialization format.
	CompactSerialize() (string, error)
}

// NestedBuilder is a utility for making Signed-Then-Encrypted JSON Web Tokens.
// Calls can be chained, and errors are accumulated until final call to
// CompactSerialize/FullSerialize.
type NestedBuilder interface {
	// Claims encodes claims into JWE/JWS form. Multiple calls will merge claims
	// into single JSON object. If you are passing private claims, make sure to set
	// struct field tags to specify the name for the JSON key to be used when
	// serializing.
	Claims(i interface{}) NestedBuilder
	// Token builds a NestedJSONWebToken from provided data.
	Token() (*NestedJSONWebToken, error)
	// FullSerialize serializes a token using the full serialization format.
	FullSerialize() (string, error)
	// CompactSerialize serializes a token using the compact serialization format.
	CompactSerialize() (string, error)
}

type builder struct {
	payload map[string]interface{}
	err     error
}

type signedBuilder struct {
	builder
	sig jose.Signer
}

type encryptedBuilder struct {
	builder
	enc jose.Encrypter
}

type nestedBuilder struct {
	builder
	sig jose.Signer
	enc jose.Encrypter
}

// Signed creates builder for signed tokens.
func Signed(sig jose.Signer) Builder {
	return &signedBuilder{
		sig: sig,
	}
}

// Encrypted creates builder for encrypted tokens.
func Encrypted(enc jose.Encrypter) Builder {
	return &encryptedBuilder{
		enc: enc,
	}
}

// SignedAndEncrypted creates builder for signed-then-encrypted tokens.
// ErrInvalidContentType will be returned if encrypter doesn't have JWT content type.
func SignedAndEncrypted(sig jose.Signer, enc jose.Encrypter) NestedBuilder {
	if contentType, _ := enc.Options().ExtraHeaders[jose.HeaderContentType].(jose.ContentType); contentType != "JWT" {
		return &nestedBuilder{
			builder: builder{
				err: ErrInvalidContentType,
			},
		}
	}
	return &nestedBuilder{
		sig: sig,
		enc: enc,
	}
}

func (b builder) claims(i interface{}) builder {
	if b.err != nil {
		return b
	}

	m, ok := i.(map[string]interface{})
	switch {
	case ok:
		return b.merge(m)
	case reflect.Indirect(reflect.ValueOf(i)).Kind() == reflect.Struct:
		m, err := normalize(i)
		if err != nil {
			return builder{
				err: err,
			}
		}
		return b.merge(m)
	default:
		return builder{
			err: ErrInvalidClaims,
		}
	}
}

func normalize(i interface{}) (map[string]interface{}, error) {
	m := make(map[string]interface{})

	raw, err := json.Marshal(i)
	if err != nil {
		return nil, err
	}

	d := json.NewDecoder(bytes.NewReader(raw))
	d.SetNumberType(json.UnmarshalJSONNumber)

	if err := d.Decode(&m); err != nil {
		return nil, err
	}

	return m, nil
}

func (b *builder) merge(m map[string]interface{}) builder {
	p := make(map[string]interface{})
	for k, v := range b.payload {
		p[k] = v
	}
	for k, v := range m {
		p[k] = v
	}

	return builder{
		payload: p,
	}
}

func (b *builder) token(p func(interface{}) ([]byte, error), h []jose.Header) (*JSONWebToken, error) {
	return &JSONWebToken{
		payload: p,
		Headers: h,
	}, nil
}

func (b *signedBuilder) Claims(i interface{}) Builder {
	return &signedBuilder{
		builder: b.builder.claims(i),
		sig:     b.sig,
	}
}

func (b *signedBuilder) Token() (*JSONWebToken, error) {
	sig, err := b.sign()
	if err != nil {
		return nil, err
	}

	h := make([]jose.Header, len(sig.Signatures))
	for i, v := range sig.Signatures {
		h[i] = v.Header
	}

	return b.builder.token(sig.Verify, h)
}

func (b *signedBuilder) CompactSerialize() (string, error) {
	sig, err := b.sign()
	if err != nil {
		return "", err
	}

	return sig.CompactSerialize()
}

func (b *signedBuilder) FullSerialize() (string, error) {
	sig, err := b.sign()
	if err != nil {
		return "", err
	}

	return sig.FullSerialize(), nil
}

func (b *signedBuilder) sign() (*jose.JSONWebSignature, error) {
	if b.err != nil {
		return nil, b.err
	}

	p, err := json.Marshal(b.payload)
	if err != nil {
		return nil, err
	}

	return b.sig.Sign(p)
}

func (b *encryptedBuilder) Claims(i interface{}) Builder {
	return &encryptedBuilder{
		builder: b.builder.claims(i),
		enc:     b.enc,
	}
}

func (b *encryptedBuilder) CompactSerialize() (string, error) {
	enc, err := b.encrypt()
	if err != nil {
		return "", err
	}

	return enc.CompactSerialize()
}

func (b *encryptedBuilder) FullSerialize() (string, error) {
	enc, err := b.encrypt()
	if err != nil {
		return "", err
	}

	return enc.FullSerialize(), nil
}

func (b *encryptedBuilder) Token() (*JSONWebToken, error) {
	enc, err := b.encrypt()
	if err != nil {
		return nil, err
	}

	return b.builder.token(enc.Decrypt, []jose.Header{enc.Header})
}

func (b *encryptedBuilder) encrypt() (*jose.JSONWebEncryption, error) {
	if b.err != nil {
		return nil, b.err
	}

	p, err := json.Marshal(b.payload)
	if err != nil {
		return nil, err
	}

	return b.enc.Encrypt(p)
}

func (b *nestedBuilder) Claims(i interface{}) NestedBuilder {
	return &nestedBuilder{
		builder: b.builder.claims(i),
		sig:     b.sig,
		enc:     b.enc,
	}
}

func (b *nestedBuilder) Token() (*NestedJSONWebToken, error) {
	enc, err := b.signAndEncrypt()
	if err != nil {
		return nil, err
	}

	return &NestedJSONWebToken{
		enc:     enc,
		Headers: []jose.Header{enc.Header},
	}, nil
}

func (b *nestedBuilder) CompactSerialize() (string, error) {
	enc, err := b.signAndEncrypt()
	if err != nil {
		return "", err
	}

	return enc.CompactSerialize()
}

func (b *nestedBuilder) FullSerialize() (string, error) {
	enc, err := b.signAndEncrypt()
	if err != nil {
		return "", err
	}

	return enc.FullSerialize(), nil
}

func (b *nestedBuilder) signAndEncrypt() (*jose.JSONWebEncryption, error) {
	if b.err != nil {
		return nil, b.err
	}

	p, err := json.Marshal(b.payload)
	if err != nil {
		return nil, err
	}

	sig, err := b.sig.Sign(p)
	if err != nil {
		return nil, err
	}

	p2, err := sig.CompactSerialize()
	if err != nil {
		return nil, err
	}

	return b.enc.Encrypt([]byte(p2))
}
