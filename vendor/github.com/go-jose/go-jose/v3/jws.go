/*-
 * Copyright 2014 Square Inc.
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

package jose

import (
	"bytes"
	"encoding/base64"
	"errors"
	"fmt"
	"strings"

	"github.com/go-jose/go-jose/v3/json"
)

// rawJSONWebSignature represents a raw JWS JSON object. Used for parsing/serializing.
type rawJSONWebSignature struct {
	Payload    *byteBuffer        `json:"payload,omitempty"`
	Signatures []rawSignatureInfo `json:"signatures,omitempty"`
	Protected  *byteBuffer        `json:"protected,omitempty"`
	Header     *rawHeader         `json:"header,omitempty"`
	Signature  *byteBuffer        `json:"signature,omitempty"`
}

// rawSignatureInfo represents a single JWS signature over the JWS payload and protected header.
type rawSignatureInfo struct {
	Protected *byteBuffer `json:"protected,omitempty"`
	Header    *rawHeader  `json:"header,omitempty"`
	Signature *byteBuffer `json:"signature,omitempty"`
}

// JSONWebSignature represents a signed JWS object after parsing.
type JSONWebSignature struct {
	payload []byte
	// Signatures attached to this object (may be more than one for multi-sig).
	// Be careful about accessing these directly, prefer to use Verify() or
	// VerifyMulti() to ensure that the data you're getting is verified.
	Signatures []Signature
}

// Signature represents a single signature over the JWS payload and protected header.
type Signature struct {
	// Merged header fields. Contains both protected and unprotected header
	// values. Prefer using Protected and Unprotected fields instead of this.
	// Values in this header may or may not have been signed and in general
	// should not be trusted.
	Header Header

	// Protected header. Values in this header were signed and
	// will be verified as part of the signature verification process.
	Protected Header

	// Unprotected header. Values in this header were not signed
	// and in general should not be trusted.
	Unprotected Header

	// The actual signature value
	Signature []byte

	protected *rawHeader
	header    *rawHeader
	original  *rawSignatureInfo
}

// ParseSigned parses a signed message in compact or JWS JSON Serialization format.
func ParseSigned(signature string) (*JSONWebSignature, error) {
	signature = stripWhitespace(signature)
	if strings.HasPrefix(signature, "{") {
		return parseSignedFull(signature)
	}

	return parseSignedCompact(signature, nil)
}

// ParseDetached parses a signed message in compact serialization format with detached payload.
func ParseDetached(signature string, payload []byte) (*JSONWebSignature, error) {
	if payload == nil {
		return nil, errors.New("go-jose/go-jose: nil payload")
	}
	return parseSignedCompact(stripWhitespace(signature), payload)
}

// Get a header value
func (sig Signature) mergedHeaders() rawHeader {
	out := rawHeader{}
	out.merge(sig.protected)
	out.merge(sig.header)
	return out
}

// Compute data to be signed
func (obj JSONWebSignature) computeAuthData(payload []byte, signature *Signature) ([]byte, error) {
	var authData bytes.Buffer

	protectedHeader := new(rawHeader)

	if signature.original != nil && signature.original.Protected != nil {
		if err := json.Unmarshal(signature.original.Protected.bytes(), protectedHeader); err != nil {
			return nil, err
		}
		authData.WriteString(signature.original.Protected.base64())
	} else if signature.protected != nil {
		protectedHeader = signature.protected
		authData.WriteString(base64.RawURLEncoding.EncodeToString(mustSerializeJSON(protectedHeader)))
	}

	needsBase64 := true

	if protectedHeader != nil {
		var err error
		if needsBase64, err = protectedHeader.getB64(); err != nil {
			needsBase64 = true
		}
	}

	authData.WriteByte('.')

	if needsBase64 {
		authData.WriteString(base64.RawURLEncoding.EncodeToString(payload))
	} else {
		authData.Write(payload)
	}

	return authData.Bytes(), nil
}

// parseSignedFull parses a message in full format.
func parseSignedFull(input string) (*JSONWebSignature, error) {
	var parsed rawJSONWebSignature
	err := json.Unmarshal([]byte(input), &parsed)
	if err != nil {
		return nil, err
	}

	return parsed.sanitized()
}

// sanitized produces a cleaned-up JWS object from the raw JSON.
func (parsed *rawJSONWebSignature) sanitized() (*JSONWebSignature, error) {
	if parsed.Payload == nil {
		return nil, fmt.Errorf("go-jose/go-jose: missing payload in JWS message")
	}

	obj := &JSONWebSignature{
		payload:    parsed.Payload.bytes(),
		Signatures: make([]Signature, len(parsed.Signatures)),
	}

	if len(parsed.Signatures) == 0 {
		// No signatures array, must be flattened serialization
		signature := Signature{}
		if parsed.Protected != nil && len(parsed.Protected.bytes()) > 0 {
			signature.protected = &rawHeader{}
			err := json.Unmarshal(parsed.Protected.bytes(), signature.protected)
			if err != nil {
				return nil, err
			}
		}

		// Check that there is not a nonce in the unprotected header
		if parsed.Header != nil && parsed.Header.getNonce() != "" {
			return nil, ErrUnprotectedNonce
		}

		signature.header = parsed.Header
		signature.Signature = parsed.Signature.bytes()
		// Make a fake "original" rawSignatureInfo to store the unprocessed
		// Protected header. This is necessary because the Protected header can
		// contain arbitrary fields not registered as part of the spec. See
		// https://tools.ietf.org/html/draft-ietf-jose-json-web-signature-41#section-4
		// If we unmarshal Protected into a rawHeader with its explicit list of fields,
		// we cannot marshal losslessly. So we have to keep around the original bytes.
		// This is used in computeAuthData, which will first attempt to use
		// the original bytes of a protected header, and fall back on marshaling the
		// header struct only if those bytes are not available.
		signature.original = &rawSignatureInfo{
			Protected: parsed.Protected,
			Header:    parsed.Header,
			Signature: parsed.Signature,
		}

		var err error
		signature.Header, err = signature.mergedHeaders().sanitized()
		if err != nil {
			return nil, err
		}

		if signature.header != nil {
			signature.Unprotected, err = signature.header.sanitized()
			if err != nil {
				return nil, err
			}
		}

		if signature.protected != nil {
			signature.Protected, err = signature.protected.sanitized()
			if err != nil {
				return nil, err
			}
		}

		// As per RFC 7515 Section 4.1.3, only public keys are allowed to be embedded.
		jwk := signature.Header.JSONWebKey
		if jwk != nil && (!jwk.Valid() || !jwk.IsPublic()) {
			return nil, errors.New("go-jose/go-jose: invalid embedded jwk, must be public key")
		}

		obj.Signatures = append(obj.Signatures, signature)
	}

	for i, sig := range parsed.Signatures {
		if sig.Protected != nil && len(sig.Protected.bytes()) > 0 {
			obj.Signatures[i].protected = &rawHeader{}
			err := json.Unmarshal(sig.Protected.bytes(), obj.Signatures[i].protected)
			if err != nil {
				return nil, err
			}
		}

		// Check that there is not a nonce in the unprotected header
		if sig.Header != nil && sig.Header.getNonce() != "" {
			return nil, ErrUnprotectedNonce
		}

		var err error
		obj.Signatures[i].Header, err = obj.Signatures[i].mergedHeaders().sanitized()
		if err != nil {
			return nil, err
		}

		if obj.Signatures[i].header != nil {
			obj.Signatures[i].Unprotected, err = obj.Signatures[i].header.sanitized()
			if err != nil {
				return nil, err
			}
		}

		if obj.Signatures[i].protected != nil {
			obj.Signatures[i].Protected, err = obj.Signatures[i].protected.sanitized()
			if err != nil {
				return nil, err
			}
		}

		obj.Signatures[i].Signature = sig.Signature.bytes()

		// As per RFC 7515 Section 4.1.3, only public keys are allowed to be embedded.
		jwk := obj.Signatures[i].Header.JSONWebKey
		if jwk != nil && (!jwk.Valid() || !jwk.IsPublic()) {
			return nil, errors.New("go-jose/go-jose: invalid embedded jwk, must be public key")
		}

		// Copy value of sig
		original := sig

		obj.Signatures[i].header = sig.Header
		obj.Signatures[i].original = &original
	}

	return obj, nil
}

// parseSignedCompact parses a message in compact format.
func parseSignedCompact(input string, payload []byte) (*JSONWebSignature, error) {
	parts := strings.Split(input, ".")
	if len(parts) != 3 {
		return nil, fmt.Errorf("go-jose/go-jose: compact JWS format must have three parts")
	}

	if parts[1] != "" && payload != nil {
		return nil, fmt.Errorf("go-jose/go-jose: payload is not detached")
	}

	rawProtected, err := base64URLDecode(parts[0])
	if err != nil {
		return nil, err
	}

	if payload == nil {
		payload, err = base64URLDecode(parts[1])
		if err != nil {
			return nil, err
		}
	}

	signature, err := base64URLDecode(parts[2])
	if err != nil {
		return nil, err
	}

	raw := &rawJSONWebSignature{
		Payload:   newBuffer(payload),
		Protected: newBuffer(rawProtected),
		Signature: newBuffer(signature),
	}
	return raw.sanitized()
}

func (obj JSONWebSignature) compactSerialize(detached bool) (string, error) {
	if len(obj.Signatures) != 1 || obj.Signatures[0].header != nil || obj.Signatures[0].protected == nil {
		return "", ErrNotSupported
	}

	serializedProtected := base64.RawURLEncoding.EncodeToString(mustSerializeJSON(obj.Signatures[0].protected))
	payload := ""
	signature := base64.RawURLEncoding.EncodeToString(obj.Signatures[0].Signature)

	if !detached {
		payload = base64.RawURLEncoding.EncodeToString(obj.payload)
	}

	return fmt.Sprintf("%s.%s.%s", serializedProtected, payload, signature), nil
}

// CompactSerialize serializes an object using the compact serialization format.
func (obj JSONWebSignature) CompactSerialize() (string, error) {
	return obj.compactSerialize(false)
}

// DetachedCompactSerialize serializes an object using the compact serialization format with detached payload.
func (obj JSONWebSignature) DetachedCompactSerialize() (string, error) {
	return obj.compactSerialize(true)
}

// FullSerialize serializes an object using the full JSON serialization format.
func (obj JSONWebSignature) FullSerialize() string {
	raw := rawJSONWebSignature{
		Payload: newBuffer(obj.payload),
	}

	if len(obj.Signatures) == 1 {
		if obj.Signatures[0].protected != nil {
			serializedProtected := mustSerializeJSON(obj.Signatures[0].protected)
			raw.Protected = newBuffer(serializedProtected)
		}
		raw.Header = obj.Signatures[0].header
		raw.Signature = newBuffer(obj.Signatures[0].Signature)
	} else {
		raw.Signatures = make([]rawSignatureInfo, len(obj.Signatures))
		for i, signature := range obj.Signatures {
			raw.Signatures[i] = rawSignatureInfo{
				Header:    signature.header,
				Signature: newBuffer(signature.Signature),
			}

			if signature.protected != nil {
				raw.Signatures[i].Protected = newBuffer(mustSerializeJSON(signature.protected))
			}
		}
	}

	return string(mustSerializeJSON(raw))
}
