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
	"fmt"
	"strings"

	"github.com/square/go-jose/json"
)

// rawJsonWebSignature represents a raw JWS JSON object. Used for parsing/serializing.
type rawJsonWebSignature struct {
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

// JsonWebSignature represents a signed JWS object after parsing.
type JsonWebSignature struct {
	payload    []byte
	Signatures []Signature
}

// Signature represents a single signature over the JWS payload and protected header.
type Signature struct {
	// Header fields, such as the signature algorithm
	Header JoseHeader

	// The actual signature value
	Signature []byte

	protected *rawHeader
	header    *rawHeader
	original  *rawSignatureInfo
}

// ParseSigned parses a signed message in compact or full serialization format.
func ParseSigned(input string) (*JsonWebSignature, error) {
	input = stripWhitespace(input)
	if strings.HasPrefix(input, "{") {
		return parseSignedFull(input)
	}

	return parseSignedCompact(input)
}

// Get a header value
func (sig Signature) mergedHeaders() rawHeader {
	out := rawHeader{}
	out.merge(sig.protected)
	out.merge(sig.header)
	return out
}

// Compute data to be signed
func (obj JsonWebSignature) computeAuthData(signature *Signature) []byte {
	var serializedProtected string

	if signature.original != nil && signature.original.Protected != nil {
		serializedProtected = signature.original.Protected.base64()
	} else if signature.protected != nil {
		serializedProtected = base64URLEncode(mustSerializeJSON(signature.protected))
	} else {
		serializedProtected = ""
	}

	return []byte(fmt.Sprintf("%s.%s",
		serializedProtected,
		base64URLEncode(obj.payload)))
}

// parseSignedFull parses a message in full format.
func parseSignedFull(input string) (*JsonWebSignature, error) {
	var parsed rawJsonWebSignature
	err := json.Unmarshal([]byte(input), &parsed)
	if err != nil {
		return nil, err
	}

	return parsed.sanitized()
}

// sanitized produces a cleaned-up JWS object from the raw JSON.
func (parsed *rawJsonWebSignature) sanitized() (*JsonWebSignature, error) {
	if parsed.Payload == nil {
		return nil, fmt.Errorf("square/go-jose: missing payload in JWS message")
	}

	obj := &JsonWebSignature{
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

		if parsed.Header != nil && parsed.Header.Nonce != "" {
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

		signature.Header = signature.mergedHeaders().sanitized()
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
		if sig.Header != nil && sig.Header.Nonce != "" {
			return nil, ErrUnprotectedNonce
		}

		obj.Signatures[i].Signature = sig.Signature.bytes()

		// Copy value of sig
		original := sig

		obj.Signatures[i].header = sig.Header
		obj.Signatures[i].original = &original
		obj.Signatures[i].Header = obj.Signatures[i].mergedHeaders().sanitized()
	}

	return obj, nil
}

// parseSignedCompact parses a message in compact format.
func parseSignedCompact(input string) (*JsonWebSignature, error) {
	parts := strings.Split(input, ".")
	if len(parts) != 3 {
		return nil, fmt.Errorf("square/go-jose: compact JWS format must have three parts")
	}

	rawProtected, err := base64URLDecode(parts[0])
	if err != nil {
		return nil, err
	}

	payload, err := base64URLDecode(parts[1])
	if err != nil {
		return nil, err
	}

	signature, err := base64URLDecode(parts[2])
	if err != nil {
		return nil, err
	}

	raw := &rawJsonWebSignature{
		Payload:   newBuffer(payload),
		Protected: newBuffer(rawProtected),
		Signature: newBuffer(signature),
	}
	return raw.sanitized()
}

// CompactSerialize serializes an object using the compact serialization format.
func (obj JsonWebSignature) CompactSerialize() (string, error) {
	if len(obj.Signatures) != 1 || obj.Signatures[0].header != nil || obj.Signatures[0].protected == nil {
		return "", ErrNotSupported
	}

	serializedProtected := mustSerializeJSON(obj.Signatures[0].protected)

	return fmt.Sprintf(
		"%s.%s.%s",
		base64URLEncode(serializedProtected),
		base64URLEncode(obj.payload),
		base64URLEncode(obj.Signatures[0].Signature)), nil
}

// FullSerialize serializes an object using the full JSON serialization format.
func (obj JsonWebSignature) FullSerialize() string {
	raw := rawJsonWebSignature{
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
