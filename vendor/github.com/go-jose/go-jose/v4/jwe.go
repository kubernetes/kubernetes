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
	"encoding/base64"
	"errors"
	"fmt"
	"strings"

	"github.com/go-jose/go-jose/v4/json"
)

// MaxRecipients is the maximum number of recipients of a JWE at parse time.
// Higher values mean more potential work done across untrusted input.
//
// If modifying MaxRecipients, do it before any parsing occurs.
var MaxRecipients = 32

// rawJSONWebEncryption represents a raw JWE JSON object. Used for parsing/serializing.
type rawJSONWebEncryption struct {
	Protected    *byteBuffer        `json:"protected,omitempty"`
	Unprotected  *rawHeader         `json:"unprotected,omitempty"`
	Header       *rawHeader         `json:"header,omitempty"`
	Recipients   []rawRecipientInfo `json:"recipients,omitempty"`
	Aad          *byteBuffer        `json:"aad,omitempty"`
	EncryptedKey *byteBuffer        `json:"encrypted_key,omitempty"`
	Iv           *byteBuffer        `json:"iv,omitempty"`
	Ciphertext   *byteBuffer        `json:"ciphertext,omitempty"`
	Tag          *byteBuffer        `json:"tag,omitempty"`
}

// rawRecipientInfo represents a raw JWE Per-Recipient header JSON object. Used for parsing/serializing.
type rawRecipientInfo struct {
	Header       *rawHeader `json:"header,omitempty"`
	EncryptedKey string     `json:"encrypted_key,omitempty"`
}

// JSONWebEncryption represents an encrypted JWE object after parsing.
type JSONWebEncryption struct {
	Header                   Header
	protected, unprotected   *rawHeader
	recipients               []recipientInfo
	aad, iv, ciphertext, tag []byte
	original                 *rawJSONWebEncryption
}

// recipientInfo represents a raw JWE Per-Recipient header JSON object after parsing.
type recipientInfo struct {
	header       *rawHeader
	encryptedKey []byte
}

// GetAuthData retrieves the (optional) authenticated data attached to the object.
func (obj JSONWebEncryption) GetAuthData() []byte {
	if obj.aad != nil {
		out := make([]byte, len(obj.aad))
		copy(out, obj.aad)
		return out
	}

	return nil
}

// Get the merged header values
func (obj JSONWebEncryption) mergedHeaders(recipient *recipientInfo) rawHeader {
	out := rawHeader{}
	out.merge(obj.protected)
	out.merge(obj.unprotected)

	if recipient != nil {
		out.merge(recipient.header)
	}

	return out
}

// Get the additional authenticated data from a JWE object.
func (obj JSONWebEncryption) computeAuthData() ([]byte, error) {
	var protected string

	switch {
	case obj.original != nil && obj.original.Protected != nil:
		protected = obj.original.Protected.base64()
	case obj.protected != nil:
		serializedProtected, err := json.Marshal(obj.protected)
		if err != nil {
			return nil, err
		}
		protected = base64.RawURLEncoding.EncodeToString(serializedProtected)
	default:
		protected = ""
	}

	output := []byte(protected)
	if obj.aad != nil {
		output = append(output, '.')
		output = append(output, []byte(base64.RawURLEncoding.EncodeToString(obj.aad))...)
	}

	return output, nil
}

func containsKeyAlgorithm(haystack []KeyAlgorithm, needle KeyAlgorithm) bool {
	for _, algorithm := range haystack {
		if algorithm == needle {
			return true
		}
	}
	return false
}

func containsContentEncryption(haystack []ContentEncryption, needle ContentEncryption) bool {
	for _, algorithm := range haystack {
		if algorithm == needle {
			return true
		}
	}
	return false
}

// ParseEncrypted parses an encrypted message in JWE Compact or JWE JSON Serialization.
//
// https://datatracker.ietf.org/doc/html/rfc7516#section-3.1
// https://datatracker.ietf.org/doc/html/rfc7516#section-3.2
//
// The keyAlgorithms and contentEncryption parameters contain the allowable algorithms in the "alg" and "enc"
// header parameters respectively. They must be non-empty. Non-matching algorithms cause a parse error.
//
// Deprecated: use ParseEncryptedJSON or ParseEncryptedCompact depending on which serialization you expect.
func ParseEncrypted(input string,
	keyEncryptionAlgorithms []KeyAlgorithm,
	contentEncryption []ContentEncryption,
) (*JSONWebEncryption, error) {
	input = stripWhitespace(input)
	if strings.HasPrefix(input, "{") {
		return ParseEncryptedJSON(input, keyEncryptionAlgorithms, contentEncryption)
	}

	return ParseEncryptedCompact(input, keyEncryptionAlgorithms, contentEncryption)
}

// ParseEncryptedJSON parses a message in JWE JSON Serialization. Messages with
// more than [MaxRecipients] recipients are rejected.
//
// https://datatracker.ietf.org/doc/html/rfc7516#section-3.2
//
// The keyAlgorithms and contentEncryption parameters contain the allowable algorithms in the "alg" and "enc"
// header parameters respectively. They must be non-empty. Any non-allowed algorithms will cause parsing to fail.
func ParseEncryptedJSON(
	input string,
	keyEncryptionAlgorithms []KeyAlgorithm,
	contentEncryption []ContentEncryption,
) (*JSONWebEncryption, error) {
	if len(input) == 0 {
		return nil, errEmptyInput
	}

	var parsed rawJSONWebEncryption
	err := json.Unmarshal([]byte(input), &parsed)
	if err != nil {
		return nil, err
	}

	return parsed.sanitized(keyEncryptionAlgorithms, contentEncryption)
}

// sanitized produces a cleaned-up JWE object from the raw JSON.
func (parsed *rawJSONWebEncryption) sanitized(
	keyEncryptionAlgorithms []KeyAlgorithm,
	contentEncryption []ContentEncryption,
) (*JSONWebEncryption, error) {
	if len(keyEncryptionAlgorithms) == 0 {
		return nil, errors.New("go-jose/go-jose: no key algorithms provided")
	}
	if len(contentEncryption) == 0 {
		return nil, errors.New("go-jose/go-jose: no content encryption algorithms provided")
	}
	if len(parsed.Recipients) > MaxRecipients {
		return nil, fmt.Errorf("go-jose/go-jose: too many recipients in JWE; maximum is %d", MaxRecipients)
	}

	obj := &JSONWebEncryption{
		original:    parsed,
		unprotected: parsed.Unprotected,
	}

	// Check that there is not a nonce in the unprotected headers
	if parsed.Unprotected != nil {
		if nonce := parsed.Unprotected.getNonce(); nonce != "" {
			return nil, ErrUnprotectedNonce
		}
	}
	if parsed.Header != nil {
		if nonce := parsed.Header.getNonce(); nonce != "" {
			return nil, ErrUnprotectedNonce
		}
	}

	protected, err := parseProtectedHeader(parsed.Protected)
	if err != nil {
		return nil, err
	}
	obj.protected = protected

	if len(parsed.Recipients) == 0 {
		obj.recipients = []recipientInfo{
			{
				header:       parsed.Header,
				encryptedKey: parsed.EncryptedKey.bytes(),
			},
		}
	} else {
		obj.recipients = make([]recipientInfo, len(parsed.Recipients))
		for r := range parsed.Recipients {
			encryptedKey, err := base64.RawURLEncoding.DecodeString(parsed.Recipients[r].EncryptedKey)
			if err != nil {
				return nil, err
			}

			// Check that there is not a nonce in the unprotected header
			if parsed.Recipients[r].Header != nil && parsed.Recipients[r].Header.getNonce() != "" {
				return nil, ErrUnprotectedNonce
			}

			obj.recipients[r].header = parsed.Recipients[r].Header
			obj.recipients[r].encryptedKey = encryptedKey
		}
	}

	for i, recipient := range obj.recipients {
		headers := obj.mergedHeaders(&recipient)
		if headers.getAlgorithm() == "" {
			return nil, fmt.Errorf(`go-jose/go-jose: recipient %d: missing header "alg"`, i)
		}
		if headers.getEncryption() == "" {
			return nil, fmt.Errorf(`go-jose/go-jose: recipient %d: missing header "enc"`, i)
		}
		err := validateAlgEnc(headers, keyEncryptionAlgorithms, contentEncryption)
		if err != nil {
			return nil, fmt.Errorf("go-jose/go-jose: recipient %d: %s", i, err)
		}

	}

	if obj.protected != nil {
		err := validateAlgEnc(*obj.protected, keyEncryptionAlgorithms, contentEncryption)
		if err != nil {
			return nil, fmt.Errorf("go-jose/go-jose: protected header: %s", err)
		}
	}
	if obj.unprotected != nil {
		err := validateAlgEnc(*obj.unprotected, keyEncryptionAlgorithms, contentEncryption)
		if err != nil {
			return nil, fmt.Errorf("go-jose/go-jose: unprotected header: %s", err)
		}
	}

	// Note: this must be called _after_ we parse the protected header,
	// otherwise fields from the protected header will not get picked up.
	mergedHeaders := obj.mergedHeaders(nil)
	obj.Header, err = mergedHeaders.sanitized()
	if err != nil {
		return nil, fmt.Errorf("go-jose/go-jose: cannot sanitize merged headers: %w", err)
	}

	obj.iv = parsed.Iv.bytes()
	obj.ciphertext = parsed.Ciphertext.bytes()
	obj.tag = parsed.Tag.bytes()
	obj.aad = parsed.Aad.bytes()

	return obj, nil
}

func validateAlgEnc(headers rawHeader, keyAlgorithms []KeyAlgorithm, contentEncryption []ContentEncryption) error {
	alg := headers.getAlgorithm()
	enc := headers.getEncryption()
	if alg != "" && !containsKeyAlgorithm(keyAlgorithms, alg) {
		return fmt.Errorf("unexpected key algorithm %q; expected %q", alg, keyAlgorithms)
	}
	if enc != "" && !containsContentEncryption(contentEncryption, enc) {
		return fmt.Errorf("unexpected content encryption algorithm %q; expected %q", enc, contentEncryption)
	}
	return nil
}

// ParseEncryptedCompact parses a message in JWE Compact Serialization.
//
// https://datatracker.ietf.org/doc/html/rfc7516#section-3.1
//
// The keyAlgorithms and contentEncryption parameters contain the allowable algorithms in the "alg" and "enc"
// header parameters respectively. They must be non-empty. Any non-allowed algorithms will cause parsing to fail.
func ParseEncryptedCompact(
	input string,
	keyAlgorithms []KeyAlgorithm,
	contentEncryption []ContentEncryption,
) (*JSONWebEncryption, error) {
	var parts [5]string
	var ok bool

	if len(input) == 0 {
		return nil, errEmptyInput
	}

	for i := range 4 {
		parts[i], input, ok = strings.Cut(input, ".")
		if !ok {
			return nil, errors.New("go-jose/go-jose: compact JWE format must have five parts")
		}
	}
	// Validate that the last part does not contain more dots
	if strings.ContainsRune(input, '.') {
		return nil, errors.New("go-jose/go-jose: compact JWE format must have five parts")
	}
	parts[4] = input

	rawProtected, err := base64.RawURLEncoding.DecodeString(parts[0])
	if err != nil {
		return nil, err
	}

	encryptedKey, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return nil, err
	}

	iv, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil {
		return nil, err
	}

	ciphertext, err := base64.RawURLEncoding.DecodeString(parts[3])
	if err != nil {
		return nil, err
	}

	tag, err := base64.RawURLEncoding.DecodeString(parts[4])
	if err != nil {
		return nil, err
	}

	raw := &rawJSONWebEncryption{
		Protected:    newBuffer(rawProtected),
		EncryptedKey: newBuffer(encryptedKey),
		Iv:           newBuffer(iv),
		Ciphertext:   newBuffer(ciphertext),
		Tag:          newBuffer(tag),
	}

	return raw.sanitized(keyAlgorithms, contentEncryption)
}

// CompactSerialize serializes an object using the compact serialization format.
func (obj JSONWebEncryption) CompactSerialize() (string, error) {
	if len(obj.recipients) != 1 || obj.unprotected != nil ||
		obj.protected == nil || obj.recipients[0].header != nil {
		return "", ErrNotSupported
	}

	serializedProtected, err := json.Marshal(obj.protected)
	if err != nil {
		return "", err
	}

	return base64JoinWithDots(
		serializedProtected,
		obj.recipients[0].encryptedKey,
		obj.iv,
		obj.ciphertext,
		obj.tag,
	), nil
}

// FullSerialize serializes an object using the full JSON serialization format.
//
// It returns "" if obj cannot be serialized, for example if it has no recipients.
func (obj JSONWebEncryption) FullSerialize() string {
	if len(obj.recipients) == 0 {
		return ""
	}

	raw := rawJSONWebEncryption{
		Unprotected:  obj.unprotected,
		Iv:           newBuffer(obj.iv),
		Ciphertext:   newBuffer(obj.ciphertext),
		EncryptedKey: newBuffer(obj.recipients[0].encryptedKey),
		Tag:          newBuffer(obj.tag),
		Aad:          newBuffer(obj.aad),
		Recipients:   []rawRecipientInfo{},
	}

	if len(obj.recipients) > 1 {
		for _, recipient := range obj.recipients {
			info := rawRecipientInfo{
				Header:       recipient.header,
				EncryptedKey: base64.RawURLEncoding.EncodeToString(recipient.encryptedKey),
			}
			raw.Recipients = append(raw.Recipients, info)
		}
	} else {
		// Use flattened serialization
		raw.Header = obj.recipients[0].header
		raw.EncryptedKey = newBuffer(obj.recipients[0].encryptedKey)
	}

	if obj.protected != nil {
		serializedProtected, err := json.Marshal(obj.protected)
		if err != nil {
			return ""
		}
		raw.Protected = newBuffer(serializedProtected)
	}

	out, err := json.Marshal(raw)
	if err != nil {
		return ""
	}
	return string(out)
}
