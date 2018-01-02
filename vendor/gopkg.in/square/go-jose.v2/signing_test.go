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
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"fmt"
	"io"
	"reflect"
	"testing"

	"gopkg.in/square/go-jose.v2/json"
)

type staticNonceSource string

func (sns staticNonceSource) Nonce() (string, error) {
	return string(sns), nil
}

func RoundtripJWS(sigAlg SignatureAlgorithm, serializer func(*JSONWebSignature) (string, error), corrupter func(*JSONWebSignature), signingKey interface{}, verificationKey interface{}, nonce string) error {
	opts := &SignerOptions{}
	if nonce != "" {
		opts.NonceSource = staticNonceSource(nonce)
	}

	signer, err := NewSigner(SigningKey{Algorithm: sigAlg, Key: signingKey}, opts)
	if err != nil {
		return fmt.Errorf("error on new signer: %s", err)
	}

	input := []byte("Lorem ipsum dolor sit amet")
	obj, err := signer.Sign(input)
	if err != nil {
		return fmt.Errorf("error on sign: %s", err)
	}

	msg, err := serializer(obj)
	if err != nil {
		return fmt.Errorf("error on serialize: %s", err)
	}

	obj, err = ParseSigned(msg)
	if err != nil {
		return fmt.Errorf("error on parse: %s", err)
	}

	// (Maybe) mangle the object
	corrupter(obj)

	output, err := obj.Verify(verificationKey)
	if err != nil {
		return fmt.Errorf("error on verify: %s", err)
	}

	// Check that verify works with embedded keys (if present)
	for i, sig := range obj.Signatures {
		if sig.Header.JSONWebKey != nil {
			_, err = obj.Verify(sig.Header.JSONWebKey)
			if err != nil {
				return fmt.Errorf("error on verify with embedded key %d: %s", i, err)
			}
		}

		// Check that the nonce correctly round-tripped (if present)
		if sig.Header.Nonce != nonce {
			return fmt.Errorf("Incorrect nonce returned: [%s]", sig.Header.Nonce)
		}
	}

	if bytes.Compare(output, input) != 0 {
		return fmt.Errorf("input/output do not match, got '%s', expected '%s'", output, input)
	}

	return nil
}

func TestRoundtripsJWS(t *testing.T) {
	// Test matrix
	sigAlgs := []SignatureAlgorithm{RS256, RS384, RS512, PS256, PS384, PS512, HS256, HS384, HS512, ES256, ES384, ES512, EdDSA}

	serializers := []func(*JSONWebSignature) (string, error){
		func(obj *JSONWebSignature) (string, error) { return obj.CompactSerialize() },
		func(obj *JSONWebSignature) (string, error) { return obj.FullSerialize(), nil },
	}

	corrupter := func(obj *JSONWebSignature) {}

	for _, alg := range sigAlgs {
		signingKey, verificationKey := GenerateSigningTestKey(alg)

		for i, serializer := range serializers {
			err := RoundtripJWS(alg, serializer, corrupter, signingKey, verificationKey, "test_nonce")
			if err != nil {
				t.Error(err, alg, i)
			}
		}
	}
}

func TestRoundtripsJWSCorruptSignature(t *testing.T) {
	// Test matrix
	sigAlgs := []SignatureAlgorithm{RS256, RS384, RS512, PS256, PS384, PS512, HS256, HS384, HS512, ES256, ES384, ES512, EdDSA}

	serializers := []func(*JSONWebSignature) (string, error){
		func(obj *JSONWebSignature) (string, error) { return obj.CompactSerialize() },
		func(obj *JSONWebSignature) (string, error) { return obj.FullSerialize(), nil },
	}

	corrupters := []func(*JSONWebSignature){
		func(obj *JSONWebSignature) {
			// Changes bytes in signature
			obj.Signatures[0].Signature[10]++
		},
		func(obj *JSONWebSignature) {
			// Set totally invalid signature
			obj.Signatures[0].Signature = []byte("###")
		},
	}

	// Test all different configurations
	for _, alg := range sigAlgs {
		signingKey, verificationKey := GenerateSigningTestKey(alg)

		for i, serializer := range serializers {
			for j, corrupter := range corrupters {
				err := RoundtripJWS(alg, serializer, corrupter, signingKey, verificationKey, "test_nonce")
				if err == nil {
					t.Error("failed to detect corrupt signature", err, alg, i, j)
				}
			}
		}
	}
}

func TestSignerWithBrokenRand(t *testing.T) {
	sigAlgs := []SignatureAlgorithm{RS256, RS384, RS512, PS256, PS384, PS512}

	serializer := func(obj *JSONWebSignature) (string, error) { return obj.CompactSerialize() }
	corrupter := func(obj *JSONWebSignature) {}

	// Break rand reader
	readers := []func() io.Reader{
		// Totally broken
		func() io.Reader { return bytes.NewReader([]byte{}) },
		// Not enough bytes
		func() io.Reader { return io.LimitReader(rand.Reader, 20) },
	}

	defer resetRandReader()

	for _, alg := range sigAlgs {
		signingKey, verificationKey := GenerateSigningTestKey(alg)
		for i, getReader := range readers {
			randReader = getReader()
			err := RoundtripJWS(alg, serializer, corrupter, signingKey, verificationKey, "test_nonce")
			if err == nil {
				t.Error("signer should fail if rand is broken", alg, i)
			}
		}
	}
}

func TestJWSInvalidKey(t *testing.T) {
	signingKey0, verificationKey0 := GenerateSigningTestKey(RS256)
	_, verificationKey1 := GenerateSigningTestKey(ES256)
	_, verificationKey2 := GenerateSigningTestKey(EdDSA)

	signer, err := NewSigner(SigningKey{Algorithm: RS256, Key: signingKey0}, nil)
	if err != nil {
		panic(err)
	}

	input := []byte("Lorem ipsum dolor sit amet")
	obj, err := signer.Sign(input)
	if err != nil {
		panic(err)
	}

	// Must work with correct key
	_, err = obj.Verify(verificationKey0)
	if err != nil {
		t.Error("error on verify", err)
	}

	// Must not work with incorrect key
	_, err = obj.Verify(verificationKey1)
	if err == nil {
		t.Error("verification should fail with incorrect key")
	}

	// Must not work with incorrect key
	_, err = obj.Verify(verificationKey2)
	if err == nil {
		t.Error("verification should fail with incorrect key")
	}

	// Must not work with invalid key
	_, err = obj.Verify("")
	if err == nil {
		t.Error("verification should fail with incorrect key")
	}
}

func TestMultiRecipientJWS(t *testing.T) {
	sharedKey := []byte{
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
	}
	jwkSharedKey := JSONWebKey{
		KeyID: "123",
		Key:   sharedKey,
	}

	signer, err := NewMultiSigner([]SigningKey{
		{RS256, rsaTestKey},
		{HS384, sharedKey},
		{HS512, jwkSharedKey},
	}, nil)
	if err != nil {
		t.Fatal("error creating signer: ", err)
	}

	input := []byte("Lorem ipsum dolor sit amet")
	obj, err := signer.Sign(input)
	if err != nil {
		t.Fatal("error on sign: ", err)
	}

	_, err = obj.CompactSerialize()
	if err == nil {
		t.Fatal("message with multiple recipient was compact serialized")
	}

	msg := obj.FullSerialize()

	obj, err = ParseSigned(msg)
	if err != nil {
		t.Fatal("error on parse: ", err)
	}

	i, _, output, err := obj.VerifyMulti(&rsaTestKey.PublicKey)
	if err != nil {
		t.Fatal("error on verify: ", err)
	}

	if i != 0 {
		t.Fatal("signature index should be 0 for RSA key")
	}

	if bytes.Compare(output, input) != 0 {
		t.Fatal("input/output do not match", output, input)
	}

	i, _, output, err = obj.VerifyMulti(sharedKey)
	if err != nil {
		t.Fatal("error on verify: ", err)
	}

	if i != 1 {
		t.Fatal("signature index should be 1 for EC key")
	}

	if bytes.Compare(output, input) != 0 {
		t.Fatal("input/output do not match", output, input)
	}
}

func GenerateSigningTestKey(sigAlg SignatureAlgorithm) (sig, ver interface{}) {
	switch sigAlg {
	case EdDSA:
		sig = ed25519PrivateKey
		ver = ed25519PublicKey
	case RS256, RS384, RS512, PS256, PS384, PS512:
		sig = rsaTestKey
		ver = &rsaTestKey.PublicKey
	case HS256, HS384, HS512:
		sig, _, _ = randomKeyGenerator{size: 16}.genKey()
		ver = sig
	case ES256:
		key, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
		sig = key
		ver = &key.PublicKey
	case ES384:
		key, _ := ecdsa.GenerateKey(elliptic.P384(), rand.Reader)
		sig = key
		ver = &key.PublicKey
	case ES512:
		key, _ := ecdsa.GenerateKey(elliptic.P521(), rand.Reader)
		sig = key
		ver = &key.PublicKey
	default:
		panic("Must update test case")
	}

	return
}

func TestInvalidSignerAlg(t *testing.T) {
	_, err := NewSigner(SigningKey{"XYZ", nil}, nil)
	if err == nil {
		t.Error("should not accept invalid algorithm")
	}

	_, err = NewSigner(SigningKey{"XYZ", []byte{}}, nil)
	if err == nil {
		t.Error("should not accept invalid algorithm")
	}
}

func TestInvalidJWS(t *testing.T) {
	signer, err := NewSigner(SigningKey{PS256, rsaTestKey}, nil)
	if err != nil {
		panic(err)
	}

	obj, err := signer.Sign([]byte("Lorem ipsum dolor sit amet"))
	obj.Signatures[0].header = &rawHeader{}
	obj.Signatures[0].header.set(headerCritical, []string{"TEST"})

	_, err = obj.Verify(&rsaTestKey.PublicKey)
	if err == nil {
		t.Error("should not verify message with unknown crit header")
	}

	// Try without alg header
	obj.Signatures[0].protected = &rawHeader{}
	obj.Signatures[0].header = &rawHeader{}

	_, err = obj.Verify(&rsaTestKey.PublicKey)
	if err == nil {
		t.Error("should not verify message with missing headers")
	}
}

func TestSignerKid(t *testing.T) {
	kid := "DEADBEEF"
	payload := []byte("Lorem ipsum dolor sit amet")

	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Error("problem generating test signing key", err)
	}

	basejwk := JSONWebKey{Key: key}
	jsonbar, err := basejwk.MarshalJSON()
	if err != nil {
		t.Error("problem marshalling base JWK", err)
	}

	var jsonmsi map[string]interface{}
	err = json.Unmarshal(jsonbar, &jsonmsi)
	if err != nil {
		t.Error("problem unmarshalling base JWK", err)
	}
	jsonmsi["kid"] = kid
	jsonbar2, err := json.Marshal(jsonmsi)
	if err != nil {
		t.Error("problem marshalling kided JWK", err)
	}

	var jwk JSONWebKey
	err = jwk.UnmarshalJSON(jsonbar2)
	if err != nil {
		t.Error("problem unmarshalling kided JWK", err)
	}

	signer, err := NewSigner(SigningKey{ES256, &jwk}, nil)
	if err != nil {
		t.Error("problem creating signer with *JSONWebKey", err)
	}
	signed, err := signer.Sign(payload)

	serialized := signed.FullSerialize()

	parsed, err := ParseSigned(serialized)
	if err != nil {
		t.Error("problem parsing signed object", err)
	}

	if parsed.Signatures[0].Header.KeyID != kid {
		t.Error("KeyID did not survive trip")
	}

	signer, err = NewSigner(SigningKey{ES256, jwk}, nil)
	if err != nil {
		t.Error("problem creating signer with JSONWebKey", err)
	}
}

func TestEmbedJwk(t *testing.T) {
	var payload = []byte("Lorem ipsum dolor sit amet")
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Error("Failed to generate key")
	}

	signer, err := NewSigner(SigningKey{ES256, key}, &SignerOptions{EmbedJWK: true})
	if err != nil {
		t.Error("Failed to create signer")
	}

	object, err := signer.Sign(payload)
	if err != nil {
		t.Error("Failed to sign payload")
	}

	object, err = ParseSigned(object.FullSerialize())
	if err != nil {
		t.Error("Failed to parse jws")
	}

	jwk, err := object.Signatures[0].protected.getJWK()
	if jwk == nil || err != nil {
		t.Error("JWK isn't set in protected header")
	}

	// This time, sign and do not embed JWK in message
	signer, err = NewSigner(SigningKey{ES256, key}, &SignerOptions{EmbedJWK: false})

	object, err = signer.Sign(payload)
	if err != nil {
		t.Error("Failed to sign payload")
	}

	object, err = ParseSigned(object.FullSerialize())
	if err != nil {
		t.Error("Failed to parse jws")
	}

	jwk2, err := object.Signatures[0].protected.getJWK()
	if err != nil {
		t.Error("JWK is invalid in protected header")
	}
	if jwk2 != nil {
		t.Error("JWK is set in protected header")
	}
}

func TestSignerOptionsEd(t *testing.T) {
	key, _ := GenerateSigningTestKey(EdDSA)
	opts := &SignerOptions{
		EmbedJWK: true,
	}
	opts.WithContentType("JWT")
	opts.WithType("JWT")
	sig, err := NewSigner(SigningKey{EdDSA, key}, opts)
	if err != nil {
		t.Error("Failed to create signer")
	}

	if !reflect.DeepEqual(*opts, sig.Options()) {
		t.Error("Signer options do not match")
	}
}

func TestSignerOptions(t *testing.T) {
	key, _ := GenerateSigningTestKey(HS256)
	opts := &SignerOptions{
		EmbedJWK: true,
	}
	opts.WithContentType("JWT")
	opts.WithType("JWT")
	sig, err := NewSigner(SigningKey{HS256, key}, opts)
	if err != nil {
		t.Error("Failed to create signer")
	}

	if !reflect.DeepEqual(*opts, sig.Options()) {
		t.Error("Signer options do not match")
	}
}

// Test that extra headers are generated and parsed in a round trip.
func TestSignerExtraHeaderInclusion(t *testing.T) {
	var payload = []byte("Lorem ipsum dolor sit amet")

	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Error("Failed to generate key")
	}

	signer, err := NewSigner(SigningKey{ES256, key}, (&SignerOptions{}).WithContentType("foo/bar").WithHeader(HeaderKey("myCustomHeader"), "xyz"))
	if err != nil {
		t.Error("Failed to create signer", err)
	}

	object, err := signer.Sign(payload)
	if err != nil {
		t.Error("Failed to sign payload")
	}

	object, err = ParseSigned(object.FullSerialize())
	if err != nil {
		t.Error("Failed to parse jws")
	}

	correct := map[HeaderKey]interface{}{
		HeaderContentType:           "foo/bar",
		HeaderKey("myCustomHeader"): "xyz",
	}

	if !reflect.DeepEqual(object.Signatures[0].Header.ExtraHeaders, correct) {
		t.Errorf("Mismatch in extra headers: %#v", object.Signatures[0].Header.ExtraHeaders)
	}
}
