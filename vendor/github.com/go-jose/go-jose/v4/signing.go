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
	"crypto/ed25519"
	"crypto/rsa"
	"encoding/base64"
	"errors"
	"fmt"

	"github.com/go-jose/go-jose/v4/json"
)

// NonceSource represents a source of random nonces to go into JWS objects
type NonceSource interface {
	Nonce() (string, error)
}

// Signer represents a signer which takes a payload and produces a signed JWS object.
type Signer interface {
	Sign(payload []byte) (*JSONWebSignature, error)
	Options() SignerOptions
}

// SigningKey represents an algorithm/key used to sign a message.
//
// Key must have one of these types:
//   - ed25519.PrivateKey
//   - *ecdsa.PrivateKey
//   - *rsa.PrivateKey
//   - *JSONWebKey
//   - JSONWebKey
//   - []byte (an HMAC key)
//   - Any type that satisfies the OpaqueSigner interface
//
// If the key is an HMAC key, it must have at least as many bytes as the relevant hash output:
//   - HS256: 32 bytes
//   - HS384: 48 bytes
//   - HS512: 64 bytes
type SigningKey struct {
	Algorithm SignatureAlgorithm
	Key       interface{}
}

// SignerOptions represents options that can be set when creating signers.
type SignerOptions struct {
	NonceSource NonceSource
	EmbedJWK    bool

	// Optional map of additional keys to be inserted into the protected header
	// of a JWS object. Some specifications which make use of JWS like to insert
	// additional values here.
	//
	// Values will be serialized by [json.Marshal] and must be valid inputs to
	// that function.
	//
	// [json.Marshal]: https://pkg.go.dev/encoding/json#Marshal
	ExtraHeaders map[HeaderKey]interface{}
}

// WithHeader adds an arbitrary value to the ExtraHeaders map, initializing it
// if necessary, and returns the updated SignerOptions.
//
// The v argument will be serialized by [json.Marshal] and must be a valid
// input to that function.
//
// [json.Marshal]: https://pkg.go.dev/encoding/json#Marshal
func (so *SignerOptions) WithHeader(k HeaderKey, v interface{}) *SignerOptions {
	if so.ExtraHeaders == nil {
		so.ExtraHeaders = map[HeaderKey]interface{}{}
	}
	so.ExtraHeaders[k] = v
	return so
}

// WithContentType adds a content type ("cty") header and returns the updated
// SignerOptions.
func (so *SignerOptions) WithContentType(contentType ContentType) *SignerOptions {
	return so.WithHeader(HeaderContentType, contentType)
}

// WithType adds a type ("typ") header and returns the updated SignerOptions.
func (so *SignerOptions) WithType(typ ContentType) *SignerOptions {
	return so.WithHeader(HeaderType, typ)
}

// WithCritical adds the given names to the critical ("crit") header and returns
// the updated SignerOptions.
func (so *SignerOptions) WithCritical(names ...string) *SignerOptions {
	if so.ExtraHeaders[headerCritical] == nil {
		so.WithHeader(headerCritical, make([]string, 0, len(names)))
	}
	crit := so.ExtraHeaders[headerCritical].([]string)
	so.ExtraHeaders[headerCritical] = append(crit, names...)
	return so
}

// WithBase64 adds a base64url-encode payload ("b64") header and returns the updated
// SignerOptions. When the "b64" value is "false", the payload is not base64 encoded.
func (so *SignerOptions) WithBase64(b64 bool) *SignerOptions {
	if !b64 {
		so.WithHeader(headerB64, b64)
		so.WithCritical(headerB64)
	}
	return so
}

type payloadSigner interface {
	signPayload(payload []byte, alg SignatureAlgorithm) (Signature, error)
}

type payloadVerifier interface {
	verifyPayload(payload []byte, signature []byte, alg SignatureAlgorithm) error
}

type genericSigner struct {
	recipients   []recipientSigInfo
	nonceSource  NonceSource
	embedJWK     bool
	extraHeaders map[HeaderKey]interface{}
}

type recipientSigInfo struct {
	sigAlg    SignatureAlgorithm
	publicKey func() *JSONWebKey
	signer    payloadSigner
}

func staticPublicKey(jwk *JSONWebKey) func() *JSONWebKey {
	return func() *JSONWebKey {
		return jwk
	}
}

// NewSigner creates an appropriate signer based on the key type
func NewSigner(sig SigningKey, opts *SignerOptions) (Signer, error) {
	return NewMultiSigner([]SigningKey{sig}, opts)
}

// NewMultiSigner creates a signer for multiple recipients
func NewMultiSigner(sigs []SigningKey, opts *SignerOptions) (Signer, error) {
	signer := &genericSigner{recipients: []recipientSigInfo{}}

	if opts != nil {
		signer.nonceSource = opts.NonceSource
		signer.embedJWK = opts.EmbedJWK
		signer.extraHeaders = opts.ExtraHeaders
	}

	for _, sig := range sigs {
		err := signer.addRecipient(sig.Algorithm, sig.Key)
		if err != nil {
			return nil, err
		}
	}

	return signer, nil
}

// newVerifier creates a verifier based on the key type
func newVerifier(verificationKey interface{}) (payloadVerifier, error) {
	switch verificationKey := verificationKey.(type) {
	case ed25519.PublicKey:
		return &edEncrypterVerifier{
			publicKey: verificationKey,
		}, nil
	case *rsa.PublicKey:
		return &rsaEncrypterVerifier{
			publicKey: verificationKey,
		}, nil
	case *ecdsa.PublicKey:
		return &ecEncrypterVerifier{
			publicKey: verificationKey,
		}, nil
	case []byte:
		return &symmetricMac{
			key: verificationKey,
		}, nil
	case JSONWebKey:
		return newVerifier(verificationKey.Key)
	case *JSONWebKey:
		return newVerifier(verificationKey.Key)
	case OpaqueVerifier:
		return &opaqueVerifier{verifier: verificationKey}, nil
	default:
		return nil, ErrUnsupportedKeyType
	}
}

func (ctx *genericSigner) addRecipient(alg SignatureAlgorithm, signingKey interface{}) error {
	recipient, err := makeJWSRecipient(alg, signingKey)
	if err != nil {
		return err
	}

	ctx.recipients = append(ctx.recipients, recipient)
	return nil
}

func makeJWSRecipient(alg SignatureAlgorithm, signingKey interface{}) (recipientSigInfo, error) {
	switch signingKey := signingKey.(type) {
	case ed25519.PrivateKey:
		return newEd25519Signer(alg, signingKey)
	case *rsa.PrivateKey:
		return newRSASigner(alg, signingKey)
	case *ecdsa.PrivateKey:
		return newECDSASigner(alg, signingKey)
	case []byte:
		return newSymmetricSigner(alg, signingKey)
	case JSONWebKey:
		return newJWKSigner(alg, signingKey)
	case *JSONWebKey:
		return newJWKSigner(alg, *signingKey)
	case OpaqueSigner:
		return newOpaqueSigner(alg, signingKey)
	default:
		return recipientSigInfo{}, ErrUnsupportedKeyType
	}
}

func newJWKSigner(alg SignatureAlgorithm, signingKey JSONWebKey) (recipientSigInfo, error) {
	recipient, err := makeJWSRecipient(alg, signingKey.Key)
	if err != nil {
		return recipientSigInfo{}, err
	}
	if recipient.publicKey != nil && recipient.publicKey() != nil {
		// recipient.publicKey is a JWK synthesized for embedding when recipientSigInfo
		// was created for the inner key (such as a RSA or ECDSA public key). It contains
		// the pub key for embedding, but doesn't have extra params like key id.
		publicKey := signingKey
		publicKey.Key = recipient.publicKey().Key
		recipient.publicKey = staticPublicKey(&publicKey)

		// This should be impossible, but let's check anyway.
		if !recipient.publicKey().IsPublic() {
			return recipientSigInfo{}, errors.New("go-jose/go-jose: public key was unexpectedly not public")
		}
	}
	return recipient, nil
}

func (ctx *genericSigner) Sign(payload []byte) (*JSONWebSignature, error) {
	obj := &JSONWebSignature{}
	obj.payload = payload
	obj.Signatures = make([]Signature, len(ctx.recipients))

	for i, recipient := range ctx.recipients {
		protected := map[HeaderKey]interface{}{
			headerAlgorithm: string(recipient.sigAlg),
		}

		if recipient.publicKey != nil && recipient.publicKey() != nil {
			// We want to embed the JWK or set the kid header, but not both. Having a protected
			// header that contains an embedded JWK while also simultaneously containing the kid
			// header is confusing, and at least in ACME the two are considered to be mutually
			// exclusive. The fact that both can exist at the same time is a somewhat unfortunate
			// result of the JOSE spec. We've decided that this library will only include one or
			// the other to avoid this confusion.
			//
			// See https://github.com/go-jose/go-jose/issues/157 for more context.
			if ctx.embedJWK {
				protected[headerJWK] = recipient.publicKey()
			} else {
				keyID := recipient.publicKey().KeyID
				if keyID != "" {
					protected[headerKeyID] = keyID
				}
			}
		}

		if ctx.nonceSource != nil {
			nonce, err := ctx.nonceSource.Nonce()
			if err != nil {
				return nil, fmt.Errorf("go-jose/go-jose: Error generating nonce: %v", err)
			}
			protected[headerNonce] = nonce
		}

		for k, v := range ctx.extraHeaders {
			protected[k] = v
		}

		serializedProtected := mustSerializeJSON(protected)
		needsBase64 := true

		if b64, ok := protected[headerB64]; ok {
			if needsBase64, ok = b64.(bool); !ok {
				return nil, errors.New("go-jose/go-jose: Invalid b64 header parameter")
			}
		}

		var input bytes.Buffer

		input.WriteString(base64.RawURLEncoding.EncodeToString(serializedProtected))
		input.WriteByte('.')

		if needsBase64 {
			input.WriteString(base64.RawURLEncoding.EncodeToString(payload))
		} else {
			input.Write(payload)
		}

		signatureInfo, err := recipient.signer.signPayload(input.Bytes(), recipient.sigAlg)
		if err != nil {
			return nil, err
		}

		signatureInfo.protected = &rawHeader{}
		for k, v := range protected {
			b, err := json.Marshal(v)
			if err != nil {
				return nil, fmt.Errorf("go-jose/go-jose: Error marshalling item %#v: %v", k, err)
			}
			(*signatureInfo.protected)[k] = makeRawMessage(b)
		}
		obj.Signatures[i] = signatureInfo
	}

	return obj, nil
}

func (ctx *genericSigner) Options() SignerOptions {
	return SignerOptions{
		NonceSource:  ctx.nonceSource,
		EmbedJWK:     ctx.embedJWK,
		ExtraHeaders: ctx.extraHeaders,
	}
}

// Verify validates the signature on the object and returns the payload.
// This function does not support multi-signature. If you desire multi-signature
// verification use VerifyMulti instead.
//
// Be careful when verifying signatures based on embedded JWKs inside the
// payload header. You cannot assume that the key received in a payload is
// trusted.
//
// The verificationKey argument must have one of these types:
//   - ed25519.PublicKey
//   - *ecdsa.PublicKey
//   - *rsa.PublicKey
//   - *JSONWebKey
//   - JSONWebKey
//   - *JSONWebKeySet
//   - JSONWebKeySet
//   - []byte (an HMAC key)
//   - Any type that implements the OpaqueVerifier interface.
//
// If the key is an HMAC key, it must have at least as many bytes as the relevant hash output:
//   - HS256: 32 bytes
//   - HS384: 48 bytes
//   - HS512: 64 bytes
func (obj JSONWebSignature) Verify(verificationKey interface{}) ([]byte, error) {
	err := obj.DetachedVerify(obj.payload, verificationKey)
	if err != nil {
		return nil, err
	}
	return obj.payload, nil
}

// UnsafePayloadWithoutVerification returns the payload without
// verifying it. The content returned from this function cannot be
// trusted.
func (obj JSONWebSignature) UnsafePayloadWithoutVerification() []byte {
	return obj.payload
}

// DetachedVerify validates a detached signature on the given payload. In
// most cases, you will probably want to use Verify instead. DetachedVerify
// is only useful if you have a payload and signature that are separated from
// each other.
//
// The verificationKey argument must have one of the types allowed for the
// verificationKey argument of JSONWebSignature.Verify().
func (obj JSONWebSignature) DetachedVerify(payload []byte, verificationKey interface{}) error {
	key, err := tryJWKS(verificationKey, obj.headers()...)
	if err != nil {
		return err
	}
	verifier, err := newVerifier(key)
	if err != nil {
		return err
	}

	if len(obj.Signatures) > 1 {
		return errors.New("go-jose/go-jose: too many signatures in payload; expecting only one")
	}

	signature := obj.Signatures[0]

	if signature.header != nil {
		// Per https://www.rfc-editor.org/rfc/rfc7515.html#section-4.1.11,
		// 4.1.11. "crit" (Critical) Header Parameter
		// "When used, this Header Parameter MUST be integrity
		// protected; therefore, it MUST occur only within the JWS
		// Protected Header."
		err = signature.header.checkNoCritical()
		if err != nil {
			return err
		}
	}

	if signature.protected != nil {
		err = signature.protected.checkSupportedCritical(supportedCritical)
		if err != nil {
			return err
		}
	}

	input, err := obj.computeAuthData(payload, &signature)
	if err != nil {
		return ErrCryptoFailure
	}

	headers := signature.mergedHeaders()
	alg := headers.getSignatureAlgorithm()
	err = verifier.verifyPayload(input, signature.Signature, alg)
	if err == nil {
		return nil
	}

	return ErrCryptoFailure
}

// VerifyMulti validates (one of the multiple) signatures on the object and
// returns the index of the signature that was verified, along with the signature
// object and the payload. We return the signature and index to guarantee that
// callers are getting the verified value.
//
// The verificationKey argument must have one of the types allowed for the
// verificationKey argument of JSONWebSignature.Verify().
func (obj JSONWebSignature) VerifyMulti(verificationKey interface{}) (int, Signature, []byte, error) {
	idx, sig, err := obj.DetachedVerifyMulti(obj.payload, verificationKey)
	if err != nil {
		return -1, Signature{}, nil, err
	}
	return idx, sig, obj.payload, nil
}

// DetachedVerifyMulti validates a detached signature on the given payload with
// a signature/object that has potentially multiple signers. This returns the index
// of the signature that was verified, along with the signature object. We return
// the signature and index to guarantee that callers are getting the verified value.
//
// In most cases, you will probably want to use Verify or VerifyMulti instead.
// DetachedVerifyMulti is only useful if you have a payload and signature that are
// separated from each other, and the signature can have multiple signers at the
// same time.
//
// The verificationKey argument must have one of the types allowed for the
// verificationKey argument of JSONWebSignature.Verify().
func (obj JSONWebSignature) DetachedVerifyMulti(payload []byte, verificationKey interface{}) (int, Signature, error) {
	key, err := tryJWKS(verificationKey, obj.headers()...)
	if err != nil {
		return -1, Signature{}, err
	}
	verifier, err := newVerifier(key)
	if err != nil {
		return -1, Signature{}, err
	}

outer:
	for i, signature := range obj.Signatures {
		if signature.header != nil {
			// Per https://www.rfc-editor.org/rfc/rfc7515.html#section-4.1.11,
			// 4.1.11. "crit" (Critical) Header Parameter
			// "When used, this Header Parameter MUST be integrity
			// protected; therefore, it MUST occur only within the JWS
			// Protected Header."
			err = signature.header.checkNoCritical()
			if err != nil {
				continue outer
			}
		}

		if signature.protected != nil {
			// Check for only supported critical headers
			err = signature.protected.checkSupportedCritical(supportedCritical)
			if err != nil {
				continue outer
			}
		}

		input, err := obj.computeAuthData(payload, &signature)
		if err != nil {
			continue
		}

		headers := signature.mergedHeaders()
		alg := headers.getSignatureAlgorithm()
		err = verifier.verifyPayload(input, signature.Signature, alg)
		if err == nil {
			return i, signature, nil
		}
	}

	return -1, Signature{}, ErrCryptoFailure
}

func (obj JSONWebSignature) headers() []Header {
	headers := make([]Header, len(obj.Signatures))
	for i, sig := range obj.Signatures {
		headers[i] = sig.Header
	}
	return headers
}
