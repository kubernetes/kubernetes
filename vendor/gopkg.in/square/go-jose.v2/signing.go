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
	"crypto/ecdsa"
	"crypto/rsa"
	"encoding/base64"
	"errors"
	"fmt"

	"golang.org/x/crypto/ed25519"

	"gopkg.in/square/go-jose.v2/json"
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
	// additional values here. All values must be JSON-serializable.
	ExtraHeaders map[HeaderKey]interface{}
}

// WithHeader adds an arbitrary value to the ExtraHeaders map, initializing it
// if necessary. It returns itself and so can be used in a fluent style.
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
	publicKey *JSONWebKey
	signer    payloadSigner
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
	default:
		return recipientSigInfo{}, ErrUnsupportedKeyType
	}
}

func newJWKSigner(alg SignatureAlgorithm, signingKey JSONWebKey) (recipientSigInfo, error) {
	recipient, err := makeJWSRecipient(alg, signingKey.Key)
	if err != nil {
		return recipientSigInfo{}, err
	}
	if recipient.publicKey != nil {
		// recipient.publicKey is a JWK synthesized for embedding when recipientSigInfo
		// was created for the inner key (such as a RSA or ECDSA public key). It contains
		// the pub key for embedding, but doesn't have extra params like key id.
		publicKey := signingKey
		publicKey.Key = recipient.publicKey.Key
		recipient.publicKey = &publicKey

		// This should be impossible, but let's check anyway.
		if !recipient.publicKey.IsPublic() {
			return recipientSigInfo{}, errors.New("square/go-jose: public key was unexpectedly not public")
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

		if recipient.publicKey != nil {
			// We want to embed the JWK or set the kid header, but not both. Having a protected
			// header that contains an embedded JWK while also simultaneously containing the kid
			// header is confusing, and at least in ACME the two are considered to be mutually
			// exclusive. The fact that both can exist at the same time is a somewhat unfortunate
			// result of the JOSE spec. We've decided that this library will only include one or
			// the other to avoid this confusion.
			//
			// See https://github.com/square/go-jose/issues/157 for more context.
			if ctx.embedJWK {
				protected[headerJWK] = recipient.publicKey
			} else {
				protected[headerKeyID] = recipient.publicKey.KeyID
			}
		}

		if ctx.nonceSource != nil {
			nonce, err := ctx.nonceSource.Nonce()
			if err != nil {
				return nil, fmt.Errorf("square/go-jose: Error generating nonce: %v", err)
			}
			protected[headerNonce] = nonce
		}

		for k, v := range ctx.extraHeaders {
			protected[k] = v
		}

		serializedProtected := mustSerializeJSON(protected)

		input := []byte(fmt.Sprintf("%s.%s",
			base64.RawURLEncoding.EncodeToString(serializedProtected),
			base64.RawURLEncoding.EncodeToString(payload)))

		signatureInfo, err := recipient.signer.signPayload(input, recipient.sigAlg)
		if err != nil {
			return nil, err
		}

		signatureInfo.protected = &rawHeader{}
		for k, v := range protected {
			b, err := json.Marshal(v)
			if err != nil {
				return nil, fmt.Errorf("square/go-jose: Error marshalling item %#v: %v", k, err)
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
// This function does not support multi-signature, if you desire multi-sig
// verification use VerifyMulti instead.
//
// Be careful when verifying signatures based on embedded JWKs inside the
// payload header. You cannot assume that the key received in a payload is
// trusted.
func (obj JSONWebSignature) Verify(verificationKey interface{}) ([]byte, error) {
	verifier, err := newVerifier(verificationKey)
	if err != nil {
		return nil, err
	}

	if len(obj.Signatures) > 1 {
		return nil, errors.New("square/go-jose: too many signatures in payload; expecting only one")
	}

	signature := obj.Signatures[0]
	headers := signature.mergedHeaders()
	critical, err := headers.getCritical()
	if err != nil {
		return nil, err
	}
	if len(critical) > 0 {
		// Unsupported crit header
		return nil, ErrCryptoFailure
	}

	input := obj.computeAuthData(&signature)
	alg := headers.getSignatureAlgorithm()
	err = verifier.verifyPayload(input, signature.Signature, alg)
	if err == nil {
		return obj.payload, nil
	}

	return nil, ErrCryptoFailure
}

// VerifyMulti validates (one of the multiple) signatures on the object and
// returns the index of the signature that was verified, along with the signature
// object and the payload. We return the signature and index to guarantee that
// callers are getting the verified value.
func (obj JSONWebSignature) VerifyMulti(verificationKey interface{}) (int, Signature, []byte, error) {
	verifier, err := newVerifier(verificationKey)
	if err != nil {
		return -1, Signature{}, nil, err
	}

	for i, signature := range obj.Signatures {
		headers := signature.mergedHeaders()
		critical, err := headers.getCritical()
		if err != nil {
			continue
		}
		if len(critical) > 0 {
			// Unsupported crit header
			continue
		}

		input := obj.computeAuthData(&signature)
		alg := headers.getSignatureAlgorithm()
		err = verifier.verifyPayload(input, signature.Signature, alg)
		if err == nil {
			return i, signature, obj.payload, nil
		}
	}

	return -1, Signature{}, nil, ErrCryptoFailure
}
