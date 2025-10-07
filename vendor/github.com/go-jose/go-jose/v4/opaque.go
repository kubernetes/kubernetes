/*-
 * Copyright 2018 Square Inc.
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

// OpaqueSigner is an interface that supports signing payloads with opaque
// private key(s). Private key operations performed by implementers may, for
// example, occur in a hardware module. An OpaqueSigner may rotate signing keys
// transparently to the user of this interface.
type OpaqueSigner interface {
	// Public returns the public key of the current signing key.
	Public() *JSONWebKey
	// Algs returns a list of supported signing algorithms.
	Algs() []SignatureAlgorithm
	// SignPayload signs a payload with the current signing key using the given
	// algorithm.
	SignPayload(payload []byte, alg SignatureAlgorithm) ([]byte, error)
}

type opaqueSigner struct {
	signer OpaqueSigner
}

func newOpaqueSigner(alg SignatureAlgorithm, signer OpaqueSigner) (recipientSigInfo, error) {
	var algSupported bool
	for _, salg := range signer.Algs() {
		if alg == salg {
			algSupported = true
			break
		}
	}
	if !algSupported {
		return recipientSigInfo{}, ErrUnsupportedAlgorithm
	}

	return recipientSigInfo{
		sigAlg:    alg,
		publicKey: signer.Public,
		signer: &opaqueSigner{
			signer: signer,
		},
	}, nil
}

func (o *opaqueSigner) signPayload(payload []byte, alg SignatureAlgorithm) (Signature, error) {
	out, err := o.signer.SignPayload(payload, alg)
	if err != nil {
		return Signature{}, err
	}

	return Signature{
		Signature: out,
		protected: &rawHeader{},
	}, nil
}

// OpaqueVerifier is an interface that supports verifying payloads with opaque
// public key(s). An OpaqueSigner may rotate signing keys transparently to the
// user of this interface.
type OpaqueVerifier interface {
	VerifyPayload(payload []byte, signature []byte, alg SignatureAlgorithm) error
}

type opaqueVerifier struct {
	verifier OpaqueVerifier
}

func (o *opaqueVerifier) verifyPayload(payload []byte, signature []byte, alg SignatureAlgorithm) error {
	return o.verifier.VerifyPayload(payload, signature, alg)
}

// OpaqueKeyEncrypter is an interface that supports encrypting keys with an opaque key.
//
// Note: this cannot currently be implemented outside this package because of its
// unexported method.
type OpaqueKeyEncrypter interface {
	// KeyID returns the kid
	KeyID() string
	// Algs returns a list of supported key encryption algorithms.
	Algs() []KeyAlgorithm
	// encryptKey encrypts the CEK using the given algorithm.
	encryptKey(cek []byte, alg KeyAlgorithm) (recipientInfo, error)
}

type opaqueKeyEncrypter struct {
	encrypter OpaqueKeyEncrypter
}

func newOpaqueKeyEncrypter(alg KeyAlgorithm, encrypter OpaqueKeyEncrypter) (recipientKeyInfo, error) {
	var algSupported bool
	for _, salg := range encrypter.Algs() {
		if alg == salg {
			algSupported = true
			break
		}
	}
	if !algSupported {
		return recipientKeyInfo{}, ErrUnsupportedAlgorithm
	}

	return recipientKeyInfo{
		keyID:  encrypter.KeyID(),
		keyAlg: alg,
		keyEncrypter: &opaqueKeyEncrypter{
			encrypter: encrypter,
		},
	}, nil
}

func (oke *opaqueKeyEncrypter) encryptKey(cek []byte, alg KeyAlgorithm) (recipientInfo, error) {
	return oke.encrypter.encryptKey(cek, alg)
}

// OpaqueKeyDecrypter is an interface that supports decrypting keys with an opaque key.
type OpaqueKeyDecrypter interface {
	DecryptKey(encryptedKey []byte, header Header) ([]byte, error)
}

type opaqueKeyDecrypter struct {
	decrypter OpaqueKeyDecrypter
}

func (okd *opaqueKeyDecrypter) decryptKey(headers rawHeader, recipient *recipientInfo, generator keyGenerator) ([]byte, error) {
	mergedHeaders := rawHeader{}
	mergedHeaders.merge(&headers)
	mergedHeaders.merge(recipient.header)

	header, err := mergedHeaders.sanitized()
	if err != nil {
		return nil, err
	}

	return okd.decrypter.DecryptKey(recipient.encryptedKey, header)
}
