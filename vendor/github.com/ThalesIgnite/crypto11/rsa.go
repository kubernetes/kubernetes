// Copyright 2016, 2017 Thales e-Security, Inc
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

package crypto11

import (
	"crypto"
	"crypto/rsa"
	"errors"
	"io"
	"math/big"

	"github.com/miekg/pkcs11"
)

// errMalformedRSAPublicKey is returned when an RSA public key is not in a suitable form.
//
// Currently this means that the public exponent is either bigger than
// 32 bits, or less than 2.
var errMalformedRSAPublicKey = errors.New("malformed RSA public key")

// errUnsupportedRSAOptions is returned when an unsupported RSA option is requested.
//
// Currently this means a nontrivial SessionKeyLen when decrypting; or
// an unsupported hash function; or crypto.rsa.PSSSaltLengthAuto was
// requested.
var errUnsupportedRSAOptions = errors.New("unsupported RSA option value")

// pkcs11PrivateKeyRSA contains a reference to a loaded PKCS#11 RSA private key object.
type pkcs11PrivateKeyRSA struct {
	pkcs11PrivateKey
}

// Export the public key corresponding to a private RSA key.
func exportRSAPublicKey(session *pkcs11Session, pubHandle pkcs11.ObjectHandle) (crypto.PublicKey, error) {
	template := []*pkcs11.Attribute{
		pkcs11.NewAttribute(pkcs11.CKA_MODULUS, nil),
		pkcs11.NewAttribute(pkcs11.CKA_PUBLIC_EXPONENT, nil),
	}
	exported, err := session.ctx.GetAttributeValue(session.handle, pubHandle, template)
	if err != nil {
		return nil, err
	}
	var modulus = new(big.Int)
	modulus.SetBytes(exported[0].Value)
	var bigExponent = new(big.Int)
	bigExponent.SetBytes(exported[1].Value)
	if bigExponent.BitLen() > 32 {
		return nil, errMalformedRSAPublicKey
	}
	if bigExponent.Sign() < 1 {
		return nil, errMalformedRSAPublicKey
	}
	exponent := int(bigExponent.Uint64())
	result := rsa.PublicKey{
		N: modulus,
		E: exponent,
	}
	if result.E < 2 {
		return nil, errMalformedRSAPublicKey
	}
	return &result, nil
}

// GenerateRSAKeyPair creates an RSA key pair on the token. The id parameter is used to
// set CKA_ID and must be non-nil. RSA private keys are generated with both sign and decrypt
// permissions, and a public exponent of 65537.
func (c *Context) GenerateRSAKeyPair(id []byte, bits int) (SignerDecrypter, error) {
	if c.closed.Get() {
		return nil, errClosed
	}

	public, err := NewAttributeSetWithID(id)
	if err != nil {
		return nil, err
	}
	// Copy the AttributeSet to allow modifications.
	private := public.Copy()

	return c.GenerateRSAKeyPairWithAttributes(public, private, bits)
}

// GenerateRSAKeyPairWithLabel creates an RSA key pair on the token. The id and label parameters are used to
// set CKA_ID and CKA_LABEL respectively and must be non-nil. RSA private keys are generated with both sign and decrypt
// permissions, and a public exponent of 65537.
func (c *Context) GenerateRSAKeyPairWithLabel(id, label []byte, bits int) (SignerDecrypter, error) {
	if c.closed.Get() {
		return nil, errClosed
	}

	public, err := NewAttributeSetWithIDAndLabel(id, label)
	if err != nil {
		return nil, err
	}
	// Copy the AttributeSet to allow modifications.
	private := public.Copy()

	return c.GenerateRSAKeyPairWithAttributes(public, private, bits)
}

// GenerateRSAKeyPairWithAttributes generates an RSA key pair on the token. After this function returns, public and
// private will contain the attributes applied to the key pair. If required attributes are missing, they will be set to
// a default value.
func (c *Context) GenerateRSAKeyPairWithAttributes(public, private AttributeSet, bits int) (SignerDecrypter, error) {
	if c.closed.Get() {
		return nil, errClosed
	}

	var k SignerDecrypter

	err := c.withSession(func(session *pkcs11Session) error {

		public.AddIfNotPresent([]*pkcs11.Attribute{
			pkcs11.NewAttribute(pkcs11.CKA_CLASS, pkcs11.CKO_PUBLIC_KEY),
			pkcs11.NewAttribute(pkcs11.CKA_KEY_TYPE, pkcs11.CKK_RSA),
			pkcs11.NewAttribute(pkcs11.CKA_TOKEN, true),
			pkcs11.NewAttribute(pkcs11.CKA_VERIFY, true),
			pkcs11.NewAttribute(pkcs11.CKA_ENCRYPT, true),
			pkcs11.NewAttribute(pkcs11.CKA_PUBLIC_EXPONENT, []byte{1, 0, 1}),
			pkcs11.NewAttribute(pkcs11.CKA_MODULUS_BITS, bits),
		})
		private.AddIfNotPresent([]*pkcs11.Attribute{
			pkcs11.NewAttribute(pkcs11.CKA_TOKEN, true),
			pkcs11.NewAttribute(pkcs11.CKA_SIGN, true),
			pkcs11.NewAttribute(pkcs11.CKA_DECRYPT, true),
			pkcs11.NewAttribute(pkcs11.CKA_SENSITIVE, true),
			pkcs11.NewAttribute(pkcs11.CKA_EXTRACTABLE, false),
		})

		mech := []*pkcs11.Mechanism{pkcs11.NewMechanism(pkcs11.CKM_RSA_PKCS_KEY_PAIR_GEN, nil)}
		pubHandle, privHandle, err := session.ctx.GenerateKeyPair(session.handle,
			mech,
			public.ToSlice(),
			private.ToSlice())
		if err != nil {
			return err
		}

		pub, err := exportRSAPublicKey(session, pubHandle)
		if err != nil {
			return err
		}
		k = &pkcs11PrivateKeyRSA{
			pkcs11PrivateKey: pkcs11PrivateKey{
				pkcs11Object: pkcs11Object{
					handle:  privHandle,
					context: c,
				},
				pubKeyHandle: pubHandle,
				pubKey:       pub,
			}}
		return nil
	})
	return k, err
}

// Decrypt decrypts a message using a RSA key.
//
// This completes the implemention of crypto.Decrypter for pkcs11PrivateKeyRSA.
//
// Note that the SessionKeyLen option (for PKCS#1v1.5 decryption) is not supported.
//
// The underlying PKCS#11 implementation may impose further restrictions.
func (priv *pkcs11PrivateKeyRSA) Decrypt(rand io.Reader, ciphertext []byte, options crypto.DecrypterOpts) (plaintext []byte, err error) {
	err = priv.context.withSession(func(session *pkcs11Session) error {
		if options == nil {
			plaintext, err = decryptPKCS1v15(session, priv, ciphertext, 0)
		} else {
			switch o := options.(type) {
			case *rsa.PKCS1v15DecryptOptions:
				plaintext, err = decryptPKCS1v15(session, priv, ciphertext, o.SessionKeyLen)
			case *rsa.OAEPOptions:
				plaintext, err = decryptOAEP(session, priv, ciphertext, o.Hash, o.Label)
			default:
				err = errUnsupportedRSAOptions
			}
		}
		return err
	})
	return plaintext, err
}

func decryptPKCS1v15(session *pkcs11Session, key *pkcs11PrivateKeyRSA, ciphertext []byte, sessionKeyLen int) ([]byte, error) {
	if sessionKeyLen != 0 {
		return nil, errUnsupportedRSAOptions
	}
	mech := []*pkcs11.Mechanism{pkcs11.NewMechanism(pkcs11.CKM_RSA_PKCS, nil)}
	if err := session.ctx.DecryptInit(session.handle, mech, key.handle); err != nil {
		return nil, err
	}
	return session.ctx.Decrypt(session.handle, ciphertext)
}

func decryptOAEP(session *pkcs11Session, key *pkcs11PrivateKeyRSA, ciphertext []byte, hashFunction crypto.Hash,
	label []byte) ([]byte, error) {

	hashAlg, mgfAlg, _, err := hashToPKCS11(hashFunction)
	if err != nil {
		return nil, err
	}

	mech := pkcs11.NewMechanism(pkcs11.CKM_RSA_PKCS_OAEP,
		pkcs11.NewOAEPParams(hashAlg, mgfAlg, pkcs11.CKZ_DATA_SPECIFIED, label))

	err = session.ctx.DecryptInit(session.handle, []*pkcs11.Mechanism{mech}, key.handle)
	if err != nil {
		return nil, err
	}
	return session.ctx.Decrypt(session.handle, ciphertext)
}

func hashToPKCS11(hashFunction crypto.Hash) (hashAlg uint, mgfAlg uint, hashLen uint, err error) {
	switch hashFunction {
	case crypto.SHA1:
		return pkcs11.CKM_SHA_1, pkcs11.CKG_MGF1_SHA1, 20, nil
	case crypto.SHA224:
		return pkcs11.CKM_SHA224, pkcs11.CKG_MGF1_SHA224, 28, nil
	case crypto.SHA256:
		return pkcs11.CKM_SHA256, pkcs11.CKG_MGF1_SHA256, 32, nil
	case crypto.SHA384:
		return pkcs11.CKM_SHA384, pkcs11.CKG_MGF1_SHA384, 48, nil
	case crypto.SHA512:
		return pkcs11.CKM_SHA512, pkcs11.CKG_MGF1_SHA512, 64, nil
	default:
		return 0, 0, 0, errUnsupportedRSAOptions
	}
}

func signPSS(session *pkcs11Session, key *pkcs11PrivateKeyRSA, digest []byte, opts *rsa.PSSOptions) ([]byte, error) {
	var hMech, mgf, hLen, sLen uint
	var err error
	if hMech, mgf, hLen, err = hashToPKCS11(opts.Hash); err != nil {
		return nil, err
	}
	switch opts.SaltLength {
	case rsa.PSSSaltLengthAuto: // parseltongue constant
		// TODO we could (in principle) work out the biggest
		// possible size from the key, but until someone has
		// the effort to do that...
		return nil, errUnsupportedRSAOptions
	case rsa.PSSSaltLengthEqualsHash:
		sLen = hLen
	default:
		sLen = uint(opts.SaltLength)
	}
	// TODO this is pretty horrible, maybe the PKCS#11 wrapper
	// could be improved to help us out here
	parameters := concat(ulongToBytes(hMech),
		ulongToBytes(mgf),
		ulongToBytes(sLen))
	mech := []*pkcs11.Mechanism{pkcs11.NewMechanism(pkcs11.CKM_RSA_PKCS_PSS, parameters)}
	if err = session.ctx.SignInit(session.handle, mech, key.handle); err != nil {
		return nil, err
	}
	return session.ctx.Sign(session.handle, digest)
}

var pkcs1Prefix = map[crypto.Hash][]byte{
	crypto.SHA1:   {0x30, 0x21, 0x30, 0x09, 0x06, 0x05, 0x2b, 0x0e, 0x03, 0x02, 0x1a, 0x05, 0x00, 0x04, 0x14},
	crypto.SHA224: {0x30, 0x2d, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x04, 0x05, 0x00, 0x04, 0x1c},
	crypto.SHA256: {0x30, 0x31, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01, 0x05, 0x00, 0x04, 0x20},
	crypto.SHA384: {0x30, 0x41, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x02, 0x05, 0x00, 0x04, 0x30},
	crypto.SHA512: {0x30, 0x51, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x03, 0x05, 0x00, 0x04, 0x40},
}

func signPKCS1v15(session *pkcs11Session, key *pkcs11PrivateKeyRSA, digest []byte, hash crypto.Hash) (signature []byte, err error) {
	/* Calculate T for EMSA-PKCS1-v1_5. */
	oid := pkcs1Prefix[hash]
	T := make([]byte, len(oid)+len(digest))
	copy(T[0:len(oid)], oid)
	copy(T[len(oid):], digest)
	mech := []*pkcs11.Mechanism{pkcs11.NewMechanism(pkcs11.CKM_RSA_PKCS, nil)}
	err = session.ctx.SignInit(session.handle, mech, key.handle)
	if err == nil {
		signature, err = session.ctx.Sign(session.handle, T)
	}
	return
}

// Sign signs a message using a RSA key.
//
// This completes the implemention of crypto.Signer for pkcs11PrivateKeyRSA.
//
// PKCS#11 expects to pick its own random data where necessary for signatures, so the rand argument is ignored.
//
// Note that (at present) the crypto.rsa.PSSSaltLengthAuto option is
// not supported. The caller must either use
// crypto.rsa.PSSSaltLengthEqualsHash (recommended) or pass an
// explicit salt length. Moreover the underlying PKCS#11
// implementation may impose further restrictions.
func (priv *pkcs11PrivateKeyRSA) Sign(rand io.Reader, digest []byte, opts crypto.SignerOpts) (signature []byte, err error) {
	err = priv.context.withSession(func(session *pkcs11Session) error {
		switch opts.(type) {
		case *rsa.PSSOptions:
			signature, err = signPSS(session, priv, digest, opts.(*rsa.PSSOptions))
		default: /* PKCS1-v1_5 */
			signature, err = signPKCS1v15(session, priv, digest, opts.HashFunc())
		}
		return err
	})

	if err != nil {
		return nil, err
	}

	return signature, err
}
