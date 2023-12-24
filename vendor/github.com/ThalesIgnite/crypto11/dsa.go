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
	"crypto/dsa"
	"io"
	"math/big"

	"github.com/pkg/errors"

	pkcs11 "github.com/miekg/pkcs11"
)

// pkcs11PrivateKeyDSA contains a reference to a loaded PKCS#11 DSA private key object.
type pkcs11PrivateKeyDSA struct {
	pkcs11PrivateKey
}

// Export the public key corresponding to a private DSA key.
func exportDSAPublicKey(session *pkcs11Session, pubHandle pkcs11.ObjectHandle) (crypto.PublicKey, error) {
	template := []*pkcs11.Attribute{
		pkcs11.NewAttribute(pkcs11.CKA_PRIME, nil),
		pkcs11.NewAttribute(pkcs11.CKA_SUBPRIME, nil),
		pkcs11.NewAttribute(pkcs11.CKA_BASE, nil),
		pkcs11.NewAttribute(pkcs11.CKA_VALUE, nil),
	}
	exported, err := session.ctx.GetAttributeValue(session.handle, pubHandle, template)
	if err != nil {
		return nil, err
	}
	var p, q, g, y big.Int
	p.SetBytes(exported[0].Value)
	q.SetBytes(exported[1].Value)
	g.SetBytes(exported[2].Value)
	y.SetBytes(exported[3].Value)
	result := dsa.PublicKey{
		Parameters: dsa.Parameters{
			P: &p,
			Q: &q,
			G: &g,
		},
		Y: &y,
	}
	return &result, nil
}

func notNilBytes(obj []byte, name string) error {
	if obj == nil {
		return errors.Errorf("%s cannot be nil", name)
	}
	return nil
}

// GenerateDSAKeyPair creates a DSA key pair on the token. The id parameter is used to
// set CKA_ID and must be non-nil.
func (c *Context) GenerateDSAKeyPair(id []byte, params *dsa.Parameters) (Signer, error) {
	if c.closed.Get() {
		return nil, errClosed
	}

	public, err := NewAttributeSetWithID(id)
	if err != nil {
		return nil, err
	}
	// Copy the AttributeSet to allow modifications.
	private := public.Copy()

	return c.GenerateDSAKeyPairWithAttributes(public, private, params)
}

// GenerateDSAKeyPairWithLabel creates a DSA key pair on the token. The id and label parameters are used to
// set CKA_ID and CKA_LABEL respectively and must be non-nil.
func (c *Context) GenerateDSAKeyPairWithLabel(id, label []byte, params *dsa.Parameters) (Signer, error) {
	if c.closed.Get() {
		return nil, errClosed
	}

	public, err := NewAttributeSetWithIDAndLabel(id, label)
	if err != nil {
		return nil, err
	}
	// Copy the AttributeSet to allow modifications.
	private := public.Copy()

	return c.GenerateDSAKeyPairWithAttributes(public, private, params)
}

// GenerateDSAKeyPairWithAttributes creates a DSA key pair on the token. After this function returns, public and private
// will contain the attributes applied to the key pair. If required attributes are missing, they will be set to a
// default value.
func (c *Context) GenerateDSAKeyPairWithAttributes(public, private AttributeSet, params *dsa.Parameters) (Signer, error) {
	if c.closed.Get() {
		return nil, errClosed
	}

	var k Signer
	err := c.withSession(func(session *pkcs11Session) error {
		p := params.P.Bytes()
		q := params.Q.Bytes()
		g := params.G.Bytes()

		public.AddIfNotPresent([]*pkcs11.Attribute{
			pkcs11.NewAttribute(pkcs11.CKA_CLASS, pkcs11.CKO_PUBLIC_KEY),
			pkcs11.NewAttribute(pkcs11.CKA_KEY_TYPE, pkcs11.CKK_DSA),
			pkcs11.NewAttribute(pkcs11.CKA_TOKEN, true),
			pkcs11.NewAttribute(pkcs11.CKA_VERIFY, true),
			pkcs11.NewAttribute(pkcs11.CKA_PRIME, p),
			pkcs11.NewAttribute(pkcs11.CKA_SUBPRIME, q),
			pkcs11.NewAttribute(pkcs11.CKA_BASE, g),
		})
		private.AddIfNotPresent([]*pkcs11.Attribute{
			pkcs11.NewAttribute(pkcs11.CKA_TOKEN, true),
			pkcs11.NewAttribute(pkcs11.CKA_SIGN, true),
			pkcs11.NewAttribute(pkcs11.CKA_SENSITIVE, true),
			pkcs11.NewAttribute(pkcs11.CKA_EXTRACTABLE, false),
		})

		mech := []*pkcs11.Mechanism{pkcs11.NewMechanism(pkcs11.CKM_DSA_KEY_PAIR_GEN, nil)}
		pubHandle, privHandle, err := session.ctx.GenerateKeyPair(session.handle,
			mech,
			public.ToSlice(),
			private.ToSlice())
		if err != nil {
			return err
		}
		pub, err := exportDSAPublicKey(session, pubHandle)
		if err != nil {
			return err
		}
		k = &pkcs11PrivateKeyDSA{
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

// Sign signs a message using a DSA key.
//
// This completes the implemention of crypto.Signer for pkcs11PrivateKeyDSA.
//
// PKCS#11 expects to pick its own random data for signatures, so the rand argument is ignored.
//
// The return value is a DER-encoded byteblock.
func (signer *pkcs11PrivateKeyDSA) Sign(rand io.Reader, digest []byte, opts crypto.SignerOpts) (signature []byte, err error) {
	return signer.context.dsaGeneric(signer.handle, pkcs11.CKM_DSA, digest)
}
