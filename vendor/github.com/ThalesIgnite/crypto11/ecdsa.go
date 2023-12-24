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
	"bytes"
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"encoding/asn1"
	"io"
	"math/big"

	"github.com/miekg/pkcs11"
	"github.com/pkg/errors"
)

// errUnsupportedEllipticCurve is returned when an elliptic curve
// unsupported by crypto11 is specified.  Note that the error behavior
// for an elliptic curve unsupported by the underlying PKCS#11
// implementation will be different.
var errUnsupportedEllipticCurve = errors.New("unsupported elliptic curve")

// pkcs11PrivateKeyECDSA contains a reference to a loaded PKCS#11 ECDSA private key object.
type pkcs11PrivateKeyECDSA struct {
	pkcs11PrivateKey
}

// Information about an Elliptic Curve
type curveInfo struct {
	// ASN.1 marshaled OID
	oid []byte

	// Curve definition in Go form
	curve elliptic.Curve
}

// ASN.1 marshal some value and panic on error
func mustMarshal(val interface{}) []byte {
	if b, err := asn1.Marshal(val); err != nil {
		panic(err)
	} else {
		return b
	}
}

// Note: some of these are outside what crypto/elliptic currently
// knows about. So I'm making a (reasonable) assumption about what
// they will be called if they are either added or if someone
// specifies them explicitly.
//
// For public key export, the curve has to be a known one, otherwise
// you're stuffed. This is probably better fixed by adding well-known
// curves to crypto/elliptic rather than having a private copy here.
var wellKnownCurves = map[string]curveInfo{
	"P-192": {
		mustMarshal(asn1.ObjectIdentifier{1, 2, 840, 10045, 3, 1, 1}),
		nil,
	},
	"P-224": {
		mustMarshal(asn1.ObjectIdentifier{1, 3, 132, 0, 33}),
		elliptic.P224(),
	},
	"P-256": {
		mustMarshal(asn1.ObjectIdentifier{1, 2, 840, 10045, 3, 1, 7}),
		elliptic.P256(),
	},
	"P-384": {
		mustMarshal(asn1.ObjectIdentifier{1, 3, 132, 0, 34}),
		elliptic.P384(),
	},
	"P-521": {
		mustMarshal(asn1.ObjectIdentifier{1, 3, 132, 0, 35}),
		elliptic.P521(),
	},

	"K-163": {
		mustMarshal(asn1.ObjectIdentifier{1, 3, 132, 0, 1}),
		nil,
	},
	"K-233": {
		mustMarshal(asn1.ObjectIdentifier{1, 3, 132, 0, 26}),
		nil,
	},
	"K-283": {
		mustMarshal(asn1.ObjectIdentifier{1, 3, 132, 0, 16}),
		nil,
	},
	"K-409": {
		mustMarshal(asn1.ObjectIdentifier{1, 3, 132, 0, 36}),
		nil,
	},
	"K-571": {
		mustMarshal(asn1.ObjectIdentifier{1, 3, 132, 0, 38}),
		nil,
	},

	"B-163": {
		mustMarshal(asn1.ObjectIdentifier{1, 3, 132, 0, 15}),
		nil,
	},
	"B-233": {
		mustMarshal(asn1.ObjectIdentifier{1, 3, 132, 0, 27}),
		nil,
	},
	"B-283": {
		mustMarshal(asn1.ObjectIdentifier{1, 3, 132, 0, 17}),
		nil,
	},
	"B-409": {
		mustMarshal(asn1.ObjectIdentifier{1, 3, 132, 0, 37}),
		nil,
	},
	"B-571": {
		mustMarshal(asn1.ObjectIdentifier{1, 3, 132, 0, 39}),
		nil,
	},
}

func marshalEcParams(c elliptic.Curve) ([]byte, error) {
	if ci, ok := wellKnownCurves[c.Params().Name]; ok {
		return ci.oid, nil
	}
	// TODO use ANSI X9.62 ECParameters representation instead
	return nil, errUnsupportedEllipticCurve
}

func unmarshalEcParams(b []byte) (elliptic.Curve, error) {
	// See if it's a well-known curve
	for _, ci := range wellKnownCurves {
		if bytes.Equal(b, ci.oid) {
			if ci.curve != nil {
				return ci.curve, nil
			}
			return nil, errUnsupportedEllipticCurve
		}
	}
	// TODO try ANSI X9.62 ECParameters representation
	return nil, errUnsupportedEllipticCurve
}

func unmarshalEcPoint(b []byte, c elliptic.Curve) (*big.Int, *big.Int, error) {
	var pointBytes []byte
	extra, err := asn1.Unmarshal(b, &pointBytes)
	if err != nil {
		return nil, nil, errors.WithMessage(err, "elliptic curve point is invalid ASN.1")
	}

	if len(extra) > 0 {
		// We weren't expecting extra data
		return nil, nil, errors.New("unexpected data found when parsing elliptic curve point")
	}

	x, y := elliptic.Unmarshal(c, pointBytes)
	if x == nil || y == nil {
		return nil, nil, errors.New("failed to parse elliptic curve point")
	}
	return x, y, nil
}

// Export the public key corresponding to a private ECDSA key.
func exportECDSAPublicKey(session *pkcs11Session, pubHandle pkcs11.ObjectHandle) (crypto.PublicKey, error) {
	var err error
	var attributes []*pkcs11.Attribute
	var pub ecdsa.PublicKey
	template := []*pkcs11.Attribute{
		pkcs11.NewAttribute(pkcs11.CKA_ECDSA_PARAMS, nil),
		pkcs11.NewAttribute(pkcs11.CKA_EC_POINT, nil),
	}
	if attributes, err = session.ctx.GetAttributeValue(session.handle, pubHandle, template); err != nil {
		return nil, err
	}
	if pub.Curve, err = unmarshalEcParams(attributes[0].Value); err != nil {
		return nil, err
	}
	if pub.X, pub.Y, err = unmarshalEcPoint(attributes[1].Value, pub.Curve); err != nil {
		return nil, err
	}
	return &pub, nil
}

// GenerateECDSAKeyPair creates a ECDSA key pair on the token using curve c. The id parameter is used to
// set CKA_ID and must be non-nil. Only a limited set of named elliptic curves are supported. The
// underlying PKCS#11 implementation may impose further restrictions.
func (c *Context) GenerateECDSAKeyPair(id []byte, curve elliptic.Curve) (Signer, error) {
	if c.closed.Get() {
		return nil, errClosed
	}

	public, err := NewAttributeSetWithID(id)
	if err != nil {
		return nil, err
	}
	// Copy the AttributeSet to allow modifications.
	private := public.Copy()

	return c.GenerateECDSAKeyPairWithAttributes(public, private, curve)
}

// GenerateECDSAKeyPairWithLabel creates a ECDSA key pair on the token using curve c. The id and label parameters are used to
// set CKA_ID and CKA_LABEL respectively and must be non-nil. Only a limited set of named elliptic curves are supported. The
// underlying PKCS#11 implementation may impose further restrictions.
func (c *Context) GenerateECDSAKeyPairWithLabel(id, label []byte, curve elliptic.Curve) (Signer, error) {
	if c.closed.Get() {
		return nil, errClosed
	}

	public, err := NewAttributeSetWithIDAndLabel(id, label)
	if err != nil {
		return nil, err
	}
	// Copy the AttributeSet to allow modifications.
	private := public.Copy()

	return c.GenerateECDSAKeyPairWithAttributes(public, private, curve)
}

// GenerateECDSAKeyPairWithAttributes generates an ECDSA key pair on the token. After this function returns, public and
// private will contain the attributes applied to the key pair. If required attributes are missing, they will be set to
// a default value.
func (c *Context) GenerateECDSAKeyPairWithAttributes(public, private AttributeSet, curve elliptic.Curve) (Signer, error) {
	if c.closed.Get() {
		return nil, errClosed
	}

	var k Signer
	err := c.withSession(func(session *pkcs11Session) error {

		parameters, err := marshalEcParams(curve)
		if err != nil {
			return err
		}
		public.AddIfNotPresent([]*pkcs11.Attribute{
			pkcs11.NewAttribute(pkcs11.CKA_CLASS, pkcs11.CKO_PUBLIC_KEY),
			pkcs11.NewAttribute(pkcs11.CKA_KEY_TYPE, pkcs11.CKK_ECDSA),
			pkcs11.NewAttribute(pkcs11.CKA_TOKEN, true),
			pkcs11.NewAttribute(pkcs11.CKA_VERIFY, true),
			pkcs11.NewAttribute(pkcs11.CKA_ECDSA_PARAMS, parameters),
		})
		private.AddIfNotPresent([]*pkcs11.Attribute{
			pkcs11.NewAttribute(pkcs11.CKA_TOKEN, true),
			pkcs11.NewAttribute(pkcs11.CKA_SIGN, true),
			pkcs11.NewAttribute(pkcs11.CKA_SENSITIVE, true),
			pkcs11.NewAttribute(pkcs11.CKA_EXTRACTABLE, false),
		})

		mech := []*pkcs11.Mechanism{pkcs11.NewMechanism(pkcs11.CKM_ECDSA_KEY_PAIR_GEN, nil)}
		pubHandle, privHandle, err := session.ctx.GenerateKeyPair(session.handle,
			mech,
			public.ToSlice(),
			private.ToSlice())
		if err != nil {
			return err
		}

		pub, err := exportECDSAPublicKey(session, pubHandle)
		if err != nil {
			return err
		}
		k = &pkcs11PrivateKeyECDSA{
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

// Sign signs a message using an ECDSA key.
//
// This completes the implemention of crypto.Signer for pkcs11PrivateKeyECDSA.
//
// PKCS#11 expects to pick its own random data where necessary for signatures, so the rand argument is ignored.
//
// The return value is a DER-encoded byteblock.
func (signer *pkcs11PrivateKeyECDSA) Sign(rand io.Reader, digest []byte, opts crypto.SignerOpts) ([]byte, error) {
	return signer.context.dsaGeneric(signer.handle, pkcs11.CKM_ECDSA, digest)
}
