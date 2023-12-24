// Copyright 2018 Thales e-Security, Inc
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
	"errors"

	"github.com/miekg/pkcs11"
)

// SymmetricGenParams holds a consistent (key type, mechanism) key generation pair.
type SymmetricGenParams struct {
	// Key type (CKK_...)
	KeyType uint

	// Key generation mechanism (CKM_..._KEY_GEN)
	GenMech uint
}

// SymmetricCipher represents information about a symmetric cipher.
type SymmetricCipher struct {
	// Possible key generation parameters
	// (For HMAC this varies between PKCS#11 implementations.)
	GenParams []SymmetricGenParams

	// Block size in bytes
	BlockSize int

	// True if encryption supported
	Encrypt bool

	// True if MAC supported
	MAC bool

	// ECB mechanism (CKM_..._ECB)
	ECBMech uint

	// CBC mechanism (CKM_..._CBC)
	CBCMech uint

	// CBC mechanism with PKCS#7 padding (CKM_..._CBC)
	CBCPKCSMech uint

	// GCM mechanism (CKM_..._GCM)
	GCMMech uint
}

// CipherAES describes the AES cipher. Use this with the
// GenerateSecretKey... functions.
var CipherAES = &SymmetricCipher{
	GenParams: []SymmetricGenParams{
		{
			KeyType: pkcs11.CKK_AES,
			GenMech: pkcs11.CKM_AES_KEY_GEN,
		},
	},
	BlockSize:   16,
	Encrypt:     true,
	MAC:         false,
	ECBMech:     pkcs11.CKM_AES_ECB,
	CBCMech:     pkcs11.CKM_AES_CBC,
	CBCPKCSMech: pkcs11.CKM_AES_CBC_PAD,
	GCMMech:     pkcs11.CKM_AES_GCM,
}

// CipherDES3 describes the three-key triple-DES cipher. Use this with the
// GenerateSecretKey... functions.
var CipherDES3 = &SymmetricCipher{
	GenParams: []SymmetricGenParams{
		{
			KeyType: pkcs11.CKK_DES3,
			GenMech: pkcs11.CKM_DES3_KEY_GEN,
		},
	},
	BlockSize:   8,
	Encrypt:     true,
	MAC:         false,
	ECBMech:     pkcs11.CKM_DES3_ECB,
	CBCMech:     pkcs11.CKM_DES3_CBC,
	CBCPKCSMech: pkcs11.CKM_DES3_CBC_PAD,
	GCMMech:     0,
}

// CipherGeneric describes the CKK_GENERIC_SECRET key type. Use this with the
// GenerateSecretKey... functions.
//
// The spec promises that this mechanism can be used to perform HMAC
// operations, although implementations vary;
// CipherHMACSHA1 and so on may give better results.
var CipherGeneric = &SymmetricCipher{
	GenParams: []SymmetricGenParams{
		{
			KeyType: pkcs11.CKK_GENERIC_SECRET,
			GenMech: pkcs11.CKM_GENERIC_SECRET_KEY_GEN,
		},
	},
	BlockSize: 64,
	Encrypt:   false,
	MAC:       true,
	ECBMech:   0,
	CBCMech:   0,
	GCMMech:   0,
}

// CipherHMACSHA1 describes the CKK_SHA_1_HMAC key type. Use this with the
// GenerateSecretKey... functions.
var CipherHMACSHA1 = &SymmetricCipher{
	GenParams: []SymmetricGenParams{
		{
			KeyType: pkcs11.CKK_SHA_1_HMAC,
			GenMech: CKM_NC_SHA_1_HMAC_KEY_GEN,
		},
		{
			KeyType: pkcs11.CKK_GENERIC_SECRET,
			GenMech: pkcs11.CKM_GENERIC_SECRET_KEY_GEN,
		},
	},
	BlockSize: 64,
	Encrypt:   false,
	MAC:       true,
	ECBMech:   0,
	CBCMech:   0,
	GCMMech:   0,
}

// CipherHMACSHA224 describes the CKK_SHA224_HMAC key type. Use this with the
// GenerateSecretKey... functions.
var CipherHMACSHA224 = &SymmetricCipher{
	GenParams: []SymmetricGenParams{
		{
			KeyType: pkcs11.CKK_SHA224_HMAC,
			GenMech: CKM_NC_SHA224_HMAC_KEY_GEN,
		},
		{
			KeyType: pkcs11.CKK_GENERIC_SECRET,
			GenMech: pkcs11.CKM_GENERIC_SECRET_KEY_GEN,
		},
	},
	BlockSize: 64,
	Encrypt:   false,
	MAC:       true,
	ECBMech:   0,
	CBCMech:   0,
	GCMMech:   0,
}

// CipherHMACSHA256 describes the CKK_SHA256_HMAC key type. Use this with the
// GenerateSecretKey... functions.
var CipherHMACSHA256 = &SymmetricCipher{
	GenParams: []SymmetricGenParams{
		{
			KeyType: pkcs11.CKK_SHA256_HMAC,
			GenMech: CKM_NC_SHA256_HMAC_KEY_GEN,
		},
		{
			KeyType: pkcs11.CKK_GENERIC_SECRET,
			GenMech: pkcs11.CKM_GENERIC_SECRET_KEY_GEN,
		},
	},
	BlockSize: 64,
	Encrypt:   false,
	MAC:       true,
	ECBMech:   0,
	CBCMech:   0,
	GCMMech:   0,
}

// CipherHMACSHA384 describes the CKK_SHA384_HMAC key type. Use this with the
// GenerateSecretKey... functions.
var CipherHMACSHA384 = &SymmetricCipher{
	GenParams: []SymmetricGenParams{
		{
			KeyType: pkcs11.CKK_SHA384_HMAC,
			GenMech: CKM_NC_SHA384_HMAC_KEY_GEN,
		},
		{
			KeyType: pkcs11.CKK_GENERIC_SECRET,
			GenMech: pkcs11.CKM_GENERIC_SECRET_KEY_GEN,
		},
	},
	BlockSize: 64,
	Encrypt:   false,
	MAC:       true,
	ECBMech:   0,
	CBCMech:   0,
	GCMMech:   0,
}

// CipherHMACSHA512 describes the CKK_SHA512_HMAC key type. Use this with the
// GenerateSecretKey... functions.
var CipherHMACSHA512 = &SymmetricCipher{
	GenParams: []SymmetricGenParams{
		{
			KeyType: pkcs11.CKK_SHA512_HMAC,
			GenMech: CKM_NC_SHA512_HMAC_KEY_GEN,
		},
		{
			KeyType: pkcs11.CKK_GENERIC_SECRET,
			GenMech: pkcs11.CKM_GENERIC_SECRET_KEY_GEN,
		},
	},
	BlockSize: 128,
	Encrypt:   false,
	MAC:       true,
	ECBMech:   0,
	CBCMech:   0,
	GCMMech:   0,
}

// Ciphers is a map of PKCS#11 key types (CKK_...) to symmetric cipher information.
var Ciphers = map[int]*SymmetricCipher{
	pkcs11.CKK_AES:            CipherAES,
	pkcs11.CKK_DES3:           CipherDES3,
	pkcs11.CKK_GENERIC_SECRET: CipherGeneric,
	pkcs11.CKK_SHA_1_HMAC:     CipherHMACSHA1,
	pkcs11.CKK_SHA224_HMAC:    CipherHMACSHA224,
	pkcs11.CKK_SHA256_HMAC:    CipherHMACSHA256,
	pkcs11.CKK_SHA384_HMAC:    CipherHMACSHA384,
	pkcs11.CKK_SHA512_HMAC:    CipherHMACSHA512,
}

// SecretKey contains a reference to a loaded PKCS#11 symmetric key object.
//
// A *SecretKey implements the cipher.Block interface, allowing it be used
// as the argument to cipher.NewCBCEncrypter and similar methods.
// For bulk operation this is very inefficient;
// using NewCBCEncrypterCloser, NewCBCEncrypter or NewCBC from this package is
// much faster.
type SecretKey struct {
	pkcs11Object

	// Symmetric cipher information
	Cipher *SymmetricCipher
}

// GenerateSecretKey creates an secret key of given length and type. The id parameter is used to
// set CKA_ID and must be non-nil.
func (c *Context) GenerateSecretKey(id []byte, bits int, cipher *SymmetricCipher) (*SecretKey, error) {
	if c.closed.Get() {
		return nil, errClosed
	}

	template, err := NewAttributeSetWithID(id)
	if err != nil {
		return nil, err
	}
	return c.GenerateSecretKeyWithAttributes(template, bits, cipher)
}

// GenerateSecretKey creates an secret key of given length and type. The id and label parameters are used to
// set CKA_ID and CKA_LABEL respectively and must be non-nil.
func (c *Context) GenerateSecretKeyWithLabel(id, label []byte, bits int, cipher *SymmetricCipher) (*SecretKey, error) {
	if c.closed.Get() {
		return nil, errClosed
	}

	template, err := NewAttributeSetWithIDAndLabel(id, label)
	if err != nil {
		return nil, err
	}
	return c.GenerateSecretKeyWithAttributes(template, bits, cipher)

}

// GenerateSecretKeyWithAttributes creates an secret key of given length and type. After this function returns, template
// will contain the attributes applied to the key. If required attributes are missing, they will be set to a default
// value.
func (c *Context) GenerateSecretKeyWithAttributes(template AttributeSet, bits int, cipher *SymmetricCipher) (k *SecretKey, err error) {
	if c.closed.Get() {
		return nil, errClosed
	}

	err = c.withSession(func(session *pkcs11Session) error {

		// CKK_*_HMAC exists but there is no specific corresponding CKM_*_KEY_GEN
		// mechanism. Therefore we attempt both CKM_GENERIC_SECRET_KEY_GEN and
		// vendor-specific mechanisms.

		template.AddIfNotPresent([]*pkcs11.Attribute{
			pkcs11.NewAttribute(pkcs11.CKA_CLASS, pkcs11.CKO_SECRET_KEY),
			pkcs11.NewAttribute(pkcs11.CKA_TOKEN, true),
			pkcs11.NewAttribute(pkcs11.CKA_SIGN, cipher.MAC),
			pkcs11.NewAttribute(pkcs11.CKA_VERIFY, cipher.MAC),
			pkcs11.NewAttribute(pkcs11.CKA_ENCRYPT, cipher.Encrypt), // Not supported on CloudHSM
			pkcs11.NewAttribute(pkcs11.CKA_DECRYPT, cipher.Encrypt), // Not supported on CloudHSM
			pkcs11.NewAttribute(pkcs11.CKA_SENSITIVE, true),
			pkcs11.NewAttribute(pkcs11.CKA_EXTRACTABLE, false),
		})
		if bits > 0 {
			_ = template.Set(pkcs11.CKA_VALUE_LEN, bits/8) // safe for an int
		}

		for n, genMech := range cipher.GenParams {

			_ = template.Set(CkaKeyType, genMech.KeyType)

			mech := []*pkcs11.Mechanism{pkcs11.NewMechanism(genMech.GenMech, nil)}

			privHandle, err := session.ctx.GenerateKey(session.handle, mech, template.ToSlice())
			if err == nil {
				k = &SecretKey{pkcs11Object{privHandle, c}, cipher}
				return nil
			}

			// As a special case, AWS CloudHSM does not accept CKA_ENCRYPT and CKA_DECRYPT on a
			// Generic Secret key. If we are in that special case, try again without those attributes.
			if e, ok := err.(pkcs11.Error); ok && e == pkcs11.CKR_ARGUMENTS_BAD && genMech.GenMech == pkcs11.CKM_GENERIC_SECRET_KEY_GEN {
				adjustedTemplate := template.Copy()
				adjustedTemplate.Unset(CkaEncrypt)
				adjustedTemplate.Unset(CkaDecrypt)

				privHandle, err = session.ctx.GenerateKey(session.handle, mech, adjustedTemplate.ToSlice())
				if err == nil {
					// Store the actual attributes
					template.cloneFrom(adjustedTemplate)

					k = &SecretKey{pkcs11Object{privHandle, c}, cipher}
					return nil
				}
			}

			if n == len(cipher.GenParams)-1 {
				// If we have tried all available gen params, we should return a sensible error. So we skip the
				// retry logic below and return directly.
				return err
			}

			// nShield returns CKR_TEMPLATE_INCONSISTENT if if doesn't like the CKK/CKM combination.
			// AWS CloudHSM returns CKR_ATTRIBUTE_VALUE_INVALID in the same circumstances.
			if e, ok := err.(pkcs11.Error); ok &&
				e == pkcs11.CKR_TEMPLATE_INCONSISTENT || e == pkcs11.CKR_ATTRIBUTE_VALUE_INVALID {
				continue
			}

			return err
		}

		// We can only get here if there were no GenParams
		return errors.New("cipher must have GenParams")
	})
	return
}

// Delete deletes the secret key from the token.
func (key *SecretKey) Delete() error {
	return key.pkcs11Object.Delete()
}
