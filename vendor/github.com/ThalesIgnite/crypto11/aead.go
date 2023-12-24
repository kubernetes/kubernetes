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
	"crypto/cipher"
	"errors"
	"fmt"

	"github.com/miekg/pkcs11"
)

// cipher.AEAD ----------------------------------------------------------

// A PaddingMode is used by a block cipher (see NewCBC).
type PaddingMode int

const (
	// PaddingNone represents a block cipher with no padding.
	PaddingNone PaddingMode = iota

	// PaddingPKCS represents a block cipher used with PKCS#7 padding.
	PaddingPKCS
)

var errBadGCMNonceSize = errors.New("nonce slice too small to hold IV")

type genericAead struct {
	key *SecretKey

	overhead int

	nonceSize int

	// Note - if the GCMParams result is non-nil, the caller must call Free() on the params when
	// finished.
	makeMech func(nonce []byte, additionalData []byte, encrypt bool) ([]*pkcs11.Mechanism, *pkcs11.GCMParams, error)
}

// NewGCM returns a given cipher wrapped in Galois Counter Mode, with the standard
// nonce length.
//
// This depends on the HSM supporting the CKM_*_GCM mechanism. If it is not supported
// then you must use cipher.NewGCM; it will be slow.
func (key *SecretKey) NewGCM() (cipher.AEAD, error) {
	if key.Cipher.GCMMech == 0 {
		return nil, fmt.Errorf("GCM not implemented for key type %#x", key.Cipher.GenParams[0].KeyType)
	}

	g := genericAead{
		key:       key,
		overhead:  16,
		nonceSize: key.context.cfg.GCMIVLength,
		makeMech: func(nonce []byte, additionalData []byte, encrypt bool) ([]*pkcs11.Mechanism, *pkcs11.GCMParams, error) {
			var params *pkcs11.GCMParams

			if (encrypt && key.context.cfg.UseGCMIVFromHSM &&
				!key.context.cfg.GCMIVFromHSMControl.SupplyIvForHSMGCMEncrypt) || (!encrypt &&
				key.context.cfg.UseGCMIVFromHSM && !key.context.cfg.GCMIVFromHSMControl.SupplyIvForHSMGCMDecrypt) {
				params = pkcs11.NewGCMParams(nil, additionalData, 16*8 /*bits*/)
			} else {
				params = pkcs11.NewGCMParams(nonce, additionalData, 16*8 /*bits*/)
			}
			return []*pkcs11.Mechanism{pkcs11.NewMechanism(key.Cipher.GCMMech, params)}, params, nil
		},
	}
	return g, nil
}

// NewCBC returns a given cipher wrapped in CBC mode.
//
// Despite the cipher.AEAD return type, there is no support for additional data and no authentication.
// This method exists to provide a convenient way to do bulk (possibly padded) CBC encryption.
// Think carefully before passing the cipher.AEAD to any consumer that expects authentication.
func (key *SecretKey) NewCBC(paddingMode PaddingMode) (cipher.AEAD, error) {

	var pkcsMech uint

	switch paddingMode {
	case PaddingNone:
		pkcsMech = key.Cipher.CBCMech
	case PaddingPKCS:
		pkcsMech = key.Cipher.CBCPKCSMech
	default:
		return nil, errors.New("unrecognized padding mode")
	}

	g := genericAead{
		key:       key,
		overhead:  0,
		nonceSize: key.BlockSize(),
		makeMech: func(nonce []byte, additionalData []byte, encrypt bool) ([]*pkcs11.Mechanism, *pkcs11.GCMParams, error) {
			if len(additionalData) > 0 {
				return nil, nil, errors.New("additional data not supported for CBC mode")
			}

			return []*pkcs11.Mechanism{pkcs11.NewMechanism(pkcsMech, nonce)}, nil, nil
		},
	}

	return g, nil
}

func (g genericAead) NonceSize() int {
	return g.nonceSize
}

func (g genericAead) Overhead() int {
	return g.overhead
}

func (g genericAead) Seal(dst, nonce, plaintext, additionalData []byte) []byte {

	var result []byte
	if err := g.key.context.withSession(func(session *pkcs11Session) (err error) {
		mech, params, err := g.makeMech(nonce, additionalData, true)

		if err != nil {
			return err
		}
		defer params.Free()

		if err = session.ctx.EncryptInit(session.handle, mech, g.key.handle); err != nil {
			err = fmt.Errorf("C_EncryptInit: %v", err)
			return
		}
		if result, err = session.ctx.Encrypt(session.handle, plaintext); err != nil {
			err = fmt.Errorf("C_Encrypt: %v", err)
			return
		}

		if g.key.context.cfg.UseGCMIVFromHSM && g.key.context.cfg.GCMIVFromHSMControl.SupplyIvForHSMGCMEncrypt {
			if len(nonce) != len(params.IV()) {
				return errBadGCMNonceSize
			}
		}

		return
	}); err != nil {
		panic(err)
	} else {
		dst = append(dst, result...)
	}
	return dst
}

func (g genericAead) Open(dst, nonce, ciphertext, additionalData []byte) ([]byte, error) {
	var result []byte
	if err := g.key.context.withSession(func(session *pkcs11Session) (err error) {
		mech, params, err := g.makeMech(nonce, additionalData, false)
		if err != nil {
			return
		}
		defer params.Free()

		if err = session.ctx.DecryptInit(session.handle, mech, g.key.handle); err != nil {
			err = fmt.Errorf("C_DecryptInit: %v", err)
			return
		}
		if result, err = session.ctx.Decrypt(session.handle, ciphertext); err != nil {
			err = fmt.Errorf("C_Decrypt: %v", err)
			return
		}
		return
	}); err != nil {
		return nil, err
	}
	dst = append(dst, result...)
	return dst, nil
}
