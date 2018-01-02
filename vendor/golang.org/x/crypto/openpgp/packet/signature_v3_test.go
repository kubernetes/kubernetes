// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packet

import (
	"bytes"
	"crypto"
	"encoding/hex"
	"io"
	"io/ioutil"
	"testing"

	"golang.org/x/crypto/openpgp/armor"
)

func TestSignatureV3Read(t *testing.T) {
	r := v3KeyReader(t)
	Read(r)                // Skip public key
	Read(r)                // Skip uid
	packet, err := Read(r) // Signature
	if err != nil {
		t.Error(err)
		return
	}
	sig, ok := packet.(*SignatureV3)
	if !ok || sig.SigType != SigTypeGenericCert || sig.PubKeyAlgo != PubKeyAlgoRSA || sig.Hash != crypto.MD5 {
		t.Errorf("failed to parse, got: %#v", packet)
	}
}

func TestSignatureV3Reserialize(t *testing.T) {
	r := v3KeyReader(t)
	Read(r) // Skip public key
	Read(r) // Skip uid
	packet, err := Read(r)
	if err != nil {
		t.Error(err)
		return
	}
	sig := packet.(*SignatureV3)
	out := new(bytes.Buffer)
	if err = sig.Serialize(out); err != nil {
		t.Errorf("error reserializing: %s", err)
		return
	}
	expected, err := ioutil.ReadAll(v3KeyReader(t))
	if err != nil {
		t.Error(err)
		return
	}
	expected = expected[4+141+4+39:] // See pgpdump offsets below, this is where the sig starts
	if !bytes.Equal(expected, out.Bytes()) {
		t.Errorf("output doesn't match input (got vs expected):\n%s\n%s", hex.Dump(out.Bytes()), hex.Dump(expected))
	}
}

func v3KeyReader(t *testing.T) io.Reader {
	armorBlock, err := armor.Decode(bytes.NewBufferString(keySigV3Armor))
	if err != nil {
		t.Fatalf("armor Decode failed: %v", err)
	}
	return armorBlock.Body
}

// keySigV3Armor is some V3 public key I found in an SKS dump.
// Old: Public Key Packet(tag 6)(141 bytes)
//      Ver 4 - new
//      Public key creation time - Fri Sep 16 17:13:54 CDT 1994
//      Pub alg - unknown(pub 0)
//      Unknown public key(pub 0)
// Old: User ID Packet(tag 13)(39 bytes)
//      User ID - Armin M. Warda <warda@nephilim.ruhr.de>
// Old: Signature Packet(tag 2)(149 bytes)
//      Ver 4 - new
//      Sig type - unknown(05)
//      Pub alg - ElGamal Encrypt-Only(pub 16)
//      Hash alg - unknown(hash 46)
//      Hashed Sub: unknown(sub 81, critical)(1988 bytes)
const keySigV3Armor = `-----BEGIN PGP PUBLIC KEY BLOCK-----
Version: SKS 1.0.10

mI0CLnoYogAAAQQA1qwA2SuJwfQ5bCQ6u5t20ulnOtY0gykf7YjiK4LiVeRBwHjGq7v30tGV
5Qti7qqRW4Ww7CDCJc4sZMFnystucR2vLkXaSoNWoFm4Fg47NiisDdhDezHwbVPW6OpCFNSi
ZAamtj4QAUBu8j4LswafrJqZqR9336/V3g8Yil2l48kABRG0J0FybWluIE0uIFdhcmRhIDx3
YXJkYUBuZXBoaWxpbS5ydWhyLmRlPoiVAgUQLok2xwXR6zmeWEiZAQE/DgP/WgxPQh40/Po4
gSkWZCDAjNdph7zexvAb0CcUWahcwiBIgg3U5ErCx9I5CNVA9U+s8bNrDZwgSIeBzp3KhWUx
524uhGgm6ZUTOAIKA6CbV6pfqoLpJnRYvXYQU5mIWsNa99wcu2qu18OeEDnztb7aLA6Ra9OF
YFCbq4EjXRoOrYM=
=LPjs
-----END PGP PUBLIC KEY BLOCK-----`
