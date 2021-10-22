// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packet

import (
	"bytes"
	"crypto"
	"crypto/rsa"
	"encoding/hex"
	"fmt"
	"io"
	"math/big"
	"testing"
)

func bigFromBase10(s string) *big.Int {
	b, ok := new(big.Int).SetString(s, 10)
	if !ok {
		panic("bigFromBase10 failed")
	}
	return b
}

var encryptedKeyPub = rsa.PublicKey{
	E: 65537,
	N: bigFromBase10("115804063926007623305902631768113868327816898845124614648849934718568541074358183759250136204762053879858102352159854352727097033322663029387610959884180306668628526686121021235757016368038585212410610742029286439607686208110250133174279811431933746643015923132833417396844716207301518956640020862630546868823"),
}

var encryptedKeyRSAPriv = &rsa.PrivateKey{
	PublicKey: encryptedKeyPub,
	D:         bigFromBase10("32355588668219869544751561565313228297765464314098552250409557267371233892496951383426602439009993875125222579159850054973310859166139474359774543943714622292329487391199285040721944491839695981199720170366763547754915493640685849961780092241140181198779299712578774460837139360803883139311171713302987058393"),
}

var encryptedKeyPriv = &PrivateKey{
	PublicKey: PublicKey{
		PubKeyAlgo: PubKeyAlgoRSA,
	},
	PrivateKey: encryptedKeyRSAPriv,
}

func TestDecryptingEncryptedKey(t *testing.T) {
	for i, encryptedKeyHex := range []string{
		"c18c032a67d68660df41c70104005789d0de26b6a50c985a02a13131ca829c413a35d0e6fa8d6842599252162808ac7439c72151c8c6183e76923fe3299301414d0c25a2f06a2257db3839e7df0ec964773f6e4c4ac7ff3b48c444237166dd46ba8ff443a5410dc670cb486672fdbe7c9dfafb75b4fea83af3a204fe2a7dfa86bd20122b4f3d2646cbeecb8f7be8",
		// MPI can be shorter than the length of the key.
		"c18b032a67d68660df41c70103f8e520c52ae9807183c669ce26e772e482dc5d8cf60e6f59316e145be14d2e5221ee69550db1d5618a8cb002a719f1f0b9345bde21536d410ec90ba86cac37748dec7933eb7f9873873b2d61d3321d1cd44535014f6df58f7bc0c7afb5edc38e1a974428997d2f747f9a173bea9ca53079b409517d332df62d805564cffc9be6",
	} {
		const expectedKeyHex = "d930363f7e0308c333b9618617ea728963d8df993665ae7be1092d4926fd864b"

		p, err := Read(readerFromHex(encryptedKeyHex))
		if err != nil {
			t.Errorf("#%d: error from Read: %s", i, err)
			return
		}
		ek, ok := p.(*EncryptedKey)
		if !ok {
			t.Errorf("#%d: didn't parse an EncryptedKey, got %#v", i, p)
			return
		}

		if ek.KeyId != 0x2a67d68660df41c7 || ek.Algo != PubKeyAlgoRSA {
			t.Errorf("#%d: unexpected EncryptedKey contents: %#v", i, ek)
			return
		}

		err = ek.Decrypt(encryptedKeyPriv, nil)
		if err != nil {
			t.Errorf("#%d: error from Decrypt: %s", i, err)
			return
		}

		if ek.CipherFunc != CipherAES256 {
			t.Errorf("#%d: unexpected EncryptedKey contents: %#v", i, ek)
			return
		}

		keyHex := fmt.Sprintf("%x", ek.Key)
		if keyHex != expectedKeyHex {
			t.Errorf("#%d: bad key, got %s want %s", i, keyHex, expectedKeyHex)
		}
	}
}

type rsaDecrypter struct {
	rsaPrivateKey *rsa.PrivateKey
	decryptCount  int
}

func (r *rsaDecrypter) Public() crypto.PublicKey {
	return &r.rsaPrivateKey.PublicKey
}

func (r *rsaDecrypter) Decrypt(rand io.Reader, msg []byte, opts crypto.DecrypterOpts) (plaintext []byte, err error) {
	r.decryptCount++
	return r.rsaPrivateKey.Decrypt(rand, msg, opts)
}

func TestRSADecrypter(t *testing.T) {
	const encryptedKeyHex = "c18c032a67d68660df41c70104005789d0de26b6a50c985a02a13131ca829c413a35d0e6fa8d6842599252162808ac7439c72151c8c6183e76923fe3299301414d0c25a2f06a2257db3839e7df0ec964773f6e4c4ac7ff3b48c444237166dd46ba8ff443a5410dc670cb486672fdbe7c9dfafb75b4fea83af3a204fe2a7dfa86bd20122b4f3d2646cbeecb8f7be8"

	const expectedKeyHex = "d930363f7e0308c333b9618617ea728963d8df993665ae7be1092d4926fd864b"

	p, err := Read(readerFromHex(encryptedKeyHex))
	if err != nil {
		t.Errorf("error from Read: %s", err)
		return
	}
	ek, ok := p.(*EncryptedKey)
	if !ok {
		t.Errorf("didn't parse an EncryptedKey, got %#v", p)
		return
	}

	if ek.KeyId != 0x2a67d68660df41c7 || ek.Algo != PubKeyAlgoRSA {
		t.Errorf("unexpected EncryptedKey contents: %#v", ek)
		return
	}

	customDecrypter := &rsaDecrypter{
		rsaPrivateKey: encryptedKeyRSAPriv,
	}

	customKeyPriv := &PrivateKey{
		PublicKey: PublicKey{
			PubKeyAlgo: PubKeyAlgoRSA,
		},
		PrivateKey: customDecrypter,
	}

	err = ek.Decrypt(customKeyPriv, nil)
	if err != nil {
		t.Errorf("error from Decrypt: %s", err)
		return
	}

	if ek.CipherFunc != CipherAES256 {
		t.Errorf("unexpected EncryptedKey contents: %#v", ek)
		return
	}

	keyHex := fmt.Sprintf("%x", ek.Key)
	if keyHex != expectedKeyHex {
		t.Errorf("bad key, got %s want %s", keyHex, expectedKeyHex)
	}

	if customDecrypter.decryptCount != 1 {
		t.Errorf("Expected customDecrypter.Decrypt() to be called 1 time, but was called %d times", customDecrypter.decryptCount)
	}
}

func TestEncryptingEncryptedKey(t *testing.T) {
	key := []byte{1, 2, 3, 4}
	const expectedKeyHex = "01020304"
	const keyId = 42

	pub := &PublicKey{
		PublicKey:  &encryptedKeyPub,
		KeyId:      keyId,
		PubKeyAlgo: PubKeyAlgoRSAEncryptOnly,
	}

	buf := new(bytes.Buffer)
	err := SerializeEncryptedKey(buf, pub, CipherAES128, key, nil)
	if err != nil {
		t.Errorf("error writing encrypted key packet: %s", err)
	}

	p, err := Read(buf)
	if err != nil {
		t.Errorf("error from Read: %s", err)
		return
	}
	ek, ok := p.(*EncryptedKey)
	if !ok {
		t.Errorf("didn't parse an EncryptedKey, got %#v", p)
		return
	}

	if ek.KeyId != keyId || ek.Algo != PubKeyAlgoRSAEncryptOnly {
		t.Errorf("unexpected EncryptedKey contents: %#v", ek)
		return
	}

	err = ek.Decrypt(encryptedKeyPriv, nil)
	if err != nil {
		t.Errorf("error from Decrypt: %s", err)
		return
	}

	if ek.CipherFunc != CipherAES128 {
		t.Errorf("unexpected EncryptedKey contents: %#v", ek)
		return
	}

	keyHex := fmt.Sprintf("%x", ek.Key)
	if keyHex != expectedKeyHex {
		t.Errorf("bad key, got %s want %s", keyHex, expectedKeyHex)
	}
}

func TestSerializingEncryptedKey(t *testing.T) {
	const encryptedKeyHex = "c18c032a67d68660df41c70104005789d0de26b6a50c985a02a13131ca829c413a35d0e6fa8d6842599252162808ac7439c72151c8c6183e76923fe3299301414d0c25a2f06a2257db3839e7df0ec964773f6e4c4ac7ff3b48c444237166dd46ba8ff443a5410dc670cb486672fdbe7c9dfafb75b4fea83af3a204fe2a7dfa86bd20122b4f3d2646cbeecb8f7be8"

	p, err := Read(readerFromHex(encryptedKeyHex))
	if err != nil {
		t.Fatalf("error from Read: %s", err)
	}
	ek, ok := p.(*EncryptedKey)
	if !ok {
		t.Fatalf("didn't parse an EncryptedKey, got %#v", p)
	}

	var buf bytes.Buffer
	ek.Serialize(&buf)

	if bufHex := hex.EncodeToString(buf.Bytes()); bufHex != encryptedKeyHex {
		t.Fatalf("serialization of encrypted key differed from original. Original was %s, but reserialized as %s", encryptedKeyHex, bufHex)
	}
}
