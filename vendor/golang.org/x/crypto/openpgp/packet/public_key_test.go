// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packet

import (
	"bytes"
	"encoding/hex"
	"testing"
	"time"
)

var pubKeyTests = []struct {
	hexData        string
	hexFingerprint string
	creationTime   time.Time
	pubKeyAlgo     PublicKeyAlgorithm
	keyId          uint64
	keyIdString    string
	keyIdShort     string
}{
	{rsaPkDataHex, rsaFingerprintHex, time.Unix(0x4d3c5c10, 0), PubKeyAlgoRSA, 0xa34d7e18c20c31bb, "A34D7E18C20C31BB", "C20C31BB"},
	{dsaPkDataHex, dsaFingerprintHex, time.Unix(0x4d432f89, 0), PubKeyAlgoDSA, 0x8e8fbe54062f19ed, "8E8FBE54062F19ED", "062F19ED"},
	{ecdsaPkDataHex, ecdsaFingerprintHex, time.Unix(0x5071c294, 0), PubKeyAlgoECDSA, 0x43fe956c542ca00b, "43FE956C542CA00B", "542CA00B"},
}

func TestPublicKeyRead(t *testing.T) {
	for i, test := range pubKeyTests {
		packet, err := Read(readerFromHex(test.hexData))
		if err != nil {
			t.Errorf("#%d: Read error: %s", i, err)
			continue
		}
		pk, ok := packet.(*PublicKey)
		if !ok {
			t.Errorf("#%d: failed to parse, got: %#v", i, packet)
			continue
		}
		if pk.PubKeyAlgo != test.pubKeyAlgo {
			t.Errorf("#%d: bad public key algorithm got:%x want:%x", i, pk.PubKeyAlgo, test.pubKeyAlgo)
		}
		if !pk.CreationTime.Equal(test.creationTime) {
			t.Errorf("#%d: bad creation time got:%v want:%v", i, pk.CreationTime, test.creationTime)
		}
		expectedFingerprint, _ := hex.DecodeString(test.hexFingerprint)
		if !bytes.Equal(expectedFingerprint, pk.Fingerprint[:]) {
			t.Errorf("#%d: bad fingerprint got:%x want:%x", i, pk.Fingerprint[:], expectedFingerprint)
		}
		if pk.KeyId != test.keyId {
			t.Errorf("#%d: bad keyid got:%x want:%x", i, pk.KeyId, test.keyId)
		}
		if g, e := pk.KeyIdString(), test.keyIdString; g != e {
			t.Errorf("#%d: bad KeyIdString got:%q want:%q", i, g, e)
		}
		if g, e := pk.KeyIdShortString(), test.keyIdShort; g != e {
			t.Errorf("#%d: bad KeyIdShortString got:%q want:%q", i, g, e)
		}
	}
}

func TestPublicKeySerialize(t *testing.T) {
	for i, test := range pubKeyTests {
		packet, err := Read(readerFromHex(test.hexData))
		if err != nil {
			t.Errorf("#%d: Read error: %s", i, err)
			continue
		}
		pk, ok := packet.(*PublicKey)
		if !ok {
			t.Errorf("#%d: failed to parse, got: %#v", i, packet)
			continue
		}
		serializeBuf := bytes.NewBuffer(nil)
		err = pk.Serialize(serializeBuf)
		if err != nil {
			t.Errorf("#%d: failed to serialize: %s", i, err)
			continue
		}

		packet, err = Read(serializeBuf)
		if err != nil {
			t.Errorf("#%d: Read error (from serialized data): %s", i, err)
			continue
		}
		pk, ok = packet.(*PublicKey)
		if !ok {
			t.Errorf("#%d: failed to parse serialized data, got: %#v", i, packet)
			continue
		}
	}
}

func TestEcc384Serialize(t *testing.T) {
	r := readerFromHex(ecc384PubHex)
	var w bytes.Buffer
	for i := 0; i < 2; i++ {
		// Public key
		p, err := Read(r)
		if err != nil {
			t.Error(err)
		}
		pubkey := p.(*PublicKey)
		if !bytes.Equal(pubkey.ec.oid, []byte{0x2b, 0x81, 0x04, 0x00, 0x22}) {
			t.Errorf("Unexpected pubkey OID: %x", pubkey.ec.oid)
		}
		if !bytes.Equal(pubkey.ec.p.bytes[:5], []byte{0x04, 0xf6, 0xb8, 0xc5, 0xac}) {
			t.Errorf("Unexpected pubkey P[:5]: %x", pubkey.ec.p.bytes)
		}
		if pubkey.KeyId != 0x098033880F54719F {
			t.Errorf("Unexpected pubkey ID: %x", pubkey.KeyId)
		}
		err = pubkey.Serialize(&w)
		if err != nil {
			t.Error(err)
		}
		// User ID
		p, err = Read(r)
		if err != nil {
			t.Error(err)
		}
		uid := p.(*UserId)
		if uid.Id != "ec_dsa_dh_384 <openpgp@brainhub.org>" {
			t.Error("Unexpected UID:", uid.Id)
		}
		err = uid.Serialize(&w)
		if err != nil {
			t.Error(err)
		}
		// User ID Sig
		p, err = Read(r)
		if err != nil {
			t.Error(err)
		}
		uidSig := p.(*Signature)
		err = pubkey.VerifyUserIdSignature(uid.Id, pubkey, uidSig)
		if err != nil {
			t.Error(err, ": UID")
		}
		err = uidSig.Serialize(&w)
		if err != nil {
			t.Error(err)
		}
		// Subkey
		p, err = Read(r)
		if err != nil {
			t.Error(err)
		}
		subkey := p.(*PublicKey)
		if !bytes.Equal(subkey.ec.oid, []byte{0x2b, 0x81, 0x04, 0x00, 0x22}) {
			t.Errorf("Unexpected subkey OID: %x", subkey.ec.oid)
		}
		if !bytes.Equal(subkey.ec.p.bytes[:5], []byte{0x04, 0x2f, 0xaa, 0x84, 0x02}) {
			t.Errorf("Unexpected subkey P[:5]: %x", subkey.ec.p.bytes)
		}
		if subkey.ecdh.KdfHash != 0x09 {
			t.Error("Expected KDF hash function SHA384 (0x09), got", subkey.ecdh.KdfHash)
		}
		if subkey.ecdh.KdfAlgo != 0x09 {
			t.Error("Expected KDF symmetric alg AES256 (0x09), got", subkey.ecdh.KdfAlgo)
		}
		if subkey.KeyId != 0xAA8B938F9A201946 {
			t.Errorf("Unexpected subkey ID: %x", subkey.KeyId)
		}
		err = subkey.Serialize(&w)
		if err != nil {
			t.Error(err)
		}
		// Subkey Sig
		p, err = Read(r)
		if err != nil {
			t.Error(err)
		}
		subkeySig := p.(*Signature)
		err = pubkey.VerifyKeySignature(subkey, subkeySig)
		if err != nil {
			t.Error(err)
		}
		err = subkeySig.Serialize(&w)
		if err != nil {
			t.Error(err)
		}
		// Now read back what we've written again
		r = bytes.NewBuffer(w.Bytes())
		w.Reset()
	}
}

const rsaFingerprintHex = "5fb74b1d03b1e3cb31bc2f8aa34d7e18c20c31bb"

const rsaPkDataHex = "988d044d3c5c10010400b1d13382944bd5aba23a4312968b5095d14f947f600eb478e14a6fcb16b0e0cac764884909c020bc495cfcc39a935387c661507bdb236a0612fb582cac3af9b29cc2c8c70090616c41b662f4da4c1201e195472eb7f4ae1ccbcbf9940fe21d985e379a5563dde5b9a23d35f1cfaa5790da3b79db26f23695107bfaca8e7b5bcd0011010001"

const dsaFingerprintHex = "eece4c094db002103714c63c8e8fbe54062f19ed"

const dsaPkDataHex = "9901a2044d432f89110400cd581334f0d7a1e1bdc8b9d6d8c0baf68793632735d2bb0903224cbaa1dfbf35a60ee7a13b92643421e1eb41aa8d79bea19a115a677f6b8ba3c7818ce53a6c2a24a1608bd8b8d6e55c5090cbde09dd26e356267465ae25e69ec8bdd57c7bbb2623e4d73336f73a0a9098f7f16da2e25252130fd694c0e8070c55a812a423ae7f00a0ebf50e70c2f19c3520a551bd4b08d30f23530d3d03ff7d0bf4a53a64a09dc5e6e6e35854b7d70c882b0c60293401958b1bd9e40abec3ea05ba87cf64899299d4bd6aa7f459c201d3fbbd6c82004bdc5e8a9eb8082d12054cc90fa9d4ec251a843236a588bf49552441817436c4f43326966fe85447d4e6d0acf8fa1ef0f014730770603ad7634c3088dc52501c237328417c31c89ed70400b2f1a98b0bf42f11fefc430704bebbaa41d9f355600c3facee1e490f64208e0e094ea55e3a598a219a58500bf78ac677b670a14f4e47e9cf8eab4f368cc1ddcaa18cc59309d4cc62dd4f680e73e6cc3e1ce87a84d0925efbcb26c575c093fc42eecf45135fabf6403a25c2016e1774c0484e440a18319072c617cc97ac0a3bb0"

const ecdsaFingerprintHex = "9892270b38b8980b05c8d56d43fe956c542ca00b"

const ecdsaPkDataHex = "9893045071c29413052b8104002304230401f4867769cedfa52c325018896245443968e52e51d0c2df8d939949cb5b330f2921711fbee1c9b9dddb95d15cb0255e99badeddda7cc23d9ddcaacbc290969b9f24019375d61c2e4e3b36953a28d8b2bc95f78c3f1d592fb24499be348656a7b17e3963187b4361afe497bc5f9f81213f04069f8e1fb9e6a6290ae295ca1a92b894396cb4"

// Source: https://sites.google.com/site/brainhub/pgpecckeys#TOC-ECC-NIST-P-384-key
const ecc384PubHex = `99006f044d53059213052b81040022030304f6b8c5aced5b84ef9f4a209db2e4a9dfb70d28cb8c10ecd57674a9fa5a67389942b62d5e51367df4c7bfd3f8e500feecf07ed265a621a8ebbbe53e947ec78c677eba143bd1533c2b350e1c29f82313e1e1108eba063be1e64b10e6950e799c2db42465635f6473615f64685f333834203c6f70656e70677040627261696e6875622e6f72673e8900cb04101309005305024d530592301480000000002000077072656665727265642d656d61696c2d656e636f64696e67407067702e636f6d7067706d696d65040b090807021901051b03000000021602051e010000000415090a08000a0910098033880f54719fca2b0180aa37350968bd5f115afd8ce7bc7b103822152dbff06d0afcda835329510905b98cb469ba208faab87c7412b799e7b633017f58364ea480e8a1a3f253a0c5f22c446e8be9a9fce6210136ee30811abbd49139de28b5bdf8dc36d06ae748579e9ff503b90073044d53059212052b810400220303042faa84024a20b6735c4897efa5bfb41bf85b7eefeab5ca0cb9ffc8ea04a46acb25534a577694f9e25340a4ab5223a9dd1eda530c8aa2e6718db10d7e672558c7736fe09369ea5739a2a3554bf16d41faa50562f11c6d39bbd5dffb6b9a9ec9180301090989008404181309000c05024d530592051b0c000000000a0910098033880f54719f80970180eee7a6d8fcee41ee4f9289df17f9bcf9d955dca25c583b94336f3a2b2d4986dc5cf417b8d2dc86f741a9e1a6d236c0e3017d1c76575458a0cfb93ae8a2b274fcc65ceecd7a91eec83656ba13219969f06945b48c56bd04152c3a0553c5f2f4bd1267`
