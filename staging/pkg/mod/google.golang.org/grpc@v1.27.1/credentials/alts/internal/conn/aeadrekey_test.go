/*
 *
 * Copyright 2018 gRPC authors.
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
 *
 */

package conn

import (
	"bytes"
	"encoding/hex"
	"testing"
)

// cryptoTestVector is struct for a rekey test vector
type rekeyAEADTestVector struct {
	desc                                   string
	key, nonce, plaintext, aad, ciphertext []byte
}

// Test encrypt and decrypt using (adapted) test vectors for AES-GCM.
func TestAES128GCMRekeyEncrypt(t *testing.T) {
	for _, test := range []rekeyAEADTestVector{
		// NIST vectors from:
		// http://csrc.nist.gov/groups/ST/toolkit/BCM/documents/proposedmodes/gcm/gcm-revised-spec.pdf
		//
		// IEEE vectors from:
		// http://www.ieee802.org/1/files/public/docs2011/bn-randall-test-vectors-0511-v1.pdf
		//
		// Key expanded by setting
		// expandedKey = (key ||
		//                key ^ {0x01,..,0x01} ||
		//                key ^ {0x02,..,0x02})[0:44].
		{
			desc:       "Derived from NIST test vector 1",
			key:        dehex("0000000000000000000000000000000001010101010101010101010101010101020202020202020202020202"),
			nonce:      dehex("000000000000000000000000"),
			aad:        dehex(""),
			plaintext:  dehex(""),
			ciphertext: dehex("85e873e002f6ebdc4060954eb8675508"),
		},
		{
			desc:       "Derived from NIST test vector 2",
			key:        dehex("0000000000000000000000000000000001010101010101010101010101010101020202020202020202020202"),
			nonce:      dehex("000000000000000000000000"),
			aad:        dehex(""),
			plaintext:  dehex("00000000000000000000000000000000"),
			ciphertext: dehex("51e9a8cb23ca2512c8256afff8e72d681aca19a1148ac115e83df4888cc00d11"),
		},
		{
			desc:       "Derived from NIST test vector 3",
			key:        dehex("feffe9928665731c6d6a8f9467308308fffee8938764721d6c6b8e9566318209fcfdeb908467711e6f688d96"),
			nonce:      dehex("cafebabefacedbaddecaf888"),
			aad:        dehex(""),
			plaintext:  dehex("d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a721c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b391aafd255"),
			ciphertext: dehex("1018ed5a1402a86516d6576d70b2ffccca261b94df88b58f53b64dfba435d18b2f6e3b7869f9353d4ac8cf09afb1663daa7b4017e6fc2c177c0c087c0df1162129952213cee1bc6e9c8495dd705e1f3d"),
		},
		{
			desc:       "Derived from NIST test vector 4",
			key:        dehex("feffe9928665731c6d6a8f9467308308fffee8938764721d6c6b8e9566318209fcfdeb908467711e6f688d96"),
			nonce:      dehex("cafebabefacedbaddecaf888"),
			aad:        dehex("feedfacedeadbeeffeedfacedeadbeefabaddad2"),
			plaintext:  dehex("d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a721c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39"),
			ciphertext: dehex("1018ed5a1402a86516d6576d70b2ffccca261b94df88b58f53b64dfba435d18b2f6e3b7869f9353d4ac8cf09afb1663daa7b4017e6fc2c177c0c087c4764565d077e9124001ddb27fc0848c5"),
		},
		{
			desc:       "Derived from adapted NIST test vector 4 for KDF counter boundary (flip nonce bit 15)",
			key:        dehex("feffe9928665731c6d6a8f9467308308fffee8938764721d6c6b8e9566318209fcfdeb908467711e6f688d96"),
			nonce:      dehex("ca7ebabefacedbaddecaf888"),
			aad:        dehex("feedfacedeadbeeffeedfacedeadbeefabaddad2"),
			plaintext:  dehex("d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a721c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39"),
			ciphertext: dehex("e650d3c0fb879327f2d03287fa93cd07342b136215adbca00c3bd5099ec41832b1d18e0423ed26bb12c6cd09debb29230a94c0cee15903656f85edb6fc509b1b28216382172ecbcc31e1e9b1"),
		},
		{
			desc:       "Derived from adapted NIST test vector 4 for KDF counter boundary (flip nonce bit 16)",
			key:        dehex("feffe9928665731c6d6a8f9467308308fffee8938764721d6c6b8e9566318209fcfdeb908467711e6f688d96"),
			nonce:      dehex("cafebbbefacedbaddecaf888"),
			aad:        dehex("feedfacedeadbeeffeedfacedeadbeefabaddad2"),
			plaintext:  dehex("d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a721c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39"),
			ciphertext: dehex("c0121e6c954d0767f96630c33450999791b2da2ad05c4190169ccad9ac86ff1c721e3d82f2ad22ab463bab4a0754b7dd68ca4de7ea2531b625eda01f89312b2ab957d5c7f8568dd95fcdcd1f"),
		},
		{
			desc:       "Derived from adapted NIST test vector 4 for KDF counter boundary (flip nonce bit 63)",
			key:        dehex("feffe9928665731c6d6a8f9467308308fffee8938764721d6c6b8e9566318209fcfdeb908467711e6f688d96"),
			nonce:      dehex("cafebabefacedb2ddecaf888"),
			aad:        dehex("feedfacedeadbeeffeedfacedeadbeefabaddad2"),
			plaintext:  dehex("d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a721c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39"),
			ciphertext: dehex("8af37ea5684a4d81d4fd817261fd9743099e7e6a025eaacf8e54b124fb5743149e05cb89f4a49467fe2e5e5965f29a19f99416b0016b54585d12553783ba59e9f782e82e097c336bf7989f08"),
		},
		{
			desc:       "Derived from adapted NIST test vector 4 for KDF counter boundary (flip nonce bit 64)",
			key:        dehex("feffe9928665731c6d6a8f9467308308fffee8938764721d6c6b8e9566318209fcfdeb908467711e6f688d96"),
			nonce:      dehex("cafebabefacedbaddfcaf888"),
			aad:        dehex("feedfacedeadbeeffeedfacedeadbeefabaddad2"),
			plaintext:  dehex("d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a721c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39"),
			ciphertext: dehex("fbd528448d0346bfa878634864d407a35a039de9db2f1feb8e965b3ae9356ce6289441d77f8f0df294891f37ea438b223e3bf2bdc53d4c5a74fb680bb312a8dec6f7252cbcd7f5799750ad78"),
		},
		{
			desc:       "Derived from IEEE 2.1.1 54-byte auth",
			key:        dehex("ad7a2bd03eac835a6f620fdcb506b345ac7b2ad13fad825b6e630eddb407b244af7829d23cae81586d600dde"),
			nonce:      dehex("12153524c0895e81b2c28465"),
			aad:        dehex("d609b1f056637a0d46df998d88e5222ab2c2846512153524c0895e8108000f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f30313233340001"),
			plaintext:  dehex(""),
			ciphertext: dehex("3ea0b584f3c85e93f9320ea591699efb"),
		},
		{
			desc:       "Derived from IEEE 2.1.2 54-byte auth",
			key:        dehex("e3c08a8f06c6e3ad95a70557b23f75483ce33021a9c72b7025666204c69c0b72e1c2888d04c4e1af97a50755"),
			nonce:      dehex("12153524c0895e81b2c28465"),
			aad:        dehex("d609b1f056637a0d46df998d88e5222ab2c2846512153524c0895e8108000f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f30313233340001"),
			plaintext:  dehex(""),
			ciphertext: dehex("294e028bf1fe6f14c4e8f7305c933eb5"),
		},
		{
			desc:       "Derived from IEEE 2.2.1 60-byte crypt",
			key:        dehex("ad7a2bd03eac835a6f620fdcb506b345ac7b2ad13fad825b6e630eddb407b244af7829d23cae81586d600dde"),
			nonce:      dehex("12153524c0895e81b2c28465"),
			aad:        dehex("d609b1f056637a0d46df998d88e52e00b2c2846512153524c0895e81"),
			plaintext:  dehex("08000f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a0002"),
			ciphertext: dehex("db3d25719c6b0a3ca6145c159d5c6ed9aff9c6e0b79f17019ea923b8665ddf52137ad611f0d1bf417a7ca85e45afe106ff9c7569d335d086ae6c03f00987ccd6"),
		},
		{
			desc:       "Derived from IEEE 2.2.2 60-byte crypt",
			key:        dehex("e3c08a8f06c6e3ad95a70557b23f75483ce33021a9c72b7025666204c69c0b72e1c2888d04c4e1af97a50755"),
			nonce:      dehex("12153524c0895e81b2c28465"),
			aad:        dehex("d609b1f056637a0d46df998d88e52e00b2c2846512153524c0895e81"),
			plaintext:  dehex("08000f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a0002"),
			ciphertext: dehex("1641f28ec13afcc8f7903389787201051644914933e9202bb9d06aa020c2a67ef51dfe7bc00a856c55b8f8133e77f659132502bad63f5713d57d0c11e0f871ed"),
		},
		{
			desc:       "Derived from IEEE 2.3.1 60-byte auth",
			key:        dehex("071b113b0ca743fecccf3d051f737382061a103a0da642ffcdce3c041e727283051913390ea541fccecd3f07"),
			nonce:      dehex("f0761e8dcd3d000176d457ed"),
			aad:        dehex("e20106d7cd0df0761e8dcd3d88e5400076d457ed08000f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a0003"),
			plaintext:  dehex(""),
			ciphertext: dehex("58837a10562b0f1f8edbe58ca55811d3"),
		},
		{
			desc:       "Derived from IEEE 2.3.2 60-byte auth",
			key:        dehex("691d3ee909d7f54167fd1ca0b5d769081f2bde1aee655fdbab80bd5295ae6be76b1f3ceb0bd5f74365ff1ea2"),
			nonce:      dehex("f0761e8dcd3d000176d457ed"),
			aad:        dehex("e20106d7cd0df0761e8dcd3d88e5400076d457ed08000f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a0003"),
			plaintext:  dehex(""),
			ciphertext: dehex("c2722ff6ca29a257718a529d1f0c6a3b"),
		},
		{
			desc:       "Derived from IEEE 2.4.1 54-byte crypt",
			key:        dehex("071b113b0ca743fecccf3d051f737382061a103a0da642ffcdce3c041e727283051913390ea541fccecd3f07"),
			nonce:      dehex("f0761e8dcd3d000176d457ed"),
			aad:        dehex("e20106d7cd0df0761e8dcd3d88e54c2a76d457ed"),
			plaintext:  dehex("08000f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f30313233340004"),
			ciphertext: dehex("fd96b715b93a13346af51e8acdf792cdc7b2686f8574c70e6b0cbf16291ded427ad73fec48cd298e0528a1f4c644a949fc31dc9279706ddba33f"),
		},
		{
			desc:       "Derived from IEEE 2.4.2 54-byte crypt",
			key:        dehex("691d3ee909d7f54167fd1ca0b5d769081f2bde1aee655fdbab80bd5295ae6be76b1f3ceb0bd5f74365ff1ea2"),
			nonce:      dehex("f0761e8dcd3d000176d457ed"),
			aad:        dehex("e20106d7cd0df0761e8dcd3d88e54c2a76d457ed"),
			plaintext:  dehex("08000f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f30313233340004"),
			ciphertext: dehex("b68f6300c2e9ae833bdc070e24021a3477118e78ccf84e11a485d861476c300f175353d5cdf92008a4f878e6cc3577768085c50a0e98fda6cbb8"),
		},
		{
			desc:       "Derived from IEEE 2.5.1 65-byte auth",
			key:        dehex("013fe00b5f11be7f866d0cbbc55a7a90003ee10a5e10bf7e876c0dbac45b7b91033de2095d13bc7d846f0eb9"),
			nonce:      dehex("7cfde9f9e33724c68932d612"),
			aad:        dehex("84c5d513d2aaf6e5bbd2727788e523008932d6127cfde9f9e33724c608000f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f0005"),
			plaintext:  dehex(""),
			ciphertext: dehex("cca20eecda6283f09bb3543dd99edb9b"),
		},
		{
			desc:       "Derived from IEEE 2.5.2 65-byte auth",
			key:        dehex("83c093b58de7ffe1c0da926ac43fb3609ac1c80fee1b624497ef942e2f79a82381c291b78fe5fde3c2d89068"),
			nonce:      dehex("7cfde9f9e33724c68932d612"),
			aad:        dehex("84c5d513d2aaf6e5bbd2727788e523008932d6127cfde9f9e33724c608000f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f0005"),
			plaintext:  dehex(""),
			ciphertext: dehex("b232cc1da5117bf15003734fa599d271"),
		},
		{
			desc:       "Derived from IEEE  2.6.1 61-byte crypt",
			key:        dehex("013fe00b5f11be7f866d0cbbc55a7a90003ee10a5e10bf7e876c0dbac45b7b91033de2095d13bc7d846f0eb9"),
			nonce:      dehex("7cfde9f9e33724c68932d612"),
			aad:        dehex("84c5d513d2aaf6e5bbd2727788e52f008932d6127cfde9f9e33724c6"),
			plaintext:  dehex("08000f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b0006"),
			ciphertext: dehex("ff1910d35ad7e5657890c7c560146fd038707f204b66edbc3d161f8ace244b985921023c436e3a1c3532ecd5d09a056d70be583f0d10829d9387d07d33d872e490"),
		},
		{
			desc:       "Derived from IEEE 2.6.2 61-byte crypt",
			key:        dehex("83c093b58de7ffe1c0da926ac43fb3609ac1c80fee1b624497ef942e2f79a82381c291b78fe5fde3c2d89068"),
			nonce:      dehex("7cfde9f9e33724c68932d612"),
			aad:        dehex("84c5d513d2aaf6e5bbd2727788e52f008932d6127cfde9f9e33724c6"),
			plaintext:  dehex("08000f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b0006"),
			ciphertext: dehex("0db4cf956b5f97eca4eab82a6955307f9ae02a32dd7d93f83d66ad04e1cfdc5182ad12abdea5bbb619a1bd5fb9a573590fba908e9c7a46c1f7ba0905d1b55ffda4"),
		},
		{
			desc:       "Derived from IEEE 2.7.1 79-byte crypt",
			key:        dehex("88ee087fd95da9fbf6725aa9d757b0cd89ef097ed85ca8faf7735ba8d656b1cc8aec0a7ddb5fabf9f47058ab"),
			nonce:      dehex("7ae8e2ca4ec500012e58495c"),
			aad:        dehex("68f2e77696ce7ae8e2ca4ec588e541002e58495c08000f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f404142434445464748494a4b4c4d0007"),
			plaintext:  dehex(""),
			ciphertext: dehex("813f0e630f96fb2d030f58d83f5cdfd0"),
		},
		{
			desc:       "Derived from IEEE 2.7.2 79-byte crypt",
			key:        dehex("4c973dbc7364621674f8b5b89e5c15511fced9216490fb1c1a2caa0ffe0407e54e953fbe7166601476fab7ba"),
			nonce:      dehex("7ae8e2ca4ec500012e58495c"),
			aad:        dehex("68f2e77696ce7ae8e2ca4ec588e541002e58495c08000f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f404142434445464748494a4b4c4d0007"),
			plaintext:  dehex(""),
			ciphertext: dehex("77e5a44c21eb07188aacbd74d1980e97"),
		},
		{
			desc:       "Derived from IEEE 2.8.1 61-byte crypt",
			key:        dehex("88ee087fd95da9fbf6725aa9d757b0cd89ef097ed85ca8faf7735ba8d656b1cc8aec0a7ddb5fabf9f47058ab"),
			nonce:      dehex("7ae8e2ca4ec500012e58495c"),
			aad:        dehex("68f2e77696ce7ae8e2ca4ec588e54d002e58495c"),
			plaintext:  dehex("08000f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f404142434445464748490008"),
			ciphertext: dehex("958ec3f6d60afeda99efd888f175e5fcd4c87b9bcc5c2f5426253a8b506296c8c43309ab2adb5939462541d95e80811e04e706b1498f2c407c7fb234f8cc01a647550ee6b557b35a7e3945381821f4"),
		},
		{
			desc:       "Derived from IEEE 2.8.2 61-byte crypt",
			key:        dehex("4c973dbc7364621674f8b5b89e5c15511fced9216490fb1c1a2caa0ffe0407e54e953fbe7166601476fab7ba"),
			nonce:      dehex("7ae8e2ca4ec500012e58495c"),
			aad:        dehex("68f2e77696ce7ae8e2ca4ec588e54d002e58495c"),
			plaintext:  dehex("08000f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f404142434445464748490008"),
			ciphertext: dehex("b44d072011cd36d272a9b7a98db9aa90cbc5c67b93ddce67c854503214e2e896ec7e9db649ed4bcf6f850aac0223d0cf92c83db80795c3a17ecc1248bb00591712b1ae71e268164196252162810b00"),
		}} {
		aead, err := newRekeyAEAD(test.key)
		if err != nil {
			t.Fatal("unexpected failure in newRekeyAEAD: ", err.Error())
		}
		if got := aead.Seal(nil, test.nonce, test.plaintext, test.aad); !bytes.Equal(got, test.ciphertext) {
			t.Errorf("Unexpected ciphertext for test vector '%s':\nciphertext=%s\nwant=      %s",
				test.desc, hex.EncodeToString(got), hex.EncodeToString(test.ciphertext))
		}
		if got, err := aead.Open(nil, test.nonce, test.ciphertext, test.aad); err != nil || !bytes.Equal(got, test.plaintext) {
			t.Errorf("Unexpected plaintext for test vector '%s':\nplaintext=%s (err=%v)\nwant=     %s",
				test.desc, hex.EncodeToString(got), err, hex.EncodeToString(test.plaintext))
		}

	}
}

func dehex(s string) []byte {
	if len(s) == 0 {
		return make([]byte, 0)
	}
	b, err := hex.DecodeString(s)
	if err != nil {
		panic(err)
	}
	return b
}
