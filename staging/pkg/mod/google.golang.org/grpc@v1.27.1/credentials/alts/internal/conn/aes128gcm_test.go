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
	"testing"

	core "google.golang.org/grpc/credentials/alts/internal"
)

// cryptoTestVector is struct for a GCM test vector
type cryptoTestVector struct {
	key, counter, plaintext, ciphertext, tag []byte
	allocateDst                              bool
}

// getGCMCryptoPair outputs a client/server pair on aes128gcm.
func getGCMCryptoPair(key []byte, counter []byte, t *testing.T) (ALTSRecordCrypto, ALTSRecordCrypto) {
	client, err := NewAES128GCM(core.ClientSide, key)
	if err != nil {
		t.Fatalf("NewAES128GCM(ClientSide, key) = %v", err)
	}
	server, err := NewAES128GCM(core.ServerSide, key)
	if err != nil {
		t.Fatalf("NewAES128GCM(ServerSide, key) = %v", err)
	}
	// set counter if provided.
	if counter != nil {
		if CounterSide(counter) == core.ClientSide {
			client.(*aes128gcm).outCounter = CounterFromValue(counter, overflowLenAES128GCM)
			server.(*aes128gcm).inCounter = CounterFromValue(counter, overflowLenAES128GCM)
		} else {
			server.(*aes128gcm).outCounter = CounterFromValue(counter, overflowLenAES128GCM)
			client.(*aes128gcm).inCounter = CounterFromValue(counter, overflowLenAES128GCM)
		}
	}
	return client, server
}

func testGCMEncryptionDecryption(sender ALTSRecordCrypto, receiver ALTSRecordCrypto, test *cryptoTestVector, withCounter bool, t *testing.T) {
	// Ciphertext is: counter + encrypted text + tag.
	ciphertext := []byte(nil)
	if withCounter {
		ciphertext = append(ciphertext, test.counter...)
	}
	ciphertext = append(ciphertext, test.ciphertext...)
	ciphertext = append(ciphertext, test.tag...)

	// Decrypt.
	if got, err := receiver.Decrypt(nil, ciphertext); err != nil || !bytes.Equal(got, test.plaintext) {
		t.Errorf("key=%v\ncounter=%v\ntag=%v\nciphertext=%v\nDecrypt = %v, %v\nwant: %v",
			test.key, test.counter, test.tag, test.ciphertext, got, err, test.plaintext)
	}

	// Encrypt.
	var dst []byte
	if test.allocateDst {
		dst = make([]byte, len(test.plaintext)+sender.EncryptionOverhead())
	}
	if got, err := sender.Encrypt(dst[:0], test.plaintext); err != nil || !bytes.Equal(got, ciphertext) {
		t.Errorf("key=%v\ncounter=%v\nplaintext=%v\nEncrypt = %v, %v\nwant: %v",
			test.key, test.counter, test.plaintext, got, err, ciphertext)
	}
}

// Test encrypt and decrypt using test vectors for aes128gcm.
func TestAES128GCMEncrypt(t *testing.T) {
	for _, test := range []cryptoTestVector{
		{
			key:         dehex("11754cd72aec309bf52f7687212e8957"),
			counter:     dehex("3c819d9a9bed087615030b65"),
			plaintext:   nil,
			ciphertext:  nil,
			tag:         dehex("250327c674aaf477aef2675748cf6971"),
			allocateDst: false,
		},
		{
			key:         dehex("ca47248ac0b6f8372a97ac43508308ed"),
			counter:     dehex("ffd2b598feabc9019262d2be"),
			plaintext:   nil,
			ciphertext:  nil,
			tag:         dehex("60d20404af527d248d893ae495707d1a"),
			allocateDst: false,
		},
		{
			key:         dehex("7fddb57453c241d03efbed3ac44e371c"),
			counter:     dehex("ee283a3fc75575e33efd4887"),
			plaintext:   dehex("d5de42b461646c255c87bd2962d3b9a2"),
			ciphertext:  dehex("2ccda4a5415cb91e135c2a0f78c9b2fd"),
			tag:         dehex("b36d1df9b9d5e596f83e8b7f52971cb3"),
			allocateDst: false,
		},
		{
			key:         dehex("ab72c77b97cb5fe9a382d9fe81ffdbed"),
			counter:     dehex("54cc7dc2c37ec006bcc6d1da"),
			plaintext:   dehex("007c5e5b3e59df24a7c355584fc1518d"),
			ciphertext:  dehex("0e1bde206a07a9c2c1b65300f8c64997"),
			tag:         dehex("2b4401346697138c7a4891ee59867d0c"),
			allocateDst: false,
		},
		{
			key:         dehex("11754cd72aec309bf52f7687212e8957"),
			counter:     dehex("3c819d9a9bed087615030b65"),
			plaintext:   nil,
			ciphertext:  nil,
			tag:         dehex("250327c674aaf477aef2675748cf6971"),
			allocateDst: true,
		},
		{
			key:         dehex("ca47248ac0b6f8372a97ac43508308ed"),
			counter:     dehex("ffd2b598feabc9019262d2be"),
			plaintext:   nil,
			ciphertext:  nil,
			tag:         dehex("60d20404af527d248d893ae495707d1a"),
			allocateDst: true,
		},
		{
			key:         dehex("7fddb57453c241d03efbed3ac44e371c"),
			counter:     dehex("ee283a3fc75575e33efd4887"),
			plaintext:   dehex("d5de42b461646c255c87bd2962d3b9a2"),
			ciphertext:  dehex("2ccda4a5415cb91e135c2a0f78c9b2fd"),
			tag:         dehex("b36d1df9b9d5e596f83e8b7f52971cb3"),
			allocateDst: true,
		},
		{
			key:         dehex("ab72c77b97cb5fe9a382d9fe81ffdbed"),
			counter:     dehex("54cc7dc2c37ec006bcc6d1da"),
			plaintext:   dehex("007c5e5b3e59df24a7c355584fc1518d"),
			ciphertext:  dehex("0e1bde206a07a9c2c1b65300f8c64997"),
			tag:         dehex("2b4401346697138c7a4891ee59867d0c"),
			allocateDst: true,
		},
	} {
		// Test encryption and decryption for aes128gcm.
		client, server := getGCMCryptoPair(test.key, test.counter, t)
		if CounterSide(test.counter) == core.ClientSide {
			testGCMEncryptionDecryption(client, server, &test, false, t)
		} else {
			testGCMEncryptionDecryption(server, client, &test, false, t)
		}
	}
}

func testGCMEncryptRoundtrip(client ALTSRecordCrypto, server ALTSRecordCrypto, t *testing.T) {
	// Encrypt.
	const plaintext = "This is plaintext."
	var err error
	buf := []byte(plaintext)
	buf, err = client.Encrypt(buf[:0], buf)
	if err != nil {
		t.Fatal("Encrypting with client-side context: unexpected error", err, "\n",
			"Plaintext:", []byte(plaintext))
	}

	// Encrypt a second message.
	const plaintext2 = "This is a second plaintext."
	buf2 := []byte(plaintext2)
	buf2, err = client.Encrypt(buf2[:0], buf2)
	if err != nil {
		t.Fatal("Encrypting with client-side context: unexpected error", err, "\n",
			"Plaintext:", []byte(plaintext2))
	}

	// Decryption fails: cannot decrypt second message before first.
	if got, err := server.Decrypt(nil, buf2); err == nil {
		t.Error("Decrypting client-side ciphertext with a client-side context unexpectedly succeeded; want unexpected counter error:\n",
			"  Original plaintext:", []byte(plaintext2), "\n",
			"  Ciphertext:", buf2, "\n",
			"  Decrypted plaintext:", got)
	}

	// Decryption fails: wrong counter space.
	if got, err := client.Decrypt(nil, buf); err == nil {
		t.Error("Decrypting client-side ciphertext with a client-side context unexpectedly succeeded; want counter space error:\n",
			"  Original plaintext:", []byte(plaintext), "\n",
			"  Ciphertext:", buf, "\n",
			"  Decrypted plaintext:", got)
	}

	// Decrypt first message.
	ciphertext := append([]byte(nil), buf...)
	buf, err = server.Decrypt(buf[:0], buf)
	if err != nil || string(buf) != plaintext {
		t.Fatal("Decrypting client-side ciphertext with a server-side context did not produce original content:\n",
			"  Original plaintext:", []byte(plaintext), "\n",
			"  Ciphertext:", ciphertext, "\n",
			"  Decryption error:", err, "\n",
			"  Decrypted plaintext:", buf)
	}

	// Decryption fails: replay attack.
	if got, err := server.Decrypt(nil, buf); err == nil {
		t.Error("Decrypting client-side ciphertext with a client-side context unexpectedly succeeded; want unexpected counter error:\n",
			"  Original plaintext:", []byte(plaintext), "\n",
			"  Ciphertext:", buf, "\n",
			"  Decrypted plaintext:", got)
	}
}

// Test encrypt and decrypt on roundtrip messages for aes128gcm.
func TestAES128GCMEncryptRoundtrip(t *testing.T) {
	// Test for aes128gcm.
	key := make([]byte, 16)
	client, server := getGCMCryptoPair(key, nil, t)
	testGCMEncryptRoundtrip(client, server, t)
}
