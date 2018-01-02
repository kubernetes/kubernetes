// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"bytes"
	"crypto"
	"crypto/aes"
	"crypto/rand"
	"testing"
)

func TestDefaultCiphersExist(t *testing.T) {
	for _, cipherAlgo := range supportedCiphers {
		if _, ok := cipherModes[cipherAlgo]; !ok {
			t.Errorf("default cipher %q is unknown", cipherAlgo)
		}
	}
}

func TestPacketCiphers(t *testing.T) {
	// Still test aes128cbc cipher although it's commented out.
	cipherModes[aes128cbcID] = &streamCipherMode{16, aes.BlockSize, 0, nil}
	defer delete(cipherModes, aes128cbcID)

	for cipher := range cipherModes {
		for mac := range macModes {
			kr := &kexResult{Hash: crypto.SHA1}
			algs := directionAlgorithms{
				Cipher:      cipher,
				MAC:         mac,
				Compression: "none",
			}
			client, err := newPacketCipher(clientKeys, algs, kr)
			if err != nil {
				t.Errorf("newPacketCipher(client, %q, %q): %v", cipher, mac, err)
				continue
			}
			server, err := newPacketCipher(clientKeys, algs, kr)
			if err != nil {
				t.Errorf("newPacketCipher(client, %q, %q): %v", cipher, mac, err)
				continue
			}

			want := "bla bla"
			input := []byte(want)
			buf := &bytes.Buffer{}
			if err := client.writePacket(0, buf, rand.Reader, input); err != nil {
				t.Errorf("writePacket(%q, %q): %v", cipher, mac, err)
				continue
			}

			packet, err := server.readPacket(0, buf)
			if err != nil {
				t.Errorf("readPacket(%q, %q): %v", cipher, mac, err)
				continue
			}

			if string(packet) != want {
				t.Errorf("roundtrip(%q, %q): got %q, want %q", cipher, mac, packet, want)
			}
		}
	}
}

func TestCBCOracleCounterMeasure(t *testing.T) {
	cipherModes[aes128cbcID] = &streamCipherMode{16, aes.BlockSize, 0, nil}
	defer delete(cipherModes, aes128cbcID)

	kr := &kexResult{Hash: crypto.SHA1}
	algs := directionAlgorithms{
		Cipher:      aes128cbcID,
		MAC:         "hmac-sha1",
		Compression: "none",
	}
	client, err := newPacketCipher(clientKeys, algs, kr)
	if err != nil {
		t.Fatalf("newPacketCipher(client): %v", err)
	}

	want := "bla bla"
	input := []byte(want)
	buf := &bytes.Buffer{}
	if err := client.writePacket(0, buf, rand.Reader, input); err != nil {
		t.Errorf("writePacket: %v", err)
	}

	packetSize := buf.Len()
	buf.Write(make([]byte, 2*maxPacket))

	// We corrupt each byte, but this usually will only test the
	// 'packet too large' or 'MAC failure' cases.
	lastRead := -1
	for i := 0; i < packetSize; i++ {
		server, err := newPacketCipher(clientKeys, algs, kr)
		if err != nil {
			t.Fatalf("newPacketCipher(client): %v", err)
		}

		fresh := &bytes.Buffer{}
		fresh.Write(buf.Bytes())
		fresh.Bytes()[i] ^= 0x01

		before := fresh.Len()
		_, err = server.readPacket(0, fresh)
		if err == nil {
			t.Errorf("corrupt byte %d: readPacket succeeded ", i)
			continue
		}
		if _, ok := err.(cbcError); !ok {
			t.Errorf("corrupt byte %d: got %v (%T), want cbcError", i, err, err)
			continue
		}

		after := fresh.Len()
		bytesRead := before - after
		if bytesRead < maxPacket {
			t.Errorf("corrupt byte %d: read %d bytes, want more than %d", i, bytesRead, maxPacket)
			continue
		}

		if i > 0 && bytesRead != lastRead {
			t.Errorf("corrupt byte %d: read %d bytes, want %d bytes read", i, bytesRead, lastRead)
		}
		lastRead = bytesRead
	}
}
