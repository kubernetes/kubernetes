// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"bytes"
	"crypto"
	"crypto/rand"
	"testing"
)

func TestDefaultCiphersExist(t *testing.T) {
	for _, cipherAlgo := range supportedCiphers {
		if _, ok := cipherModes[cipherAlgo]; !ok {
			t.Errorf("supported cipher %q is unknown", cipherAlgo)
		}
	}
	for _, cipherAlgo := range preferredCiphers {
		if _, ok := cipherModes[cipherAlgo]; !ok {
			t.Errorf("preferred cipher %q is unknown", cipherAlgo)
		}
	}
}

func TestPacketCiphers(t *testing.T) {
	defaultMac := "hmac-sha2-256"
	defaultCipher := "aes128-ctr"
	for cipher := range cipherModes {
		t.Run("cipher="+cipher,
			func(t *testing.T) { testPacketCipher(t, cipher, defaultMac) })
	}
	for mac := range macModes {
		t.Run("mac="+mac,
			func(t *testing.T) { testPacketCipher(t, defaultCipher, mac) })
	}
}

func testPacketCipher(t *testing.T, cipher, mac string) {
	kr := &kexResult{Hash: crypto.SHA1}
	algs := directionAlgorithms{
		Cipher:      cipher,
		MAC:         mac,
		Compression: "none",
	}
	client, err := newPacketCipher(clientKeys, algs, kr)
	if err != nil {
		t.Fatalf("newPacketCipher(client, %q, %q): %v", cipher, mac, err)
	}
	server, err := newPacketCipher(clientKeys, algs, kr)
	if err != nil {
		t.Fatalf("newPacketCipher(client, %q, %q): %v", cipher, mac, err)
	}

	want := "bla bla"
	input := []byte(want)
	buf := &bytes.Buffer{}
	if err := client.writeCipherPacket(0, buf, rand.Reader, input); err != nil {
		t.Fatalf("writeCipherPacket(%q, %q): %v", cipher, mac, err)
	}

	packet, err := server.readCipherPacket(0, buf)
	if err != nil {
		t.Fatalf("readCipherPacket(%q, %q): %v", cipher, mac, err)
	}

	if string(packet) != want {
		t.Errorf("roundtrip(%q, %q): got %q, want %q", cipher, mac, packet, want)
	}
}

func TestCBCOracleCounterMeasure(t *testing.T) {
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
	if err := client.writeCipherPacket(0, buf, rand.Reader, input); err != nil {
		t.Errorf("writeCipherPacket: %v", err)
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
		_, err = server.readCipherPacket(0, fresh)
		if err == nil {
			t.Errorf("corrupt byte %d: readCipherPacket succeeded ", i)
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
