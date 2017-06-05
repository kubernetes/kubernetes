// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package otr

import (
	"bufio"
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"math/big"
	"os"
	"os/exec"
	"testing"
)

var isQueryTests = []struct {
	msg             string
	expectedVersion int
}{
	{"foo", 0},
	{"?OtR", 0},
	{"?OtR?", 0},
	{"?OTR?", 0},
	{"?OTRv?", 0},
	{"?OTRv1?", 0},
	{"?OTR?v1?", 0},
	{"?OTR?v?", 0},
	{"?OTR?v2?", 2},
	{"?OTRv2?", 2},
	{"?OTRv23?", 2},
	{"?OTRv23 ?", 0},
}

func TestIsQuery(t *testing.T) {
	for i, test := range isQueryTests {
		version := isQuery([]byte(test.msg))
		if version != test.expectedVersion {
			t.Errorf("#%d: got %d, want %d", i, version, test.expectedVersion)
		}
	}
}

var alicePrivateKeyHex = "000000000080c81c2cb2eb729b7e6fd48e975a932c638b3a9055478583afa46755683e30102447f6da2d8bec9f386bbb5da6403b0040fee8650b6ab2d7f32c55ab017ae9b6aec8c324ab5844784e9a80e194830d548fb7f09a0410df2c4d5c8bc2b3e9ad484e65412be689cf0834694e0839fb2954021521ffdffb8f5c32c14dbf2020b3ce7500000014da4591d58def96de61aea7b04a8405fe1609308d000000808ddd5cb0b9d66956e3dea5a915d9aba9d8a6e7053b74dadb2fc52f9fe4e5bcc487d2305485ed95fed026ad93f06ebb8c9e8baf693b7887132c7ffdd3b0f72f4002ff4ed56583ca7c54458f8c068ca3e8a4dfa309d1dd5d34e2a4b68e6f4338835e5e0fb4317c9e4c7e4806dafda3ef459cd563775a586dd91b1319f72621bf3f00000080b8147e74d8c45e6318c37731b8b33b984a795b3653c2cd1d65cc99efe097cb7eb2fa49569bab5aab6e8a1c261a27d0f7840a5e80b317e6683042b59b6dceca2879c6ffc877a465be690c15e4a42f9a7588e79b10faac11b1ce3741fcef7aba8ce05327a2c16d279ee1b3d77eb783fb10e3356caa25635331e26dd42b8396c4d00000001420bec691fea37ecea58a5c717142f0b804452f57"

var aliceFingerprintHex = "0bb01c360424522e94ee9c346ce877a1a4288b2f"

var bobPrivateKeyHex = "000000000080a5138eb3d3eb9c1d85716faecadb718f87d31aaed1157671d7fee7e488f95e8e0ba60ad449ec732710a7dec5190f7182af2e2f98312d98497221dff160fd68033dd4f3a33b7c078d0d9f66e26847e76ca7447d4bab35486045090572863d9e4454777f24d6706f63e02548dfec2d0a620af37bbc1d24f884708a212c343b480d00000014e9c58f0ea21a5e4dfd9f44b6a9f7f6a9961a8fa9000000803c4d111aebd62d3c50c2889d420a32cdf1e98b70affcc1fcf44d59cca2eb019f6b774ef88153fb9b9615441a5fe25ea2d11b74ce922ca0232bd81b3c0fcac2a95b20cb6e6c0c5c1ace2e26f65dc43c751af0edbb10d669890e8ab6beea91410b8b2187af1a8347627a06ecea7e0f772c28aae9461301e83884860c9b656c722f0000008065af8625a555ea0e008cd04743671a3cda21162e83af045725db2eb2bb52712708dc0cc1a84c08b3649b88a966974bde27d8612c2861792ec9f08786a246fcadd6d8d3a81a32287745f309238f47618c2bd7612cb8b02d940571e0f30b96420bcd462ff542901b46109b1e5ad6423744448d20a57818a8cbb1647d0fea3b664e0000001440f9f2eb554cb00d45a5826b54bfa419b6980e48"

func TestKeySerialization(t *testing.T) {
	var priv PrivateKey
	alicePrivateKey, _ := hex.DecodeString(alicePrivateKeyHex)
	rest, ok := priv.Parse(alicePrivateKey)
	if !ok {
		t.Error("failed to parse private key")
	}
	if len(rest) > 0 {
		t.Error("data remaining after parsing private key")
	}

	out := priv.Serialize(nil)
	if !bytes.Equal(alicePrivateKey, out) {
		t.Errorf("serialization (%x) is not equal to original (%x)", out, alicePrivateKey)
	}

	aliceFingerprint, _ := hex.DecodeString(aliceFingerprintHex)
	fingerprint := priv.PublicKey.Fingerprint()
	if !bytes.Equal(aliceFingerprint, fingerprint) {
		t.Errorf("fingerprint (%x) is not equal to expected value (%x)", fingerprint, aliceFingerprint)
	}
}

const libOTRPrivateKey = `(privkeys
 (account
(name "foo@example.com")
(protocol prpl-jabber)
(private-key 
 (dsa 
  (p #00FC07ABCF0DC916AFF6E9AE47BEF60C7AB9B4D6B2469E436630E36F8A489BE812486A09F30B71224508654940A835301ACC525A4FF133FC152CC53DCC59D65C30A54F1993FE13FE63E5823D4C746DB21B90F9B9C00B49EC7404AB1D929BA7FBA12F2E45C6E0A651689750E8528AB8C031D3561FECEE72EBB4A090D450A9B7A857#)
  (q #00997BD266EF7B1F60A5C23F3A741F2AEFD07A2081#)
  (g #535E360E8A95EBA46A4F7DE50AD6E9B2A6DB785A66B64EB9F20338D2A3E8FB0E94725848F1AA6CC567CB83A1CC517EC806F2E92EAE71457E80B2210A189B91250779434B41FC8A8873F6DB94BEA7D177F5D59E7E114EE10A49CFD9CEF88AE43387023B672927BA74B04EB6BBB5E57597766A2F9CE3857D7ACE3E1E3BC1FC6F26#)
  (y #0AC8670AD767D7A8D9D14CC1AC6744CD7D76F993B77FFD9E39DF01E5A6536EF65E775FCEF2A983E2A19BD6415500F6979715D9FD1257E1FE2B6F5E1E74B333079E7C880D39868462A93454B41877BE62E5EF0A041C2EE9C9E76BD1E12AE25D9628DECB097025DD625EF49C3258A1A3C0FF501E3DC673B76D7BABF349009B6ECF#)
  (x #14D0345A3562C480A039E3C72764F72D79043216#)
  )
 )
 )
)`

func TestParseLibOTRPrivateKey(t *testing.T) {
	var priv PrivateKey

	if !priv.Import([]byte(libOTRPrivateKey)) {
		t.Fatalf("Failed to import sample private key")
	}
}

func TestSignVerify(t *testing.T) {
	var priv PrivateKey
	alicePrivateKey, _ := hex.DecodeString(alicePrivateKeyHex)
	_, ok := priv.Parse(alicePrivateKey)
	if !ok {
		t.Error("failed to parse private key")
	}

	var msg [32]byte
	rand.Reader.Read(msg[:])

	sig := priv.Sign(rand.Reader, msg[:])
	rest, ok := priv.PublicKey.Verify(msg[:], sig)
	if !ok {
		t.Errorf("signature (%x) of %x failed to verify", sig, msg[:])
	} else if len(rest) > 0 {
		t.Error("signature data remains after verification")
	}

	sig[10] ^= 80
	_, ok = priv.PublicKey.Verify(msg[:], sig)
	if ok {
		t.Errorf("corrupted signature (%x) of %x verified", sig, msg[:])
	}
}

func setupConversation(t *testing.T) (alice, bob *Conversation) {
	alicePrivateKey, _ := hex.DecodeString(alicePrivateKeyHex)
	bobPrivateKey, _ := hex.DecodeString(bobPrivateKeyHex)

	alice, bob = new(Conversation), new(Conversation)

	alice.PrivateKey = new(PrivateKey)
	bob.PrivateKey = new(PrivateKey)
	alice.PrivateKey.Parse(alicePrivateKey)
	bob.PrivateKey.Parse(bobPrivateKey)
	alice.FragmentSize = 100
	bob.FragmentSize = 100

	if alice.IsEncrypted() {
		t.Error("Alice believes that the conversation is secure before we've started")
	}
	if bob.IsEncrypted() {
		t.Error("Bob believes that the conversation is secure before we've started")
	}

	performHandshake(t, alice, bob)
	return alice, bob
}

func performHandshake(t *testing.T, alice, bob *Conversation) {
	var alicesMessage, bobsMessage [][]byte
	var out []byte
	var aliceChange, bobChange SecurityChange
	var err error
	alicesMessage = append(alicesMessage, []byte(QueryMessage))

	for round := 0; len(alicesMessage) > 0 || len(bobsMessage) > 0; round++ {
		bobsMessage = nil
		for i, msg := range alicesMessage {
			out, _, bobChange, bobsMessage, err = bob.Receive(msg)
			if len(out) > 0 {
				t.Errorf("Bob generated output during key exchange, round %d, message %d", round, i)
			}
			if err != nil {
				t.Fatalf("Bob returned an error, round %d, message %d (%x): %s", round, i, msg, err)
			}
			if len(bobsMessage) > 0 && i != len(alicesMessage)-1 {
				t.Errorf("Bob produced output while processing a fragment, round %d, message %d", round, i)
			}
		}

		alicesMessage = nil
		for i, msg := range bobsMessage {
			out, _, aliceChange, alicesMessage, err = alice.Receive(msg)
			if len(out) > 0 {
				t.Errorf("Alice generated output during key exchange, round %d, message %d", round, i)
			}
			if err != nil {
				t.Fatalf("Alice returned an error, round %d, message %d (%x): %s", round, i, msg, err)
			}
			if len(alicesMessage) > 0 && i != len(bobsMessage)-1 {
				t.Errorf("Alice produced output while processing a fragment, round %d, message %d", round, i)
			}
		}
	}

	if aliceChange != NewKeys {
		t.Errorf("Alice terminated without signaling new keys")
	}
	if bobChange != NewKeys {
		t.Errorf("Bob terminated without signaling new keys")
	}

	if !bytes.Equal(alice.SSID[:], bob.SSID[:]) {
		t.Errorf("Session identifiers don't match. Alice has %x, Bob has %x", alice.SSID[:], bob.SSID[:])
	}

	if !alice.IsEncrypted() {
		t.Error("Alice doesn't believe that the conversation is secure")
	}
	if !bob.IsEncrypted() {
		t.Error("Bob doesn't believe that the conversation is secure")
	}
}

const (
	firstRoundTrip = iota
	subsequentRoundTrip
	noMACKeyCheck
)

func roundTrip(t *testing.T, alice, bob *Conversation, message []byte, macKeyCheck int) {
	alicesMessage, err := alice.Send(message)
	if err != nil {
		t.Errorf("Error from Alice sending message: %s", err)
	}

	if len(alice.oldMACs) != 0 {
		t.Errorf("Alice has not revealed all MAC keys")
	}

	for i, msg := range alicesMessage {
		out, encrypted, _, _, err := bob.Receive(msg)

		if err != nil {
			t.Errorf("Error generated while processing test message: %s", err.Error())
		}
		if len(out) > 0 {
			if i != len(alicesMessage)-1 {
				t.Fatal("Bob produced a message while processing a fragment of Alice's")
			}
			if !encrypted {
				t.Errorf("Message was not marked as encrypted")
			}
			if !bytes.Equal(out, message) {
				t.Errorf("Message corrupted: got %x, want %x", out, message)
			}
		}
	}

	switch macKeyCheck {
	case firstRoundTrip:
		if len(bob.oldMACs) != 0 {
			t.Errorf("Bob should not have MAC keys to reveal")
		}
	case subsequentRoundTrip:
		if len(bob.oldMACs) != 40 {
			t.Errorf("Bob has %d bytes of MAC keys to reveal, but should have 40", len(bob.oldMACs))
		}
	}

	bobsMessage, err := bob.Send(message)
	if err != nil {
		t.Errorf("Error from Bob sending message: %s", err)
	}

	if len(bob.oldMACs) != 0 {
		t.Errorf("Bob has not revealed all MAC keys")
	}

	for i, msg := range bobsMessage {
		out, encrypted, _, _, err := alice.Receive(msg)

		if err != nil {
			t.Errorf("Error generated while processing test message: %s", err.Error())
		}
		if len(out) > 0 {
			if i != len(bobsMessage)-1 {
				t.Fatal("Alice produced a message while processing a fragment of Bob's")
			}
			if !encrypted {
				t.Errorf("Message was not marked as encrypted")
			}
			if !bytes.Equal(out, message) {
				t.Errorf("Message corrupted: got %x, want %x", out, message)
			}
		}
	}

	switch macKeyCheck {
	case firstRoundTrip:
		if len(alice.oldMACs) != 20 {
			t.Errorf("Alice has %d bytes of MAC keys to reveal, but should have 20", len(alice.oldMACs))
		}
	case subsequentRoundTrip:
		if len(alice.oldMACs) != 40 {
			t.Errorf("Alice has %d bytes of MAC keys to reveal, but should have 40", len(alice.oldMACs))
		}
	}
}

func TestConversation(t *testing.T) {
	alice, bob := setupConversation(t)

	var testMessages = [][]byte{
		[]byte("hello"), []byte("bye"),
	}

	roundTripType := firstRoundTrip

	for _, testMessage := range testMessages {
		roundTrip(t, alice, bob, testMessage, roundTripType)
		roundTripType = subsequentRoundTrip
	}
}

func TestGoodSMP(t *testing.T) {
	var alice, bob Conversation

	alice.smp.secret = new(big.Int).SetInt64(42)
	bob.smp.secret = alice.smp.secret

	var alicesMessages, bobsMessages []tlv
	var aliceComplete, bobComplete bool
	var err error
	var out tlv

	alicesMessages = alice.startSMP("")
	for round := 0; len(alicesMessages) > 0 || len(bobsMessages) > 0; round++ {
		bobsMessages = bobsMessages[:0]
		for i, msg := range alicesMessages {
			out, bobComplete, err = bob.processSMP(msg)
			if err != nil {
				t.Errorf("Error from Bob in round %d: %s", round, err)
			}
			if bobComplete && i != len(alicesMessages)-1 {
				t.Errorf("Bob returned a completed signal before processing all of Alice's messages in round %d", round)
			}
			if out.typ != 0 {
				bobsMessages = append(bobsMessages, out)
			}
		}

		alicesMessages = alicesMessages[:0]
		for i, msg := range bobsMessages {
			out, aliceComplete, err = alice.processSMP(msg)
			if err != nil {
				t.Errorf("Error from Alice in round %d: %s", round, err)
			}
			if aliceComplete && i != len(bobsMessages)-1 {
				t.Errorf("Alice returned a completed signal before processing all of Bob's messages in round %d", round)
			}
			if out.typ != 0 {
				alicesMessages = append(alicesMessages, out)
			}
		}
	}

	if !aliceComplete || !bobComplete {
		t.Errorf("SMP completed without both sides reporting success: alice: %v, bob: %v\n", aliceComplete, bobComplete)
	}
}

func TestBadSMP(t *testing.T) {
	var alice, bob Conversation

	alice.smp.secret = new(big.Int).SetInt64(42)
	bob.smp.secret = new(big.Int).SetInt64(43)

	var alicesMessages, bobsMessages []tlv

	alicesMessages = alice.startSMP("")
	for round := 0; len(alicesMessages) > 0 || len(bobsMessages) > 0; round++ {
		bobsMessages = bobsMessages[:0]
		for _, msg := range alicesMessages {
			out, complete, _ := bob.processSMP(msg)
			if complete {
				t.Errorf("Bob signaled completion in round %d", round)
			}
			if out.typ != 0 {
				bobsMessages = append(bobsMessages, out)
			}
		}

		alicesMessages = alicesMessages[:0]
		for _, msg := range bobsMessages {
			out, complete, _ := alice.processSMP(msg)
			if complete {
				t.Errorf("Alice signaled completion in round %d", round)
			}
			if out.typ != 0 {
				alicesMessages = append(alicesMessages, out)
			}
		}
	}
}

func TestRehandshaking(t *testing.T) {
	alice, bob := setupConversation(t)
	roundTrip(t, alice, bob, []byte("test"), firstRoundTrip)
	roundTrip(t, alice, bob, []byte("test 2"), subsequentRoundTrip)
	roundTrip(t, alice, bob, []byte("test 3"), subsequentRoundTrip)
	roundTrip(t, alice, bob, []byte("test 4"), subsequentRoundTrip)
	roundTrip(t, alice, bob, []byte("test 5"), subsequentRoundTrip)
	roundTrip(t, alice, bob, []byte("test 6"), subsequentRoundTrip)
	roundTrip(t, alice, bob, []byte("test 7"), subsequentRoundTrip)
	roundTrip(t, alice, bob, []byte("test 8"), subsequentRoundTrip)
	performHandshake(t, alice, bob)
	roundTrip(t, alice, bob, []byte("test"), noMACKeyCheck)
	roundTrip(t, alice, bob, []byte("test 2"), noMACKeyCheck)
}

func TestAgainstLibOTR(t *testing.T) {
	// This test requires otr.c.test to be built as /tmp/a.out.
	// If enabled, this tests runs forever performing OTR handshakes in a
	// loop.
	return

	alicePrivateKey, _ := hex.DecodeString(alicePrivateKeyHex)
	var alice Conversation
	alice.PrivateKey = new(PrivateKey)
	alice.PrivateKey.Parse(alicePrivateKey)

	cmd := exec.Command("/tmp/a.out")
	cmd.Stderr = os.Stderr

	out, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	in := bufio.NewReader(stdout)

	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}

	out.Write([]byte(QueryMessage))
	out.Write([]byte("\n"))
	var expectedText = []byte("test message")

	for {
		line, isPrefix, err := in.ReadLine()
		if isPrefix {
			t.Fatal("line from subprocess too long")
		}
		if err != nil {
			t.Fatal(err)
		}
		text, encrypted, change, alicesMessage, err := alice.Receive(line)
		if err != nil {
			t.Fatal(err)
		}
		for _, msg := range alicesMessage {
			out.Write(msg)
			out.Write([]byte("\n"))
		}
		if change == NewKeys {
			alicesMessage, err := alice.Send([]byte("Go -> libotr test message"))
			if err != nil {
				t.Fatalf("error sending message: %s", err.Error())
			} else {
				for _, msg := range alicesMessage {
					out.Write(msg)
					out.Write([]byte("\n"))
				}
			}
		}
		if len(text) > 0 {
			if !bytes.Equal(text, expectedText) {
				t.Fatalf("expected %x, but got %x", expectedText, text)
			}
			if !encrypted {
				t.Fatal("message wasn't encrypted")
			}
		}
	}
}
