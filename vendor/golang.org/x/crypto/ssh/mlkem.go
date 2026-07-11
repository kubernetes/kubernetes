// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"crypto"
	"crypto/mlkem"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"

	"golang.org/x/crypto/curve25519"
)

// mlkem768WithCurve25519sha256 implements the hybrid ML-KEM768 with
// curve25519-sha256 key exchange method, as described by
// draft-kampanakis-curdle-ssh-pq-ke-05 section 2.3.3.
type mlkem768WithCurve25519sha256 struct{}

func (kex *mlkem768WithCurve25519sha256) Client(c packetConn, rand io.Reader, magics *handshakeMagics) (*kexResult, error) {
	var c25519kp curve25519KeyPair
	if err := c25519kp.generate(rand); err != nil {
		return nil, err
	}

	seed := make([]byte, mlkem.SeedSize)
	if _, err := io.ReadFull(rand, seed); err != nil {
		return nil, err
	}

	mlkemDk, err := mlkem.NewDecapsulationKey768(seed)
	if err != nil {
		return nil, err
	}

	hybridKey := append(mlkemDk.EncapsulationKey().Bytes(), c25519kp.pub[:]...)
	if err := c.writePacket(Marshal(&kexECDHInitMsg{hybridKey})); err != nil {
		return nil, err
	}

	packet, err := c.readPacket()
	if err != nil {
		return nil, err
	}

	var reply kexECDHReplyMsg
	if err = Unmarshal(packet, &reply); err != nil {
		return nil, err
	}

	if len(reply.EphemeralPubKey) != mlkem.CiphertextSize768+32 {
		return nil, errors.New("ssh: peer's mlkem768x25519 public value has wrong length")
	}

	// Perform KEM decapsulate operation to obtain shared key from ML-KEM.
	mlkem768Secret, err := mlkemDk.Decapsulate(reply.EphemeralPubKey[:mlkem.CiphertextSize768])
	if err != nil {
		return nil, err
	}

	// Complete Curve25519 ECDH to obtain its shared key.
	c25519Secret, err := curve25519.X25519(c25519kp.priv[:], reply.EphemeralPubKey[mlkem.CiphertextSize768:])
	if err != nil {
		return nil, fmt.Errorf("ssh: peer's mlkem768x25519 public value is not valid: %w", err)
	}
	// Compute actual shared key.
	h := sha256.New()
	h.Write(mlkem768Secret)
	h.Write(c25519Secret)
	secret := h.Sum(nil)

	h.Reset()
	magics.write(h)
	writeString(h, reply.HostKey)
	writeString(h, hybridKey)
	writeString(h, reply.EphemeralPubKey)

	K := make([]byte, stringLength(len(secret)))
	marshalString(K, secret)
	h.Write(K)

	return &kexResult{
		H:         h.Sum(nil),
		K:         K,
		HostKey:   reply.HostKey,
		Signature: reply.Signature,
		Hash:      crypto.SHA256,
	}, nil
}

func (kex *mlkem768WithCurve25519sha256) Server(c packetConn, rand io.Reader, magics *handshakeMagics, priv AlgorithmSigner, algo string) (*kexResult, error) {
	packet, err := c.readPacket()
	if err != nil {
		return nil, err
	}

	var kexInit kexECDHInitMsg
	if err = Unmarshal(packet, &kexInit); err != nil {
		return nil, err
	}

	if len(kexInit.ClientPubKey) != mlkem.EncapsulationKeySize768+32 {
		return nil, errors.New("ssh: peer's ML-KEM768/curve25519 public value has wrong length")
	}

	encapsulationKey, err := mlkem.NewEncapsulationKey768(kexInit.ClientPubKey[:mlkem.EncapsulationKeySize768])
	if err != nil {
		return nil, fmt.Errorf("ssh: peer's ML-KEM768 encapsulation key is not valid: %w", err)
	}
	// Perform KEM encapsulate operation to obtain ciphertext and shared key.
	mlkem768Secret, mlkem768Ciphertext := encapsulationKey.Encapsulate()

	// Perform server side of Curve25519 ECDH to obtain server public value and
	// shared key.
	var c25519kp curve25519KeyPair
	if err := c25519kp.generate(rand); err != nil {
		return nil, err
	}
	c25519Secret, err := curve25519.X25519(c25519kp.priv[:], kexInit.ClientPubKey[mlkem.EncapsulationKeySize768:])
	if err != nil {
		return nil, fmt.Errorf("ssh: peer's ML-KEM768/curve25519 public value is not valid: %w", err)
	}
	hybridKey := append(mlkem768Ciphertext, c25519kp.pub[:]...)

	// Compute actual shared key.
	h := sha256.New()
	h.Write(mlkem768Secret)
	h.Write(c25519Secret)
	secret := h.Sum(nil)

	hostKeyBytes := priv.PublicKey().Marshal()

	h.Reset()
	magics.write(h)
	writeString(h, hostKeyBytes)
	writeString(h, kexInit.ClientPubKey)
	writeString(h, hybridKey)

	K := make([]byte, stringLength(len(secret)))
	marshalString(K, secret)
	h.Write(K)

	H := h.Sum(nil)

	sig, err := signAndMarshal(priv, rand, H, algo)
	if err != nil {
		return nil, err
	}

	reply := kexECDHReplyMsg{
		EphemeralPubKey: hybridKey,
		HostKey:         hostKeyBytes,
		Signature:       sig,
	}
	if err := c.writePacket(Marshal(&reply)); err != nil {
		return nil, err
	}
	return &kexResult{
		H:         H,
		K:         K,
		HostKey:   hostKeyBytes,
		Signature: sig,
		Hash:      crypto.SHA256,
	}, nil
}
