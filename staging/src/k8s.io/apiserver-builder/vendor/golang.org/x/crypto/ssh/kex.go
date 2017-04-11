// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/subtle"
	"errors"
	"io"
	"math/big"

	"golang.org/x/crypto/curve25519"
)

const (
	kexAlgoDH1SHA1          = "diffie-hellman-group1-sha1"
	kexAlgoDH14SHA1         = "diffie-hellman-group14-sha1"
	kexAlgoECDH256          = "ecdh-sha2-nistp256"
	kexAlgoECDH384          = "ecdh-sha2-nistp384"
	kexAlgoECDH521          = "ecdh-sha2-nistp521"
	kexAlgoCurve25519SHA256 = "curve25519-sha256@libssh.org"
)

// kexResult captures the outcome of a key exchange.
type kexResult struct {
	// Session hash. See also RFC 4253, section 8.
	H []byte

	// Shared secret. See also RFC 4253, section 8.
	K []byte

	// Host key as hashed into H.
	HostKey []byte

	// Signature of H.
	Signature []byte

	// A cryptographic hash function that matches the security
	// level of the key exchange algorithm. It is used for
	// calculating H, and for deriving keys from H and K.
	Hash crypto.Hash

	// The session ID, which is the first H computed. This is used
	// to derive key material inside the transport.
	SessionID []byte
}

// handshakeMagics contains data that is always included in the
// session hash.
type handshakeMagics struct {
	clientVersion, serverVersion []byte
	clientKexInit, serverKexInit []byte
}

func (m *handshakeMagics) write(w io.Writer) {
	writeString(w, m.clientVersion)
	writeString(w, m.serverVersion)
	writeString(w, m.clientKexInit)
	writeString(w, m.serverKexInit)
}

// kexAlgorithm abstracts different key exchange algorithms.
type kexAlgorithm interface {
	// Server runs server-side key agreement, signing the result
	// with a hostkey.
	Server(p packetConn, rand io.Reader, magics *handshakeMagics, s Signer) (*kexResult, error)

	// Client runs the client-side key agreement. Caller is
	// responsible for verifying the host key signature.
	Client(p packetConn, rand io.Reader, magics *handshakeMagics) (*kexResult, error)
}

// dhGroup is a multiplicative group suitable for implementing Diffie-Hellman key agreement.
type dhGroup struct {
	g, p *big.Int
}

func (group *dhGroup) diffieHellman(theirPublic, myPrivate *big.Int) (*big.Int, error) {
	if theirPublic.Sign() <= 0 || theirPublic.Cmp(group.p) >= 0 {
		return nil, errors.New("ssh: DH parameter out of bounds")
	}
	return new(big.Int).Exp(theirPublic, myPrivate, group.p), nil
}

func (group *dhGroup) Client(c packetConn, randSource io.Reader, magics *handshakeMagics) (*kexResult, error) {
	hashFunc := crypto.SHA1

	x, err := rand.Int(randSource, group.p)
	if err != nil {
		return nil, err
	}
	X := new(big.Int).Exp(group.g, x, group.p)
	kexDHInit := kexDHInitMsg{
		X: X,
	}
	if err := c.writePacket(Marshal(&kexDHInit)); err != nil {
		return nil, err
	}

	packet, err := c.readPacket()
	if err != nil {
		return nil, err
	}

	var kexDHReply kexDHReplyMsg
	if err = Unmarshal(packet, &kexDHReply); err != nil {
		return nil, err
	}

	kInt, err := group.diffieHellman(kexDHReply.Y, x)
	if err != nil {
		return nil, err
	}

	h := hashFunc.New()
	magics.write(h)
	writeString(h, kexDHReply.HostKey)
	writeInt(h, X)
	writeInt(h, kexDHReply.Y)
	K := make([]byte, intLength(kInt))
	marshalInt(K, kInt)
	h.Write(K)

	return &kexResult{
		H:         h.Sum(nil),
		K:         K,
		HostKey:   kexDHReply.HostKey,
		Signature: kexDHReply.Signature,
		Hash:      crypto.SHA1,
	}, nil
}

func (group *dhGroup) Server(c packetConn, randSource io.Reader, magics *handshakeMagics, priv Signer) (result *kexResult, err error) {
	hashFunc := crypto.SHA1
	packet, err := c.readPacket()
	if err != nil {
		return
	}
	var kexDHInit kexDHInitMsg
	if err = Unmarshal(packet, &kexDHInit); err != nil {
		return
	}

	y, err := rand.Int(randSource, group.p)
	if err != nil {
		return
	}

	Y := new(big.Int).Exp(group.g, y, group.p)
	kInt, err := group.diffieHellman(kexDHInit.X, y)
	if err != nil {
		return nil, err
	}

	hostKeyBytes := priv.PublicKey().Marshal()

	h := hashFunc.New()
	magics.write(h)
	writeString(h, hostKeyBytes)
	writeInt(h, kexDHInit.X)
	writeInt(h, Y)

	K := make([]byte, intLength(kInt))
	marshalInt(K, kInt)
	h.Write(K)

	H := h.Sum(nil)

	// H is already a hash, but the hostkey signing will apply its
	// own key-specific hash algorithm.
	sig, err := signAndMarshal(priv, randSource, H)
	if err != nil {
		return nil, err
	}

	kexDHReply := kexDHReplyMsg{
		HostKey:   hostKeyBytes,
		Y:         Y,
		Signature: sig,
	}
	packet = Marshal(&kexDHReply)

	err = c.writePacket(packet)
	return &kexResult{
		H:         H,
		K:         K,
		HostKey:   hostKeyBytes,
		Signature: sig,
		Hash:      crypto.SHA1,
	}, nil
}

// ecdh performs Elliptic Curve Diffie-Hellman key exchange as
// described in RFC 5656, section 4.
type ecdh struct {
	curve elliptic.Curve
}

func (kex *ecdh) Client(c packetConn, rand io.Reader, magics *handshakeMagics) (*kexResult, error) {
	ephKey, err := ecdsa.GenerateKey(kex.curve, rand)
	if err != nil {
		return nil, err
	}

	kexInit := kexECDHInitMsg{
		ClientPubKey: elliptic.Marshal(kex.curve, ephKey.PublicKey.X, ephKey.PublicKey.Y),
	}

	serialized := Marshal(&kexInit)
	if err := c.writePacket(serialized); err != nil {
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

	x, y, err := unmarshalECKey(kex.curve, reply.EphemeralPubKey)
	if err != nil {
		return nil, err
	}

	// generate shared secret
	secret, _ := kex.curve.ScalarMult(x, y, ephKey.D.Bytes())

	h := ecHash(kex.curve).New()
	magics.write(h)
	writeString(h, reply.HostKey)
	writeString(h, kexInit.ClientPubKey)
	writeString(h, reply.EphemeralPubKey)
	K := make([]byte, intLength(secret))
	marshalInt(K, secret)
	h.Write(K)

	return &kexResult{
		H:         h.Sum(nil),
		K:         K,
		HostKey:   reply.HostKey,
		Signature: reply.Signature,
		Hash:      ecHash(kex.curve),
	}, nil
}

// unmarshalECKey parses and checks an EC key.
func unmarshalECKey(curve elliptic.Curve, pubkey []byte) (x, y *big.Int, err error) {
	x, y = elliptic.Unmarshal(curve, pubkey)
	if x == nil {
		return nil, nil, errors.New("ssh: elliptic.Unmarshal failure")
	}
	if !validateECPublicKey(curve, x, y) {
		return nil, nil, errors.New("ssh: public key not on curve")
	}
	return x, y, nil
}

// validateECPublicKey checks that the point is a valid public key for
// the given curve. See [SEC1], 3.2.2
func validateECPublicKey(curve elliptic.Curve, x, y *big.Int) bool {
	if x.Sign() == 0 && y.Sign() == 0 {
		return false
	}

	if x.Cmp(curve.Params().P) >= 0 {
		return false
	}

	if y.Cmp(curve.Params().P) >= 0 {
		return false
	}

	if !curve.IsOnCurve(x, y) {
		return false
	}

	// We don't check if N * PubKey == 0, since
	//
	// - the NIST curves have cofactor = 1, so this is implicit.
	// (We don't foresee an implementation that supports non NIST
	// curves)
	//
	// - for ephemeral keys, we don't need to worry about small
	// subgroup attacks.
	return true
}

func (kex *ecdh) Server(c packetConn, rand io.Reader, magics *handshakeMagics, priv Signer) (result *kexResult, err error) {
	packet, err := c.readPacket()
	if err != nil {
		return nil, err
	}

	var kexECDHInit kexECDHInitMsg
	if err = Unmarshal(packet, &kexECDHInit); err != nil {
		return nil, err
	}

	clientX, clientY, err := unmarshalECKey(kex.curve, kexECDHInit.ClientPubKey)
	if err != nil {
		return nil, err
	}

	// We could cache this key across multiple users/multiple
	// connection attempts, but the benefit is small. OpenSSH
	// generates a new key for each incoming connection.
	ephKey, err := ecdsa.GenerateKey(kex.curve, rand)
	if err != nil {
		return nil, err
	}

	hostKeyBytes := priv.PublicKey().Marshal()

	serializedEphKey := elliptic.Marshal(kex.curve, ephKey.PublicKey.X, ephKey.PublicKey.Y)

	// generate shared secret
	secret, _ := kex.curve.ScalarMult(clientX, clientY, ephKey.D.Bytes())

	h := ecHash(kex.curve).New()
	magics.write(h)
	writeString(h, hostKeyBytes)
	writeString(h, kexECDHInit.ClientPubKey)
	writeString(h, serializedEphKey)

	K := make([]byte, intLength(secret))
	marshalInt(K, secret)
	h.Write(K)

	H := h.Sum(nil)

	// H is already a hash, but the hostkey signing will apply its
	// own key-specific hash algorithm.
	sig, err := signAndMarshal(priv, rand, H)
	if err != nil {
		return nil, err
	}

	reply := kexECDHReplyMsg{
		EphemeralPubKey: serializedEphKey,
		HostKey:         hostKeyBytes,
		Signature:       sig,
	}

	serialized := Marshal(&reply)
	if err := c.writePacket(serialized); err != nil {
		return nil, err
	}

	return &kexResult{
		H:         H,
		K:         K,
		HostKey:   reply.HostKey,
		Signature: sig,
		Hash:      ecHash(kex.curve),
	}, nil
}

var kexAlgoMap = map[string]kexAlgorithm{}

func init() {
	// This is the group called diffie-hellman-group1-sha1 in RFC
	// 4253 and Oakley Group 2 in RFC 2409.
	p, _ := new(big.Int).SetString("FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE65381FFFFFFFFFFFFFFFF", 16)
	kexAlgoMap[kexAlgoDH1SHA1] = &dhGroup{
		g: new(big.Int).SetInt64(2),
		p: p,
	}

	// This is the group called diffie-hellman-group14-sha1 in RFC
	// 4253 and Oakley Group 14 in RFC 3526.
	p, _ = new(big.Int).SetString("FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA051015728E5A8AACAA68FFFFFFFFFFFFFFFF", 16)

	kexAlgoMap[kexAlgoDH14SHA1] = &dhGroup{
		g: new(big.Int).SetInt64(2),
		p: p,
	}

	kexAlgoMap[kexAlgoECDH521] = &ecdh{elliptic.P521()}
	kexAlgoMap[kexAlgoECDH384] = &ecdh{elliptic.P384()}
	kexAlgoMap[kexAlgoECDH256] = &ecdh{elliptic.P256()}
	kexAlgoMap[kexAlgoCurve25519SHA256] = &curve25519sha256{}
}

// curve25519sha256 implements the curve25519-sha256@libssh.org key
// agreement protocol, as described in
// https://git.libssh.org/projects/libssh.git/tree/doc/curve25519-sha256@libssh.org.txt
type curve25519sha256 struct{}

type curve25519KeyPair struct {
	priv [32]byte
	pub  [32]byte
}

func (kp *curve25519KeyPair) generate(rand io.Reader) error {
	if _, err := io.ReadFull(rand, kp.priv[:]); err != nil {
		return err
	}
	curve25519.ScalarBaseMult(&kp.pub, &kp.priv)
	return nil
}

// curve25519Zeros is just an array of 32 zero bytes so that we have something
// convenient to compare against in order to reject curve25519 points with the
// wrong order.
var curve25519Zeros [32]byte

func (kex *curve25519sha256) Client(c packetConn, rand io.Reader, magics *handshakeMagics) (*kexResult, error) {
	var kp curve25519KeyPair
	if err := kp.generate(rand); err != nil {
		return nil, err
	}
	if err := c.writePacket(Marshal(&kexECDHInitMsg{kp.pub[:]})); err != nil {
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
	if len(reply.EphemeralPubKey) != 32 {
		return nil, errors.New("ssh: peer's curve25519 public value has wrong length")
	}

	var servPub, secret [32]byte
	copy(servPub[:], reply.EphemeralPubKey)
	curve25519.ScalarMult(&secret, &kp.priv, &servPub)
	if subtle.ConstantTimeCompare(secret[:], curve25519Zeros[:]) == 1 {
		return nil, errors.New("ssh: peer's curve25519 public value has wrong order")
	}

	h := crypto.SHA256.New()
	magics.write(h)
	writeString(h, reply.HostKey)
	writeString(h, kp.pub[:])
	writeString(h, reply.EphemeralPubKey)

	kInt := new(big.Int).SetBytes(secret[:])
	K := make([]byte, intLength(kInt))
	marshalInt(K, kInt)
	h.Write(K)

	return &kexResult{
		H:         h.Sum(nil),
		K:         K,
		HostKey:   reply.HostKey,
		Signature: reply.Signature,
		Hash:      crypto.SHA256,
	}, nil
}

func (kex *curve25519sha256) Server(c packetConn, rand io.Reader, magics *handshakeMagics, priv Signer) (result *kexResult, err error) {
	packet, err := c.readPacket()
	if err != nil {
		return
	}
	var kexInit kexECDHInitMsg
	if err = Unmarshal(packet, &kexInit); err != nil {
		return
	}

	if len(kexInit.ClientPubKey) != 32 {
		return nil, errors.New("ssh: peer's curve25519 public value has wrong length")
	}

	var kp curve25519KeyPair
	if err := kp.generate(rand); err != nil {
		return nil, err
	}

	var clientPub, secret [32]byte
	copy(clientPub[:], kexInit.ClientPubKey)
	curve25519.ScalarMult(&secret, &kp.priv, &clientPub)
	if subtle.ConstantTimeCompare(secret[:], curve25519Zeros[:]) == 1 {
		return nil, errors.New("ssh: peer's curve25519 public value has wrong order")
	}

	hostKeyBytes := priv.PublicKey().Marshal()

	h := crypto.SHA256.New()
	magics.write(h)
	writeString(h, hostKeyBytes)
	writeString(h, kexInit.ClientPubKey)
	writeString(h, kp.pub[:])

	kInt := new(big.Int).SetBytes(secret[:])
	K := make([]byte, intLength(kInt))
	marshalInt(K, kInt)
	h.Write(K)

	H := h.Sum(nil)

	sig, err := signAndMarshal(priv, rand, H)
	if err != nil {
		return nil, err
	}

	reply := kexECDHReplyMsg{
		EphemeralPubKey: kp.pub[:],
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
