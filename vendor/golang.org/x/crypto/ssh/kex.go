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
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math/big"

	"golang.org/x/crypto/curve25519"
)

const (
	// This is the group called diffie-hellman-group1-sha1 in RFC 4253 and
	// Oakley Group 2 in RFC 2409.
	oakleyGroup2 = "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE65381FFFFFFFFFFFFFFFF"
	// This is the group called diffie-hellman-group14-sha1 in RFC 4253 and
	// Oakley Group 14 in RFC 3526.
	oakleyGroup14 = "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA051015728E5A8AACAA68FFFFFFFFFFFFFFFF"
	// This is the group called diffie-hellman-group15-sha512 in RFC 8268 and
	// Oakley Group 15 in RFC 3526.
	oakleyGroup15 = "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA051015728E5A8AAAC42DAD33170D04507A33A85521ABDF1CBA64ECFB850458DBEF0A8AEA71575D060C7DB3970F85A6E1E4C7ABF5AE8CDB0933D71E8C94E04A25619DCEE3D2261AD2EE6BF12FFA06D98A0864D87602733EC86A64521F2B18177B200CBBE117577A615D6C770988C0BAD946E208E24FA074E5AB3143DB5BFCE0FD108E4B82D120A93AD2CAFFFFFFFFFFFFFFFF"
	// This is the group called diffie-hellman-group16-sha512 in RFC 8268 and
	// Oakley Group 16 in RFC 3526.
	oakleyGroup16 = "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA051015728E5A8AAAC42DAD33170D04507A33A85521ABDF1CBA64ECFB850458DBEF0A8AEA71575D060C7DB3970F85A6E1E4C7ABF5AE8CDB0933D71E8C94E04A25619DCEE3D2261AD2EE6BF12FFA06D98A0864D87602733EC86A64521F2B18177B200CBBE117577A615D6C770988C0BAD946E208E24FA074E5AB3143DB5BFCE0FD108E4B82D120A92108011A723C12A787E6D788719A10BDBA5B2699C327186AF4E23C1A946834B6150BDA2583E9CA2AD44CE8DBBBC2DB04DE8EF92E8EFC141FBECAA6287C59474E6BC05D99B2964FA090C3A2233BA186515BE7ED1F612970CEE2D7AFB81BDD762170481CD0069127D5B05AA993B4EA988D8FDDC186FFB7DC90A6C08F4DF435C934063199FFFFFFFFFFFFFFFF"
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
	// with a hostkey. algo is the negotiated algorithm, and may
	// be a certificate type.
	Server(p packetConn, rand io.Reader, magics *handshakeMagics, s AlgorithmSigner, algo string) (*kexResult, error)

	// Client runs the client-side key agreement. Caller is
	// responsible for verifying the host key signature.
	Client(p packetConn, rand io.Reader, magics *handshakeMagics) (*kexResult, error)
}

// dhGroup is a multiplicative group suitable for implementing Diffie-Hellman key agreement.
type dhGroup struct {
	g, p, pMinus1 *big.Int
	hashFunc      crypto.Hash
}

func (group *dhGroup) diffieHellman(theirPublic, myPrivate *big.Int) (*big.Int, error) {
	if theirPublic.Cmp(bigOne) <= 0 || theirPublic.Cmp(group.pMinus1) >= 0 {
		return nil, errors.New("ssh: DH parameter out of bounds")
	}
	return new(big.Int).Exp(theirPublic, myPrivate, group.p), nil
}

func (group *dhGroup) Client(c packetConn, randSource io.Reader, magics *handshakeMagics) (*kexResult, error) {
	var x *big.Int
	for {
		var err error
		if x, err = rand.Int(randSource, group.pMinus1); err != nil {
			return nil, err
		}
		if x.Sign() > 0 {
			break
		}
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

	ki, err := group.diffieHellman(kexDHReply.Y, x)
	if err != nil {
		return nil, err
	}

	h := group.hashFunc.New()
	magics.write(h)
	writeString(h, kexDHReply.HostKey)
	writeInt(h, X)
	writeInt(h, kexDHReply.Y)
	K := make([]byte, intLength(ki))
	marshalInt(K, ki)
	h.Write(K)

	return &kexResult{
		H:         h.Sum(nil),
		K:         K,
		HostKey:   kexDHReply.HostKey,
		Signature: kexDHReply.Signature,
		Hash:      group.hashFunc,
	}, nil
}

func (group *dhGroup) Server(c packetConn, randSource io.Reader, magics *handshakeMagics, priv AlgorithmSigner, algo string) (result *kexResult, err error) {
	packet, err := c.readPacket()
	if err != nil {
		return
	}
	var kexDHInit kexDHInitMsg
	if err = Unmarshal(packet, &kexDHInit); err != nil {
		return
	}

	var y *big.Int
	for {
		if y, err = rand.Int(randSource, group.pMinus1); err != nil {
			return
		}
		if y.Sign() > 0 {
			break
		}
	}

	Y := new(big.Int).Exp(group.g, y, group.p)
	ki, err := group.diffieHellman(kexDHInit.X, y)
	if err != nil {
		return nil, err
	}

	hostKeyBytes := priv.PublicKey().Marshal()

	h := group.hashFunc.New()
	magics.write(h)
	writeString(h, hostKeyBytes)
	writeInt(h, kexDHInit.X)
	writeInt(h, Y)

	K := make([]byte, intLength(ki))
	marshalInt(K, ki)
	h.Write(K)

	H := h.Sum(nil)

	// H is already a hash, but the hostkey signing will apply its
	// own key-specific hash algorithm.
	sig, err := signAndMarshal(priv, randSource, H, algo)
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
		Hash:      group.hashFunc,
	}, err
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

func (kex *ecdh) Server(c packetConn, rand io.Reader, magics *handshakeMagics, priv AlgorithmSigner, algo string) (result *kexResult, err error) {
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
	sig, err := signAndMarshal(priv, rand, H, algo)
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

// ecHash returns the hash to match the given elliptic curve, see RFC
// 5656, section 6.2.1
func ecHash(curve elliptic.Curve) crypto.Hash {
	bitSize := curve.Params().BitSize
	switch {
	case bitSize <= 256:
		return crypto.SHA256
	case bitSize <= 384:
		return crypto.SHA384
	}
	return crypto.SHA512
}

var kexAlgoMap = map[string]kexAlgorithm{}

func init() {
	p, _ := new(big.Int).SetString(oakleyGroup2, 16)
	kexAlgoMap[InsecureKeyExchangeDH1SHA1] = &dhGroup{
		g:        new(big.Int).SetInt64(2),
		p:        p,
		pMinus1:  new(big.Int).Sub(p, bigOne),
		hashFunc: crypto.SHA1,
	}

	p, _ = new(big.Int).SetString(oakleyGroup14, 16)
	group14 := &dhGroup{
		g:       new(big.Int).SetInt64(2),
		p:       p,
		pMinus1: new(big.Int).Sub(p, bigOne),
	}

	kexAlgoMap[InsecureKeyExchangeDH14SHA1] = &dhGroup{
		g: group14.g, p: group14.p, pMinus1: group14.pMinus1,
		hashFunc: crypto.SHA1,
	}
	kexAlgoMap[KeyExchangeDH14SHA256] = &dhGroup{
		g: group14.g, p: group14.p, pMinus1: group14.pMinus1,
		hashFunc: crypto.SHA256,
	}

	p, _ = new(big.Int).SetString(oakleyGroup16, 16)

	kexAlgoMap[KeyExchangeDH16SHA512] = &dhGroup{
		g:        new(big.Int).SetInt64(2),
		p:        p,
		pMinus1:  new(big.Int).Sub(p, bigOne),
		hashFunc: crypto.SHA512,
	}

	kexAlgoMap[KeyExchangeECDHP521] = &ecdh{elliptic.P521()}
	kexAlgoMap[KeyExchangeECDHP384] = &ecdh{elliptic.P384()}
	kexAlgoMap[KeyExchangeECDHP256] = &ecdh{elliptic.P256()}
	kexAlgoMap[KeyExchangeCurve25519] = &curve25519sha256{}
	kexAlgoMap[keyExchangeCurve25519LibSSH] = &curve25519sha256{}
	kexAlgoMap[InsecureKeyExchangeDHGEXSHA1] = &dhGEXSHA{hashFunc: crypto.SHA1}
	kexAlgoMap[KeyExchangeDHGEXSHA256] = &dhGEXSHA{hashFunc: crypto.SHA256}
}

// curve25519sha256 implements the curve25519-sha256 (formerly known as
// curve25519-sha256@libssh.org) key exchange method, as described in RFC 8731.
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

	ki := new(big.Int).SetBytes(secret[:])
	K := make([]byte, intLength(ki))
	marshalInt(K, ki)
	h.Write(K)

	return &kexResult{
		H:         h.Sum(nil),
		K:         K,
		HostKey:   reply.HostKey,
		Signature: reply.Signature,
		Hash:      crypto.SHA256,
	}, nil
}

func (kex *curve25519sha256) Server(c packetConn, rand io.Reader, magics *handshakeMagics, priv AlgorithmSigner, algo string) (result *kexResult, err error) {
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

	ki := new(big.Int).SetBytes(secret[:])
	K := make([]byte, intLength(ki))
	marshalInt(K, ki)
	h.Write(K)

	H := h.Sum(nil)

	sig, err := signAndMarshal(priv, rand, H, algo)
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

// dhGEXSHA implements the diffie-hellman-group-exchange-sha1 and
// diffie-hellman-group-exchange-sha256 key agreement protocols,
// as described in RFC 4419
type dhGEXSHA struct {
	hashFunc crypto.Hash
}

const (
	dhGroupExchangeMinimumBits   = 2048
	dhGroupExchangePreferredBits = 2048
	dhGroupExchangeMaximumBits   = 8192
)

func (gex *dhGEXSHA) Client(c packetConn, randSource io.Reader, magics *handshakeMagics) (*kexResult, error) {
	// Send GexRequest
	kexDHGexRequest := kexDHGexRequestMsg{
		MinBits:       dhGroupExchangeMinimumBits,
		PreferredBits: dhGroupExchangePreferredBits,
		MaxBits:       dhGroupExchangeMaximumBits,
	}
	if err := c.writePacket(Marshal(&kexDHGexRequest)); err != nil {
		return nil, err
	}

	// Receive GexGroup
	packet, err := c.readPacket()
	if err != nil {
		return nil, err
	}

	var msg kexDHGexGroupMsg
	if err = Unmarshal(packet, &msg); err != nil {
		return nil, err
	}

	// reject if p's bit length < dhGroupExchangeMinimumBits or > dhGroupExchangeMaximumBits
	if msg.P.BitLen() < dhGroupExchangeMinimumBits || msg.P.BitLen() > dhGroupExchangeMaximumBits {
		return nil, fmt.Errorf("ssh: server-generated gex p is out of range (%d bits)", msg.P.BitLen())
	}

	// Check if g is safe by verifying that 1 < g < p-1
	pMinusOne := new(big.Int).Sub(msg.P, bigOne)
	if msg.G.Cmp(bigOne) <= 0 || msg.G.Cmp(pMinusOne) >= 0 {
		return nil, fmt.Errorf("ssh: server provided gex g is not safe")
	}

	// Send GexInit
	pHalf := new(big.Int).Rsh(msg.P, 1)
	x, err := rand.Int(randSource, pHalf)
	if err != nil {
		return nil, err
	}
	X := new(big.Int).Exp(msg.G, x, msg.P)
	kexDHGexInit := kexDHGexInitMsg{
		X: X,
	}
	if err := c.writePacket(Marshal(&kexDHGexInit)); err != nil {
		return nil, err
	}

	// Receive GexReply
	packet, err = c.readPacket()
	if err != nil {
		return nil, err
	}

	var kexDHGexReply kexDHGexReplyMsg
	if err = Unmarshal(packet, &kexDHGexReply); err != nil {
		return nil, err
	}

	if kexDHGexReply.Y.Cmp(bigOne) <= 0 || kexDHGexReply.Y.Cmp(pMinusOne) >= 0 {
		return nil, errors.New("ssh: DH parameter out of bounds")
	}
	kInt := new(big.Int).Exp(kexDHGexReply.Y, x, msg.P)

	// Check if k is safe by verifying that k > 1 and k < p - 1
	if kInt.Cmp(bigOne) <= 0 || kInt.Cmp(pMinusOne) >= 0 {
		return nil, fmt.Errorf("ssh: derived k is not safe")
	}

	h := gex.hashFunc.New()
	magics.write(h)
	writeString(h, kexDHGexReply.HostKey)
	binary.Write(h, binary.BigEndian, uint32(dhGroupExchangeMinimumBits))
	binary.Write(h, binary.BigEndian, uint32(dhGroupExchangePreferredBits))
	binary.Write(h, binary.BigEndian, uint32(dhGroupExchangeMaximumBits))
	writeInt(h, msg.P)
	writeInt(h, msg.G)
	writeInt(h, X)
	writeInt(h, kexDHGexReply.Y)
	K := make([]byte, intLength(kInt))
	marshalInt(K, kInt)
	h.Write(K)

	return &kexResult{
		H:         h.Sum(nil),
		K:         K,
		HostKey:   kexDHGexReply.HostKey,
		Signature: kexDHGexReply.Signature,
		Hash:      gex.hashFunc,
	}, nil
}

// Server half implementation of the Diffie Hellman Key Exchange with SHA1 and SHA256.
func (gex *dhGEXSHA) Server(c packetConn, randSource io.Reader, magics *handshakeMagics, priv AlgorithmSigner, algo string) (result *kexResult, err error) {
	// Receive GexRequest
	packet, err := c.readPacket()
	if err != nil {
		return
	}
	var kexDHGexRequest kexDHGexRequestMsg
	if err = Unmarshal(packet, &kexDHGexRequest); err != nil {
		return
	}
	// We check that the request received is valid and that the MaxBits
	// requested are at least equal to our supported minimum. This is the same
	// check done in OpenSSH:
	// https://github.com/openssh/openssh-portable/blob/80a2f64b/kexgexs.c#L94
	//
	// Furthermore, we also check that the required MinBits are less than or
	// equal to 4096 because we can use up to Oakley Group 16.
	if kexDHGexRequest.MaxBits < kexDHGexRequest.MinBits || kexDHGexRequest.PreferredBits < kexDHGexRequest.MinBits ||
		kexDHGexRequest.MaxBits < kexDHGexRequest.PreferredBits || kexDHGexRequest.MaxBits < dhGroupExchangeMinimumBits ||
		kexDHGexRequest.MinBits > 4096 {
		return nil, fmt.Errorf("ssh: DH GEX request out of range, min: %d, max: %d, preferred: %d", kexDHGexRequest.MinBits,
			kexDHGexRequest.MaxBits, kexDHGexRequest.PreferredBits)
	}

	var p *big.Int
	// We hardcode sending Oakley Group 14 (2048 bits), Oakley Group 15 (3072
	// bits) or Oakley Group 16 (4096 bits), based on the requested max size.
	if kexDHGexRequest.MaxBits < 3072 {
		p, _ = new(big.Int).SetString(oakleyGroup14, 16)
	} else if kexDHGexRequest.MaxBits < 4096 {
		p, _ = new(big.Int).SetString(oakleyGroup15, 16)
	} else {
		p, _ = new(big.Int).SetString(oakleyGroup16, 16)
	}

	g := big.NewInt(2)
	msg := &kexDHGexGroupMsg{
		P: p,
		G: g,
	}
	if err := c.writePacket(Marshal(msg)); err != nil {
		return nil, err
	}

	// Receive GexInit
	packet, err = c.readPacket()
	if err != nil {
		return
	}
	var kexDHGexInit kexDHGexInitMsg
	if err = Unmarshal(packet, &kexDHGexInit); err != nil {
		return
	}

	pHalf := new(big.Int).Rsh(p, 1)

	y, err := rand.Int(randSource, pHalf)
	if err != nil {
		return
	}
	Y := new(big.Int).Exp(g, y, p)

	pMinusOne := new(big.Int).Sub(p, bigOne)
	if kexDHGexInit.X.Cmp(bigOne) <= 0 || kexDHGexInit.X.Cmp(pMinusOne) >= 0 {
		return nil, errors.New("ssh: DH parameter out of bounds")
	}
	kInt := new(big.Int).Exp(kexDHGexInit.X, y, p)

	hostKeyBytes := priv.PublicKey().Marshal()

	h := gex.hashFunc.New()
	magics.write(h)
	writeString(h, hostKeyBytes)
	binary.Write(h, binary.BigEndian, kexDHGexRequest.MinBits)
	binary.Write(h, binary.BigEndian, kexDHGexRequest.PreferredBits)
	binary.Write(h, binary.BigEndian, kexDHGexRequest.MaxBits)
	writeInt(h, p)
	writeInt(h, g)
	writeInt(h, kexDHGexInit.X)
	writeInt(h, Y)

	K := make([]byte, intLength(kInt))
	marshalInt(K, kInt)
	h.Write(K)

	H := h.Sum(nil)

	// H is already a hash, but the hostkey signing will apply its
	// own key-specific hash algorithm.
	sig, err := signAndMarshal(priv, randSource, H, algo)
	if err != nil {
		return nil, err
	}

	kexDHGexReply := kexDHGexReplyMsg{
		HostKey:   hostKeyBytes,
		Y:         Y,
		Signature: sig,
	}
	packet = Marshal(&kexDHGexReply)

	err = c.writePacket(packet)

	return &kexResult{
		H:         H,
		K:         K,
		HostKey:   hostKeyBytes,
		Signature: sig,
		Hash:      gex.hashFunc,
	}, err
}
