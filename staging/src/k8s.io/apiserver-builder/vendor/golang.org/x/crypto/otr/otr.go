// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package otr implements the Off The Record protocol as specified in
// http://www.cypherpunks.ca/otr/Protocol-v2-3.1.0.html
package otr // import "golang.org/x/crypto/otr"

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/dsa"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha1"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"hash"
	"io"
	"math/big"
	"strconv"
)

// SecurityChange describes a change in the security state of a Conversation.
type SecurityChange int

const (
	NoChange SecurityChange = iota
	// NewKeys indicates that a key exchange has completed. This occurs
	// when a conversation first becomes encrypted, and when the keys are
	// renegotiated within an encrypted conversation.
	NewKeys
	// SMPSecretNeeded indicates that the peer has started an
	// authentication and that we need to supply a secret. Call SMPQuestion
	// to get the optional, human readable challenge and then Authenticate
	// to supply the matching secret.
	SMPSecretNeeded
	// SMPComplete indicates that an authentication completed. The identity
	// of the peer has now been confirmed.
	SMPComplete
	// SMPFailed indicates that an authentication failed.
	SMPFailed
	// ConversationEnded indicates that the peer ended the secure
	// conversation.
	ConversationEnded
)

// QueryMessage can be sent to a peer to start an OTR conversation.
var QueryMessage = "?OTRv2?"

// ErrorPrefix can be used to make an OTR error by appending an error message
// to it.
var ErrorPrefix = "?OTR Error:"

var (
	fragmentPartSeparator = []byte(",")
	fragmentPrefix        = []byte("?OTR,")
	msgPrefix             = []byte("?OTR:")
	queryMarker           = []byte("?OTR")
)

// isQuery attempts to parse an OTR query from msg and returns the greatest
// common version, or 0 if msg is not an OTR query.
func isQuery(msg []byte) (greatestCommonVersion int) {
	pos := bytes.Index(msg, queryMarker)
	if pos == -1 {
		return 0
	}
	for i, c := range msg[pos+len(queryMarker):] {
		if i == 0 {
			if c == '?' {
				// Indicates support for version 1, but we don't
				// implement that.
				continue
			}

			if c != 'v' {
				// Invalid message
				return 0
			}

			continue
		}

		if c == '?' {
			// End of message
			return
		}

		if c == ' ' || c == '\t' {
			// Probably an invalid message
			return 0
		}

		if c == '2' {
			greatestCommonVersion = 2
		}
	}

	return 0
}

const (
	statePlaintext = iota
	stateEncrypted
	stateFinished
)

const (
	authStateNone = iota
	authStateAwaitingDHKey
	authStateAwaitingRevealSig
	authStateAwaitingSig
)

const (
	msgTypeDHCommit  = 2
	msgTypeData      = 3
	msgTypeDHKey     = 10
	msgTypeRevealSig = 17
	msgTypeSig       = 18
)

const (
	// If the requested fragment size is less than this, it will be ignored.
	minFragmentSize = 18
	// Messages are padded to a multiple of this number of bytes.
	paddingGranularity = 256
	// The number of bytes in a Diffie-Hellman private value (320-bits).
	dhPrivateBytes = 40
	// The number of bytes needed to represent an element of the DSA
	// subgroup (160-bits).
	dsaSubgroupBytes = 20
	// The number of bytes of the MAC that are sent on the wire (160-bits).
	macPrefixBytes = 20
)

// These are the global, common group parameters for OTR.
var (
	p       *big.Int // group prime
	g       *big.Int // group generator
	q       *big.Int // group order
	pMinus2 *big.Int
)

func init() {
	p, _ = new(big.Int).SetString("FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA237327FFFFFFFFFFFFFFFF", 16)
	q, _ = new(big.Int).SetString("7FFFFFFFFFFFFFFFE487ED5110B4611A62633145C06E0E68948127044533E63A0105DF531D89CD9128A5043CC71A026EF7CA8CD9E69D218D98158536F92F8A1BA7F09AB6B6A8E122F242DABB312F3F637A262174D31BF6B585FFAE5B7A035BF6F71C35FDAD44CFD2D74F9208BE258FF324943328F6722D9EE1003E5C50B1DF82CC6D241B0E2AE9CD348B1FD47E9267AFC1B2AE91EE51D6CB0E3179AB1042A95DCF6A9483B84B4B36B3861AA7255E4C0278BA36046511B993FFFFFFFFFFFFFFFF", 16)
	g = new(big.Int).SetInt64(2)
	pMinus2 = new(big.Int).Sub(p, g)
}

// Conversation represents a relation with a peer. The zero value is a valid
// Conversation, although PrivateKey must be set.
//
// When communicating with a peer, all inbound messages should be passed to
// Conversation.Receive and all outbound messages to Conversation.Send. The
// Conversation will take care of maintaining the encryption state and
// negotiating encryption as needed.
type Conversation struct {
	// PrivateKey contains the private key to use to sign key exchanges.
	PrivateKey *PrivateKey

	// Rand can be set to override the entropy source. Otherwise,
	// crypto/rand will be used.
	Rand io.Reader
	// If FragmentSize is set, all messages produced by Receive and Send
	// will be fragmented into messages of, at most, this number of bytes.
	FragmentSize int

	// Once Receive has returned NewKeys once, the following fields are
	// valid.
	SSID           [8]byte
	TheirPublicKey PublicKey

	state, authState int

	r       [16]byte
	x, y    *big.Int
	gx, gy  *big.Int
	gxBytes []byte
	digest  [sha256.Size]byte

	revealKeys, sigKeys akeKeys

	myKeyId         uint32
	myCurrentDHPub  *big.Int
	myCurrentDHPriv *big.Int
	myLastDHPub     *big.Int
	myLastDHPriv    *big.Int

	theirKeyId        uint32
	theirCurrentDHPub *big.Int
	theirLastDHPub    *big.Int

	keySlots [4]keySlot

	myCounter    [8]byte
	theirLastCtr [8]byte
	oldMACs      []byte

	k, n int // fragment state
	frag []byte

	smp smpState
}

// A keySlot contains key material for a specific (their keyid, my keyid) pair.
type keySlot struct {
	// used is true if this slot is valid. If false, it's free for reuse.
	used                   bool
	theirKeyId             uint32
	myKeyId                uint32
	sendAESKey, recvAESKey []byte
	sendMACKey, recvMACKey []byte
	theirLastCtr           [8]byte
}

// akeKeys are generated during key exchange. There's one set for the reveal
// signature message and another for the signature message. In the protocol
// spec the latter are indicated with a prime mark.
type akeKeys struct {
	c      [16]byte
	m1, m2 [32]byte
}

func (c *Conversation) rand() io.Reader {
	if c.Rand != nil {
		return c.Rand
	}
	return rand.Reader
}

func (c *Conversation) randMPI(buf []byte) *big.Int {
	_, err := io.ReadFull(c.rand(), buf)
	if err != nil {
		panic("otr: short read from random source")
	}

	return new(big.Int).SetBytes(buf)
}

// tlv represents the type-length value from the protocol.
type tlv struct {
	typ, length uint16
	data        []byte
}

const (
	tlvTypePadding          = 0
	tlvTypeDisconnected     = 1
	tlvTypeSMP1             = 2
	tlvTypeSMP2             = 3
	tlvTypeSMP3             = 4
	tlvTypeSMP4             = 5
	tlvTypeSMPAbort         = 6
	tlvTypeSMP1WithQuestion = 7
)

// Receive handles a message from a peer. It returns a human readable message,
// an indicator of whether that message was encrypted, a hint about the
// encryption state and zero or more messages to send back to the peer.
// These messages do not need to be passed to Send before transmission.
func (c *Conversation) Receive(in []byte) (out []byte, encrypted bool, change SecurityChange, toSend [][]byte, err error) {
	if bytes.HasPrefix(in, fragmentPrefix) {
		in, err = c.processFragment(in)
		if in == nil || err != nil {
			return
		}
	}

	if bytes.HasPrefix(in, msgPrefix) && in[len(in)-1] == '.' {
		in = in[len(msgPrefix) : len(in)-1]
	} else if version := isQuery(in); version > 0 {
		c.authState = authStateAwaitingDHKey
		c.reset()
		toSend = c.encode(c.generateDHCommit())
		return
	} else {
		// plaintext message
		out = in
		return
	}

	msg := make([]byte, base64.StdEncoding.DecodedLen(len(in)))
	msgLen, err := base64.StdEncoding.Decode(msg, in)
	if err != nil {
		err = errors.New("otr: invalid base64 encoding in message")
		return
	}
	msg = msg[:msgLen]

	// The first two bytes are the protocol version (2)
	if len(msg) < 3 || msg[0] != 0 || msg[1] != 2 {
		err = errors.New("otr: invalid OTR message")
		return
	}

	msgType := int(msg[2])
	msg = msg[3:]

	switch msgType {
	case msgTypeDHCommit:
		switch c.authState {
		case authStateNone:
			c.authState = authStateAwaitingRevealSig
			if err = c.processDHCommit(msg); err != nil {
				return
			}
			c.reset()
			toSend = c.encode(c.generateDHKey())
			return
		case authStateAwaitingDHKey:
			// This is a 'SYN-crossing'. The greater digest wins.
			var cmp int
			if cmp, err = c.compareToDHCommit(msg); err != nil {
				return
			}
			if cmp > 0 {
				// We win. Retransmit DH commit.
				toSend = c.encode(c.serializeDHCommit())
				return
			} else {
				// They win. We forget about our DH commit.
				c.authState = authStateAwaitingRevealSig
				if err = c.processDHCommit(msg); err != nil {
					return
				}
				c.reset()
				toSend = c.encode(c.generateDHKey())
				return
			}
		case authStateAwaitingRevealSig:
			if err = c.processDHCommit(msg); err != nil {
				return
			}
			toSend = c.encode(c.serializeDHKey())
		case authStateAwaitingSig:
			if err = c.processDHCommit(msg); err != nil {
				return
			}
			c.reset()
			toSend = c.encode(c.generateDHKey())
			c.authState = authStateAwaitingRevealSig
		default:
			panic("bad state")
		}
	case msgTypeDHKey:
		switch c.authState {
		case authStateAwaitingDHKey:
			var isSame bool
			if isSame, err = c.processDHKey(msg); err != nil {
				return
			}
			if isSame {
				err = errors.New("otr: unexpected duplicate DH key")
				return
			}
			toSend = c.encode(c.generateRevealSig())
			c.authState = authStateAwaitingSig
		case authStateAwaitingSig:
			var isSame bool
			if isSame, err = c.processDHKey(msg); err != nil {
				return
			}
			if isSame {
				toSend = c.encode(c.serializeDHKey())
			}
		}
	case msgTypeRevealSig:
		if c.authState != authStateAwaitingRevealSig {
			return
		}
		if err = c.processRevealSig(msg); err != nil {
			return
		}
		toSend = c.encode(c.generateSig())
		c.authState = authStateNone
		c.state = stateEncrypted
		change = NewKeys
	case msgTypeSig:
		if c.authState != authStateAwaitingSig {
			return
		}
		if err = c.processSig(msg); err != nil {
			return
		}
		c.authState = authStateNone
		c.state = stateEncrypted
		change = NewKeys
	case msgTypeData:
		if c.state != stateEncrypted {
			err = errors.New("otr: encrypted message received without encrypted session established")
			return
		}
		var tlvs []tlv
		out, tlvs, err = c.processData(msg)
		encrypted = true

	EachTLV:
		for _, inTLV := range tlvs {
			switch inTLV.typ {
			case tlvTypeDisconnected:
				change = ConversationEnded
				c.state = stateFinished
				break EachTLV
			case tlvTypeSMP1, tlvTypeSMP2, tlvTypeSMP3, tlvTypeSMP4, tlvTypeSMPAbort, tlvTypeSMP1WithQuestion:
				var reply tlv
				var complete bool
				reply, complete, err = c.processSMP(inTLV)
				if err == smpSecretMissingError {
					err = nil
					change = SMPSecretNeeded
					c.smp.saved = &inTLV
					return
				}
				if err == smpFailureError {
					err = nil
					change = SMPFailed
				} else if complete {
					change = SMPComplete
				}
				if reply.typ != 0 {
					toSend = c.encode(c.generateData(nil, &reply))
				}
				break EachTLV
			default:
				// skip unknown TLVs
			}
		}
	default:
		err = errors.New("otr: unknown message type " + strconv.Itoa(msgType))
	}

	return
}

// Send takes a human readable message from the local user, possibly encrypts
// it and returns zero one or more messages to send to the peer.
func (c *Conversation) Send(msg []byte) ([][]byte, error) {
	switch c.state {
	case statePlaintext:
		return [][]byte{msg}, nil
	case stateEncrypted:
		return c.encode(c.generateData(msg, nil)), nil
	case stateFinished:
		return nil, errors.New("otr: cannot send message because secure conversation has finished")
	}

	return nil, errors.New("otr: cannot send message in current state")
}

// SMPQuestion returns the human readable challenge question from the peer.
// It's only valid after Receive has returned SMPSecretNeeded.
func (c *Conversation) SMPQuestion() string {
	return c.smp.question
}

// Authenticate begins an authentication with the peer. Authentication involves
// an optional challenge message and a shared secret. The authentication
// proceeds until either Receive returns SMPComplete, SMPSecretNeeded (which
// indicates that a new authentication is happening and thus this one was
// aborted) or SMPFailed.
func (c *Conversation) Authenticate(question string, mutualSecret []byte) (toSend [][]byte, err error) {
	if c.state != stateEncrypted {
		err = errors.New("otr: can't authenticate a peer without a secure conversation established")
		return
	}

	if c.smp.saved != nil {
		c.calcSMPSecret(mutualSecret, false /* they started it */)

		var out tlv
		var complete bool
		out, complete, err = c.processSMP(*c.smp.saved)
		if complete {
			panic("SMP completed on the first message")
		}
		c.smp.saved = nil
		if out.typ != 0 {
			toSend = c.encode(c.generateData(nil, &out))
		}
		return
	}

	c.calcSMPSecret(mutualSecret, true /* we started it */)
	outs := c.startSMP(question)
	for _, out := range outs {
		toSend = append(toSend, c.encode(c.generateData(nil, &out))...)
	}
	return
}

// End ends a secure conversation by generating a termination message for
// the peer and switches to unencrypted communication.
func (c *Conversation) End() (toSend [][]byte) {
	switch c.state {
	case statePlaintext:
		return nil
	case stateEncrypted:
		c.state = statePlaintext
		return c.encode(c.generateData(nil, &tlv{typ: tlvTypeDisconnected}))
	case stateFinished:
		c.state = statePlaintext
		return nil
	}
	panic("unreachable")
}

// IsEncrypted returns true if a message passed to Send would be encrypted
// before transmission. This result remains valid until the next call to
// Receive or End, which may change the state of the Conversation.
func (c *Conversation) IsEncrypted() bool {
	return c.state == stateEncrypted
}

var fragmentError = errors.New("otr: invalid OTR fragment")

// processFragment processes a fragmented OTR message and possibly returns a
// complete message. Fragmented messages look like "?OTR,k,n,msg," where k is
// the fragment number (starting from 1), n is the number of fragments in this
// message and msg is a substring of the base64 encoded message.
func (c *Conversation) processFragment(in []byte) (out []byte, err error) {
	in = in[len(fragmentPrefix):] // remove "?OTR,"
	parts := bytes.Split(in, fragmentPartSeparator)
	if len(parts) != 4 || len(parts[3]) != 0 {
		return nil, fragmentError
	}

	k, err := strconv.Atoi(string(parts[0]))
	if err != nil {
		return nil, fragmentError
	}

	n, err := strconv.Atoi(string(parts[1]))
	if err != nil {
		return nil, fragmentError
	}

	if k < 1 || n < 1 || k > n {
		return nil, fragmentError
	}

	if k == 1 {
		c.frag = append(c.frag[:0], parts[2]...)
		c.k, c.n = k, n
	} else if n == c.n && k == c.k+1 {
		c.frag = append(c.frag, parts[2]...)
		c.k++
	} else {
		c.frag = c.frag[:0]
		c.n, c.k = 0, 0
	}

	if c.n > 0 && c.k == c.n {
		c.n, c.k = 0, 0
		return c.frag, nil
	}

	return nil, nil
}

func (c *Conversation) generateDHCommit() []byte {
	_, err := io.ReadFull(c.rand(), c.r[:])
	if err != nil {
		panic("otr: short read from random source")
	}

	var xBytes [dhPrivateBytes]byte
	c.x = c.randMPI(xBytes[:])
	c.gx = new(big.Int).Exp(g, c.x, p)
	c.gy = nil
	c.gxBytes = appendMPI(nil, c.gx)

	h := sha256.New()
	h.Write(c.gxBytes)
	h.Sum(c.digest[:0])

	aesCipher, err := aes.NewCipher(c.r[:])
	if err != nil {
		panic(err.Error())
	}

	var iv [aes.BlockSize]byte
	ctr := cipher.NewCTR(aesCipher, iv[:])
	ctr.XORKeyStream(c.gxBytes, c.gxBytes)

	return c.serializeDHCommit()
}

func (c *Conversation) serializeDHCommit() []byte {
	var ret []byte
	ret = appendU16(ret, 2) // protocol version
	ret = append(ret, msgTypeDHCommit)
	ret = appendData(ret, c.gxBytes)
	ret = appendData(ret, c.digest[:])
	return ret
}

func (c *Conversation) processDHCommit(in []byte) error {
	var ok1, ok2 bool
	c.gxBytes, in, ok1 = getData(in)
	digest, in, ok2 := getData(in)
	if !ok1 || !ok2 || len(in) > 0 {
		return errors.New("otr: corrupt DH commit message")
	}
	copy(c.digest[:], digest)
	return nil
}

func (c *Conversation) compareToDHCommit(in []byte) (int, error) {
	_, in, ok1 := getData(in)
	digest, in, ok2 := getData(in)
	if !ok1 || !ok2 || len(in) > 0 {
		return 0, errors.New("otr: corrupt DH commit message")
	}
	return bytes.Compare(c.digest[:], digest), nil
}

func (c *Conversation) generateDHKey() []byte {
	var yBytes [dhPrivateBytes]byte
	c.y = c.randMPI(yBytes[:])
	c.gy = new(big.Int).Exp(g, c.y, p)
	return c.serializeDHKey()
}

func (c *Conversation) serializeDHKey() []byte {
	var ret []byte
	ret = appendU16(ret, 2) // protocol version
	ret = append(ret, msgTypeDHKey)
	ret = appendMPI(ret, c.gy)
	return ret
}

func (c *Conversation) processDHKey(in []byte) (isSame bool, err error) {
	gy, in, ok := getMPI(in)
	if !ok {
		err = errors.New("otr: corrupt DH key message")
		return
	}
	if gy.Cmp(g) < 0 || gy.Cmp(pMinus2) > 0 {
		err = errors.New("otr: DH value out of range")
		return
	}
	if c.gy != nil {
		isSame = c.gy.Cmp(gy) == 0
		return
	}
	c.gy = gy
	return
}

func (c *Conversation) generateEncryptedSignature(keys *akeKeys, xFirst bool) ([]byte, []byte) {
	var xb []byte
	xb = c.PrivateKey.PublicKey.Serialize(xb)

	var verifyData []byte
	if xFirst {
		verifyData = appendMPI(verifyData, c.gx)
		verifyData = appendMPI(verifyData, c.gy)
	} else {
		verifyData = appendMPI(verifyData, c.gy)
		verifyData = appendMPI(verifyData, c.gx)
	}
	verifyData = append(verifyData, xb...)
	verifyData = appendU32(verifyData, c.myKeyId)

	mac := hmac.New(sha256.New, keys.m1[:])
	mac.Write(verifyData)
	mb := mac.Sum(nil)

	xb = appendU32(xb, c.myKeyId)
	xb = append(xb, c.PrivateKey.Sign(c.rand(), mb)...)

	aesCipher, err := aes.NewCipher(keys.c[:])
	if err != nil {
		panic(err.Error())
	}
	var iv [aes.BlockSize]byte
	ctr := cipher.NewCTR(aesCipher, iv[:])
	ctr.XORKeyStream(xb, xb)

	mac = hmac.New(sha256.New, keys.m2[:])
	encryptedSig := appendData(nil, xb)
	mac.Write(encryptedSig)

	return encryptedSig, mac.Sum(nil)
}

func (c *Conversation) generateRevealSig() []byte {
	s := new(big.Int).Exp(c.gy, c.x, p)
	c.calcAKEKeys(s)
	c.myKeyId++

	encryptedSig, mac := c.generateEncryptedSignature(&c.revealKeys, true /* gx comes first */)

	c.myCurrentDHPub = c.gx
	c.myCurrentDHPriv = c.x
	c.rotateDHKeys()
	incCounter(&c.myCounter)

	var ret []byte
	ret = appendU16(ret, 2)
	ret = append(ret, msgTypeRevealSig)
	ret = appendData(ret, c.r[:])
	ret = append(ret, encryptedSig...)
	ret = append(ret, mac[:20]...)
	return ret
}

func (c *Conversation) processEncryptedSig(encryptedSig, theirMAC []byte, keys *akeKeys, xFirst bool) error {
	mac := hmac.New(sha256.New, keys.m2[:])
	mac.Write(appendData(nil, encryptedSig))
	myMAC := mac.Sum(nil)[:20]

	if len(myMAC) != len(theirMAC) || subtle.ConstantTimeCompare(myMAC, theirMAC) == 0 {
		return errors.New("bad signature MAC in encrypted signature")
	}

	aesCipher, err := aes.NewCipher(keys.c[:])
	if err != nil {
		panic(err.Error())
	}
	var iv [aes.BlockSize]byte
	ctr := cipher.NewCTR(aesCipher, iv[:])
	ctr.XORKeyStream(encryptedSig, encryptedSig)

	sig := encryptedSig
	sig, ok1 := c.TheirPublicKey.Parse(sig)
	keyId, sig, ok2 := getU32(sig)
	if !ok1 || !ok2 {
		return errors.New("otr: corrupt encrypted signature")
	}

	var verifyData []byte
	if xFirst {
		verifyData = appendMPI(verifyData, c.gx)
		verifyData = appendMPI(verifyData, c.gy)
	} else {
		verifyData = appendMPI(verifyData, c.gy)
		verifyData = appendMPI(verifyData, c.gx)
	}
	verifyData = c.TheirPublicKey.Serialize(verifyData)
	verifyData = appendU32(verifyData, keyId)

	mac = hmac.New(sha256.New, keys.m1[:])
	mac.Write(verifyData)
	mb := mac.Sum(nil)

	sig, ok1 = c.TheirPublicKey.Verify(mb, sig)
	if !ok1 {
		return errors.New("bad signature in encrypted signature")
	}
	if len(sig) > 0 {
		return errors.New("corrupt encrypted signature")
	}

	c.theirKeyId = keyId
	zero(c.theirLastCtr[:])
	return nil
}

func (c *Conversation) processRevealSig(in []byte) error {
	r, in, ok1 := getData(in)
	encryptedSig, in, ok2 := getData(in)
	theirMAC := in
	if !ok1 || !ok2 || len(theirMAC) != 20 {
		return errors.New("otr: corrupt reveal signature message")
	}

	aesCipher, err := aes.NewCipher(r)
	if err != nil {
		return errors.New("otr: cannot create AES cipher from reveal signature message: " + err.Error())
	}
	var iv [aes.BlockSize]byte
	ctr := cipher.NewCTR(aesCipher, iv[:])
	ctr.XORKeyStream(c.gxBytes, c.gxBytes)
	h := sha256.New()
	h.Write(c.gxBytes)
	digest := h.Sum(nil)
	if len(digest) != len(c.digest) || subtle.ConstantTimeCompare(digest, c.digest[:]) == 0 {
		return errors.New("otr: bad commit MAC in reveal signature message")
	}
	var rest []byte
	c.gx, rest, ok1 = getMPI(c.gxBytes)
	if !ok1 || len(rest) > 0 {
		return errors.New("otr: gx corrupt after decryption")
	}
	if c.gx.Cmp(g) < 0 || c.gx.Cmp(pMinus2) > 0 {
		return errors.New("otr: DH value out of range")
	}
	s := new(big.Int).Exp(c.gx, c.y, p)
	c.calcAKEKeys(s)

	if err := c.processEncryptedSig(encryptedSig, theirMAC, &c.revealKeys, true /* gx comes first */); err != nil {
		return errors.New("otr: in reveal signature message: " + err.Error())
	}

	c.theirCurrentDHPub = c.gx
	c.theirLastDHPub = nil

	return nil
}

func (c *Conversation) generateSig() []byte {
	c.myKeyId++

	encryptedSig, mac := c.generateEncryptedSignature(&c.sigKeys, false /* gy comes first */)

	c.myCurrentDHPub = c.gy
	c.myCurrentDHPriv = c.y
	c.rotateDHKeys()
	incCounter(&c.myCounter)

	var ret []byte
	ret = appendU16(ret, 2)
	ret = append(ret, msgTypeSig)
	ret = append(ret, encryptedSig...)
	ret = append(ret, mac[:macPrefixBytes]...)
	return ret
}

func (c *Conversation) processSig(in []byte) error {
	encryptedSig, in, ok1 := getData(in)
	theirMAC := in
	if !ok1 || len(theirMAC) != macPrefixBytes {
		return errors.New("otr: corrupt signature message")
	}

	if err := c.processEncryptedSig(encryptedSig, theirMAC, &c.sigKeys, false /* gy comes first */); err != nil {
		return errors.New("otr: in signature message: " + err.Error())
	}

	c.theirCurrentDHPub = c.gy
	c.theirLastDHPub = nil

	return nil
}

func (c *Conversation) rotateDHKeys() {
	// evict slots using our retired key id
	for i := range c.keySlots {
		slot := &c.keySlots[i]
		if slot.used && slot.myKeyId == c.myKeyId-1 {
			slot.used = false
			c.oldMACs = append(c.oldMACs, slot.recvMACKey...)
		}
	}

	c.myLastDHPriv = c.myCurrentDHPriv
	c.myLastDHPub = c.myCurrentDHPub

	var xBytes [dhPrivateBytes]byte
	c.myCurrentDHPriv = c.randMPI(xBytes[:])
	c.myCurrentDHPub = new(big.Int).Exp(g, c.myCurrentDHPriv, p)
	c.myKeyId++
}

func (c *Conversation) processData(in []byte) (out []byte, tlvs []tlv, err error) {
	origIn := in
	flags, in, ok1 := getU8(in)
	theirKeyId, in, ok2 := getU32(in)
	myKeyId, in, ok3 := getU32(in)
	y, in, ok4 := getMPI(in)
	counter, in, ok5 := getNBytes(in, 8)
	encrypted, in, ok6 := getData(in)
	macedData := origIn[:len(origIn)-len(in)]
	theirMAC, in, ok7 := getNBytes(in, macPrefixBytes)
	_, in, ok8 := getData(in)
	if !ok1 || !ok2 || !ok3 || !ok4 || !ok5 || !ok6 || !ok7 || !ok8 || len(in) > 0 {
		err = errors.New("otr: corrupt data message")
		return
	}

	ignoreErrors := flags&1 != 0

	slot, err := c.calcDataKeys(myKeyId, theirKeyId)
	if err != nil {
		if ignoreErrors {
			err = nil
		}
		return
	}

	mac := hmac.New(sha1.New, slot.recvMACKey)
	mac.Write([]byte{0, 2, 3})
	mac.Write(macedData)
	myMAC := mac.Sum(nil)
	if len(myMAC) != len(theirMAC) || subtle.ConstantTimeCompare(myMAC, theirMAC) == 0 {
		if !ignoreErrors {
			err = errors.New("otr: bad MAC on data message")
		}
		return
	}

	if bytes.Compare(counter, slot.theirLastCtr[:]) <= 0 {
		err = errors.New("otr: counter regressed")
		return
	}
	copy(slot.theirLastCtr[:], counter)

	var iv [aes.BlockSize]byte
	copy(iv[:], counter)
	aesCipher, err := aes.NewCipher(slot.recvAESKey)
	if err != nil {
		panic(err.Error())
	}
	ctr := cipher.NewCTR(aesCipher, iv[:])
	ctr.XORKeyStream(encrypted, encrypted)
	decrypted := encrypted

	if myKeyId == c.myKeyId {
		c.rotateDHKeys()
	}
	if theirKeyId == c.theirKeyId {
		// evict slots using their retired key id
		for i := range c.keySlots {
			slot := &c.keySlots[i]
			if slot.used && slot.theirKeyId == theirKeyId-1 {
				slot.used = false
				c.oldMACs = append(c.oldMACs, slot.recvMACKey...)
			}
		}

		c.theirLastDHPub = c.theirCurrentDHPub
		c.theirKeyId++
		c.theirCurrentDHPub = y
	}

	if nulPos := bytes.IndexByte(decrypted, 0); nulPos >= 0 {
		out = decrypted[:nulPos]
		tlvData := decrypted[nulPos+1:]
		for len(tlvData) > 0 {
			var t tlv
			var ok1, ok2, ok3 bool

			t.typ, tlvData, ok1 = getU16(tlvData)
			t.length, tlvData, ok2 = getU16(tlvData)
			t.data, tlvData, ok3 = getNBytes(tlvData, int(t.length))
			if !ok1 || !ok2 || !ok3 {
				err = errors.New("otr: corrupt tlv data")
			}
			tlvs = append(tlvs, t)
		}
	} else {
		out = decrypted
	}

	return
}

func (c *Conversation) generateData(msg []byte, extra *tlv) []byte {
	slot, err := c.calcDataKeys(c.myKeyId-1, c.theirKeyId)
	if err != nil {
		panic("otr: failed to generate sending keys: " + err.Error())
	}

	var plaintext []byte
	plaintext = append(plaintext, msg...)
	plaintext = append(plaintext, 0)

	padding := paddingGranularity - ((len(plaintext) + 4) % paddingGranularity)
	plaintext = appendU16(plaintext, tlvTypePadding)
	plaintext = appendU16(plaintext, uint16(padding))
	for i := 0; i < padding; i++ {
		plaintext = append(plaintext, 0)
	}

	if extra != nil {
		plaintext = appendU16(plaintext, extra.typ)
		plaintext = appendU16(plaintext, uint16(len(extra.data)))
		plaintext = append(plaintext, extra.data...)
	}

	encrypted := make([]byte, len(plaintext))

	var iv [aes.BlockSize]byte
	copy(iv[:], c.myCounter[:])
	aesCipher, err := aes.NewCipher(slot.sendAESKey)
	if err != nil {
		panic(err.Error())
	}
	ctr := cipher.NewCTR(aesCipher, iv[:])
	ctr.XORKeyStream(encrypted, plaintext)

	var ret []byte
	ret = appendU16(ret, 2)
	ret = append(ret, msgTypeData)
	ret = append(ret, 0 /* flags */)
	ret = appendU32(ret, c.myKeyId-1)
	ret = appendU32(ret, c.theirKeyId)
	ret = appendMPI(ret, c.myCurrentDHPub)
	ret = append(ret, c.myCounter[:]...)
	ret = appendData(ret, encrypted)

	mac := hmac.New(sha1.New, slot.sendMACKey)
	mac.Write(ret)
	ret = append(ret, mac.Sum(nil)[:macPrefixBytes]...)
	ret = appendData(ret, c.oldMACs)
	c.oldMACs = nil
	incCounter(&c.myCounter)

	return ret
}

func incCounter(counter *[8]byte) {
	for i := 7; i >= 0; i-- {
		counter[i]++
		if counter[i] > 0 {
			break
		}
	}
}

// calcDataKeys computes the keys used to encrypt a data message given the key
// IDs.
func (c *Conversation) calcDataKeys(myKeyId, theirKeyId uint32) (slot *keySlot, err error) {
	// Check for a cache hit.
	for i := range c.keySlots {
		slot = &c.keySlots[i]
		if slot.used && slot.theirKeyId == theirKeyId && slot.myKeyId == myKeyId {
			return
		}
	}

	// Find an empty slot to write into.
	slot = nil
	for i := range c.keySlots {
		if !c.keySlots[i].used {
			slot = &c.keySlots[i]
			break
		}
	}
	if slot == nil {
		return nil, errors.New("otr: internal error: no more key slots")
	}

	var myPriv, myPub, theirPub *big.Int

	if myKeyId == c.myKeyId {
		myPriv = c.myCurrentDHPriv
		myPub = c.myCurrentDHPub
	} else if myKeyId == c.myKeyId-1 {
		myPriv = c.myLastDHPriv
		myPub = c.myLastDHPub
	} else {
		err = errors.New("otr: peer requested keyid " + strconv.FormatUint(uint64(myKeyId), 10) + " when I'm on " + strconv.FormatUint(uint64(c.myKeyId), 10))
		return
	}

	if theirKeyId == c.theirKeyId {
		theirPub = c.theirCurrentDHPub
	} else if theirKeyId == c.theirKeyId-1 && c.theirLastDHPub != nil {
		theirPub = c.theirLastDHPub
	} else {
		err = errors.New("otr: peer requested keyid " + strconv.FormatUint(uint64(myKeyId), 10) + " when they're on " + strconv.FormatUint(uint64(c.myKeyId), 10))
		return
	}

	var sendPrefixByte, recvPrefixByte [1]byte

	if myPub.Cmp(theirPub) > 0 {
		// we're the high end
		sendPrefixByte[0], recvPrefixByte[0] = 1, 2
	} else {
		// we're the low end
		sendPrefixByte[0], recvPrefixByte[0] = 2, 1
	}

	s := new(big.Int).Exp(theirPub, myPriv, p)
	sBytes := appendMPI(nil, s)

	h := sha1.New()
	h.Write(sendPrefixByte[:])
	h.Write(sBytes)
	slot.sendAESKey = h.Sum(slot.sendAESKey[:0])[:16]

	h.Reset()
	h.Write(slot.sendAESKey)
	slot.sendMACKey = h.Sum(slot.sendMACKey[:0])

	h.Reset()
	h.Write(recvPrefixByte[:])
	h.Write(sBytes)
	slot.recvAESKey = h.Sum(slot.recvAESKey[:0])[:16]

	h.Reset()
	h.Write(slot.recvAESKey)
	slot.recvMACKey = h.Sum(slot.recvMACKey[:0])

	slot.theirKeyId = theirKeyId
	slot.myKeyId = myKeyId
	slot.used = true

	zero(slot.theirLastCtr[:])
	return
}

func (c *Conversation) calcAKEKeys(s *big.Int) {
	mpi := appendMPI(nil, s)
	h := sha256.New()

	var cBytes [32]byte
	hashWithPrefix(c.SSID[:], 0, mpi, h)

	hashWithPrefix(cBytes[:], 1, mpi, h)
	copy(c.revealKeys.c[:], cBytes[:16])
	copy(c.sigKeys.c[:], cBytes[16:])

	hashWithPrefix(c.revealKeys.m1[:], 2, mpi, h)
	hashWithPrefix(c.revealKeys.m2[:], 3, mpi, h)
	hashWithPrefix(c.sigKeys.m1[:], 4, mpi, h)
	hashWithPrefix(c.sigKeys.m2[:], 5, mpi, h)
}

func hashWithPrefix(out []byte, prefix byte, in []byte, h hash.Hash) {
	h.Reset()
	var p [1]byte
	p[0] = prefix
	h.Write(p[:])
	h.Write(in)
	if len(out) == h.Size() {
		h.Sum(out[:0])
	} else {
		digest := h.Sum(nil)
		copy(out, digest)
	}
}

func (c *Conversation) encode(msg []byte) [][]byte {
	b64 := make([]byte, base64.StdEncoding.EncodedLen(len(msg))+len(msgPrefix)+1)
	base64.StdEncoding.Encode(b64[len(msgPrefix):], msg)
	copy(b64, msgPrefix)
	b64[len(b64)-1] = '.'

	if c.FragmentSize < minFragmentSize || len(b64) <= c.FragmentSize {
		// We can encode this in a single fragment.
		return [][]byte{b64}
	}

	// We have to fragment this message.
	var ret [][]byte
	bytesPerFragment := c.FragmentSize - minFragmentSize
	numFragments := (len(b64) + bytesPerFragment) / bytesPerFragment

	for i := 0; i < numFragments; i++ {
		frag := []byte("?OTR," + strconv.Itoa(i+1) + "," + strconv.Itoa(numFragments) + ",")
		todo := bytesPerFragment
		if todo > len(b64) {
			todo = len(b64)
		}
		frag = append(frag, b64[:todo]...)
		b64 = b64[todo:]
		frag = append(frag, ',')
		ret = append(ret, frag)
	}

	return ret
}

func (c *Conversation) reset() {
	c.myKeyId = 0

	for i := range c.keySlots {
		c.keySlots[i].used = false
	}
}

type PublicKey struct {
	dsa.PublicKey
}

func (pk *PublicKey) Parse(in []byte) ([]byte, bool) {
	var ok bool
	var pubKeyType uint16

	if pubKeyType, in, ok = getU16(in); !ok || pubKeyType != 0 {
		return nil, false
	}
	if pk.P, in, ok = getMPI(in); !ok {
		return nil, false
	}
	if pk.Q, in, ok = getMPI(in); !ok {
		return nil, false
	}
	if pk.G, in, ok = getMPI(in); !ok {
		return nil, false
	}
	if pk.Y, in, ok = getMPI(in); !ok {
		return nil, false
	}

	return in, true
}

func (pk *PublicKey) Serialize(in []byte) []byte {
	in = appendU16(in, 0)
	in = appendMPI(in, pk.P)
	in = appendMPI(in, pk.Q)
	in = appendMPI(in, pk.G)
	in = appendMPI(in, pk.Y)
	return in
}

// Fingerprint returns the 20-byte, binary fingerprint of the PublicKey.
func (pk *PublicKey) Fingerprint() []byte {
	b := pk.Serialize(nil)
	h := sha1.New()
	h.Write(b[2:])
	return h.Sum(nil)
}

func (pk *PublicKey) Verify(hashed, sig []byte) ([]byte, bool) {
	if len(sig) != 2*dsaSubgroupBytes {
		return nil, false
	}
	r := new(big.Int).SetBytes(sig[:dsaSubgroupBytes])
	s := new(big.Int).SetBytes(sig[dsaSubgroupBytes:])
	ok := dsa.Verify(&pk.PublicKey, hashed, r, s)
	return sig[dsaSubgroupBytes*2:], ok
}

type PrivateKey struct {
	PublicKey
	dsa.PrivateKey
}

func (priv *PrivateKey) Sign(rand io.Reader, hashed []byte) []byte {
	r, s, err := dsa.Sign(rand, &priv.PrivateKey, hashed)
	if err != nil {
		panic(err.Error())
	}
	rBytes := r.Bytes()
	sBytes := s.Bytes()
	if len(rBytes) > dsaSubgroupBytes || len(sBytes) > dsaSubgroupBytes {
		panic("DSA signature too large")
	}

	out := make([]byte, 2*dsaSubgroupBytes)
	copy(out[dsaSubgroupBytes-len(rBytes):], rBytes)
	copy(out[len(out)-len(sBytes):], sBytes)
	return out
}

func (priv *PrivateKey) Serialize(in []byte) []byte {
	in = priv.PublicKey.Serialize(in)
	in = appendMPI(in, priv.PrivateKey.X)
	return in
}

func (priv *PrivateKey) Parse(in []byte) ([]byte, bool) {
	in, ok := priv.PublicKey.Parse(in)
	if !ok {
		return in, ok
	}
	priv.PrivateKey.PublicKey = priv.PublicKey.PublicKey
	priv.PrivateKey.X, in, ok = getMPI(in)
	return in, ok
}

func (priv *PrivateKey) Generate(rand io.Reader) {
	if err := dsa.GenerateParameters(&priv.PrivateKey.PublicKey.Parameters, rand, dsa.L1024N160); err != nil {
		panic(err.Error())
	}
	if err := dsa.GenerateKey(&priv.PrivateKey, rand); err != nil {
		panic(err.Error())
	}
	priv.PublicKey.PublicKey = priv.PrivateKey.PublicKey
}

func notHex(r rune) bool {
	if r >= '0' && r <= '9' ||
		r >= 'a' && r <= 'f' ||
		r >= 'A' && r <= 'F' {
		return false
	}

	return true
}

// Import parses the contents of a libotr private key file.
func (priv *PrivateKey) Import(in []byte) bool {
	mpiStart := []byte(" #")

	mpis := make([]*big.Int, 5)

	for i := 0; i < len(mpis); i++ {
		start := bytes.Index(in, mpiStart)
		if start == -1 {
			return false
		}
		in = in[start+len(mpiStart):]
		end := bytes.IndexFunc(in, notHex)
		if end == -1 {
			return false
		}
		hexBytes := in[:end]
		in = in[end:]

		if len(hexBytes)&1 != 0 {
			return false
		}

		mpiBytes := make([]byte, len(hexBytes)/2)
		if _, err := hex.Decode(mpiBytes, hexBytes); err != nil {
			return false
		}

		mpis[i] = new(big.Int).SetBytes(mpiBytes)
	}

	priv.PrivateKey.P = mpis[0]
	priv.PrivateKey.Q = mpis[1]
	priv.PrivateKey.G = mpis[2]
	priv.PrivateKey.Y = mpis[3]
	priv.PrivateKey.X = mpis[4]
	priv.PublicKey.PublicKey = priv.PrivateKey.PublicKey

	a := new(big.Int).Exp(priv.PrivateKey.G, priv.PrivateKey.X, priv.PrivateKey.P)
	return a.Cmp(priv.PrivateKey.Y) == 0
}

func getU8(in []byte) (uint8, []byte, bool) {
	if len(in) < 1 {
		return 0, in, false
	}
	return in[0], in[1:], true
}

func getU16(in []byte) (uint16, []byte, bool) {
	if len(in) < 2 {
		return 0, in, false
	}
	r := uint16(in[0])<<8 | uint16(in[1])
	return r, in[2:], true
}

func getU32(in []byte) (uint32, []byte, bool) {
	if len(in) < 4 {
		return 0, in, false
	}
	r := uint32(in[0])<<24 | uint32(in[1])<<16 | uint32(in[2])<<8 | uint32(in[3])
	return r, in[4:], true
}

func getMPI(in []byte) (*big.Int, []byte, bool) {
	l, in, ok := getU32(in)
	if !ok || uint32(len(in)) < l {
		return nil, in, false
	}
	r := new(big.Int).SetBytes(in[:l])
	return r, in[l:], true
}

func getData(in []byte) ([]byte, []byte, bool) {
	l, in, ok := getU32(in)
	if !ok || uint32(len(in)) < l {
		return nil, in, false
	}
	return in[:l], in[l:], true
}

func getNBytes(in []byte, n int) ([]byte, []byte, bool) {
	if len(in) < n {
		return nil, in, false
	}
	return in[:n], in[n:], true
}

func appendU16(out []byte, v uint16) []byte {
	out = append(out, byte(v>>8), byte(v))
	return out
}

func appendU32(out []byte, v uint32) []byte {
	out = append(out, byte(v>>24), byte(v>>16), byte(v>>8), byte(v))
	return out
}

func appendData(out, v []byte) []byte {
	out = appendU32(out, uint32(len(v)))
	out = append(out, v...)
	return out
}

func appendMPI(out []byte, v *big.Int) []byte {
	vBytes := v.Bytes()
	out = appendU32(out, uint32(len(vBytes)))
	out = append(out, vBytes...)
	return out
}

func appendMPIs(out []byte, mpis ...*big.Int) []byte {
	for _, mpi := range mpis {
		out = appendMPI(out, mpi)
	}
	return out
}

func zero(b []byte) {
	for i := range b {
		b[i] = 0
	}
}
