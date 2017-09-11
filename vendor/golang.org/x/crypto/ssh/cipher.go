// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/des"
	"crypto/rc4"
	"crypto/subtle"
	"encoding/binary"
	"errors"
	"fmt"
	"hash"
	"io"
	"io/ioutil"
)

const (
	packetSizeMultiple = 16 // TODO(huin) this should be determined by the cipher.

	// RFC 4253 section 6.1 defines a minimum packet size of 32768 that implementations
	// MUST be able to process (plus a few more kilobytes for padding and mac). The RFC
	// indicates implementations SHOULD be able to handle larger packet sizes, but then
	// waffles on about reasonable limits.
	//
	// OpenSSH caps their maxPacket at 256kB so we choose to do
	// the same. maxPacket is also used to ensure that uint32
	// length fields do not overflow, so it should remain well
	// below 4G.
	maxPacket = 256 * 1024
)

// noneCipher implements cipher.Stream and provides no encryption. It is used
// by the transport before the first key-exchange.
type noneCipher struct{}

func (c noneCipher) XORKeyStream(dst, src []byte) {
	copy(dst, src)
}

func newAESCTR(key, iv []byte) (cipher.Stream, error) {
	c, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	return cipher.NewCTR(c, iv), nil
}

func newRC4(key, iv []byte) (cipher.Stream, error) {
	return rc4.NewCipher(key)
}

type streamCipherMode struct {
	keySize    int
	ivSize     int
	skip       int
	createFunc func(key, iv []byte) (cipher.Stream, error)
}

func (c *streamCipherMode) createStream(key, iv []byte) (cipher.Stream, error) {
	if len(key) < c.keySize {
		panic("ssh: key length too small for cipher")
	}
	if len(iv) < c.ivSize {
		panic("ssh: iv too small for cipher")
	}

	stream, err := c.createFunc(key[:c.keySize], iv[:c.ivSize])
	if err != nil {
		return nil, err
	}

	var streamDump []byte
	if c.skip > 0 {
		streamDump = make([]byte, 512)
	}

	for remainingToDump := c.skip; remainingToDump > 0; {
		dumpThisTime := remainingToDump
		if dumpThisTime > len(streamDump) {
			dumpThisTime = len(streamDump)
		}
		stream.XORKeyStream(streamDump[:dumpThisTime], streamDump[:dumpThisTime])
		remainingToDump -= dumpThisTime
	}

	return stream, nil
}

// cipherModes documents properties of supported ciphers. Ciphers not included
// are not supported and will not be negotiated, even if explicitly requested in
// ClientConfig.Crypto.Ciphers.
var cipherModes = map[string]*streamCipherMode{
	// Ciphers from RFC4344, which introduced many CTR-based ciphers. Algorithms
	// are defined in the order specified in the RFC.
	"aes128-ctr": {16, aes.BlockSize, 0, newAESCTR},
	"aes192-ctr": {24, aes.BlockSize, 0, newAESCTR},
	"aes256-ctr": {32, aes.BlockSize, 0, newAESCTR},

	// Ciphers from RFC4345, which introduces security-improved arcfour ciphers.
	// They are defined in the order specified in the RFC.
	"arcfour128": {16, 0, 1536, newRC4},
	"arcfour256": {32, 0, 1536, newRC4},

	// Cipher defined in RFC 4253, which describes SSH Transport Layer Protocol.
	// Note that this cipher is not safe, as stated in RFC 4253: "Arcfour (and
	// RC4) has problems with weak keys, and should be used with caution."
	// RFC4345 introduces improved versions of Arcfour.
	"arcfour": {16, 0, 0, newRC4},

	// AES-GCM is not a stream cipher, so it is constructed with a
	// special case. If we add any more non-stream ciphers, we
	// should invest a cleaner way to do this.
	gcmCipherID: {16, 12, 0, nil},

	// CBC mode is insecure and so is not included in the default config.
	// (See http://www.isg.rhul.ac.uk/~kp/SandPfinal.pdf). If absolutely
	// needed, it's possible to specify a custom Config to enable it.
	// You should expect that an active attacker can recover plaintext if
	// you do.
	aes128cbcID: {16, aes.BlockSize, 0, nil},

	// 3des-cbc is insecure and is disabled by default.
	tripledescbcID: {24, des.BlockSize, 0, nil},
}

// prefixLen is the length of the packet prefix that contains the packet length
// and number of padding bytes.
const prefixLen = 5

// streamPacketCipher is a packetCipher using a stream cipher.
type streamPacketCipher struct {
	mac    hash.Hash
	cipher cipher.Stream
	etm    bool

	// The following members are to avoid per-packet allocations.
	prefix      [prefixLen]byte
	seqNumBytes [4]byte
	padding     [2 * packetSizeMultiple]byte
	packetData  []byte
	macResult   []byte
}

// readPacket reads and decrypt a single packet from the reader argument.
func (s *streamPacketCipher) readPacket(seqNum uint32, r io.Reader) ([]byte, error) {
	if _, err := io.ReadFull(r, s.prefix[:]); err != nil {
		return nil, err
	}

	var encryptedPaddingLength [1]byte
	if s.mac != nil && s.etm {
		copy(encryptedPaddingLength[:], s.prefix[4:5])
		s.cipher.XORKeyStream(s.prefix[4:5], s.prefix[4:5])
	} else {
		s.cipher.XORKeyStream(s.prefix[:], s.prefix[:])
	}

	length := binary.BigEndian.Uint32(s.prefix[0:4])
	paddingLength := uint32(s.prefix[4])

	var macSize uint32
	if s.mac != nil {
		s.mac.Reset()
		binary.BigEndian.PutUint32(s.seqNumBytes[:], seqNum)
		s.mac.Write(s.seqNumBytes[:])
		if s.etm {
			s.mac.Write(s.prefix[:4])
			s.mac.Write(encryptedPaddingLength[:])
		} else {
			s.mac.Write(s.prefix[:])
		}
		macSize = uint32(s.mac.Size())
	}

	if length <= paddingLength+1 {
		return nil, errors.New("ssh: invalid packet length, packet too small")
	}

	if length > maxPacket {
		return nil, errors.New("ssh: invalid packet length, packet too large")
	}

	// the maxPacket check above ensures that length-1+macSize
	// does not overflow.
	if uint32(cap(s.packetData)) < length-1+macSize {
		s.packetData = make([]byte, length-1+macSize)
	} else {
		s.packetData = s.packetData[:length-1+macSize]
	}

	if _, err := io.ReadFull(r, s.packetData); err != nil {
		return nil, err
	}
	mac := s.packetData[length-1:]
	data := s.packetData[:length-1]

	if s.mac != nil && s.etm {
		s.mac.Write(data)
	}

	s.cipher.XORKeyStream(data, data)

	if s.mac != nil {
		if !s.etm {
			s.mac.Write(data)
		}
		s.macResult = s.mac.Sum(s.macResult[:0])
		if subtle.ConstantTimeCompare(s.macResult, mac) != 1 {
			return nil, errors.New("ssh: MAC failure")
		}
	}

	return s.packetData[:length-paddingLength-1], nil
}

// writePacket encrypts and sends a packet of data to the writer argument
func (s *streamPacketCipher) writePacket(seqNum uint32, w io.Writer, rand io.Reader, packet []byte) error {
	if len(packet) > maxPacket {
		return errors.New("ssh: packet too large")
	}

	aadlen := 0
	if s.mac != nil && s.etm {
		// packet length is not encrypted for EtM modes
		aadlen = 4
	}

	paddingLength := packetSizeMultiple - (prefixLen+len(packet)-aadlen)%packetSizeMultiple
	if paddingLength < 4 {
		paddingLength += packetSizeMultiple
	}

	length := len(packet) + 1 + paddingLength
	binary.BigEndian.PutUint32(s.prefix[:], uint32(length))
	s.prefix[4] = byte(paddingLength)
	padding := s.padding[:paddingLength]
	if _, err := io.ReadFull(rand, padding); err != nil {
		return err
	}

	if s.mac != nil {
		s.mac.Reset()
		binary.BigEndian.PutUint32(s.seqNumBytes[:], seqNum)
		s.mac.Write(s.seqNumBytes[:])

		if s.etm {
			// For EtM algorithms, the packet length must stay unencrypted,
			// but the following data (padding length) must be encrypted
			s.cipher.XORKeyStream(s.prefix[4:5], s.prefix[4:5])
		}

		s.mac.Write(s.prefix[:])

		if !s.etm {
			// For non-EtM algorithms, the algorithm is applied on unencrypted data
			s.mac.Write(packet)
			s.mac.Write(padding)
		}
	}

	if !(s.mac != nil && s.etm) {
		// For EtM algorithms, the padding length has already been encrypted
		// and the packet length must remain unencrypted
		s.cipher.XORKeyStream(s.prefix[:], s.prefix[:])
	}

	s.cipher.XORKeyStream(packet, packet)
	s.cipher.XORKeyStream(padding, padding)

	if s.mac != nil && s.etm {
		// For EtM algorithms, packet and padding must be encrypted
		s.mac.Write(packet)
		s.mac.Write(padding)
	}

	if _, err := w.Write(s.prefix[:]); err != nil {
		return err
	}
	if _, err := w.Write(packet); err != nil {
		return err
	}
	if _, err := w.Write(padding); err != nil {
		return err
	}

	if s.mac != nil {
		s.macResult = s.mac.Sum(s.macResult[:0])
		if _, err := w.Write(s.macResult); err != nil {
			return err
		}
	}

	return nil
}

type gcmCipher struct {
	aead   cipher.AEAD
	prefix [4]byte
	iv     []byte
	buf    []byte
}

func newGCMCipher(iv, key, macKey []byte) (packetCipher, error) {
	c, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}

	aead, err := cipher.NewGCM(c)
	if err != nil {
		return nil, err
	}

	return &gcmCipher{
		aead: aead,
		iv:   iv,
	}, nil
}

const gcmTagSize = 16

func (c *gcmCipher) writePacket(seqNum uint32, w io.Writer, rand io.Reader, packet []byte) error {
	// Pad out to multiple of 16 bytes. This is different from the
	// stream cipher because that encrypts the length too.
	padding := byte(packetSizeMultiple - (1+len(packet))%packetSizeMultiple)
	if padding < 4 {
		padding += packetSizeMultiple
	}

	length := uint32(len(packet) + int(padding) + 1)
	binary.BigEndian.PutUint32(c.prefix[:], length)
	if _, err := w.Write(c.prefix[:]); err != nil {
		return err
	}

	if cap(c.buf) < int(length) {
		c.buf = make([]byte, length)
	} else {
		c.buf = c.buf[:length]
	}

	c.buf[0] = padding
	copy(c.buf[1:], packet)
	if _, err := io.ReadFull(rand, c.buf[1+len(packet):]); err != nil {
		return err
	}
	c.buf = c.aead.Seal(c.buf[:0], c.iv, c.buf, c.prefix[:])
	if _, err := w.Write(c.buf); err != nil {
		return err
	}
	c.incIV()

	return nil
}

func (c *gcmCipher) incIV() {
	for i := 4 + 7; i >= 4; i-- {
		c.iv[i]++
		if c.iv[i] != 0 {
			break
		}
	}
}

func (c *gcmCipher) readPacket(seqNum uint32, r io.Reader) ([]byte, error) {
	if _, err := io.ReadFull(r, c.prefix[:]); err != nil {
		return nil, err
	}
	length := binary.BigEndian.Uint32(c.prefix[:])
	if length > maxPacket {
		return nil, errors.New("ssh: max packet length exceeded.")
	}

	if cap(c.buf) < int(length+gcmTagSize) {
		c.buf = make([]byte, length+gcmTagSize)
	} else {
		c.buf = c.buf[:length+gcmTagSize]
	}

	if _, err := io.ReadFull(r, c.buf); err != nil {
		return nil, err
	}

	plain, err := c.aead.Open(c.buf[:0], c.iv, c.buf, c.prefix[:])
	if err != nil {
		return nil, err
	}
	c.incIV()

	padding := plain[0]
	if padding < 4 {
		// padding is a byte, so it automatically satisfies
		// the maximum size, which is 255.
		return nil, fmt.Errorf("ssh: illegal padding %d", padding)
	}

	if int(padding+1) >= len(plain) {
		return nil, fmt.Errorf("ssh: padding %d too large", padding)
	}
	plain = plain[1 : length-uint32(padding)]
	return plain, nil
}

// cbcCipher implements aes128-cbc cipher defined in RFC 4253 section 6.1
type cbcCipher struct {
	mac       hash.Hash
	macSize   uint32
	decrypter cipher.BlockMode
	encrypter cipher.BlockMode

	// The following members are to avoid per-packet allocations.
	seqNumBytes [4]byte
	packetData  []byte
	macResult   []byte

	// Amount of data we should still read to hide which
	// verification error triggered.
	oracleCamouflage uint32
}

func newCBCCipher(c cipher.Block, iv, key, macKey []byte, algs directionAlgorithms) (packetCipher, error) {
	cbc := &cbcCipher{
		mac:        macModes[algs.MAC].new(macKey),
		decrypter:  cipher.NewCBCDecrypter(c, iv),
		encrypter:  cipher.NewCBCEncrypter(c, iv),
		packetData: make([]byte, 1024),
	}
	if cbc.mac != nil {
		cbc.macSize = uint32(cbc.mac.Size())
	}

	return cbc, nil
}

func newAESCBCCipher(iv, key, macKey []byte, algs directionAlgorithms) (packetCipher, error) {
	c, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}

	cbc, err := newCBCCipher(c, iv, key, macKey, algs)
	if err != nil {
		return nil, err
	}

	return cbc, nil
}

func newTripleDESCBCCipher(iv, key, macKey []byte, algs directionAlgorithms) (packetCipher, error) {
	c, err := des.NewTripleDESCipher(key)
	if err != nil {
		return nil, err
	}

	cbc, err := newCBCCipher(c, iv, key, macKey, algs)
	if err != nil {
		return nil, err
	}

	return cbc, nil
}

func maxUInt32(a, b int) uint32 {
	if a > b {
		return uint32(a)
	}
	return uint32(b)
}

const (
	cbcMinPacketSizeMultiple = 8
	cbcMinPacketSize         = 16
	cbcMinPaddingSize        = 4
)

// cbcError represents a verification error that may leak information.
type cbcError string

func (e cbcError) Error() string { return string(e) }

func (c *cbcCipher) readPacket(seqNum uint32, r io.Reader) ([]byte, error) {
	p, err := c.readPacketLeaky(seqNum, r)
	if err != nil {
		if _, ok := err.(cbcError); ok {
			// Verification error: read a fixed amount of
			// data, to make distinguishing between
			// failing MAC and failing length check more
			// difficult.
			io.CopyN(ioutil.Discard, r, int64(c.oracleCamouflage))
		}
	}
	return p, err
}

func (c *cbcCipher) readPacketLeaky(seqNum uint32, r io.Reader) ([]byte, error) {
	blockSize := c.decrypter.BlockSize()

	// Read the header, which will include some of the subsequent data in the
	// case of block ciphers - this is copied back to the payload later.
	// How many bytes of payload/padding will be read with this first read.
	firstBlockLength := uint32((prefixLen + blockSize - 1) / blockSize * blockSize)
	firstBlock := c.packetData[:firstBlockLength]
	if _, err := io.ReadFull(r, firstBlock); err != nil {
		return nil, err
	}

	c.oracleCamouflage = maxPacket + 4 + c.macSize - firstBlockLength

	c.decrypter.CryptBlocks(firstBlock, firstBlock)
	length := binary.BigEndian.Uint32(firstBlock[:4])
	if length > maxPacket {
		return nil, cbcError("ssh: packet too large")
	}
	if length+4 < maxUInt32(cbcMinPacketSize, blockSize) {
		// The minimum size of a packet is 16 (or the cipher block size, whichever
		// is larger) bytes.
		return nil, cbcError("ssh: packet too small")
	}
	// The length of the packet (including the length field but not the MAC) must
	// be a multiple of the block size or 8, whichever is larger.
	if (length+4)%maxUInt32(cbcMinPacketSizeMultiple, blockSize) != 0 {
		return nil, cbcError("ssh: invalid packet length multiple")
	}

	paddingLength := uint32(firstBlock[4])
	if paddingLength < cbcMinPaddingSize || length <= paddingLength+1 {
		return nil, cbcError("ssh: invalid packet length")
	}

	// Positions within the c.packetData buffer:
	macStart := 4 + length
	paddingStart := macStart - paddingLength

	// Entire packet size, starting before length, ending at end of mac.
	entirePacketSize := macStart + c.macSize

	// Ensure c.packetData is large enough for the entire packet data.
	if uint32(cap(c.packetData)) < entirePacketSize {
		// Still need to upsize and copy, but this should be rare at runtime, only
		// on upsizing the packetData buffer.
		c.packetData = make([]byte, entirePacketSize)
		copy(c.packetData, firstBlock)
	} else {
		c.packetData = c.packetData[:entirePacketSize]
	}

	if n, err := io.ReadFull(r, c.packetData[firstBlockLength:]); err != nil {
		return nil, err
	} else {
		c.oracleCamouflage -= uint32(n)
	}

	remainingCrypted := c.packetData[firstBlockLength:macStart]
	c.decrypter.CryptBlocks(remainingCrypted, remainingCrypted)

	mac := c.packetData[macStart:]
	if c.mac != nil {
		c.mac.Reset()
		binary.BigEndian.PutUint32(c.seqNumBytes[:], seqNum)
		c.mac.Write(c.seqNumBytes[:])
		c.mac.Write(c.packetData[:macStart])
		c.macResult = c.mac.Sum(c.macResult[:0])
		if subtle.ConstantTimeCompare(c.macResult, mac) != 1 {
			return nil, cbcError("ssh: MAC failure")
		}
	}

	return c.packetData[prefixLen:paddingStart], nil
}

func (c *cbcCipher) writePacket(seqNum uint32, w io.Writer, rand io.Reader, packet []byte) error {
	effectiveBlockSize := maxUInt32(cbcMinPacketSizeMultiple, c.encrypter.BlockSize())

	// Length of encrypted portion of the packet (header, payload, padding).
	// Enforce minimum padding and packet size.
	encLength := maxUInt32(prefixLen+len(packet)+cbcMinPaddingSize, cbcMinPaddingSize)
	// Enforce block size.
	encLength = (encLength + effectiveBlockSize - 1) / effectiveBlockSize * effectiveBlockSize

	length := encLength - 4
	paddingLength := int(length) - (1 + len(packet))

	// Overall buffer contains: header, payload, padding, mac.
	// Space for the MAC is reserved in the capacity but not the slice length.
	bufferSize := encLength + c.macSize
	if uint32(cap(c.packetData)) < bufferSize {
		c.packetData = make([]byte, encLength, bufferSize)
	} else {
		c.packetData = c.packetData[:encLength]
	}

	p := c.packetData

	// Packet header.
	binary.BigEndian.PutUint32(p, length)
	p = p[4:]
	p[0] = byte(paddingLength)

	// Payload.
	p = p[1:]
	copy(p, packet)

	// Padding.
	p = p[len(packet):]
	if _, err := io.ReadFull(rand, p); err != nil {
		return err
	}

	if c.mac != nil {
		c.mac.Reset()
		binary.BigEndian.PutUint32(c.seqNumBytes[:], seqNum)
		c.mac.Write(c.seqNumBytes[:])
		c.mac.Write(c.packetData)
		// The MAC is now appended into the capacity reserved for it earlier.
		c.packetData = c.mac.Sum(c.packetData)
	}

	c.encrypter.CryptBlocks(c.packetData[:encLength], c.packetData[:encLength])

	if _, err := w.Write(c.packetData); err != nil {
		return err
	}

	return nil
}
