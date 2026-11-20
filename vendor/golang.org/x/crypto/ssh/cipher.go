// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/des"
	"crypto/fips140"
	"crypto/rc4"
	"crypto/subtle"
	"encoding/binary"
	"errors"
	"fmt"
	"hash"
	"io"
	"slices"

	"golang.org/x/crypto/chacha20"
	"golang.org/x/crypto/internal/poly1305"
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

type cipherMode struct {
	keySize int
	ivSize  int
	create  func(key, iv []byte, macKey []byte, algs DirectionAlgorithms) (packetCipher, error)
}

func streamCipherMode(skip int, createFunc func(key, iv []byte) (cipher.Stream, error)) func(key, iv []byte, macKey []byte, algs DirectionAlgorithms) (packetCipher, error) {
	return func(key, iv, macKey []byte, algs DirectionAlgorithms) (packetCipher, error) {
		stream, err := createFunc(key, iv)
		if err != nil {
			return nil, err
		}

		var streamDump []byte
		if skip > 0 {
			streamDump = make([]byte, 512)
		}

		for remainingToDump := skip; remainingToDump > 0; {
			dumpThisTime := remainingToDump
			if dumpThisTime > len(streamDump) {
				dumpThisTime = len(streamDump)
			}
			stream.XORKeyStream(streamDump[:dumpThisTime], streamDump[:dumpThisTime])
			remainingToDump -= dumpThisTime
		}

		mac := macModes[algs.MAC].new(macKey)
		return &streamPacketCipher{
			mac:       mac,
			etm:       macModes[algs.MAC].etm,
			macResult: make([]byte, mac.Size()),
			cipher:    stream,
		}, nil
	}
}

// cipherModes documents properties of supported ciphers. Ciphers not included
// are not supported and will not be negotiated, even if explicitly configured.
// When FIPS mode is enabled, only FIPS-approved algorithms are included.
var cipherModes = map[string]*cipherMode{}

func init() {
	cipherModes[CipherAES128CTR] = &cipherMode{16, aes.BlockSize, streamCipherMode(0, newAESCTR)}
	cipherModes[CipherAES192CTR] = &cipherMode{24, aes.BlockSize, streamCipherMode(0, newAESCTR)}
	cipherModes[CipherAES256CTR] = &cipherMode{32, aes.BlockSize, streamCipherMode(0, newAESCTR)}
	//  Use of GCM with arbitrary IVs is not allowed in FIPS 140-only mode,
	// we'll wire it up to NewGCMForSSH in Go 1.26.
	//
	// For now it means we'll work with fips140=on but not fips140=only.
	cipherModes[CipherAES128GCM] = &cipherMode{16, 12, newGCMCipher}
	cipherModes[CipherAES256GCM] = &cipherMode{32, 12, newGCMCipher}

	if fips140.Enabled() {
		defaultCiphers = slices.DeleteFunc(defaultCiphers, func(algo string) bool {
			_, ok := cipherModes[algo]
			return !ok
		})
		return
	}

	cipherModes[CipherChaCha20Poly1305] = &cipherMode{64, 0, newChaCha20Cipher}
	// Insecure ciphers not included in the default configuration.
	cipherModes[InsecureCipherRC4128] = &cipherMode{16, 0, streamCipherMode(1536, newRC4)}
	cipherModes[InsecureCipherRC4256] = &cipherMode{32, 0, streamCipherMode(1536, newRC4)}
	cipherModes[InsecureCipherRC4] = &cipherMode{16, 0, streamCipherMode(0, newRC4)}
	// CBC mode is insecure and so is not included in the default config.
	// (See https://www.ieee-security.org/TC/SP2013/papers/4977a526.pdf). If absolutely
	// needed, it's possible to specify a custom Config to enable it.
	// You should expect that an active attacker can recover plaintext if
	// you do.
	cipherModes[InsecureCipherAES128CBC] = &cipherMode{16, aes.BlockSize, newAESCBCCipher}
	cipherModes[InsecureCipherTripleDESCBC] = &cipherMode{24, des.BlockSize, newTripleDESCBCCipher}
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

// readCipherPacket reads and decrypt a single packet from the reader argument.
func (s *streamPacketCipher) readCipherPacket(seqNum uint32, r io.Reader) ([]byte, error) {
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

// writeCipherPacket encrypts and sends a packet of data to the writer argument
func (s *streamPacketCipher) writeCipherPacket(seqNum uint32, w io.Writer, rand io.Reader, packet []byte) error {
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

func newGCMCipher(key, iv, unusedMacKey []byte, unusedAlgs DirectionAlgorithms) (packetCipher, error) {
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

func (c *gcmCipher) writeCipherPacket(seqNum uint32, w io.Writer, rand io.Reader, packet []byte) error {
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

func (c *gcmCipher) readCipherPacket(seqNum uint32, r io.Reader) ([]byte, error) {
	if _, err := io.ReadFull(r, c.prefix[:]); err != nil {
		return nil, err
	}
	length := binary.BigEndian.Uint32(c.prefix[:])
	if length > maxPacket {
		return nil, errors.New("ssh: max packet length exceeded")
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

	if len(plain) == 0 {
		return nil, errors.New("ssh: empty packet")
	}

	padding := plain[0]
	if padding < 4 {
		// padding is a byte, so it automatically satisfies
		// the maximum size, which is 255.
		return nil, fmt.Errorf("ssh: illegal padding %d", padding)
	}

	if int(padding)+1 >= len(plain) {
		return nil, fmt.Errorf("ssh: padding %d too large", padding)
	}
	plain = plain[1 : length-uint32(padding)]
	return plain, nil
}

// cbcCipher implements the CBC ciphers defined in RFC 4253 section 6.1, in
// MAC-then-encrypt mode and, for the *-etm@openssh.com MACs, in the
// encrypt-then-MAC mode defined in [PROTOCOL], section 1.5.
//
//	0               4                prefixLen                   macStart
//	|               |                |                           |
//	+---------------+----------------+---------+-----------------+-----------+
//	| packet_length | padding_length | payload | random padding  |    mac    |
//	+---------------+----------------+---------+-----------------+-----------+
//	                |<--------- length = packet_length --------->|< macSize >|
//	                                           |< paddingLength >|
//	|<-------------------------- entirePacketSize -------------------------->|
//
//	MAC-then-encrypt (encStart = 0)
//	|<----------------------- encLength ------------------------>|
//	|<-------------- MAC'd (seqNum || plaintext) --------------->|
//
//	encrypt-then-MAC (encStart = 4)
//	                |<--------------- encLength ---------------->|
//	|<-------------- MAC'd (seqNum || ciphertext) -------------->|
//
// encLength is always a multiple of effectiveBlockSize, and in ETM mode the
// packet_length field is not part of it. The padding is only trustworthy once
// the MAC has been verified.
type cbcCipher struct {
	mac       hash.Hash
	macSize   uint32
	etm       bool
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

func newCBCCipher(c cipher.Block, _, iv, macKey []byte, algs DirectionAlgorithms) (packetCipher, error) {
	macMode := macModes[algs.MAC]
	if macMode == nil {
		return nil, fmt.Errorf("ssh: unsupported MAC %v", algs.MAC)
	}
	mac := macMode.new(macKey)
	cbc := &cbcCipher{
		mac:        mac,
		macSize:    uint32(mac.Size()),
		etm:        macMode.etm,
		decrypter:  cipher.NewCBCDecrypter(c, iv),
		encrypter:  cipher.NewCBCEncrypter(c, iv),
		packetData: make([]byte, 1024),
	}

	return cbc, nil
}

func newAESCBCCipher(key, iv, macKey []byte, algs DirectionAlgorithms) (packetCipher, error) {
	c, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}

	cbc, err := newCBCCipher(c, key, iv, macKey, algs)
	if err != nil {
		return nil, err
	}

	return cbc, nil
}

func newTripleDESCBCCipher(key, iv, macKey []byte, algs DirectionAlgorithms) (packetCipher, error) {
	c, err := des.NewTripleDESCipher(key)
	if err != nil {
		return nil, err
	}

	cbc, err := newCBCCipher(c, key, iv, macKey, algs)
	if err != nil {
		return nil, err
	}

	return cbc, nil
}

const (
	cbcMinPacketSizeMultiple = 8
	cbcMinPacketSize         = 16
	cbcMinPaddingSize        = 4
)

// cbcError represents a verification error that may leak information.
type cbcError string

func (e cbcError) Error() string { return string(e) }

func (c *cbcCipher) readCipherPacket(seqNum uint32, r io.Reader) ([]byte, error) {
	p, err := c.readCipherPacketLeaky(seqNum, r)
	if _, ok := err.(cbcError); ok {
		// Verification error: read a fixed amount of
		// data, to make distinguishing between
		// failing MAC and failing length check more
		// difficult.
		io.CopyN(io.Discard, r, int64(c.oracleCamouflage))
	}
	return p, err
}

func (c *cbcCipher) readCipherPacketLeaky(seqNum uint32, r io.Reader) ([]byte, error) {
	blockSize := c.decrypter.BlockSize()
	// The encrypted portion of a packet is a multiple of the cipher block size
	// or 8, whichever is larger.
	effectiveBlockSize := uint32(max(cbcMinPacketSizeMultiple, blockSize))
	encStart := uint32(0)
	// A packet is at least 16 bytes, or the cipher block size if larger. This does
	// not apply to EtM (but the encrypted length still can't be zero).
	minEncLength := uint32(max(cbcMinPacketSize, blockSize))
	// For EtM MACs the packet length is transmitted in the clear and is not part
	// of the encrypted data, which starts after it.
	if c.etm {
		encStart = 4
		minEncLength = effectiveBlockSize
	}

	// Read the header, which will include some of the subsequent data in the
	// case of block ciphers - this is copied back to the payload later. The
	// rounding up to the block size covers at least prefixLen bytes, so the
	// padding length is always within firstBlock.
	headerEncLength := prefixLen - encStart
	firstBlockLength := encStart + (headerEncLength+uint32(blockSize)-1)/uint32(blockSize)*uint32(blockSize)
	firstBlock := c.packetData[:firstBlockLength]
	if _, err := io.ReadFull(r, firstBlock); err != nil {
		return nil, err
	}

	c.oracleCamouflage = maxPacket + 4 + c.macSize - firstBlockLength

	c.mac.Reset()
	binary.BigEndian.PutUint32(c.seqNumBytes[:], seqNum)
	c.mac.Write(c.seqNumBytes[:])

	var length uint32
	if c.etm {
		// For EtM MACs the MAC is computed over the cleartext packet length
		// followed by the ciphertext, so it is updated before decrypting.
		length = binary.BigEndian.Uint32(firstBlock[:encStart])
		c.mac.Write(firstBlock[:encStart])
		c.mac.Write(firstBlock[encStart:])
		c.decrypter.CryptBlocks(firstBlock[encStart:], firstBlock[encStart:])
	} else {
		c.decrypter.CryptBlocks(firstBlock, firstBlock)
		length = binary.BigEndian.Uint32(firstBlock[:4])
	}

	if length > maxPacket {
		return nil, cbcError("ssh: packet too large")
	}
	// Length of the encrypted portion of the packet, which includes the packet
	// length field only for non-EtM MACs.
	encLength := 4 + length - encStart
	if encLength < minEncLength {
		return nil, cbcError("ssh: packet too small")
	}
	if encLength%effectiveBlockSize != 0 {
		return nil, cbcError("ssh: invalid packet length multiple")
	}

	// The padding length is part of the encrypted data. For EtM MACs it is not
	// acted upon before the MAC is verified, otherwise the CBC padding would
	// still be usable as an oracle, which is what EtM prevents.
	paddingLength := uint32(firstBlock[4])
	if !c.etm {
		if paddingLength < cbcMinPaddingSize || length <= paddingLength+1 {
			return nil, cbcError("ssh: invalid packet length")
		}
	}

	// Positions within the c.packetData buffer: the MAC follows the encrypted
	// data.
	macStart := encStart + encLength

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

	n, err := io.ReadFull(r, c.packetData[firstBlockLength:])
	if err != nil {
		return nil, err
	}
	c.oracleCamouflage -= uint32(n)

	remainingCrypted := c.packetData[firstBlockLength:macStart]
	if c.etm {
		c.mac.Write(remainingCrypted)
	}
	c.decrypter.CryptBlocks(remainingCrypted, remainingCrypted)

	if !c.etm {
		c.mac.Write(c.packetData[:macStart])
	}
	c.macResult = c.mac.Sum(c.macResult[:0])
	if subtle.ConstantTimeCompare(c.macResult, c.packetData[macStart:]) != 1 {
		return nil, cbcError("ssh: MAC failure")
	}

	if c.etm {
		if paddingLength < cbcMinPaddingSize || length <= paddingLength+1 {
			return nil, cbcError("ssh: invalid packet length")
		}
	}

	return c.packetData[prefixLen : macStart-paddingLength], nil
}

func (c *cbcCipher) writeCipherPacket(seqNum uint32, w io.Writer, rand io.Reader, payload []byte) error {
	blockSize := c.encrypter.BlockSize()
	// The encrypted portion of a packet is a multiple of the cipher block size
	// or 8, whichever is larger.
	effectiveBlockSize := uint32(max(cbcMinPacketSizeMultiple, blockSize))
	// For EtM MACs the packet length is transmitted in the clear and is not part
	// of the encrypted data, which starts after it.
	encStart := uint32(0)
	if c.etm {
		encStart = 4
	}
	// Length of the encrypted portion of the packet (header, payload, padding),
	// enforcing the minimum padding and the minimum packet size of RFC 4253,
	// Section 6: 16 bytes, or the cipher block size if larger, excluding the
	// MAC.
	encLength := uint32(max(prefixLen-int(encStart)+len(payload)+cbcMinPaddingSize, cbcMinPacketSize-int(encStart)))
	// Enforce block size.
	encLength = (encLength + effectiveBlockSize - 1) / effectiveBlockSize * effectiveBlockSize
	// The packet length field covers the whole packet except the field itself.
	length := encStart + encLength - 4
	paddingLength := int(length) - (1 + len(payload))

	// Overall buffer contains: header, payload, padding, mac.
	// Space for the MAC is reserved in the capacity but not the slice length.
	macStart := encStart + encLength
	entirePacketSize := macStart + c.macSize
	if uint32(cap(c.packetData)) < entirePacketSize {
		c.packetData = make([]byte, macStart, entirePacketSize)
	} else {
		c.packetData = c.packetData[:macStart]
	}

	p := c.packetData

	// Packet header.
	binary.BigEndian.PutUint32(p, length)
	p = p[4:]
	p[0] = byte(paddingLength)

	// Payload.
	p = p[1:]
	copy(p, payload)

	// Padding.
	p = p[len(payload):]
	if _, err := io.ReadFull(rand, p); err != nil {
		return err
	}

	c.mac.Reset()
	binary.BigEndian.PutUint32(c.seqNumBytes[:], seqNum)
	c.mac.Write(c.seqNumBytes[:])
	if c.etm {
		// For EtM MACs the MAC is computed over the cleartext packet length
		// followed by the ciphertext.
		c.mac.Write(c.packetData[:encStart])
		c.encrypter.CryptBlocks(c.packetData[encStart:], c.packetData[encStart:])
		c.mac.Write(c.packetData[encStart:])
	} else {
		c.mac.Write(c.packetData)
		c.encrypter.CryptBlocks(c.packetData, c.packetData)
	}
	// The MAC is now appended into the capacity reserved for it earlier.
	c.packetData = c.mac.Sum(c.packetData)

	if _, err := w.Write(c.packetData); err != nil {
		return err
	}

	return nil
}

// chacha20Poly1305Cipher implements the chacha20-poly1305@openssh.com
// AEAD, which is described here:
//
//	https://tools.ietf.org/html/draft-josefsson-ssh-chacha20-poly1305-openssh-00
//
// the methods here also implement padding, which RFC 4253 Section 6
// also requires of stream ciphers.
type chacha20Poly1305Cipher struct {
	lengthKey  [32]byte
	contentKey [32]byte
	buf        []byte
}

func newChaCha20Cipher(key, unusedIV, unusedMACKey []byte, unusedAlgs DirectionAlgorithms) (packetCipher, error) {
	if len(key) != 64 {
		panic(len(key))
	}

	c := &chacha20Poly1305Cipher{
		buf: make([]byte, 256),
	}

	copy(c.contentKey[:], key[:32])
	copy(c.lengthKey[:], key[32:])
	return c, nil
}

func (c *chacha20Poly1305Cipher) readCipherPacket(seqNum uint32, r io.Reader) ([]byte, error) {
	nonce := make([]byte, 12)
	binary.BigEndian.PutUint32(nonce[8:], seqNum)
	s, err := chacha20.NewUnauthenticatedCipher(c.contentKey[:], nonce)
	if err != nil {
		return nil, err
	}
	var polyKey, discardBuf [32]byte
	s.XORKeyStream(polyKey[:], polyKey[:])
	s.XORKeyStream(discardBuf[:], discardBuf[:]) // skip the next 32 bytes

	encryptedLength := c.buf[:4]
	if _, err := io.ReadFull(r, encryptedLength); err != nil {
		return nil, err
	}

	var lenBytes [4]byte
	ls, err := chacha20.NewUnauthenticatedCipher(c.lengthKey[:], nonce)
	if err != nil {
		return nil, err
	}
	ls.XORKeyStream(lenBytes[:], encryptedLength)

	length := binary.BigEndian.Uint32(lenBytes[:])
	if length > maxPacket {
		return nil, errors.New("ssh: invalid packet length, packet too large")
	}

	contentEnd := 4 + length
	packetEnd := contentEnd + poly1305.TagSize
	if uint32(cap(c.buf)) < packetEnd {
		c.buf = make([]byte, packetEnd)
		copy(c.buf[:], encryptedLength)
	} else {
		c.buf = c.buf[:packetEnd]
	}

	if _, err := io.ReadFull(r, c.buf[4:packetEnd]); err != nil {
		return nil, err
	}

	var mac [poly1305.TagSize]byte
	copy(mac[:], c.buf[contentEnd:packetEnd])
	if !poly1305.Verify(&mac, c.buf[:contentEnd], &polyKey) {
		return nil, errors.New("ssh: MAC failure")
	}

	plain := c.buf[4:contentEnd]
	s.XORKeyStream(plain, plain)

	if len(plain) == 0 {
		return nil, errors.New("ssh: empty packet")
	}

	padding := plain[0]
	if padding < 4 {
		// padding is a byte, so it automatically satisfies
		// the maximum size, which is 255.
		return nil, fmt.Errorf("ssh: illegal padding %d", padding)
	}

	if int(padding)+1 >= len(plain) {
		return nil, fmt.Errorf("ssh: padding %d too large", padding)
	}

	plain = plain[1 : len(plain)-int(padding)]

	return plain, nil
}

func (c *chacha20Poly1305Cipher) writeCipherPacket(seqNum uint32, w io.Writer, rand io.Reader, payload []byte) error {
	nonce := make([]byte, 12)
	binary.BigEndian.PutUint32(nonce[8:], seqNum)
	s, err := chacha20.NewUnauthenticatedCipher(c.contentKey[:], nonce)
	if err != nil {
		return err
	}
	var polyKey, discardBuf [32]byte
	s.XORKeyStream(polyKey[:], polyKey[:])
	s.XORKeyStream(discardBuf[:], discardBuf[:]) // skip the next 32 bytes

	// There is no blocksize, so fall back to multiple of 8 byte
	// padding, as described in RFC 4253, Sec 6.
	const packetSizeMultiple = 8

	padding := packetSizeMultiple - (1+len(payload))%packetSizeMultiple
	if padding < 4 {
		padding += packetSizeMultiple
	}

	// size (4 bytes), padding (1), payload, padding, tag.
	totalLength := 4 + 1 + len(payload) + padding + poly1305.TagSize
	if cap(c.buf) < totalLength {
		c.buf = make([]byte, totalLength)
	} else {
		c.buf = c.buf[:totalLength]
	}

	binary.BigEndian.PutUint32(c.buf, uint32(1+len(payload)+padding))
	ls, err := chacha20.NewUnauthenticatedCipher(c.lengthKey[:], nonce)
	if err != nil {
		return err
	}
	ls.XORKeyStream(c.buf, c.buf[:4])
	c.buf[4] = byte(padding)
	copy(c.buf[5:], payload)
	packetEnd := 5 + len(payload) + padding
	if _, err := io.ReadFull(rand, c.buf[5+len(payload):packetEnd]); err != nil {
		return err
	}

	s.XORKeyStream(c.buf[4:], c.buf[4:packetEnd])

	var mac [poly1305.TagSize]byte
	poly1305.Sum(&mac, c.buf[:packetEnd], &polyKey)

	copy(c.buf[packetEnd:], mac[:])

	if _, err := w.Write(c.buf); err != nil {
		return err
	}
	return nil
}
