// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package packet implements parsing and serialization of OpenPGP packets, as
// specified in RFC 4880.
package packet // import "golang.org/x/crypto/openpgp/packet"

import (
	"bufio"
	"crypto/aes"
	"crypto/cipher"
	"crypto/des"
	"crypto/rsa"
	"io"
	"math/big"
	"math/bits"

	"golang.org/x/crypto/cast5"
	"golang.org/x/crypto/openpgp/errors"
)

// readFull is the same as io.ReadFull except that reading zero bytes returns
// ErrUnexpectedEOF rather than EOF.
func readFull(r io.Reader, buf []byte) (n int, err error) {
	n, err = io.ReadFull(r, buf)
	if err == io.EOF {
		err = io.ErrUnexpectedEOF
	}
	return
}

// readLength reads an OpenPGP length from r. See RFC 4880, section 4.2.2.
func readLength(r io.Reader) (length int64, isPartial bool, err error) {
	var buf [4]byte
	_, err = readFull(r, buf[:1])
	if err != nil {
		return
	}
	switch {
	case buf[0] < 192:
		length = int64(buf[0])
	case buf[0] < 224:
		length = int64(buf[0]-192) << 8
		_, err = readFull(r, buf[0:1])
		if err != nil {
			return
		}
		length += int64(buf[0]) + 192
	case buf[0] < 255:
		length = int64(1) << (buf[0] & 0x1f)
		isPartial = true
	default:
		_, err = readFull(r, buf[0:4])
		if err != nil {
			return
		}
		length = int64(buf[0])<<24 |
			int64(buf[1])<<16 |
			int64(buf[2])<<8 |
			int64(buf[3])
	}
	return
}

// partialLengthReader wraps an io.Reader and handles OpenPGP partial lengths.
// The continuation lengths are parsed and removed from the stream and EOF is
// returned at the end of the packet. See RFC 4880, section 4.2.2.4.
type partialLengthReader struct {
	r         io.Reader
	remaining int64
	isPartial bool
}

func (r *partialLengthReader) Read(p []byte) (n int, err error) {
	for r.remaining == 0 {
		if !r.isPartial {
			return 0, io.EOF
		}
		r.remaining, r.isPartial, err = readLength(r.r)
		if err != nil {
			return 0, err
		}
	}

	toRead := int64(len(p))
	if toRead > r.remaining {
		toRead = r.remaining
	}

	n, err = r.r.Read(p[:int(toRead)])
	r.remaining -= int64(n)
	if n < int(toRead) && err == io.EOF {
		err = io.ErrUnexpectedEOF
	}
	return
}

// partialLengthWriter writes a stream of data using OpenPGP partial lengths.
// See RFC 4880, section 4.2.2.4.
type partialLengthWriter struct {
	w          io.WriteCloser
	lengthByte [1]byte
	sentFirst  bool
	buf        []byte
}

// RFC 4880 4.2.2.4: the first partial length MUST be at least 512 octets long.
const minFirstPartialWrite = 512

func (w *partialLengthWriter) Write(p []byte) (n int, err error) {
	off := 0
	if !w.sentFirst {
		if len(w.buf) > 0 || len(p) < minFirstPartialWrite {
			off = len(w.buf)
			w.buf = append(w.buf, p...)
			if len(w.buf) < minFirstPartialWrite {
				return len(p), nil
			}
			p = w.buf
			w.buf = nil
		}
		w.sentFirst = true
	}

	power := uint8(30)
	for len(p) > 0 {
		l := 1 << power
		if len(p) < l {
			power = uint8(bits.Len32(uint32(len(p)))) - 1
			l = 1 << power
		}
		w.lengthByte[0] = 224 + power
		_, err = w.w.Write(w.lengthByte[:])
		if err == nil {
			var m int
			m, err = w.w.Write(p[:l])
			n += m
		}
		if err != nil {
			if n < off {
				return 0, err
			}
			return n - off, err
		}
		p = p[l:]
	}
	return n - off, nil
}

func (w *partialLengthWriter) Close() error {
	if len(w.buf) > 0 {
		// In this case we can't send a 512 byte packet.
		// Just send what we have.
		p := w.buf
		w.sentFirst = true
		w.buf = nil
		if _, err := w.Write(p); err != nil {
			return err
		}
	}

	w.lengthByte[0] = 0
	_, err := w.w.Write(w.lengthByte[:])
	if err != nil {
		return err
	}
	return w.w.Close()
}

// A spanReader is an io.LimitReader, but it returns ErrUnexpectedEOF if the
// underlying Reader returns EOF before the limit has been reached.
type spanReader struct {
	r io.Reader
	n int64
}

func (l *spanReader) Read(p []byte) (n int, err error) {
	if l.n <= 0 {
		return 0, io.EOF
	}
	if int64(len(p)) > l.n {
		p = p[0:l.n]
	}
	n, err = l.r.Read(p)
	l.n -= int64(n)
	if l.n > 0 && err == io.EOF {
		err = io.ErrUnexpectedEOF
	}
	return
}

// readHeader parses a packet header and returns an io.Reader which will return
// the contents of the packet. See RFC 4880, section 4.2.
func readHeader(r io.Reader) (tag packetType, length int64, contents io.Reader, err error) {
	var buf [4]byte
	_, err = io.ReadFull(r, buf[:1])
	if err != nil {
		return
	}
	if buf[0]&0x80 == 0 {
		err = errors.StructuralError("tag byte does not have MSB set")
		return
	}
	if buf[0]&0x40 == 0 {
		// Old format packet
		tag = packetType((buf[0] & 0x3f) >> 2)
		lengthType := buf[0] & 3
		if lengthType == 3 {
			length = -1
			contents = r
			return
		}
		lengthBytes := 1 << lengthType
		_, err = readFull(r, buf[0:lengthBytes])
		if err != nil {
			return
		}
		for i := 0; i < lengthBytes; i++ {
			length <<= 8
			length |= int64(buf[i])
		}
		contents = &spanReader{r, length}
		return
	}

	// New format packet
	tag = packetType(buf[0] & 0x3f)
	length, isPartial, err := readLength(r)
	if err != nil {
		return
	}
	if isPartial {
		contents = &partialLengthReader{
			remaining: length,
			isPartial: true,
			r:         r,
		}
		length = -1
	} else {
		contents = &spanReader{r, length}
	}
	return
}

// serializeHeader writes an OpenPGP packet header to w. See RFC 4880, section
// 4.2.
func serializeHeader(w io.Writer, ptype packetType, length int) (err error) {
	var buf [6]byte
	var n int

	buf[0] = 0x80 | 0x40 | byte(ptype)
	if length < 192 {
		buf[1] = byte(length)
		n = 2
	} else if length < 8384 {
		length -= 192
		buf[1] = 192 + byte(length>>8)
		buf[2] = byte(length)
		n = 3
	} else {
		buf[1] = 255
		buf[2] = byte(length >> 24)
		buf[3] = byte(length >> 16)
		buf[4] = byte(length >> 8)
		buf[5] = byte(length)
		n = 6
	}

	_, err = w.Write(buf[:n])
	return
}

// serializeStreamHeader writes an OpenPGP packet header to w where the
// length of the packet is unknown. It returns a io.WriteCloser which can be
// used to write the contents of the packet. See RFC 4880, section 4.2.
func serializeStreamHeader(w io.WriteCloser, ptype packetType) (out io.WriteCloser, err error) {
	var buf [1]byte
	buf[0] = 0x80 | 0x40 | byte(ptype)
	_, err = w.Write(buf[:])
	if err != nil {
		return
	}
	out = &partialLengthWriter{w: w}
	return
}

// Packet represents an OpenPGP packet. Users are expected to try casting
// instances of this interface to specific packet types.
type Packet interface {
	parse(io.Reader) error
}

// consumeAll reads from the given Reader until error, returning the number of
// bytes read.
func consumeAll(r io.Reader) (n int64, err error) {
	var m int
	var buf [1024]byte

	for {
		m, err = r.Read(buf[:])
		n += int64(m)
		if err == io.EOF {
			err = nil
			return
		}
		if err != nil {
			return
		}
	}
}

// packetType represents the numeric ids of the different OpenPGP packet types. See
// http://www.iana.org/assignments/pgp-parameters/pgp-parameters.xhtml#pgp-parameters-2
type packetType uint8

const (
	packetTypeEncryptedKey              packetType = 1
	packetTypeSignature                 packetType = 2
	packetTypeSymmetricKeyEncrypted     packetType = 3
	packetTypeOnePassSignature          packetType = 4
	packetTypePrivateKey                packetType = 5
	packetTypePublicKey                 packetType = 6
	packetTypePrivateSubkey             packetType = 7
	packetTypeCompressed                packetType = 8
	packetTypeSymmetricallyEncrypted    packetType = 9
	packetTypeLiteralData               packetType = 11
	packetTypeUserId                    packetType = 13
	packetTypePublicSubkey              packetType = 14
	packetTypeUserAttribute             packetType = 17
	packetTypeSymmetricallyEncryptedMDC packetType = 18
)

// peekVersion detects the version of a public key packet about to
// be read. A bufio.Reader at the original position of the io.Reader
// is returned.
func peekVersion(r io.Reader) (bufr *bufio.Reader, ver byte, err error) {
	bufr = bufio.NewReader(r)
	var verBuf []byte
	if verBuf, err = bufr.Peek(1); err != nil {
		return
	}
	ver = verBuf[0]
	return
}

// Read reads a single OpenPGP packet from the given io.Reader. If there is an
// error parsing a packet, the whole packet is consumed from the input.
func Read(r io.Reader) (p Packet, err error) {
	tag, _, contents, err := readHeader(r)
	if err != nil {
		return
	}

	switch tag {
	case packetTypeEncryptedKey:
		p = new(EncryptedKey)
	case packetTypeSignature:
		var version byte
		// Detect signature version
		if contents, version, err = peekVersion(contents); err != nil {
			return
		}
		if version < 4 {
			p = new(SignatureV3)
		} else {
			p = new(Signature)
		}
	case packetTypeSymmetricKeyEncrypted:
		p = new(SymmetricKeyEncrypted)
	case packetTypeOnePassSignature:
		p = new(OnePassSignature)
	case packetTypePrivateKey, packetTypePrivateSubkey:
		pk := new(PrivateKey)
		if tag == packetTypePrivateSubkey {
			pk.IsSubkey = true
		}
		p = pk
	case packetTypePublicKey, packetTypePublicSubkey:
		var version byte
		if contents, version, err = peekVersion(contents); err != nil {
			return
		}
		isSubkey := tag == packetTypePublicSubkey
		if version < 4 {
			p = &PublicKeyV3{IsSubkey: isSubkey}
		} else {
			p = &PublicKey{IsSubkey: isSubkey}
		}
	case packetTypeCompressed:
		p = new(Compressed)
	case packetTypeSymmetricallyEncrypted:
		p = new(SymmetricallyEncrypted)
	case packetTypeLiteralData:
		p = new(LiteralData)
	case packetTypeUserId:
		p = new(UserId)
	case packetTypeUserAttribute:
		p = new(UserAttribute)
	case packetTypeSymmetricallyEncryptedMDC:
		se := new(SymmetricallyEncrypted)
		se.MDC = true
		p = se
	default:
		err = errors.UnknownPacketTypeError(tag)
	}
	if p != nil {
		err = p.parse(contents)
	}
	if err != nil {
		consumeAll(contents)
	}
	return
}

// SignatureType represents the different semantic meanings of an OpenPGP
// signature. See RFC 4880, section 5.2.1.
type SignatureType uint8

const (
	SigTypeBinary            SignatureType = 0
	SigTypeText                            = 1
	SigTypeGenericCert                     = 0x10
	SigTypePersonaCert                     = 0x11
	SigTypeCasualCert                      = 0x12
	SigTypePositiveCert                    = 0x13
	SigTypeSubkeyBinding                   = 0x18
	SigTypePrimaryKeyBinding               = 0x19
	SigTypeDirectSignature                 = 0x1F
	SigTypeKeyRevocation                   = 0x20
	SigTypeSubkeyRevocation                = 0x28
)

// PublicKeyAlgorithm represents the different public key system specified for
// OpenPGP. See
// http://www.iana.org/assignments/pgp-parameters/pgp-parameters.xhtml#pgp-parameters-12
type PublicKeyAlgorithm uint8

const (
	PubKeyAlgoRSA     PublicKeyAlgorithm = 1
	PubKeyAlgoElGamal PublicKeyAlgorithm = 16
	PubKeyAlgoDSA     PublicKeyAlgorithm = 17
	// RFC 6637, Section 5.
	PubKeyAlgoECDH  PublicKeyAlgorithm = 18
	PubKeyAlgoECDSA PublicKeyAlgorithm = 19

	// Deprecated in RFC 4880, Section 13.5. Use key flags instead.
	PubKeyAlgoRSAEncryptOnly PublicKeyAlgorithm = 2
	PubKeyAlgoRSASignOnly    PublicKeyAlgorithm = 3
)

// CanEncrypt returns true if it's possible to encrypt a message to a public
// key of the given type.
func (pka PublicKeyAlgorithm) CanEncrypt() bool {
	switch pka {
	case PubKeyAlgoRSA, PubKeyAlgoRSAEncryptOnly, PubKeyAlgoElGamal:
		return true
	}
	return false
}

// CanSign returns true if it's possible for a public key of the given type to
// sign a message.
func (pka PublicKeyAlgorithm) CanSign() bool {
	switch pka {
	case PubKeyAlgoRSA, PubKeyAlgoRSASignOnly, PubKeyAlgoDSA, PubKeyAlgoECDSA:
		return true
	}
	return false
}

// CipherFunction represents the different block ciphers specified for OpenPGP. See
// http://www.iana.org/assignments/pgp-parameters/pgp-parameters.xhtml#pgp-parameters-13
type CipherFunction uint8

const (
	Cipher3DES   CipherFunction = 2
	CipherCAST5  CipherFunction = 3
	CipherAES128 CipherFunction = 7
	CipherAES192 CipherFunction = 8
	CipherAES256 CipherFunction = 9
)

// KeySize returns the key size, in bytes, of cipher.
func (cipher CipherFunction) KeySize() int {
	switch cipher {
	case Cipher3DES:
		return 24
	case CipherCAST5:
		return cast5.KeySize
	case CipherAES128:
		return 16
	case CipherAES192:
		return 24
	case CipherAES256:
		return 32
	}
	return 0
}

// blockSize returns the block size, in bytes, of cipher.
func (cipher CipherFunction) blockSize() int {
	switch cipher {
	case Cipher3DES:
		return des.BlockSize
	case CipherCAST5:
		return 8
	case CipherAES128, CipherAES192, CipherAES256:
		return 16
	}
	return 0
}

// new returns a fresh instance of the given cipher.
func (cipher CipherFunction) new(key []byte) (block cipher.Block) {
	switch cipher {
	case Cipher3DES:
		block, _ = des.NewTripleDESCipher(key)
	case CipherCAST5:
		block, _ = cast5.NewCipher(key)
	case CipherAES128, CipherAES192, CipherAES256:
		block, _ = aes.NewCipher(key)
	}
	return
}

// readMPI reads a big integer from r. The bit length returned is the bit
// length that was specified in r. This is preserved so that the integer can be
// reserialized exactly.
func readMPI(r io.Reader) (mpi []byte, bitLength uint16, err error) {
	var buf [2]byte
	_, err = readFull(r, buf[0:])
	if err != nil {
		return
	}
	bitLength = uint16(buf[0])<<8 | uint16(buf[1])
	numBytes := (int(bitLength) + 7) / 8
	mpi = make([]byte, numBytes)
	_, err = readFull(r, mpi)
	// According to RFC 4880 3.2. we should check that the MPI has no leading
	// zeroes (at least when not an encrypted MPI?), but this implementation
	// does generate leading zeroes, so we keep accepting them.
	return
}

// writeMPI serializes a big integer to w.
func writeMPI(w io.Writer, bitLength uint16, mpiBytes []byte) (err error) {
	// Note that we can produce leading zeroes, in violation of RFC 4880 3.2.
	// Implementations seem to be tolerant of them, and stripping them would
	// make it complex to guarantee matching re-serialization.
	_, err = w.Write([]byte{byte(bitLength >> 8), byte(bitLength)})
	if err == nil {
		_, err = w.Write(mpiBytes)
	}
	return
}

// writeBig serializes a *big.Int to w.
func writeBig(w io.Writer, i *big.Int) error {
	return writeMPI(w, uint16(i.BitLen()), i.Bytes())
}

// padToKeySize left-pads a MPI with zeroes to match the length of the
// specified RSA public.
func padToKeySize(pub *rsa.PublicKey, b []byte) []byte {
	k := (pub.N.BitLen() + 7) / 8
	if len(b) >= k {
		return b
	}
	bb := make([]byte, k)
	copy(bb[len(bb)-len(b):], b)
	return bb
}

// CompressionAlgo Represents the different compression algorithms
// supported by OpenPGP (except for BZIP2, which is not currently
// supported). See Section 9.3 of RFC 4880.
type CompressionAlgo uint8

const (
	CompressionNone CompressionAlgo = 0
	CompressionZIP  CompressionAlgo = 1
	CompressionZLIB CompressionAlgo = 2
)
