// Package rc2 implements the RC2 cipher
/*
https://www.ietf.org/rfc/rfc2268.txt
http://people.csail.mit.edu/rivest/pubs/KRRR98.pdf

This code is licensed under the MIT license.
*/
package rc2

import (
	"crypto/cipher"
	"encoding/binary"
	"strconv"
)

// The rc2 block size in bytes
const BlockSize = 8

type rc2Cipher struct {
	k [64]uint16
}

// KeySizeError indicates the supplied key was invalid
type KeySizeError int

func (k KeySizeError) Error() string { return "rc2: invalid key size " + strconv.Itoa(int(k)) }

// EffectiveKeySizeError indicates the supplied effective key length was invalid
type EffectiveKeySizeError int

func (k EffectiveKeySizeError) Error() string {
	return "rc2: invalid effective key size " + strconv.Itoa(int(k))
}

// New returns a new rc2 cipher with the given key and effective key length t1
func New(key []byte, t1 int) (cipher.Block, error) {
	if l := len(key); l == 0 || l > 128 {
		return nil, KeySizeError(l)
	}

	if t1 < 8 || t1 > 1024 {
		return nil, EffectiveKeySizeError(t1)
	}

	return &rc2Cipher{
		k: expandKey(key, t1),
	}, nil
}

func (c *rc2Cipher) BlockSize() int { return BlockSize }

var piTable = [256]byte{
	0xd9, 0x78, 0xf9, 0xc4, 0x19, 0xdd, 0xb5, 0xed, 0x28, 0xe9, 0xfd, 0x79, 0x4a, 0xa0, 0xd8, 0x9d,
	0xc6, 0x7e, 0x37, 0x83, 0x2b, 0x76, 0x53, 0x8e, 0x62, 0x4c, 0x64, 0x88, 0x44, 0x8b, 0xfb, 0xa2,
	0x17, 0x9a, 0x59, 0xf5, 0x87, 0xb3, 0x4f, 0x13, 0x61, 0x45, 0x6d, 0x8d, 0x09, 0x81, 0x7d, 0x32,
	0xbd, 0x8f, 0x40, 0xeb, 0x86, 0xb7, 0x7b, 0x0b, 0xf0, 0x95, 0x21, 0x22, 0x5c, 0x6b, 0x4e, 0x82,
	0x54, 0xd6, 0x65, 0x93, 0xce, 0x60, 0xb2, 0x1c, 0x73, 0x56, 0xc0, 0x14, 0xa7, 0x8c, 0xf1, 0xdc,
	0x12, 0x75, 0xca, 0x1f, 0x3b, 0xbe, 0xe4, 0xd1, 0x42, 0x3d, 0xd4, 0x30, 0xa3, 0x3c, 0xb6, 0x26,
	0x6f, 0xbf, 0x0e, 0xda, 0x46, 0x69, 0x07, 0x57, 0x27, 0xf2, 0x1d, 0x9b, 0xbc, 0x94, 0x43, 0x03,
	0xf8, 0x11, 0xc7, 0xf6, 0x90, 0xef, 0x3e, 0xe7, 0x06, 0xc3, 0xd5, 0x2f, 0xc8, 0x66, 0x1e, 0xd7,
	0x08, 0xe8, 0xea, 0xde, 0x80, 0x52, 0xee, 0xf7, 0x84, 0xaa, 0x72, 0xac, 0x35, 0x4d, 0x6a, 0x2a,
	0x96, 0x1a, 0xd2, 0x71, 0x5a, 0x15, 0x49, 0x74, 0x4b, 0x9f, 0xd0, 0x5e, 0x04, 0x18, 0xa4, 0xec,
	0xc2, 0xe0, 0x41, 0x6e, 0x0f, 0x51, 0xcb, 0xcc, 0x24, 0x91, 0xaf, 0x50, 0xa1, 0xf4, 0x70, 0x39,
	0x99, 0x7c, 0x3a, 0x85, 0x23, 0xb8, 0xb4, 0x7a, 0xfc, 0x02, 0x36, 0x5b, 0x25, 0x55, 0x97, 0x31,
	0x2d, 0x5d, 0xfa, 0x98, 0xe3, 0x8a, 0x92, 0xae, 0x05, 0xdf, 0x29, 0x10, 0x67, 0x6c, 0xba, 0xc9,
	0xd3, 0x00, 0xe6, 0xcf, 0xe1, 0x9e, 0xa8, 0x2c, 0x63, 0x16, 0x01, 0x3f, 0x58, 0xe2, 0x89, 0xa9,
	0x0d, 0x38, 0x34, 0x1b, 0xab, 0x33, 0xff, 0xb0, 0xbb, 0x48, 0x0c, 0x5f, 0xb9, 0xb1, 0xcd, 0x2e,
	0xc5, 0xf3, 0xdb, 0x47, 0xe5, 0xa5, 0x9c, 0x77, 0x0a, 0xa6, 0x20, 0x68, 0xfe, 0x7f, 0xc1, 0xad,
}

func expandKey(key []byte, t1 int) [64]uint16 {

	l := make([]byte, 128)
	copy(l, key)

	var t = len(key)
	var t8 = (t1 + 7) / 8
	var tm = byte(255 % uint(1<<(8+uint(t1)-8*uint(t8))))

	for i := len(key); i < 128; i++ {
		l[i] = piTable[l[i-1]+l[uint8(i-t)]]
	}

	l[128-t8] = piTable[l[128-t8]&tm]

	for i := 127 - t8; i >= 0; i-- {
		l[i] = piTable[l[i+1]^l[i+t8]]
	}

	var k [64]uint16

	for i := range k {
		k[i] = uint16(l[2*i]) + uint16(l[2*i+1])*256
	}

	return k
}

func rotl16(x uint16, b uint) uint16 {
	return (x >> (16 - b)) | (x << b)
}

func (c *rc2Cipher) Encrypt(dst, src []byte) {

	r0 := binary.LittleEndian.Uint16(src[0:])
	r1 := binary.LittleEndian.Uint16(src[2:])
	r2 := binary.LittleEndian.Uint16(src[4:])
	r3 := binary.LittleEndian.Uint16(src[6:])

	var j int

	// These three mix blocks have not been extracted to a common function for to performance reasons.
	for j <= 16 {
		// mix r0
		r0 = r0 + c.k[j] + (r3 & r2) + ((^r3) & r1)
		r0 = rotl16(r0, 1)
		j++

		// mix r1
		r1 = r1 + c.k[j] + (r0 & r3) + ((^r0) & r2)
		r1 = rotl16(r1, 2)
		j++

		// mix r2
		r2 = r2 + c.k[j] + (r1 & r0) + ((^r1) & r3)
		r2 = rotl16(r2, 3)
		j++

		// mix r3
		r3 = r3 + c.k[j] + (r2 & r1) + ((^r2) & r0)
		r3 = rotl16(r3, 5)
		j++
	}

	r0 = r0 + c.k[r3&63]
	r1 = r1 + c.k[r0&63]
	r2 = r2 + c.k[r1&63]
	r3 = r3 + c.k[r2&63]

	for j <= 40 {
		// mix r0
		r0 = r0 + c.k[j] + (r3 & r2) + ((^r3) & r1)
		r0 = rotl16(r0, 1)
		j++

		// mix r1
		r1 = r1 + c.k[j] + (r0 & r3) + ((^r0) & r2)
		r1 = rotl16(r1, 2)
		j++

		// mix r2
		r2 = r2 + c.k[j] + (r1 & r0) + ((^r1) & r3)
		r2 = rotl16(r2, 3)
		j++

		// mix r3
		r3 = r3 + c.k[j] + (r2 & r1) + ((^r2) & r0)
		r3 = rotl16(r3, 5)
		j++
	}

	r0 = r0 + c.k[r3&63]
	r1 = r1 + c.k[r0&63]
	r2 = r2 + c.k[r1&63]
	r3 = r3 + c.k[r2&63]

	for j <= 60 {
		// mix r0
		r0 = r0 + c.k[j] + (r3 & r2) + ((^r3) & r1)
		r0 = rotl16(r0, 1)
		j++

		// mix r1
		r1 = r1 + c.k[j] + (r0 & r3) + ((^r0) & r2)
		r1 = rotl16(r1, 2)
		j++

		// mix r2
		r2 = r2 + c.k[j] + (r1 & r0) + ((^r1) & r3)
		r2 = rotl16(r2, 3)
		j++

		// mix r3
		r3 = r3 + c.k[j] + (r2 & r1) + ((^r2) & r0)
		r3 = rotl16(r3, 5)
		j++
	}

	binary.LittleEndian.PutUint16(dst[0:], r0)
	binary.LittleEndian.PutUint16(dst[2:], r1)
	binary.LittleEndian.PutUint16(dst[4:], r2)
	binary.LittleEndian.PutUint16(dst[6:], r3)
}

func (c *rc2Cipher) Decrypt(dst, src []byte) {

	r0 := binary.LittleEndian.Uint16(src[0:])
	r1 := binary.LittleEndian.Uint16(src[2:])
	r2 := binary.LittleEndian.Uint16(src[4:])
	r3 := binary.LittleEndian.Uint16(src[6:])

	j := 63

	for j >= 44 {
		// unmix r3
		r3 = rotl16(r3, 16-5)
		r3 = r3 - c.k[j] - (r2 & r1) - ((^r2) & r0)
		j--

		// unmix r2
		r2 = rotl16(r2, 16-3)
		r2 = r2 - c.k[j] - (r1 & r0) - ((^r1) & r3)
		j--

		// unmix r1
		r1 = rotl16(r1, 16-2)
		r1 = r1 - c.k[j] - (r0 & r3) - ((^r0) & r2)
		j--

		// unmix r0
		r0 = rotl16(r0, 16-1)
		r0 = r0 - c.k[j] - (r3 & r2) - ((^r3) & r1)
		j--
	}

	r3 = r3 - c.k[r2&63]
	r2 = r2 - c.k[r1&63]
	r1 = r1 - c.k[r0&63]
	r0 = r0 - c.k[r3&63]

	for j >= 20 {
		// unmix r3
		r3 = rotl16(r3, 16-5)
		r3 = r3 - c.k[j] - (r2 & r1) - ((^r2) & r0)
		j--

		// unmix r2
		r2 = rotl16(r2, 16-3)
		r2 = r2 - c.k[j] - (r1 & r0) - ((^r1) & r3)
		j--

		// unmix r1
		r1 = rotl16(r1, 16-2)
		r1 = r1 - c.k[j] - (r0 & r3) - ((^r0) & r2)
		j--

		// unmix r0
		r0 = rotl16(r0, 16-1)
		r0 = r0 - c.k[j] - (r3 & r2) - ((^r3) & r1)
		j--
	}

	r3 = r3 - c.k[r2&63]
	r2 = r2 - c.k[r1&63]
	r1 = r1 - c.k[r0&63]
	r0 = r0 - c.k[r3&63]

	for j >= 0 {
		// unmix r3
		r3 = rotl16(r3, 16-5)
		r3 = r3 - c.k[j] - (r2 & r1) - ((^r2) & r0)
		j--

		// unmix r2
		r2 = rotl16(r2, 16-3)
		r2 = r2 - c.k[j] - (r1 & r0) - ((^r1) & r3)
		j--

		// unmix r1
		r1 = rotl16(r1, 16-2)
		r1 = r1 - c.k[j] - (r0 & r3) - ((^r0) & r2)
		j--

		// unmix r0
		r0 = rotl16(r0, 16-1)
		r0 = r0 - c.k[j] - (r3 & r2) - ((^r3) & r1)
		j--
	}

	binary.LittleEndian.PutUint16(dst[0:], r0)
	binary.LittleEndian.PutUint16(dst[2:], r1)
	binary.LittleEndian.PutUint16(dst[4:], r2)
	binary.LittleEndian.PutUint16(dst[6:], r3)
}
