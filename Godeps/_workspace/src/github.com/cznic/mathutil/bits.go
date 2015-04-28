// Copyright (c) 2014 The mathutil Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mathutil

import (
	"math/big"
)

// BitLenByte returns the bit width of the non zero part of n.
func BitLenByte(n byte) int {
	return log2[n] + 1
}

// BitLenUint16 returns the bit width of the non zero part of n.
func BitLenUint16(n uint16) int {
	if b := n >> 8; b != 0 {
		return log2[b] + 8 + 1
	}

	return log2[n] + 1
}

// BitLenUint32 returns the bit width of the non zero part of n.
func BitLenUint32(n uint32) int {
	if b := n >> 24; b != 0 {
		return log2[b] + 24 + 1
	}

	if b := n >> 16; b != 0 {
		return log2[b] + 16 + 1
	}

	if b := n >> 8; b != 0 {
		return log2[b] + 8 + 1
	}

	return log2[n] + 1
}

// BitLen returns the bit width of the non zero part of n.
func BitLen(n int) int { // Should handle correctly [future] 64 bit Go ints
	if IntBits == 64 {
		return BitLenUint64(uint64(n))
	}

	if b := byte(n >> 24); b != 0 {
		return log2[b] + 24 + 1
	}

	if b := byte(n >> 16); b != 0 {
		return log2[b] + 16 + 1
	}

	if b := byte(n >> 8); b != 0 {
		return log2[b] + 8 + 1
	}

	return log2[byte(n)] + 1
}

// BitLenUint returns the bit width of the non zero part of n.
func BitLenUint(n uint) int { // Should handle correctly [future] 64 bit Go uints
	if IntBits == 64 {
		return BitLenUint64(uint64(n))
	}

	if b := n >> 24; b != 0 {
		return log2[b] + 24 + 1
	}

	if b := n >> 16; b != 0 {
		return log2[b] + 16 + 1
	}

	if b := n >> 8; b != 0 {
		return log2[b] + 8 + 1
	}

	return log2[n] + 1
}

// BitLenUint64 returns the bit width of the non zero part of n.
func BitLenUint64(n uint64) int {
	if b := n >> 56; b != 0 {
		return log2[b] + 56 + 1
	}

	if b := n >> 48; b != 0 {
		return log2[b] + 48 + 1
	}

	if b := n >> 40; b != 0 {
		return log2[b] + 40 + 1
	}

	if b := n >> 32; b != 0 {
		return log2[b] + 32 + 1
	}

	if b := n >> 24; b != 0 {
		return log2[b] + 24 + 1
	}

	if b := n >> 16; b != 0 {
		return log2[b] + 16 + 1
	}

	if b := n >> 8; b != 0 {
		return log2[b] + 8 + 1
	}

	return log2[n] + 1
}

// BitLenUintptr returns the bit width of the non zero part of n.
func BitLenUintptr(n uintptr) int {
	if b := n >> 56; b != 0 {
		return log2[b] + 56 + 1
	}

	if b := n >> 48; b != 0 {
		return log2[b] + 48 + 1
	}

	if b := n >> 40; b != 0 {
		return log2[b] + 40 + 1
	}

	if b := n >> 32; b != 0 {
		return log2[b] + 32 + 1
	}

	if b := n >> 24; b != 0 {
		return log2[b] + 24 + 1
	}

	if b := n >> 16; b != 0 {
		return log2[b] + 16 + 1
	}

	if b := n >> 8; b != 0 {
		return log2[b] + 8 + 1
	}

	return log2[n] + 1
}

// PopCountByte returns population count of n (number of bits set in n).
func PopCountByte(n byte) int {
	return int(popcnt[byte(n)])
}

// PopCountUint16 returns population count of n (number of bits set in n).
func PopCountUint16(n uint16) int {
	return int(popcnt[byte(n>>8)]) + int(popcnt[byte(n)])
}

// PopCountUint32 returns population count of n (number of bits set in n).
func PopCountUint32(n uint32) int {
	return int(popcnt[byte(n>>24)]) + int(popcnt[byte(n>>16)]) +
		int(popcnt[byte(n>>8)]) + int(popcnt[byte(n)])
}

// PopCount returns population count of n (number of bits set in n).
func PopCount(n int) int { // Should handle correctly [future] 64 bit Go ints
	if IntBits == 64 {
		return PopCountUint64(uint64(n))
	}

	return PopCountUint32(uint32(n))
}

// PopCountUint returns population count of n (number of bits set in n).
func PopCountUint(n uint) int { // Should handle correctly [future] 64 bit Go uints
	if IntBits == 64 {
		return PopCountUint64(uint64(n))
	}

	return PopCountUint32(uint32(n))
}

// PopCountUintptr returns population count of n (number of bits set in n).
func PopCountUintptr(n uintptr) int {
	if UintPtrBits == 64 {
		return PopCountUint64(uint64(n))
	}

	return PopCountUint32(uint32(n))
}

// PopCountUint64 returns population count of n (number of bits set in n).
func PopCountUint64(n uint64) int {
	return int(popcnt[byte(n>>56)]) + int(popcnt[byte(n>>48)]) +
		int(popcnt[byte(n>>40)]) + int(popcnt[byte(n>>32)]) +
		int(popcnt[byte(n>>24)]) + int(popcnt[byte(n>>16)]) +
		int(popcnt[byte(n>>8)]) + int(popcnt[byte(n)])
}

// PopCountBigInt returns population count of |n| (number of bits set in |n|).
func PopCountBigInt(n *big.Int) (r int) {
	for _, v := range n.Bits() {
		r += PopCountUintptr(uintptr(v))
	}
	return
}
