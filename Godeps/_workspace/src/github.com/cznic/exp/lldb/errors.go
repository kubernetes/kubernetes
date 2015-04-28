// Copyright 2014 The lldb Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Some errors returned by this package.
//
// Note that this package can return more errors than declared here, for
// example io.EOF from Filer.ReadAt().

package lldb

import (
	"fmt"
)

// ErrDecodeScalars is possibly returned from DecodeScalars
type ErrDecodeScalars struct {
	B []byte // Data being decoded
	I int    // offending offset
}

// Error implements the built in error type.
func (e *ErrDecodeScalars) Error() string {
	return fmt.Sprintf("DecodeScalars: corrupted data @ %d/%d", e.I, len(e.B))
}

// ErrINVAL reports invalid values passed as parameters, for example negative
// offsets where only non-negative ones are allowed or read from the DB.
type ErrINVAL struct {
	Src string
	Val interface{}
}

// Error implements the built in error type.
func (e *ErrINVAL) Error() string {
	return fmt.Sprintf("%s: %+v", e.Src, e.Val)
}

// ErrPERM is for example reported when a Filer is closed while BeginUpdate(s)
// are not balanced with EndUpdate(s)/Rollback(s) or when EndUpdate or Rollback
// is invoked which is not paired with a BeginUpdate.
type ErrPERM struct {
	Src string
}

// Error implements the built in error type.
func (e *ErrPERM) Error() string {
	return fmt.Sprintf("%s: Operation not permitted", string(e.Src))
}

// ErrTag represents an ErrILSEQ kind.
type ErrType int

// ErrILSEQ types
const (
	ErrOther ErrType = iota

	ErrAdjacentFree          // Adjacent free blocks (.Off and .Arg)
	ErrDecompress            // Used compressed block: corrupted compression
	ErrExpFreeTag            // Expected a free block tag, got .Arg
	ErrExpUsedTag            // Expected a used block tag, got .Arg
	ErrFLT                   // Free block is invalid or referenced multiple times
	ErrFLTLoad               // FLT truncated to .Off, need size >= .Arg
	ErrFLTSize               // Free block size (.Arg) doesn't belong to its list min size: .Arg2
	ErrFileSize              // File .Name size (.Arg) != 0 (mod 16)
	ErrFreeChaining          // Free block, .prev.next doesn't point back to this block
	ErrFreeTailBlock         // Last block is free
	ErrHead                  // Head of a free block list has non zero Prev (.Arg)
	ErrInvalidRelocTarget    // Reloc doesn't target (.Arg) a short or long used block
	ErrInvalidWAL            // Corrupted write ahead log. .Name: file name, .More: more
	ErrLongFreeBlkTooLong    // Long free block spans beyond EOF, size .Arg
	ErrLongFreeBlkTooShort   // Long free block must have at least 2 atoms, got only .Arg
	ErrLongFreeNextBeyondEOF // Long free block .Next (.Arg) spans beyond EOF
	ErrLongFreePrevBeyondEOF // Long free block .Prev (.Arg) spans beyond EOF
	ErrLongFreeTailTag       // Expected a long free block tail tag, got .Arg
	ErrLostFreeBlock         // Free block is not in any FLT list
	ErrNullReloc             // Used reloc block with nil target
	ErrRelocBeyondEOF        // Used reloc points (.Arg) beyond EOF
	ErrShortFreeTailTag      // Expected a short free block tail tag, got .Arg
	ErrSmall                 // Request for a free block (.Arg) returned a too small one (.Arg2) at .Off
	ErrTailTag               // Block at .Off has invalid tail CC (compression code) tag, got .Arg
	ErrUnexpReloc            // Unexpected reloc block referred to from reloc block .Arg
	ErrVerifyPadding         // Used block has nonzero padding
	ErrVerifyTailSize        // Long free block size .Arg but tail size .Arg2
	ErrVerifyUsedSpan        // Used block size (.Arg) spans beyond EOF
)

// ErrILSEQ reports a corrupted file format. Details in fields according to Type.
type ErrILSEQ struct {
	Type ErrType
	Off  int64
	Arg  int64
	Arg2 int64
	Arg3 int64
	Name string
	More interface{}
}

// Error implements the built in error type.
func (e *ErrILSEQ) Error() string {
	switch e.Type {
	case ErrAdjacentFree:
		return fmt.Sprintf("Adjacent free blocks at offset %#x and %#x", e.Off, e.Arg)
	case ErrDecompress:
		return fmt.Sprintf("Compressed block at offset %#x: Corrupted compressed content", e.Off)
	case ErrExpFreeTag:
		return fmt.Sprintf("Block at offset %#x: Expected a free block tag, got %#2x", e.Off, e.Arg)
	case ErrExpUsedTag:
		return fmt.Sprintf("Block at ofset %#x: Expected a used block tag, got %#2x", e.Off, e.Arg)
	case ErrFLT:
		return fmt.Sprintf("Free block at offset %#x is invalid or referenced multiple times", e.Off)
	case ErrFLTLoad:
		return fmt.Sprintf("FLT truncated to size %d, expected at least %d", e.Off, e.Arg)
	case ErrFLTSize:
		return fmt.Sprintf("Free block at offset %#x has size (%#x) should be at least (%#x)", e.Off, e.Arg, e.Arg2)
	case ErrFileSize:
		return fmt.Sprintf("File %q size (%#x) != 0 (mod 16)", e.Name, e.Arg)
	case ErrFreeChaining:
		return fmt.Sprintf("Free block at offset %#x: .prev.next doesn point back here.", e.Off)
	case ErrFreeTailBlock:
		return fmt.Sprintf("Free block at offset %#x: Cannot be last file block", e.Off)
	case ErrHead:
		return fmt.Sprintf("Block at offset %#x: Head of free block list has non zero .prev %#x", e.Off, e.Arg)
	case ErrInvalidRelocTarget:
		return fmt.Sprintf("Used reloc block at offset %#x: Target (%#x) is not a short or long used block", e.Off, e.Arg)
	case ErrInvalidWAL:
		return fmt.Sprintf("Corrupted write ahead log file: %q %v", e.Name, e.More)
	case ErrLongFreeBlkTooLong:
		return fmt.Sprintf("Long free block at offset %#x: Size (%#x) beyond EOF", e.Off, e.Arg)
	case ErrLongFreeBlkTooShort:
		return fmt.Sprintf("Long free block at offset %#x: Size (%#x) too small", e.Off, e.Arg)
	case ErrLongFreeNextBeyondEOF:
		return fmt.Sprintf("Long free block at offset %#x: Next (%#x) points beyond EOF", e.Off, e.Arg)
	case ErrLongFreePrevBeyondEOF:
		return fmt.Sprintf("Long free block at offset %#x: Prev (%#x) points beyond EOF", e.Off, e.Arg)
	case ErrLongFreeTailTag:
		return fmt.Sprintf("Block at offset %#x: Expected long free tail tag, got %#2x", e.Off, e.Arg)
	case ErrLostFreeBlock:
		return fmt.Sprintf("Free block at offset %#x: not in any FLT list", e.Off)
	case ErrNullReloc:
		return fmt.Sprintf("Used reloc block at offset %#x: Nil target", e.Off)
	case ErrRelocBeyondEOF:
		return fmt.Sprintf("Used reloc block at offset %#x: Link (%#x) points beyond EOF", e.Off, e.Arg)
	case ErrShortFreeTailTag:
		return fmt.Sprintf("Block at offset %#x: Expected short free tail tag, got %#2x", e.Off, e.Arg)
	case ErrSmall:
		return fmt.Sprintf("Request for of free block of size %d returned a too small (%d) one at offset %#x", e.Arg, e.Arg2, e.Off)
	case ErrTailTag:
		return fmt.Sprintf("Block at offset %#x: Invalid tail CC tag, got %#2x", e.Off, e.Arg)
	case ErrUnexpReloc:
		return fmt.Sprintf("Block at offset %#x: Unexpected reloc block. Referred to from reloc block at offset %#x", e.Off, e.Arg)
	case ErrVerifyPadding:
		return fmt.Sprintf("Used block at offset %#x: Nonzero padding", e.Off)
	case ErrVerifyTailSize:
		return fmt.Sprintf("Long free block at offset %#x: Size %#x, but tail size %#x", e.Off, e.Arg, e.Arg2)
	case ErrVerifyUsedSpan:
		return fmt.Sprintf("Used block at offset %#x: Size %#x spans beyond EOF", e.Off, e.Arg)
	}

	more := ""
	if e.More != nil {
		more = fmt.Sprintf(", %v", e.More)
	}
	off := ""
	if e.Off != 0 {
		off = fmt.Sprintf(", off: %#x", e.Off)
	}

	return fmt.Sprintf("Error%s%s", off, more)
}
