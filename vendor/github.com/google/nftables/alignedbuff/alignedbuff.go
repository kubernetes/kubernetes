// Package alignedbuff implements encoding and decoding aligned data elements
// to/from buffers in native endianess.
//
// # Note
//
// The alignment/padding as implemented in this package must match that of
// kernel's and user space C implementations for a particular architecture (bit
// size). Please see also the "dummy structure" _xt_align
// (https://elixir.bootlin.com/linux/v5.17.7/source/include/uapi/linux/netfilter/x_tables.h#L93)
// as well as the associated XT_ALIGN C preprocessor macro.
//
// In particular, we rely on the Go compiler to follow the same architecture
// alignments as the C compiler(s) on Linux.
package alignedbuff

import (
	"bytes"
	"errors"
	"fmt"
	"unsafe"

	"github.com/google/nftables/binaryutil"
)

// ErrEOF signals trying to read beyond the available payload information.
var ErrEOF = errors.New("not enough data left")

// AlignedBuff implements marshalling and unmarshalling information in
// platform/architecture native endianess and data type alignment. It
// additionally covers some of the nftables-xtables translation-specific
// idiosyncracies to the extend needed in order to properly marshal and
// unmarshal Match and Target expressions, and their Info payload in particular.
type AlignedBuff struct {
	data []byte
	pos  int
}

// New returns a new AlignedBuff for marshalling aligned data in native
// endianess.
func New() AlignedBuff {
	return AlignedBuff{}
}

// NewWithData returns a new AlignedBuff for unmarshalling the passed data in
// native endianess.
func NewWithData(data []byte) AlignedBuff {
	return AlignedBuff{data: data}
}

// Data returns the properly padded info payload data written before by calling
// the various Uint8, Uint16, ... marshalling functions.
func (a *AlignedBuff) Data() []byte {
	// The Linux kernel expects payloads to be padded to the next uint64
	// alignment.
	a.alignWrite(uint64AlignMask)
	return a.data
}

// BytesAligned32 unmarshals the given amount of bytes starting with the native
// alignment for uint32 data types. It returns ErrEOF when trying to read beyond
// the payload.
//
// BytesAligned32 is used to unmarshal IP addresses for different IP versions,
// which are always aligned the same way as the native alignment for uint32.
func (a *AlignedBuff) BytesAligned32(size int) ([]byte, error) {
	if err := a.alignCheckedRead(uint32AlignMask); err != nil {
		return nil, err
	}
	if a.pos > len(a.data)-size {
		return nil, ErrEOF
	}
	data := a.data[a.pos : a.pos+size]
	a.pos += size
	return data, nil
}

// Uint8 unmarshals an uint8 in native endianess and alignment. It returns
// ErrEOF when trying to read beyond the payload.
func (a *AlignedBuff) Uint8() (uint8, error) {
	if a.pos >= len(a.data) {
		return 0, ErrEOF
	}
	v := a.data[a.pos]
	a.pos++
	return v, nil
}

// Uint16 unmarshals an uint16 in native endianess and alignment. It returns
// ErrEOF when trying to read beyond the payload.
func (a *AlignedBuff) Uint16() (uint16, error) {
	if err := a.alignCheckedRead(uint16AlignMask); err != nil {
		return 0, err
	}
	v := binaryutil.NativeEndian.Uint16(a.data[a.pos : a.pos+2])
	a.pos += 2
	return v, nil
}

// Uint16BE unmarshals an uint16 in "network" (=big endian) endianess and native
// uint16 alignment. It returns ErrEOF when trying to read beyond the payload.
func (a *AlignedBuff) Uint16BE() (uint16, error) {
	if err := a.alignCheckedRead(uint16AlignMask); err != nil {
		return 0, err
	}
	v := binaryutil.BigEndian.Uint16(a.data[a.pos : a.pos+2])
	a.pos += 2
	return v, nil
}

// Uint32 unmarshals an uint32 in native endianess and alignment. It returns
// ErrEOF when trying to read beyond the payload.
func (a *AlignedBuff) Uint32() (uint32, error) {
	if err := a.alignCheckedRead(uint32AlignMask); err != nil {
		return 0, err
	}
	v := binaryutil.NativeEndian.Uint32(a.data[a.pos : a.pos+4])
	a.pos += 4
	return v, nil
}

// Uint64 unmarshals an uint64 in native endianess and alignment. It returns
// ErrEOF when trying to read beyond the payload.
func (a *AlignedBuff) Uint64() (uint64, error) {
	if err := a.alignCheckedRead(uint64AlignMask); err != nil {
		return 0, err
	}
	v := binaryutil.NativeEndian.Uint64(a.data[a.pos : a.pos+8])
	a.pos += 8
	return v, nil
}

// Int32 unmarshals an int32 in native endianess and alignment. It returns
// ErrEOF when trying to read beyond the payload.
func (a *AlignedBuff) Int32() (int32, error) {
	if err := a.alignCheckedRead(int32AlignMask); err != nil {
		return 0, err
	}
	v := binaryutil.Int32(a.data[a.pos : a.pos+4])
	a.pos += 4
	return v, nil
}

// String unmarshals a null terminated string
func (a *AlignedBuff) String() (string, error) {
	len := 0
	for {
		if a.data[a.pos+len] == 0x00 {
			break
		}
		len++
	}

	v := binaryutil.String(a.data[a.pos : a.pos+len])
	a.pos += len
	return v, nil
}

// StringWithLength unmarshals a string of a given length (for non-null
// terminated strings)
func (a *AlignedBuff) StringWithLength(len int) (string, error) {
	v := binaryutil.String(a.data[a.pos : a.pos+len])
	a.pos += len
	return v, nil
}

// Uint unmarshals an uint in native endianess and alignment for the C "unsigned
// int" type. It returns ErrEOF when trying to read beyond the payload. Please
// note that on 64bit platforms, the size and alignment of C's and Go's unsigned
// integer data types differ, so we encapsulate this difference here.
func (a *AlignedBuff) Uint() (uint, error) {
	switch uintSize {
	case 2:
		v, err := a.Uint16()
		return uint(v), err
	case 4:
		v, err := a.Uint32()
		return uint(v), err
	case 8:
		v, err := a.Uint64()
		return uint(v), err
	default:
		panic(fmt.Sprintf("unsupported uint size %d", uintSize))
	}
}

// PutBytesAligned32 marshals the given bytes starting with the native alignment
// for uint32 data types. It additionaly adds padding to reach the specified
// size.
//
// PutBytesAligned32 is used to marshal IP addresses for different IP versions,
// which are always aligned the same way as the native alignment for uint32.
func (a *AlignedBuff) PutBytesAligned32(data []byte, size int) {
	a.alignWrite(uint32AlignMask)
	a.data = append(a.data, data...)
	a.pos += len(data)
	if len(data) < size {
		padding := size - len(data)
		a.data = append(a.data, bytes.Repeat([]byte{0}, padding)...)
		a.pos += padding
	}
}

// PutUint8 marshals an uint8 in native endianess and alignment.
func (a *AlignedBuff) PutUint8(v uint8) {
	a.data = append(a.data, v)
	a.pos++
}

// PutUint16 marshals an uint16 in native endianess and alignment.
func (a *AlignedBuff) PutUint16(v uint16) {
	a.alignWrite(uint16AlignMask)
	a.data = append(a.data, binaryutil.NativeEndian.PutUint16(v)...)
	a.pos += 2
}

// PutUint16BE marshals an uint16 in "network" (=big endian) endianess and
// native uint16 alignment.
func (a *AlignedBuff) PutUint16BE(v uint16) {
	a.alignWrite(uint16AlignMask)
	a.data = append(a.data, binaryutil.BigEndian.PutUint16(v)...)
	a.pos += 2
}

// PutUint32 marshals an uint32 in native endianess and alignment.
func (a *AlignedBuff) PutUint32(v uint32) {
	a.alignWrite(uint32AlignMask)
	a.data = append(a.data, binaryutil.NativeEndian.PutUint32(v)...)
	a.pos += 4
}

// PutUint64 marshals an uint64 in native endianess and alignment.
func (a *AlignedBuff) PutUint64(v uint64) {
	a.alignWrite(uint64AlignMask)
	a.data = append(a.data, binaryutil.NativeEndian.PutUint64(v)...)
	a.pos += 8
}

// PutInt32 marshals an int32 in native endianess and alignment.
func (a *AlignedBuff) PutInt32(v int32) {
	a.alignWrite(int32AlignMask)
	a.data = append(a.data, binaryutil.PutInt32(v)...)
	a.pos += 4
}

// PutString marshals a string.
func (a *AlignedBuff) PutString(v string) {
	a.data = append(a.data, binaryutil.PutString(v)...)
	a.pos += len(v)
}

// PutUint marshals an uint in native endianess and alignment for the C
// "unsigned int" type. Please note that on 64bit platforms, the size and
// alignment of C's and Go's unsigned integer data types differ, so we
// encapsulate this difference here.
func (a *AlignedBuff) PutUint(v uint) {
	switch uintSize {
	case 2:
		a.PutUint16(uint16(v))
	case 4:
		a.PutUint32(uint32(v))
	case 8:
		a.PutUint64(uint64(v))
	default:
		panic(fmt.Sprintf("unsupported uint size %d", uintSize))
	}
}

// alignCheckedRead aligns the (read) position if necessary and suitable
// according to the specified alignment mask. alignCheckedRead returns an error
// if after any necessary alignment there isn't enough data left to be read into
// a value of the size corresponding to the specified alignment mask.
func (a *AlignedBuff) alignCheckedRead(m int) error {
	a.pos = (a.pos + m) & ^m
	if a.pos > len(a.data)-(m+1) {
		return ErrEOF
	}
	return nil
}

// alignWrite aligns the (write) position if necessary and suitable according to
// the specified alignment mask. It doubles as final payload padding helpmate in
// order to keep the kernel happy.
func (a *AlignedBuff) alignWrite(m int) {
	pos := (a.pos + m) & ^m
	if pos != a.pos {
		a.data = append(a.data, padding[:pos-a.pos]...)
		a.pos = pos
	}
}

// This is ... ugly.
var uint16AlignMask = int(unsafe.Alignof(uint16(0)) - 1)
var uint32AlignMask = int(unsafe.Alignof(uint32(0)) - 1)
var uint64AlignMask = int(unsafe.Alignof(uint64(0)) - 1)
var padding = bytes.Repeat([]byte{0}, uint64AlignMask)

var int32AlignMask = int(unsafe.Alignof(int32(0)) - 1)

// And this even worse.
var uintSize = unsafe.Sizeof(uint32(0))
