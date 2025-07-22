package stringbuffer

import (
	"sync"
	"unicode/utf16"
)

// TODO: worth exporting and using in mkwinsyscall?

// Uint16BufferSize is the buffer size in the pool, chosen somewhat arbitrarily to accommodate
// large path strings:
// MAX_PATH (260) + size of volume GUID prefix (49) + null terminator = 310.
const MinWStringCap = 310

// use *[]uint16 since []uint16 creates an extra allocation where the slice header
// is copied to heap and then referenced via pointer in the interface header that sync.Pool
// stores.
var pathPool = sync.Pool{ // if go1.18+ adds Pool[T], use that to store []uint16 directly
	New: func() interface{} {
		b := make([]uint16, MinWStringCap)
		return &b
	},
}

func newBuffer() []uint16 { return *(pathPool.Get().(*[]uint16)) }

// freeBuffer copies the slice header data, and puts a pointer to that in the pool.
// This avoids taking a pointer to the slice header in WString, which can be set to nil.
func freeBuffer(b []uint16) { pathPool.Put(&b) }

// WString is a wide string buffer ([]uint16) meant for storing UTF-16 encoded strings
// for interacting with Win32 APIs.
// Sizes are specified as uint32 and not int.
//
// It is not thread safe.
type WString struct {
	// type-def allows casting to []uint16 directly, use struct to prevent that and allow adding fields in the future.

	// raw buffer
	b []uint16
}

// NewWString returns a [WString] allocated from a shared pool with an
// initial capacity of at least [MinWStringCap].
// Since the buffer may have been previously used, its contents are not guaranteed to be empty.
//
// The buffer should be freed via [WString.Free]
func NewWString() *WString {
	return &WString{
		b: newBuffer(),
	}
}

func (b *WString) Free() {
	if b.empty() {
		return
	}
	freeBuffer(b.b)
	b.b = nil
}

// ResizeTo grows the buffer to at least c and returns the new capacity, freeing the
// previous buffer back into pool.
func (b *WString) ResizeTo(c uint32) uint32 {
	// already sufficient (or n is 0)
	if c <= b.Cap() {
		return b.Cap()
	}

	if c <= MinWStringCap {
		c = MinWStringCap
	}
	// allocate at-least double buffer size, as is done in [bytes.Buffer] and other places
	if c <= 2*b.Cap() {
		c = 2 * b.Cap()
	}

	b2 := make([]uint16, c)
	if !b.empty() {
		copy(b2, b.b)
		freeBuffer(b.b)
	}
	b.b = b2
	return c
}

// Buffer returns the underlying []uint16 buffer.
func (b *WString) Buffer() []uint16 {
	if b.empty() {
		return nil
	}
	return b.b
}

// Pointer returns a pointer to the first uint16 in the buffer.
// If the [WString.Free] has already been called, the pointer will be nil.
func (b *WString) Pointer() *uint16 {
	if b.empty() {
		return nil
	}
	return &b.b[0]
}

// String returns the returns the UTF-8 encoding of the UTF-16 string in the buffer.
//
// It assumes that the data is null-terminated.
func (b *WString) String() string {
	// Using [windows.UTF16ToString] would require importing "golang.org/x/sys/windows"
	// and would make this code Windows-only, which makes no sense.
	// So copy UTF16ToString code into here.
	// If other windows-specific code is added, switch to [windows.UTF16ToString]

	s := b.b
	for i, v := range s {
		if v == 0 {
			s = s[:i]
			break
		}
	}
	return string(utf16.Decode(s))
}

// Cap returns the underlying buffer capacity.
func (b *WString) Cap() uint32 {
	if b.empty() {
		return 0
	}
	return b.cap()
}

func (b *WString) cap() uint32 { return uint32(cap(b.b)) }
func (b *WString) empty() bool { return b == nil || b.cap() == 0 }
