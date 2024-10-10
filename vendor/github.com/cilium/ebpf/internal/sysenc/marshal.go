package sysenc

import (
	"bytes"
	"encoding"
	"encoding/binary"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"sync"
	"unsafe"

	"github.com/cilium/ebpf/internal"
)

// Marshal turns data into a byte slice using the system's native endianness.
//
// If possible, avoids allocations by directly using the backing memory
// of data. This means that the variable must not be modified for the lifetime
// of the returned [Buffer].
//
// Returns an error if the data can't be turned into a byte slice according to
// the behaviour of [binary.Write].
func Marshal(data any, size int) (Buffer, error) {
	if data == nil {
		return Buffer{}, errors.New("can't marshal a nil value")
	}

	var buf []byte
	var err error
	switch value := data.(type) {
	case encoding.BinaryMarshaler:
		buf, err = value.MarshalBinary()
	case string:
		buf = unsafe.Slice(unsafe.StringData(value), len(value))
	case []byte:
		buf = value
	case int16:
		buf = internal.NativeEndian.AppendUint16(make([]byte, 0, 2), uint16(value))
	case uint16:
		buf = internal.NativeEndian.AppendUint16(make([]byte, 0, 2), value)
	case int32:
		buf = internal.NativeEndian.AppendUint32(make([]byte, 0, 4), uint32(value))
	case uint32:
		buf = internal.NativeEndian.AppendUint32(make([]byte, 0, 4), value)
	case int64:
		buf = internal.NativeEndian.AppendUint64(make([]byte, 0, 8), uint64(value))
	case uint64:
		buf = internal.NativeEndian.AppendUint64(make([]byte, 0, 8), value)
	default:
		if buf := unsafeBackingMemory(data); len(buf) == size {
			return newBuffer(buf), nil
		}

		wr := internal.NewBuffer(make([]byte, 0, size))
		defer internal.PutBuffer(wr)

		err = binary.Write(wr, internal.NativeEndian, value)
		buf = wr.Bytes()
	}
	if err != nil {
		return Buffer{}, err
	}

	if len(buf) != size {
		return Buffer{}, fmt.Errorf("%T doesn't marshal to %d bytes", data, size)
	}

	return newBuffer(buf), nil
}

var bytesReaderPool = sync.Pool{
	New: func() interface{} {
		return new(bytes.Reader)
	},
}

// Unmarshal a byte slice in the system's native endianness into data.
//
// Returns an error if buf can't be unmarshalled according to the behaviour
// of [binary.Read].
func Unmarshal(data interface{}, buf []byte) error {
	switch value := data.(type) {
	case encoding.BinaryUnmarshaler:
		return value.UnmarshalBinary(buf)

	case *string:
		*value = string(buf)
		return nil

	case *[]byte:
		// Backwards compat: unmarshaling into a slice replaces the whole slice.
		*value = slices.Clone(buf)
		return nil

	default:
		if dataBuf := unsafeBackingMemory(data); len(dataBuf) == len(buf) {
			copy(dataBuf, buf)
			return nil
		}

		rd := bytesReaderPool.Get().(*bytes.Reader)
		defer bytesReaderPool.Put(rd)

		rd.Reset(buf)

		if err := binary.Read(rd, internal.NativeEndian, value); err != nil {
			return err
		}

		if rd.Len() != 0 {
			return fmt.Errorf("unmarshaling %T doesn't consume all data", data)
		}

		return nil
	}
}

// unsafeBackingMemory returns the backing memory of data if it can be used
// instead of calling into package binary.
//
// Returns nil if the value is not a pointer or a slice, or if it contains
// padding or unexported fields.
func unsafeBackingMemory(data any) []byte {
	if data == nil {
		return nil
	}

	value := reflect.ValueOf(data)
	var valueSize int
	switch value.Kind() {
	case reflect.Pointer:
		if value.IsNil() {
			return nil
		}

		if elemType := value.Type().Elem(); elemType.Kind() != reflect.Slice {
			valueSize = int(elemType.Size())
			break
		}

		// We're dealing with a pointer to a slice. Dereference and
		// handle it like a regular slice.
		value = value.Elem()
		fallthrough

	case reflect.Slice:
		valueSize = int(value.Type().Elem().Size()) * value.Len()

	default:
		// Prevent Value.UnsafePointer from panicking.
		return nil
	}

	// Some nil pointer types currently crash binary.Size. Call it after our own
	// code so that the panic isn't reachable.
	// See https://github.com/golang/go/issues/60892
	if size := binary.Size(data); size == -1 || size != valueSize {
		// The type contains padding or unsupported types.
		return nil
	}

	if hasUnexportedFields(reflect.TypeOf(data)) {
		return nil
	}

	// Reinterpret the pointer as a byte slice. This violates the unsafe.Pointer
	// rules because it's very unlikely that the source data has "an equivalent
	// memory layout". However, we can make it safe-ish because of the
	// following reasons:
	//  - There is no alignment mismatch since we cast to a type with an
	//    alignment of 1.
	//  - There are no pointers in the source type so we don't upset the GC.
	//  - The length is verified at runtime.
	return unsafe.Slice((*byte)(value.UnsafePointer()), valueSize)
}
