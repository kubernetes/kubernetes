package ebpf

import (
	"bytes"
	"encoding"
	"encoding/binary"
	"errors"
	"fmt"
	"reflect"
	"runtime"
	"sync"
	"unsafe"

	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/sys"
)

// marshalPtr converts an arbitrary value into a pointer suitable
// to be passed to the kernel.
//
// As an optimization, it returns the original value if it is an
// unsafe.Pointer.
func marshalPtr(data interface{}, length int) (sys.Pointer, error) {
	if ptr, ok := data.(unsafe.Pointer); ok {
		return sys.NewPointer(ptr), nil
	}

	buf, err := marshalBytes(data, length)
	if err != nil {
		return sys.Pointer{}, err
	}

	return sys.NewSlicePointer(buf), nil
}

// marshalBytes converts an arbitrary value into a byte buffer.
//
// Prefer using Map.marshalKey and Map.marshalValue if possible, since
// those have special cases that allow more types to be encoded.
//
// Returns an error if the given value isn't representable in exactly
// length bytes.
func marshalBytes(data interface{}, length int) (buf []byte, err error) {
	if data == nil {
		return nil, errors.New("can't marshal a nil value")
	}

	switch value := data.(type) {
	case encoding.BinaryMarshaler:
		buf, err = value.MarshalBinary()
	case string:
		buf = []byte(value)
	case []byte:
		buf = value
	case unsafe.Pointer:
		err = errors.New("can't marshal from unsafe.Pointer")
	case Map, *Map, Program, *Program:
		err = fmt.Errorf("can't marshal %T", value)
	default:
		var wr bytes.Buffer
		err = binary.Write(&wr, internal.NativeEndian, value)
		if err != nil {
			err = fmt.Errorf("encoding %T: %v", value, err)
		}
		buf = wr.Bytes()
	}
	if err != nil {
		return nil, err
	}

	if len(buf) != length {
		return nil, fmt.Errorf("%T doesn't marshal to %d bytes", data, length)
	}
	return buf, nil
}

func makeBuffer(dst interface{}, length int) (sys.Pointer, []byte) {
	if ptr, ok := dst.(unsafe.Pointer); ok {
		return sys.NewPointer(ptr), nil
	}

	buf := make([]byte, length)
	return sys.NewSlicePointer(buf), buf
}

var bytesReaderPool = sync.Pool{
	New: func() interface{} {
		return new(bytes.Reader)
	},
}

// unmarshalBytes converts a byte buffer into an arbitrary value.
//
// Prefer using Map.unmarshalKey and Map.unmarshalValue if possible, since
// those have special cases that allow more types to be encoded.
//
// The common int32 and int64 types are directly handled to avoid
// unnecessary heap allocations as happening in the default case.
func unmarshalBytes(data interface{}, buf []byte) error {
	switch value := data.(type) {
	case unsafe.Pointer:
		dst := unsafe.Slice((*byte)(value), len(buf))
		copy(dst, buf)
		runtime.KeepAlive(value)
		return nil
	case Map, *Map, Program, *Program:
		return fmt.Errorf("can't unmarshal into %T", value)
	case encoding.BinaryUnmarshaler:
		return value.UnmarshalBinary(buf)
	case *string:
		*value = string(buf)
		return nil
	case *[]byte:
		*value = buf
		return nil
	case *int32:
		if len(buf) < 4 {
			return errors.New("int32 requires 4 bytes")
		}
		*value = int32(internal.NativeEndian.Uint32(buf))
		return nil
	case *uint32:
		if len(buf) < 4 {
			return errors.New("uint32 requires 4 bytes")
		}
		*value = internal.NativeEndian.Uint32(buf)
		return nil
	case *int64:
		if len(buf) < 8 {
			return errors.New("int64 requires 8 bytes")
		}
		*value = int64(internal.NativeEndian.Uint64(buf))
		return nil
	case *uint64:
		if len(buf) < 8 {
			return errors.New("uint64 requires 8 bytes")
		}
		*value = internal.NativeEndian.Uint64(buf)
		return nil
	case string:
		return errors.New("require pointer to string")
	case []byte:
		return errors.New("require pointer to []byte")
	default:
		rd := bytesReaderPool.Get().(*bytes.Reader)
		rd.Reset(buf)
		defer bytesReaderPool.Put(rd)
		if err := binary.Read(rd, internal.NativeEndian, value); err != nil {
			return fmt.Errorf("decoding %T: %v", value, err)
		}
		return nil
	}
}

// marshalPerCPUValue encodes a slice containing one value per
// possible CPU into a buffer of bytes.
//
// Values are initialized to zero if the slice has less elements than CPUs.
//
// slice must have a type like []elementType.
func marshalPerCPUValue(slice interface{}, elemLength int) (sys.Pointer, error) {
	sliceType := reflect.TypeOf(slice)
	if sliceType.Kind() != reflect.Slice {
		return sys.Pointer{}, errors.New("per-CPU value requires slice")
	}

	possibleCPUs, err := internal.PossibleCPUs()
	if err != nil {
		return sys.Pointer{}, err
	}

	sliceValue := reflect.ValueOf(slice)
	sliceLen := sliceValue.Len()
	if sliceLen > possibleCPUs {
		return sys.Pointer{}, fmt.Errorf("per-CPU value exceeds number of CPUs")
	}

	alignedElemLength := internal.Align(elemLength, 8)
	buf := make([]byte, alignedElemLength*possibleCPUs)

	for i := 0; i < sliceLen; i++ {
		elem := sliceValue.Index(i).Interface()
		elemBytes, err := marshalBytes(elem, elemLength)
		if err != nil {
			return sys.Pointer{}, err
		}

		offset := i * alignedElemLength
		copy(buf[offset:offset+elemLength], elemBytes)
	}

	return sys.NewSlicePointer(buf), nil
}

// unmarshalPerCPUValue decodes a buffer into a slice containing one value per
// possible CPU.
//
// valueOut must have a type like *[]elementType
func unmarshalPerCPUValue(slicePtr interface{}, elemLength int, buf []byte) error {
	slicePtrType := reflect.TypeOf(slicePtr)
	if slicePtrType.Kind() != reflect.Ptr || slicePtrType.Elem().Kind() != reflect.Slice {
		return fmt.Errorf("per-cpu value requires pointer to slice")
	}

	possibleCPUs, err := internal.PossibleCPUs()
	if err != nil {
		return err
	}

	sliceType := slicePtrType.Elem()
	slice := reflect.MakeSlice(sliceType, possibleCPUs, possibleCPUs)

	sliceElemType := sliceType.Elem()
	sliceElemIsPointer := sliceElemType.Kind() == reflect.Ptr
	if sliceElemIsPointer {
		sliceElemType = sliceElemType.Elem()
	}

	step := len(buf) / possibleCPUs
	if step < elemLength {
		return fmt.Errorf("per-cpu element length is larger than available data")
	}
	for i := 0; i < possibleCPUs; i++ {
		var elem interface{}
		if sliceElemIsPointer {
			newElem := reflect.New(sliceElemType)
			slice.Index(i).Set(newElem)
			elem = newElem.Interface()
		} else {
			elem = slice.Index(i).Addr().Interface()
		}

		// Make a copy, since unmarshal can hold on to itemBytes
		elemBytes := make([]byte, elemLength)
		copy(elemBytes, buf[:elemLength])

		err := unmarshalBytes(elem, elemBytes)
		if err != nil {
			return fmt.Errorf("cpu %d: %w", i, err)
		}

		buf = buf[step:]
	}

	reflect.ValueOf(slicePtr).Elem().Set(slice)
	return nil
}
