package ebpf

import (
	"bytes"
	"encoding"
	"encoding/binary"
	"errors"
	"fmt"
	"reflect"
	"runtime"
	"unsafe"

	"github.com/cilium/ebpf/internal"
)

func marshalPtr(data interface{}, length int) (internal.Pointer, error) {
	if data == nil {
		if length == 0 {
			return internal.NewPointer(nil), nil
		}
		return internal.Pointer{}, errors.New("can't use nil as key of map")
	}

	if ptr, ok := data.(unsafe.Pointer); ok {
		return internal.NewPointer(ptr), nil
	}

	buf, err := marshalBytes(data, length)
	if err != nil {
		return internal.Pointer{}, err
	}

	return internal.NewSlicePointer(buf), nil
}

func marshalBytes(data interface{}, length int) (buf []byte, err error) {
	switch value := data.(type) {
	case encoding.BinaryMarshaler:
		buf, err = value.MarshalBinary()
	case string:
		buf = []byte(value)
	case []byte:
		buf = value
	case unsafe.Pointer:
		err = errors.New("can't marshal from unsafe.Pointer")
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

func makeBuffer(dst interface{}, length int) (internal.Pointer, []byte) {
	if ptr, ok := dst.(unsafe.Pointer); ok {
		return internal.NewPointer(ptr), nil
	}

	buf := make([]byte, length)
	return internal.NewSlicePointer(buf), buf
}

func unmarshalBytes(data interface{}, buf []byte) error {
	switch value := data.(type) {
	case unsafe.Pointer:
		sh := &reflect.SliceHeader{
			Data: uintptr(value),
			Len:  len(buf),
			Cap:  len(buf),
		}

		dst := *(*[]byte)(unsafe.Pointer(sh))
		copy(dst, buf)
		runtime.KeepAlive(value)
		return nil
	case encoding.BinaryUnmarshaler:
		return value.UnmarshalBinary(buf)
	case *string:
		*value = string(buf)
		return nil
	case *[]byte:
		*value = buf
		return nil
	case string:
		return errors.New("require pointer to string")
	case []byte:
		return errors.New("require pointer to []byte")
	default:
		rd := bytes.NewReader(buf)
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
func marshalPerCPUValue(slice interface{}, elemLength int) (internal.Pointer, error) {
	sliceType := reflect.TypeOf(slice)
	if sliceType.Kind() != reflect.Slice {
		return internal.Pointer{}, errors.New("per-CPU value requires slice")
	}

	possibleCPUs, err := internal.PossibleCPUs()
	if err != nil {
		return internal.Pointer{}, err
	}

	sliceValue := reflect.ValueOf(slice)
	sliceLen := sliceValue.Len()
	if sliceLen > possibleCPUs {
		return internal.Pointer{}, fmt.Errorf("per-CPU value exceeds number of CPUs")
	}

	alignedElemLength := align(elemLength, 8)
	buf := make([]byte, alignedElemLength*possibleCPUs)

	for i := 0; i < sliceLen; i++ {
		elem := sliceValue.Index(i).Interface()
		elemBytes, err := marshalBytes(elem, elemLength)
		if err != nil {
			return internal.Pointer{}, err
		}

		offset := i * alignedElemLength
		copy(buf[offset:offset+elemLength], elemBytes)
	}

	return internal.NewSlicePointer(buf), nil
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

func align(n, alignment int) int {
	return (int(n) + alignment - 1) / alignment * alignment
}
