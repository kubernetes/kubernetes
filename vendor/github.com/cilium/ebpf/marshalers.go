package ebpf

import (
	"encoding"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"unsafe"

	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/sys"
	"github.com/cilium/ebpf/internal/sysenc"
)

// marshalMapSyscallInput converts an arbitrary value into a pointer suitable
// to be passed to the kernel.
//
// As an optimization, it returns the original value if it is an
// unsafe.Pointer.
func marshalMapSyscallInput(data any, length int) (sys.Pointer, error) {
	if ptr, ok := data.(unsafe.Pointer); ok {
		return sys.NewPointer(ptr), nil
	}

	buf, err := sysenc.Marshal(data, length)
	if err != nil {
		return sys.Pointer{}, err
	}

	return buf.Pointer(), nil
}

func makeMapSyscallOutput(dst any, length int) sysenc.Buffer {
	if ptr, ok := dst.(unsafe.Pointer); ok {
		return sysenc.UnsafeBuffer(ptr)
	}

	_, ok := dst.(encoding.BinaryUnmarshaler)
	if ok {
		return sysenc.SyscallOutput(nil, length)
	}

	return sysenc.SyscallOutput(dst, length)
}

// appendPerCPUSlice encodes a slice containing one value per
// possible CPU into a buffer of bytes.
//
// Values are initialized to zero if the slice has less elements than CPUs.
func appendPerCPUSlice(buf []byte, slice any, possibleCPUs, elemLength, alignedElemLength int) ([]byte, error) {
	sliceType := reflect.TypeOf(slice)
	if sliceType.Kind() != reflect.Slice {
		return nil, errors.New("per-CPU value requires slice")
	}

	sliceValue := reflect.ValueOf(slice)
	sliceLen := sliceValue.Len()
	if sliceLen > possibleCPUs {
		return nil, fmt.Errorf("per-CPU value greater than number of CPUs")
	}

	// Grow increases the slice's capacity, _if_necessary_
	buf = slices.Grow(buf, alignedElemLength*possibleCPUs)
	for i := 0; i < sliceLen; i++ {
		elem := sliceValue.Index(i).Interface()
		elemBytes, err := sysenc.Marshal(elem, elemLength)
		if err != nil {
			return nil, err
		}

		buf = elemBytes.AppendTo(buf)
		buf = append(buf, make([]byte, alignedElemLength-elemLength)...)
	}

	// Ensure buf is zero-padded full size.
	buf = append(buf, make([]byte, (possibleCPUs-sliceLen)*alignedElemLength)...)

	return buf, nil
}

// marshalPerCPUValue encodes a slice containing one value per
// possible CPU into a buffer of bytes.
//
// Values are initialized to zero if the slice has less elements than CPUs.
func marshalPerCPUValue(slice any, elemLength int) (sys.Pointer, error) {
	possibleCPUs, err := PossibleCPU()
	if err != nil {
		return sys.Pointer{}, err
	}

	alignedElemLength := internal.Align(elemLength, 8)
	buf := make([]byte, 0, alignedElemLength*possibleCPUs)
	buf, err = appendPerCPUSlice(buf, slice, possibleCPUs, elemLength, alignedElemLength)
	if err != nil {
		return sys.Pointer{}, err
	}

	return sys.NewSlicePointer(buf), nil
}

// marshalBatchPerCPUValue encodes a batch-sized slice of slices containing
// one value per possible CPU into a buffer of bytes.
func marshalBatchPerCPUValue(slice any, batchLen, elemLength int) ([]byte, error) {
	sliceType := reflect.TypeOf(slice)
	if sliceType.Kind() != reflect.Slice {
		return nil, fmt.Errorf("batch value requires a slice")
	}
	sliceValue := reflect.ValueOf(slice)

	possibleCPUs, err := PossibleCPU()
	if err != nil {
		return nil, err
	}
	if sliceValue.Len() != batchLen*possibleCPUs {
		return nil, fmt.Errorf("per-CPU slice has incorrect length, expected %d, got %d",
			batchLen*possibleCPUs, sliceValue.Len())
	}
	alignedElemLength := internal.Align(elemLength, 8)
	buf := make([]byte, 0, batchLen*alignedElemLength*possibleCPUs)
	for i := 0; i < batchLen; i++ {
		batch := sliceValue.Slice(i*possibleCPUs, (i+1)*possibleCPUs).Interface()
		buf, err = appendPerCPUSlice(buf, batch, possibleCPUs, elemLength, alignedElemLength)
		if err != nil {
			return nil, fmt.Errorf("batch %d: %w", i, err)
		}
	}
	return buf, nil
}

// unmarshalPerCPUValue decodes a buffer into a slice containing one value per
// possible CPU.
//
// slice must be a literal slice and not a pointer.
func unmarshalPerCPUValue(slice any, elemLength int, buf []byte) error {
	sliceType := reflect.TypeOf(slice)
	if sliceType.Kind() != reflect.Slice {
		return fmt.Errorf("per-CPU value requires a slice")
	}

	possibleCPUs, err := PossibleCPU()
	if err != nil {
		return err
	}

	sliceValue := reflect.ValueOf(slice)
	if sliceValue.Len() != possibleCPUs {
		return fmt.Errorf("per-CPU slice has incorrect length, expected %d, got %d",
			possibleCPUs, sliceValue.Len())
	}

	sliceElemType := sliceType.Elem()
	sliceElemIsPointer := sliceElemType.Kind() == reflect.Ptr
	stride := internal.Align(elemLength, 8)
	for i := 0; i < possibleCPUs; i++ {
		var elem any
		v := sliceValue.Index(i)
		if sliceElemIsPointer {
			if !v.Elem().CanAddr() {
				return fmt.Errorf("per-CPU slice elements cannot be nil")
			}
			elem = v.Elem().Addr().Interface()
		} else {
			elem = v.Addr().Interface()
		}
		err := sysenc.Unmarshal(elem, buf[:elemLength])
		if err != nil {
			return fmt.Errorf("cpu %d: %w", i, err)
		}

		buf = buf[stride:]
	}
	return nil
}

// unmarshalBatchPerCPUValue decodes a buffer into a batch-sized slice
// containing one value per possible CPU.
//
// slice must have length batchLen * PossibleCPUs().
func unmarshalBatchPerCPUValue(slice any, batchLen, elemLength int, buf []byte) error {
	sliceType := reflect.TypeOf(slice)
	if sliceType.Kind() != reflect.Slice {
		return fmt.Errorf("batch requires a slice")
	}

	sliceValue := reflect.ValueOf(slice)
	possibleCPUs, err := PossibleCPU()
	if err != nil {
		return err
	}
	if sliceValue.Len() != batchLen*possibleCPUs {
		return fmt.Errorf("per-CPU slice has incorrect length, expected %d, got %d",
			sliceValue.Len(), batchLen*possibleCPUs)
	}

	fullValueSize := possibleCPUs * internal.Align(elemLength, 8)
	if len(buf) != batchLen*fullValueSize {
		return fmt.Errorf("input buffer has incorrect length, expected %d, got %d",
			len(buf), batchLen*fullValueSize)
	}

	for i := 0; i < batchLen; i++ {
		elem := sliceValue.Slice(i*possibleCPUs, (i+1)*possibleCPUs).Interface()
		if err := unmarshalPerCPUValue(elem, elemLength, buf[:fullValueSize]); err != nil {
			return fmt.Errorf("batch %d: %w", i, err)
		}
		buf = buf[fullValueSize:]
	}
	return nil
}
