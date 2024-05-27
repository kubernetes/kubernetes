package internal

import (
	"errors"
	"io"
	_ "unsafe"
)

type auxvPairReader interface {
	Close() error
	ReadAuxvPair() (uint64, uint64, error)
}

// See https://elixir.bootlin.com/linux/v6.5.5/source/include/uapi/linux/auxvec.h
const (
	_AT_NULL         = 0  // End of vector
	_AT_SYSINFO_EHDR = 33 // Offset to vDSO blob in process image
)

//go:linkname runtime_getAuxv runtime.getAuxv
func runtime_getAuxv() []uintptr

type auxvRuntimeReader struct {
	data  []uintptr
	index int
}

func (r *auxvRuntimeReader) Close() error {
	return nil
}

func (r *auxvRuntimeReader) ReadAuxvPair() (uint64, uint64, error) {
	if r.index >= len(r.data)+2 {
		return 0, 0, io.EOF
	}

	// we manually add the (_AT_NULL, _AT_NULL) pair at the end
	// that is not provided by the go runtime
	var tag, value uintptr
	if r.index+1 < len(r.data) {
		tag, value = r.data[r.index], r.data[r.index+1]
	} else {
		tag, value = _AT_NULL, _AT_NULL
	}
	r.index += 2
	return uint64(tag), uint64(value), nil
}

func newAuxvRuntimeReader() (auxvPairReader, error) {
	data := runtime_getAuxv()

	if len(data)%2 != 0 {
		return nil, errors.New("malformed auxv passed from runtime")
	}

	return &auxvRuntimeReader{
		data:  data,
		index: 0,
	}, nil
}
