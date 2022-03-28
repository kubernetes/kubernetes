package btf

import (
	"fmt"
	"unsafe"

	"github.com/cilium/ebpf/internal"
)

type bpfBTFInfo struct {
	btf       internal.Pointer
	btfSize   uint32
	id        uint32
	name      internal.Pointer
	nameLen   uint32
	kernelBTF uint32
}

func bpfGetBTFInfoByFD(fd *internal.FD, btf, name []byte) (*bpfBTFInfo, error) {
	info := bpfBTFInfo{
		btf:     internal.NewSlicePointer(btf),
		btfSize: uint32(len(btf)),
		name:    internal.NewSlicePointer(name),
		nameLen: uint32(len(name)),
	}
	if err := internal.BPFObjGetInfoByFD(fd, unsafe.Pointer(&info), unsafe.Sizeof(info)); err != nil {
		return nil, fmt.Errorf("can't get program info: %w", err)
	}

	return &info, nil
}
