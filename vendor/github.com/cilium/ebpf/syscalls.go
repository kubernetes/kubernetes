package ebpf

import (
	"bytes"
	"errors"
	"fmt"

	"github.com/cilium/ebpf/asm"
	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/sys"
	"github.com/cilium/ebpf/internal/unix"
)

// invalidBPFObjNameChar returns true if char may not appear in
// a BPF object name.
func invalidBPFObjNameChar(char rune) bool {
	dotAllowed := objNameAllowsDot() == nil

	switch {
	case char >= 'A' && char <= 'Z':
		return false
	case char >= 'a' && char <= 'z':
		return false
	case char >= '0' && char <= '9':
		return false
	case dotAllowed && char == '.':
		return false
	case char == '_':
		return false
	default:
		return true
	}
}

func progLoad(insns asm.Instructions, typ ProgramType, license string) (*sys.FD, error) {
	buf := bytes.NewBuffer(make([]byte, 0, insns.Size()))
	if err := insns.Marshal(buf, internal.NativeEndian); err != nil {
		return nil, err
	}
	bytecode := buf.Bytes()

	return sys.ProgLoad(&sys.ProgLoadAttr{
		ProgType: sys.ProgType(typ),
		License:  sys.NewStringPointer(license),
		Insns:    sys.NewSlicePointer(bytecode),
		InsnCnt:  uint32(len(bytecode) / asm.InstructionSize),
	})
}

var haveNestedMaps = internal.FeatureTest("nested maps", "4.12", func() error {
	_, err := sys.MapCreate(&sys.MapCreateAttr{
		MapType:    sys.MapType(ArrayOfMaps),
		KeySize:    4,
		ValueSize:  4,
		MaxEntries: 1,
		// Invalid file descriptor.
		InnerMapFd: ^uint32(0),
	})
	if errors.Is(err, unix.EINVAL) {
		return internal.ErrNotSupported
	}
	if errors.Is(err, unix.EBADF) {
		return nil
	}
	return err
})

var haveMapMutabilityModifiers = internal.FeatureTest("read- and write-only maps", "5.2", func() error {
	// This checks BPF_F_RDONLY_PROG and BPF_F_WRONLY_PROG. Since
	// BPF_MAP_FREEZE appeared in 5.2 as well we don't do a separate check.
	m, err := sys.MapCreate(&sys.MapCreateAttr{
		MapType:    sys.MapType(Array),
		KeySize:    4,
		ValueSize:  4,
		MaxEntries: 1,
		MapFlags:   unix.BPF_F_RDONLY_PROG,
	})
	if err != nil {
		return internal.ErrNotSupported
	}
	_ = m.Close()
	return nil
})

var haveMmapableMaps = internal.FeatureTest("mmapable maps", "5.5", func() error {
	// This checks BPF_F_MMAPABLE, which appeared in 5.5 for array maps.
	m, err := sys.MapCreate(&sys.MapCreateAttr{
		MapType:    sys.MapType(Array),
		KeySize:    4,
		ValueSize:  4,
		MaxEntries: 1,
		MapFlags:   unix.BPF_F_MMAPABLE,
	})
	if err != nil {
		return internal.ErrNotSupported
	}
	_ = m.Close()
	return nil
})

var haveInnerMaps = internal.FeatureTest("inner maps", "5.10", func() error {
	// This checks BPF_F_INNER_MAP, which appeared in 5.10.
	m, err := sys.MapCreate(&sys.MapCreateAttr{
		MapType:    sys.MapType(Array),
		KeySize:    4,
		ValueSize:  4,
		MaxEntries: 1,
		MapFlags:   unix.BPF_F_INNER_MAP,
	})
	if err != nil {
		return internal.ErrNotSupported
	}
	_ = m.Close()
	return nil
})

var haveNoPreallocMaps = internal.FeatureTest("prealloc maps", "4.6", func() error {
	// This checks BPF_F_NO_PREALLOC, which appeared in 4.6.
	m, err := sys.MapCreate(&sys.MapCreateAttr{
		MapType:    sys.MapType(Hash),
		KeySize:    4,
		ValueSize:  4,
		MaxEntries: 1,
		MapFlags:   unix.BPF_F_NO_PREALLOC,
	})
	if err != nil {
		return internal.ErrNotSupported
	}
	_ = m.Close()
	return nil
})

func wrapMapError(err error) error {
	if err == nil {
		return nil
	}

	if errors.Is(err, unix.ENOENT) {
		return sys.Error(ErrKeyNotExist, unix.ENOENT)
	}

	if errors.Is(err, unix.EEXIST) {
		return sys.Error(ErrKeyExist, unix.EEXIST)
	}

	if errors.Is(err, unix.ENOTSUPP) {
		return sys.Error(ErrNotSupported, unix.ENOTSUPP)
	}

	if errors.Is(err, unix.E2BIG) {
		return fmt.Errorf("key too big for map: %w", err)
	}

	return err
}

var haveObjName = internal.FeatureTest("object names", "4.15", func() error {
	attr := sys.MapCreateAttr{
		MapType:    sys.MapType(Array),
		KeySize:    4,
		ValueSize:  4,
		MaxEntries: 1,
		MapName:    sys.NewObjName("feature_test"),
	}

	fd, err := sys.MapCreate(&attr)
	if err != nil {
		return internal.ErrNotSupported
	}

	_ = fd.Close()
	return nil
})

var objNameAllowsDot = internal.FeatureTest("dot in object names", "5.2", func() error {
	if err := haveObjName(); err != nil {
		return err
	}

	attr := sys.MapCreateAttr{
		MapType:    sys.MapType(Array),
		KeySize:    4,
		ValueSize:  4,
		MaxEntries: 1,
		MapName:    sys.NewObjName(".test"),
	}

	fd, err := sys.MapCreate(&attr)
	if err != nil {
		return internal.ErrNotSupported
	}

	_ = fd.Close()
	return nil
})

var haveBatchAPI = internal.FeatureTest("map batch api", "5.6", func() error {
	var maxEntries uint32 = 2
	attr := sys.MapCreateAttr{
		MapType:    sys.MapType(Hash),
		KeySize:    4,
		ValueSize:  4,
		MaxEntries: maxEntries,
	}

	fd, err := sys.MapCreate(&attr)
	if err != nil {
		return internal.ErrNotSupported
	}
	defer fd.Close()

	keys := []uint32{1, 2}
	values := []uint32{3, 4}
	kp, _ := marshalPtr(keys, 8)
	vp, _ := marshalPtr(values, 8)

	err = sys.MapUpdateBatch(&sys.MapUpdateBatchAttr{
		MapFd:  fd.Uint(),
		Keys:   kp,
		Values: vp,
		Count:  maxEntries,
	})
	if err != nil {
		return internal.ErrNotSupported
	}
	return nil
})

var haveProbeReadKernel = internal.FeatureTest("bpf_probe_read_kernel", "5.5", func() error {
	insns := asm.Instructions{
		asm.Mov.Reg(asm.R1, asm.R10),
		asm.Add.Imm(asm.R1, -8),
		asm.Mov.Imm(asm.R2, 8),
		asm.Mov.Imm(asm.R3, 0),
		asm.FnProbeReadKernel.Call(),
		asm.Return(),
	}

	fd, err := progLoad(insns, Kprobe, "GPL")
	if err != nil {
		return internal.ErrNotSupported
	}
	_ = fd.Close()
	return nil
})

var haveBPFToBPFCalls = internal.FeatureTest("bpf2bpf calls", "4.16", func() error {
	insns := asm.Instructions{
		asm.Call.Label("prog2").WithSymbol("prog1"),
		asm.Return(),
		asm.Mov.Imm(asm.R0, 0).WithSymbol("prog2"),
		asm.Return(),
	}

	fd, err := progLoad(insns, SocketFilter, "MIT")
	if errors.Is(err, unix.EINVAL) {
		return internal.ErrNotSupported
	}
	if err != nil {
		return err
	}
	_ = fd.Close()
	return nil
})
