package ebpf

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"unsafe"

	"github.com/cilium/ebpf/asm"
	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/unix"
)

// ErrNotExist is returned when loading a non-existing map or program.
//
// Deprecated: use os.ErrNotExist instead.
var ErrNotExist = os.ErrNotExist

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

type bpfMapOpAttr struct {
	mapFd   uint32
	padding uint32
	key     internal.Pointer
	value   internal.Pointer
	flags   uint64
}

type bpfBatchMapOpAttr struct {
	inBatch   internal.Pointer
	outBatch  internal.Pointer
	keys      internal.Pointer
	values    internal.Pointer
	count     uint32
	mapFd     uint32
	elemFlags uint64
	flags     uint64
}

type bpfMapInfo struct {
	map_type                  uint32 // since 4.12 1e2709769086
	id                        uint32
	key_size                  uint32
	value_size                uint32
	max_entries               uint32
	map_flags                 uint32
	name                      internal.BPFObjName // since 4.15 ad5b177bd73f
	ifindex                   uint32              // since 4.16 52775b33bb50
	btf_vmlinux_value_type_id uint32              // since 5.6  85d33df357b6
	netns_dev                 uint64              // since 4.16 52775b33bb50
	netns_ino                 uint64
	btf_id                    uint32 // since 4.18 78958fca7ead
	btf_key_type_id           uint32 // since 4.18 9b2cf328b2ec
	btf_value_type_id         uint32
}

type bpfProgInfo struct {
	prog_type                uint32
	id                       uint32
	tag                      [unix.BPF_TAG_SIZE]byte
	jited_prog_len           uint32
	xlated_prog_len          uint32
	jited_prog_insns         internal.Pointer
	xlated_prog_insns        internal.Pointer
	load_time                uint64 // since 4.15 cb4d2b3f03d8
	created_by_uid           uint32
	nr_map_ids               uint32 // since 4.15 cb4d2b3f03d8
	map_ids                  internal.Pointer
	name                     internal.BPFObjName // since 4.15 067cae47771c
	ifindex                  uint32
	gpl_compatible           uint32
	netns_dev                uint64
	netns_ino                uint64
	nr_jited_ksyms           uint32
	nr_jited_func_lens       uint32
	jited_ksyms              internal.Pointer
	jited_func_lens          internal.Pointer
	btf_id                   uint32
	func_info_rec_size       uint32
	func_info                internal.Pointer
	nr_func_info             uint32
	nr_line_info             uint32
	line_info                internal.Pointer
	jited_line_info          internal.Pointer
	nr_jited_line_info       uint32
	line_info_rec_size       uint32
	jited_line_info_rec_size uint32
	nr_prog_tags             uint32
	prog_tags                internal.Pointer
	run_time_ns              uint64
	run_cnt                  uint64
}

type bpfProgTestRunAttr struct {
	fd          uint32
	retval      uint32
	dataSizeIn  uint32
	dataSizeOut uint32
	dataIn      internal.Pointer
	dataOut     internal.Pointer
	repeat      uint32
	duration    uint32
}

type bpfMapFreezeAttr struct {
	mapFd uint32
}

type bpfObjGetNextIDAttr struct {
	startID   uint32
	nextID    uint32
	openFlags uint32
}

func bpfProgTestRun(attr *bpfProgTestRunAttr) error {
	_, err := internal.BPF(internal.BPF_PROG_TEST_RUN, unsafe.Pointer(attr), unsafe.Sizeof(*attr))
	return err
}

var haveNestedMaps = internal.FeatureTest("nested maps", "4.12", func() error {
	_, err := internal.BPFMapCreate(&internal.BPFMapCreateAttr{
		MapType:    uint32(ArrayOfMaps),
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
	m, err := internal.BPFMapCreate(&internal.BPFMapCreateAttr{
		MapType:    uint32(Array),
		KeySize:    4,
		ValueSize:  4,
		MaxEntries: 1,
		Flags:      unix.BPF_F_RDONLY_PROG,
	})
	if err != nil {
		return internal.ErrNotSupported
	}
	_ = m.Close()
	return nil
})

var haveMmapableMaps = internal.FeatureTest("mmapable maps", "5.5", func() error {
	// This checks BPF_F_MMAPABLE, which appeared in 5.5 for array maps.
	m, err := internal.BPFMapCreate(&internal.BPFMapCreateAttr{
		MapType:    uint32(Array),
		KeySize:    4,
		ValueSize:  4,
		MaxEntries: 1,
		Flags:      unix.BPF_F_MMAPABLE,
	})
	if err != nil {
		return internal.ErrNotSupported
	}
	_ = m.Close()
	return nil
})

var haveInnerMaps = internal.FeatureTest("inner maps", "5.10", func() error {
	// This checks BPF_F_INNER_MAP, which appeared in 5.10.
	m, err := internal.BPFMapCreate(&internal.BPFMapCreateAttr{
		MapType:    uint32(Array),
		KeySize:    4,
		ValueSize:  4,
		MaxEntries: 1,
		Flags:      unix.BPF_F_INNER_MAP,
	})
	if err != nil {
		return internal.ErrNotSupported
	}
	_ = m.Close()
	return nil
})

func bpfMapLookupElem(m *internal.FD, key, valueOut internal.Pointer) error {
	fd, err := m.Value()
	if err != nil {
		return err
	}

	attr := bpfMapOpAttr{
		mapFd: fd,
		key:   key,
		value: valueOut,
	}
	_, err = internal.BPF(internal.BPF_MAP_LOOKUP_ELEM, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	return wrapMapError(err)
}

func bpfMapLookupAndDelete(m *internal.FD, key, valueOut internal.Pointer) error {
	fd, err := m.Value()
	if err != nil {
		return err
	}

	attr := bpfMapOpAttr{
		mapFd: fd,
		key:   key,
		value: valueOut,
	}
	_, err = internal.BPF(internal.BPF_MAP_LOOKUP_AND_DELETE_ELEM, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	return wrapMapError(err)
}

func bpfMapUpdateElem(m *internal.FD, key, valueOut internal.Pointer, flags uint64) error {
	fd, err := m.Value()
	if err != nil {
		return err
	}

	attr := bpfMapOpAttr{
		mapFd: fd,
		key:   key,
		value: valueOut,
		flags: flags,
	}
	_, err = internal.BPF(internal.BPF_MAP_UPDATE_ELEM, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	return wrapMapError(err)
}

func bpfMapDeleteElem(m *internal.FD, key internal.Pointer) error {
	fd, err := m.Value()
	if err != nil {
		return err
	}

	attr := bpfMapOpAttr{
		mapFd: fd,
		key:   key,
	}
	_, err = internal.BPF(internal.BPF_MAP_DELETE_ELEM, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	return wrapMapError(err)
}

func bpfMapGetNextKey(m *internal.FD, key, nextKeyOut internal.Pointer) error {
	fd, err := m.Value()
	if err != nil {
		return err
	}

	attr := bpfMapOpAttr{
		mapFd: fd,
		key:   key,
		value: nextKeyOut,
	}
	_, err = internal.BPF(internal.BPF_MAP_GET_NEXT_KEY, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	return wrapMapError(err)
}

func objGetNextID(cmd internal.BPFCmd, start uint32) (uint32, error) {
	attr := bpfObjGetNextIDAttr{
		startID: start,
	}
	_, err := internal.BPF(cmd, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	return attr.nextID, err
}

func bpfMapBatch(cmd internal.BPFCmd, m *internal.FD, inBatch, outBatch, keys, values internal.Pointer, count uint32, opts *BatchOptions) (uint32, error) {
	fd, err := m.Value()
	if err != nil {
		return 0, err
	}

	attr := bpfBatchMapOpAttr{
		inBatch:  inBatch,
		outBatch: outBatch,
		keys:     keys,
		values:   values,
		count:    count,
		mapFd:    fd,
	}
	if opts != nil {
		attr.elemFlags = opts.ElemFlags
		attr.flags = opts.Flags
	}
	_, err = internal.BPF(cmd, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	// always return count even on an error, as things like update might partially be fulfilled.
	return attr.count, wrapMapError(err)
}

func wrapMapError(err error) error {
	if err == nil {
		return nil
	}

	if errors.Is(err, unix.ENOENT) {
		return internal.SyscallError(ErrKeyNotExist, unix.ENOENT)
	}

	if errors.Is(err, unix.EEXIST) {
		return internal.SyscallError(ErrKeyExist, unix.EEXIST)
	}

	if errors.Is(err, unix.ENOTSUPP) {
		return internal.SyscallError(ErrNotSupported, unix.ENOTSUPP)
	}

	if errors.Is(err, unix.E2BIG) {
		return fmt.Errorf("key too big for map: %w", err)
	}

	return err
}

func bpfMapFreeze(m *internal.FD) error {
	fd, err := m.Value()
	if err != nil {
		return err
	}

	attr := bpfMapFreezeAttr{
		mapFd: fd,
	}
	_, err = internal.BPF(internal.BPF_MAP_FREEZE, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	return err
}

func bpfGetProgInfoByFD(fd *internal.FD, ids []MapID) (*bpfProgInfo, error) {
	var info bpfProgInfo
	if len(ids) > 0 {
		info.nr_map_ids = uint32(len(ids))
		info.map_ids = internal.NewPointer(unsafe.Pointer(&ids[0]))
	}

	if err := internal.BPFObjGetInfoByFD(fd, unsafe.Pointer(&info), unsafe.Sizeof(info)); err != nil {
		return nil, fmt.Errorf("can't get program info: %w", err)
	}
	return &info, nil
}

func bpfGetMapInfoByFD(fd *internal.FD) (*bpfMapInfo, error) {
	var info bpfMapInfo
	err := internal.BPFObjGetInfoByFD(fd, unsafe.Pointer(&info), unsafe.Sizeof(info))
	if err != nil {
		return nil, fmt.Errorf("can't get map info: %w", err)
	}
	return &info, nil
}

var haveObjName = internal.FeatureTest("object names", "4.15", func() error {
	attr := internal.BPFMapCreateAttr{
		MapType:    uint32(Array),
		KeySize:    4,
		ValueSize:  4,
		MaxEntries: 1,
		MapName:    internal.NewBPFObjName("feature_test"),
	}

	fd, err := internal.BPFMapCreate(&attr)
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

	attr := internal.BPFMapCreateAttr{
		MapType:    uint32(Array),
		KeySize:    4,
		ValueSize:  4,
		MaxEntries: 1,
		MapName:    internal.NewBPFObjName(".test"),
	}

	fd, err := internal.BPFMapCreate(&attr)
	if err != nil {
		return internal.ErrNotSupported
	}

	_ = fd.Close()
	return nil
})

var haveBatchAPI = internal.FeatureTest("map batch api", "5.6", func() error {
	var maxEntries uint32 = 2
	attr := internal.BPFMapCreateAttr{
		MapType:    uint32(Hash),
		KeySize:    4,
		ValueSize:  4,
		MaxEntries: maxEntries,
	}

	fd, err := internal.BPFMapCreate(&attr)
	if err != nil {
		return internal.ErrNotSupported
	}
	defer fd.Close()
	keys := []uint32{1, 2}
	values := []uint32{3, 4}
	kp, _ := marshalPtr(keys, 8)
	vp, _ := marshalPtr(values, 8)
	nilPtr := internal.NewPointer(nil)
	_, err = bpfMapBatch(internal.BPF_MAP_UPDATE_BATCH, fd, nilPtr, nilPtr, kp, vp, maxEntries, nil)
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
	buf := bytes.NewBuffer(make([]byte, 0, len(insns)*asm.InstructionSize))
	if err := insns.Marshal(buf, internal.NativeEndian); err != nil {
		return err
	}
	bytecode := buf.Bytes()

	fd, err := internal.BPFProgLoad(&internal.BPFProgLoadAttr{
		ProgType:     uint32(Kprobe),
		License:      internal.NewStringPointer("GPL"),
		Instructions: internal.NewSlicePointer(bytecode),
		InsCount:     uint32(len(bytecode) / asm.InstructionSize),
	})
	if err != nil {
		return internal.ErrNotSupported
	}
	_ = fd.Close()
	return nil
})
