package ebpf

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/btf"
	"github.com/cilium/ebpf/internal/unix"
)

// Generic errors returned by BPF syscalls.
var ErrNotExist = errors.New("requested object does not exist")

// bpfObjName is a null-terminated string made up of
// 'A-Za-z0-9_' characters.
type bpfObjName [unix.BPF_OBJ_NAME_LEN]byte

// newBPFObjName truncates the result if it is too long.
func newBPFObjName(name string) bpfObjName {
	var result bpfObjName
	copy(result[:unix.BPF_OBJ_NAME_LEN-1], name)
	return result
}

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

type bpfMapCreateAttr struct {
	mapType        MapType
	keySize        uint32
	valueSize      uint32
	maxEntries     uint32
	flags          uint32
	innerMapFd     uint32     // since 4.12 56f668dfe00d
	numaNode       uint32     // since 4.14 96eabe7a40aa
	mapName        bpfObjName // since 4.15 ad5b177bd73f
	mapIfIndex     uint32
	btfFd          uint32
	btfKeyTypeID   btf.TypeID
	btfValueTypeID btf.TypeID
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
	name                      bpfObjName // since 4.15 ad5b177bd73f
	ifindex                   uint32     // since 4.16 52775b33bb50
	btf_vmlinux_value_type_id uint32     // since 5.6  85d33df357b6
	netns_dev                 uint64     // since 4.16 52775b33bb50
	netns_ino                 uint64
	btf_id                    uint32 // since 4.18 78958fca7ead
	btf_key_type_id           uint32 // since 4.18 9b2cf328b2ec
	btf_value_type_id         uint32
}

type bpfProgLoadAttr struct {
	progType           ProgramType
	insCount           uint32
	instructions       internal.Pointer
	license            internal.Pointer
	logLevel           uint32
	logSize            uint32
	logBuf             internal.Pointer
	kernelVersion      uint32     // since 4.1  2541517c32be
	progFlags          uint32     // since 4.11 e07b98d9bffe
	progName           bpfObjName // since 4.15 067cae47771c
	progIfIndex        uint32     // since 4.15 1f6f4cb7ba21
	expectedAttachType AttachType // since 4.17 5e43f899b03a
	progBTFFd          uint32
	funcInfoRecSize    uint32
	funcInfo           internal.Pointer
	funcInfoCnt        uint32
	lineInfoRecSize    uint32
	lineInfo           internal.Pointer
	lineInfoCnt        uint32
	attachBTFID        btf.TypeID
	attachProgFd       uint32
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
	nr_map_ids               uint32
	map_ids                  internal.Pointer
	name                     bpfObjName // since 4.15 067cae47771c
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

type bpfGetFDByIDAttr struct {
	id   uint32
	next uint32
}

type bpfMapFreezeAttr struct {
	mapFd uint32
}

type bpfObjGetNextIDAttr struct {
	startID   uint32
	nextID    uint32
	openFlags uint32
}

func bpfProgLoad(attr *bpfProgLoadAttr) (*internal.FD, error) {
	for {
		fd, err := internal.BPF(internal.BPF_PROG_LOAD, unsafe.Pointer(attr), unsafe.Sizeof(*attr))
		// As of ~4.20 the verifier can be interrupted by a signal,
		// and returns EAGAIN in that case.
		if err == unix.EAGAIN {
			continue
		}

		if err != nil {
			return nil, err
		}

		return internal.NewFD(uint32(fd)), nil
	}
}

func bpfProgTestRun(attr *bpfProgTestRunAttr) error {
	_, err := internal.BPF(internal.BPF_PROG_TEST_RUN, unsafe.Pointer(attr), unsafe.Sizeof(*attr))
	return err
}

func bpfMapCreate(attr *bpfMapCreateAttr) (*internal.FD, error) {
	fd, err := internal.BPF(internal.BPF_MAP_CREATE, unsafe.Pointer(attr), unsafe.Sizeof(*attr))
	if err != nil {
		return nil, err
	}

	return internal.NewFD(uint32(fd)), nil
}

var haveNestedMaps = internal.FeatureTest("nested maps", "4.12", func() error {
	_, err := bpfMapCreate(&bpfMapCreateAttr{
		mapType:    ArrayOfMaps,
		keySize:    4,
		valueSize:  4,
		maxEntries: 1,
		// Invalid file descriptor.
		innerMapFd: ^uint32(0),
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
	m, err := bpfMapCreate(&bpfMapCreateAttr{
		mapType:    Array,
		keySize:    4,
		valueSize:  4,
		maxEntries: 1,
		flags:      unix.BPF_F_RDONLY_PROG,
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
	return attr.nextID, wrapObjError(err)
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

func wrapObjError(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, unix.ENOENT) {
		return fmt.Errorf("%w", ErrNotExist)
	}

	return errors.New(err.Error())
}

func wrapMapError(err error) error {
	if err == nil {
		return nil
	}

	if errors.Is(err, unix.ENOENT) {
		return ErrKeyNotExist
	}

	if errors.Is(err, unix.EEXIST) {
		return ErrKeyExist
	}

	if errors.Is(err, unix.ENOTSUPP) {
		return ErrNotSupported
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

func bpfGetProgInfoByFD(fd *internal.FD) (*bpfProgInfo, error) {
	var info bpfProgInfo
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
	attr := bpfMapCreateAttr{
		mapType:    Array,
		keySize:    4,
		valueSize:  4,
		maxEntries: 1,
		mapName:    newBPFObjName("feature_test"),
	}

	fd, err := bpfMapCreate(&attr)
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

	attr := bpfMapCreateAttr{
		mapType:    Array,
		keySize:    4,
		valueSize:  4,
		maxEntries: 1,
		mapName:    newBPFObjName(".test"),
	}

	fd, err := bpfMapCreate(&attr)
	if err != nil {
		return internal.ErrNotSupported
	}

	_ = fd.Close()
	return nil
})

var haveBatchAPI = internal.FeatureTest("map batch api", "5.6", func() error {
	var maxEntries uint32 = 2
	attr := bpfMapCreateAttr{
		mapType:    Hash,
		keySize:    4,
		valueSize:  4,
		maxEntries: maxEntries,
	}

	fd, err := bpfMapCreate(&attr)
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

func bpfObjGetFDByID(cmd internal.BPFCmd, id uint32) (*internal.FD, error) {
	attr := bpfGetFDByIDAttr{
		id: id,
	}
	ptr, err := internal.BPF(cmd, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	return internal.NewFD(uint32(ptr)), wrapObjError(err)
}
