package ebpf

import (
	"errors"
	"fmt"
	"os"
	"unsafe"

	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/btf"
	"github.com/cilium/ebpf/internal/unix"
)

// Generic errors returned by BPF syscalls.
var (
	ErrNotExist = errors.New("requested object does not exist")
)

// bpfObjName is a null-terminated string made up of
// 'A-Za-z0-9_' characters.
type bpfObjName [unix.BPF_OBJ_NAME_LEN]byte

// newBPFObjName truncates the result if it is too long.
func newBPFObjName(name string) bpfObjName {
	var result bpfObjName
	copy(result[:unix.BPF_OBJ_NAME_LEN-1], name)
	return result
}

func invalidBPFObjNameChar(char rune) bool {
	dotAllowed := objNameAllowsDot() == nil

	switch {
	case char >= 'A' && char <= 'Z':
		fallthrough
	case char >= 'a' && char <= 'z':
		fallthrough
	case char >= '0' && char <= '9':
		fallthrough
	case dotAllowed && char == '.':
		fallthrough
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

type bpfMapInfo struct {
	mapType    uint32
	id         uint32
	keySize    uint32
	valueSize  uint32
	maxEntries uint32
	flags      uint32
	mapName    bpfObjName // since 4.15 ad5b177bd73f
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
	progType     uint32
	id           uint32
	tag          [unix.BPF_TAG_SIZE]byte
	jitedLen     uint32
	xlatedLen    uint32
	jited        internal.Pointer
	xlated       internal.Pointer
	loadTime     uint64 // since 4.15 cb4d2b3f03d8
	createdByUID uint32
	nrMapIDs     uint32
	mapIds       internal.Pointer
	name         bpfObjName
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

type bpfObjGetInfoByFDAttr struct {
	fd      uint32
	infoLen uint32
	info    internal.Pointer // May be either bpfMapInfo or bpfProgInfo
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
	if errors.Is(err, os.ErrPermission) {
		return nil, errors.New("permission denied or insufficient rlimit to lock memory for map")
	}

	if err != nil {
		return nil, err
	}

	return internal.NewFD(uint32(fd)), nil
}

var haveNestedMaps = internal.FeatureTest("nested maps", "4.12", func() (bool, error) {
	inner, err := bpfMapCreate(&bpfMapCreateAttr{
		mapType:    Array,
		keySize:    4,
		valueSize:  4,
		maxEntries: 1,
	})
	if err != nil {
		return false, err
	}
	defer inner.Close()

	innerFd, _ := inner.Value()
	nested, err := bpfMapCreate(&bpfMapCreateAttr{
		mapType:    ArrayOfMaps,
		keySize:    4,
		valueSize:  4,
		maxEntries: 1,
		innerMapFd: innerFd,
	})
	if err != nil {
		return false, nil
	}

	_ = nested.Close()
	return true, nil
})

var haveMapMutabilityModifiers = internal.FeatureTest("read- and write-only maps", "5.2", func() (bool, error) {
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
		return false, nil
	}
	_ = m.Close()
	return true, nil
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

	return errors.New(err.Error())
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

func bpfGetObjectInfoByFD(fd *internal.FD, info unsafe.Pointer, size uintptr) error {
	value, err := fd.Value()
	if err != nil {
		return err
	}

	// available from 4.13
	attr := bpfObjGetInfoByFDAttr{
		fd:      value,
		infoLen: uint32(size),
		info:    internal.NewPointer(info),
	}
	_, err = internal.BPF(internal.BPF_OBJ_GET_INFO_BY_FD, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	if err != nil {
		return fmt.Errorf("fd %d: %w", fd, err)
	}
	return nil
}

func bpfGetProgInfoByFD(fd *internal.FD) (*bpfProgInfo, error) {
	var info bpfProgInfo
	if err := bpfGetObjectInfoByFD(fd, unsafe.Pointer(&info), unsafe.Sizeof(info)); err != nil {
		return nil, fmt.Errorf("can't get program info: %w", err)
	}
	return &info, nil
}

func bpfGetMapInfoByFD(fd *internal.FD) (*bpfMapInfo, error) {
	var info bpfMapInfo
	err := bpfGetObjectInfoByFD(fd, unsafe.Pointer(&info), unsafe.Sizeof(info))
	if err != nil {
		return nil, fmt.Errorf("can't get map info: %w", err)
	}
	return &info, nil
}

var haveObjName = internal.FeatureTest("object names", "4.15", func() (bool, error) {
	attr := bpfMapCreateAttr{
		mapType:    Array,
		keySize:    4,
		valueSize:  4,
		maxEntries: 1,
		mapName:    newBPFObjName("feature_test"),
	}

	fd, err := bpfMapCreate(&attr)
	if err != nil {
		return false, nil
	}

	_ = fd.Close()
	return true, nil
})

var objNameAllowsDot = internal.FeatureTest("dot in object names", "5.2", func() (bool, error) {
	if err := haveObjName(); err != nil {
		return false, err
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
		return false, nil
	}

	_ = fd.Close()
	return true, nil
})

func bpfObjGetFDByID(cmd internal.BPFCmd, id uint32) (*internal.FD, error) {
	attr := bpfGetFDByIDAttr{
		id: id,
	}
	ptr, err := internal.BPF(cmd, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	return internal.NewFD(uint32(ptr)), wrapObjError(err)
}
