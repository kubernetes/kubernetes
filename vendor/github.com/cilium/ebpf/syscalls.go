package ebpf

import (
	"bytes"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"unsafe"

	"github.com/cilium/ebpf/internal/unix"

	"github.com/pkg/errors"
)

var errClosedFd = errors.New("use of closed file descriptor")

type bpfFD struct {
	raw int64
}

func newBPFFD(value uint32) *bpfFD {
	fd := &bpfFD{int64(value)}
	runtime.SetFinalizer(fd, (*bpfFD).close)
	return fd
}

func (fd *bpfFD) String() string {
	return strconv.FormatInt(fd.raw, 10)
}

func (fd *bpfFD) value() (uint32, error) {
	if fd.raw < 0 {
		return 0, errClosedFd
	}

	return uint32(fd.raw), nil
}

func (fd *bpfFD) close() error {
	if fd.raw < 0 {
		return nil
	}

	value := int(fd.raw)
	fd.raw = -1

	fd.forget()
	return unix.Close(value)
}

func (fd *bpfFD) forget() {
	runtime.SetFinalizer(fd, nil)
}

func (fd *bpfFD) dup() (*bpfFD, error) {
	if fd.raw < 0 {
		return nil, errClosedFd
	}

	dup, err := unix.FcntlInt(uintptr(fd.raw), unix.F_DUPFD_CLOEXEC, 0)
	if err != nil {
		return nil, errors.Wrap(err, "can't dup fd")
	}

	return newBPFFD(uint32(dup)), nil
}

// bpfObjName is a null-terminated string made up of
// 'A-Za-z0-9_' characters.
type bpfObjName [unix.BPF_OBJ_NAME_LEN]byte

// newBPFObjName truncates the result if it is too long.
func newBPFObjName(name string) (bpfObjName, error) {
	idx := strings.IndexFunc(name, invalidBPFObjNameChar)
	if idx != -1 {
		return bpfObjName{}, errors.Errorf("invalid character '%c' in name '%s'", name[idx], name)
	}

	var result bpfObjName
	copy(result[:unix.BPF_OBJ_NAME_LEN-1], name)
	return result, nil
}

func invalidBPFObjNameChar(char rune) bool {
	switch {
	case char >= 'A' && char <= 'Z':
		fallthrough
	case char >= 'a' && char <= 'z':
		fallthrough
	case char >= '0' && char <= '9':
		fallthrough
	case char == '_':
		return false
	default:
		return true
	}
}

type bpfMapCreateAttr struct {
	mapType    MapType
	keySize    uint32
	valueSize  uint32
	maxEntries uint32
	flags      uint32
	innerMapFd uint32     // since 4.12 56f668dfe00d
	numaNode   uint32     // since 4.14 96eabe7a40aa
	mapName    bpfObjName // since 4.15 ad5b177bd73f
}

type bpfMapOpAttr struct {
	mapFd   uint32
	padding uint32
	key     syscallPtr
	value   syscallPtr
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

type bpfPinObjAttr struct {
	fileName syscallPtr
	fd       uint32
	padding  uint32
}

type bpfProgLoadAttr struct {
	progType           ProgramType
	insCount           uint32
	instructions       syscallPtr
	license            syscallPtr
	logLevel           uint32
	logSize            uint32
	logBuf             syscallPtr
	kernelVersion      uint32     // since 4.1  2541517c32be
	progFlags          uint32     // since 4.11 e07b98d9bffe
	progName           bpfObjName // since 4.15 067cae47771c
	progIfIndex        uint32     // since 4.15 1f6f4cb7ba21
	expectedAttachType AttachType // since 4.17 5e43f899b03a
}

type bpfProgInfo struct {
	progType     uint32
	id           uint32
	tag          [unix.BPF_TAG_SIZE]byte
	jitedLen     uint32
	xlatedLen    uint32
	jited        syscallPtr
	xlated       syscallPtr
	loadTime     uint64 // since 4.15 cb4d2b3f03d8
	createdByUID uint32
	nrMapIDs     uint32
	mapIds       syscallPtr
	name         bpfObjName
}

type bpfProgTestRunAttr struct {
	fd          uint32
	retval      uint32
	dataSizeIn  uint32
	dataSizeOut uint32
	dataIn      syscallPtr
	dataOut     syscallPtr
	repeat      uint32
	duration    uint32
}

type bpfProgAlterAttr struct {
	targetFd    uint32
	attachBpfFd uint32
	attachType  uint32
	attachFlags uint32
}

type bpfObjGetInfoByFDAttr struct {
	fd      uint32
	infoLen uint32
	info    syscallPtr // May be either bpfMapInfo or bpfProgInfo
}

type bpfGetFDByIDAttr struct {
	id   uint32
	next uint32
}

func newPtr(ptr unsafe.Pointer) syscallPtr {
	return syscallPtr{ptr: ptr}
}

func bpfProgLoad(attr *bpfProgLoadAttr) (*bpfFD, error) {
	for {
		fd, err := bpfCall(_ProgLoad, unsafe.Pointer(attr), unsafe.Sizeof(*attr))
		// As of ~4.20 the verifier can be interrupted by a signal,
		// and returns EAGAIN in that case.
		if err == unix.EAGAIN {
			continue
		}

		if err != nil {
			return nil, err
		}

		return newBPFFD(uint32(fd)), nil
	}
}

func bpfProgAlter(cmd int, attr *bpfProgAlterAttr) error {
	_, err := bpfCall(cmd, unsafe.Pointer(attr), unsafe.Sizeof(*attr))
	return err
}

func bpfMapCreate(attr *bpfMapCreateAttr) (*bpfFD, error) {
	fd, err := bpfCall(_MapCreate, unsafe.Pointer(attr), unsafe.Sizeof(*attr))
	if err != nil {
		return nil, err
	}

	return newBPFFD(uint32(fd)), nil
}

func bpfMapLookupElem(m *bpfFD, key, valueOut syscallPtr) error {
	fd, err := m.value()
	if err != nil {
		return err
	}

	attr := bpfMapOpAttr{
		mapFd: fd,
		key:   key,
		value: valueOut,
	}
	_, err = bpfCall(_MapLookupElem, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	return err
}

func bpfMapUpdateElem(m *bpfFD, key, valueOut syscallPtr, flags uint64) error {
	fd, err := m.value()
	if err != nil {
		return err
	}

	attr := bpfMapOpAttr{
		mapFd: fd,
		key:   key,
		value: valueOut,
		flags: flags,
	}
	_, err = bpfCall(_MapUpdateElem, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	return err
}

func bpfMapDeleteElem(m *bpfFD, key syscallPtr) error {
	fd, err := m.value()
	if err != nil {
		return err
	}

	attr := bpfMapOpAttr{
		mapFd: fd,
		key:   key,
	}
	_, err = bpfCall(_MapDeleteElem, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	return err
}

func bpfMapGetNextKey(m *bpfFD, key, nextKeyOut syscallPtr) error {
	fd, err := m.value()
	if err != nil {
		return err
	}

	attr := bpfMapOpAttr{
		mapFd: fd,
		key:   key,
		value: nextKeyOut,
	}
	_, err = bpfCall(_MapGetNextKey, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	return err
}

const bpfFSType = 0xcafe4a11

func bpfPinObject(fileName string, fd *bpfFD) error {
	dirName := filepath.Dir(fileName)
	var statfs unix.Statfs_t
	if err := unix.Statfs(dirName, &statfs); err != nil {
		return err
	}
	if uint64(statfs.Type) != bpfFSType {
		return errors.Errorf("%s is not on a bpf filesystem", fileName)
	}

	value, err := fd.value()
	if err != nil {
		return err
	}

	_, err = bpfCall(_ObjPin, unsafe.Pointer(&bpfPinObjAttr{
		fileName: newPtr(unsafe.Pointer(&[]byte(fileName)[0])),
		fd:       value,
	}), 16)
	return errors.Wrapf(err, "pin object %s", fileName)
}

func bpfGetObject(fileName string) (*bpfFD, error) {
	ptr, err := bpfCall(_ObjGet, unsafe.Pointer(&bpfPinObjAttr{
		fileName: newPtr(unsafe.Pointer(&[]byte(fileName)[0])),
	}), 16)
	if err != nil {
		return nil, errors.Wrapf(err, "get object %s", fileName)
	}
	return newBPFFD(uint32(ptr)), nil
}

func bpfGetObjectInfoByFD(fd *bpfFD, info unsafe.Pointer, size uintptr) error {
	value, err := fd.value()
	if err != nil {
		return err
	}

	// available from 4.13
	attr := bpfObjGetInfoByFDAttr{
		fd:      value,
		infoLen: uint32(size),
		info:    newPtr(info),
	}
	_, err = bpfCall(_ObjGetInfoByFD, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	return errors.Wrapf(err, "fd %d", value)
}

func bpfGetProgInfoByFD(fd *bpfFD) (*bpfProgInfo, error) {
	var info bpfProgInfo
	err := bpfGetObjectInfoByFD(fd, unsafe.Pointer(&info), unsafe.Sizeof(info))
	return &info, errors.Wrap(err, "can't get program info")
}

func bpfGetMapInfoByFD(fd *bpfFD) (*bpfMapInfo, error) {
	var info bpfMapInfo
	err := bpfGetObjectInfoByFD(fd, unsafe.Pointer(&info), unsafe.Sizeof(info))
	return &info, errors.Wrap(err, "can't get map info:")
}

var haveObjName = featureTest{
	Fn: func() bool {
		name, err := newBPFObjName("feature_test")
		if err != nil {
			// This really is a fatal error, but it should be caught
			// by the unit tests not working.
			return false
		}

		attr := bpfMapCreateAttr{
			mapType:    Array,
			keySize:    4,
			valueSize:  4,
			maxEntries: 1,
			mapName:    name,
		}

		fd, err := bpfMapCreate(&attr)
		if err != nil {
			return false
		}

		_ = fd.close()
		return true
	},
}

func bpfGetMapFDByID(id uint32) (*bpfFD, error) {
	// available from 4.13
	attr := bpfGetFDByIDAttr{
		id: id,
	}
	ptr, err := bpfCall(_MapGetFDByID, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	if err != nil {
		return nil, errors.Wrapf(err, "can't get fd for map id %d", id)
	}
	return newBPFFD(uint32(ptr)), nil
}

func bpfGetProgramFDByID(id uint32) (*bpfFD, error) {
	// available from 4.13
	attr := bpfGetFDByIDAttr{
		id: id,
	}
	ptr, err := bpfCall(_ProgGetFDByID, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	if err != nil {
		return nil, errors.Wrapf(err, "can't get fd for program id %d", id)
	}
	return newBPFFD(uint32(ptr)), nil
}

func bpfCall(cmd int, attr unsafe.Pointer, size uintptr) (uintptr, error) {
	r1, _, errNo := unix.Syscall(unix.SYS_BPF, uintptr(cmd), uintptr(attr), size)
	runtime.KeepAlive(attr)

	var err error
	if errNo != 0 {
		err = errNo
	}

	return r1, err
}

func convertCString(in []byte) string {
	inLen := bytes.IndexByte(in, 0)
	if inLen == -1 {
		return ""
	}
	return string(in[:inLen])
}
