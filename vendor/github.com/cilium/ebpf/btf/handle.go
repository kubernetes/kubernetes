package btf

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"os"

	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/sys"
	"github.com/cilium/ebpf/internal/unix"
)

// Handle is a reference to BTF loaded into the kernel.
type Handle struct {
	fd *sys.FD

	// Size of the raw BTF in bytes.
	size uint32

	needsKernelBase bool
}

// NewHandle loads BTF into the kernel.
//
// Returns ErrNotSupported if BTF is not supported.
func NewHandle(spec *Spec) (*Handle, error) {
	if spec.byteOrder != nil && spec.byteOrder != internal.NativeEndian {
		return nil, fmt.Errorf("can't load %s BTF on %s", spec.byteOrder, internal.NativeEndian)
	}

	enc := newEncoder(kernelEncoderOptions, newStringTableBuilderFromTable(spec.strings))

	for _, typ := range spec.types {
		_, err := enc.Add(typ)
		if err != nil {
			return nil, fmt.Errorf("add %s: %w", typ, err)
		}
	}

	btf, err := enc.Encode()
	if err != nil {
		return nil, fmt.Errorf("marshal BTF: %w", err)
	}

	return newHandleFromRawBTF(btf)
}

func newHandleFromRawBTF(btf []byte) (*Handle, error) {
	if uint64(len(btf)) > math.MaxUint32 {
		return nil, errors.New("BTF exceeds the maximum size")
	}

	attr := &sys.BtfLoadAttr{
		Btf:     sys.NewSlicePointer(btf),
		BtfSize: uint32(len(btf)),
	}

	fd, err := sys.BtfLoad(attr)
	if err == nil {
		return &Handle{fd, attr.BtfSize, false}, nil
	}

	if err := haveBTF(); err != nil {
		return nil, err
	}

	logBuf := make([]byte, 64*1024)
	attr.BtfLogBuf = sys.NewSlicePointer(logBuf)
	attr.BtfLogSize = uint32(len(logBuf))
	attr.BtfLogLevel = 1

	// Up until at least kernel 6.0, the BTF verifier does not return ENOSPC
	// if there are other verification errors. ENOSPC is only returned when
	// the BTF blob is correct, a log was requested, and the provided buffer
	// is too small.
	_, ve := sys.BtfLoad(attr)
	return nil, internal.ErrorWithLog("load btf", err, logBuf, errors.Is(ve, unix.ENOSPC))
}

// NewHandleFromID returns the BTF handle for a given id.
//
// Prefer calling [ebpf.Program.Handle] or [ebpf.Map.Handle] if possible.
//
// Returns ErrNotExist, if there is no BTF with the given id.
//
// Requires CAP_SYS_ADMIN.
func NewHandleFromID(id ID) (*Handle, error) {
	fd, err := sys.BtfGetFdById(&sys.BtfGetFdByIdAttr{
		Id: uint32(id),
	})
	if err != nil {
		return nil, fmt.Errorf("get FD for ID %d: %w", id, err)
	}

	info, err := newHandleInfoFromFD(fd)
	if err != nil {
		_ = fd.Close()
		return nil, err
	}

	return &Handle{fd, info.size, info.IsModule()}, nil
}

// Spec parses the kernel BTF into Go types.
func (h *Handle) Spec() (*Spec, error) {
	var btfInfo sys.BtfInfo
	btfBuffer := make([]byte, h.size)
	btfInfo.Btf, btfInfo.BtfSize = sys.NewSlicePointerLen(btfBuffer)

	if err := sys.ObjInfo(h.fd, &btfInfo); err != nil {
		return nil, err
	}

	if !h.needsKernelBase {
		return loadRawSpec(bytes.NewReader(btfBuffer), internal.NativeEndian, nil, nil)
	}

	base, fallback, err := kernelSpec()
	if err != nil {
		return nil, fmt.Errorf("load BTF base: %w", err)
	}

	if fallback {
		return nil, fmt.Errorf("can't load split BTF without access to /sys")
	}

	return loadRawSpec(bytes.NewReader(btfBuffer), internal.NativeEndian, base.types, base.strings)
}

// Close destroys the handle.
//
// Subsequent calls to FD will return an invalid value.
func (h *Handle) Close() error {
	if h == nil {
		return nil
	}

	return h.fd.Close()
}

// FD returns the file descriptor for the handle.
func (h *Handle) FD() int {
	return h.fd.Int()
}

// Info returns metadata about the handle.
func (h *Handle) Info() (*HandleInfo, error) {
	return newHandleInfoFromFD(h.fd)
}

// HandleInfo describes a Handle.
type HandleInfo struct {
	// ID of this handle in the kernel. The ID is only valid as long as the
	// associated handle is kept alive.
	ID ID

	// Name is an identifying name for the BTF, currently only used by the
	// kernel.
	Name string

	// IsKernel is true if the BTF originated with the kernel and not
	// userspace.
	IsKernel bool

	// Size of the raw BTF in bytes.
	size uint32
}

func newHandleInfoFromFD(fd *sys.FD) (*HandleInfo, error) {
	// We invoke the syscall once with a empty BTF and name buffers to get size
	// information to allocate buffers. Then we invoke it a second time with
	// buffers to receive the data.
	var btfInfo sys.BtfInfo
	if err := sys.ObjInfo(fd, &btfInfo); err != nil {
		return nil, fmt.Errorf("get BTF info for fd %s: %w", fd, err)
	}

	if btfInfo.NameLen > 0 {
		// NameLen doesn't account for the terminating NUL.
		btfInfo.NameLen++
	}

	// Don't pull raw BTF by default, since it may be quite large.
	btfSize := btfInfo.BtfSize
	btfInfo.BtfSize = 0

	nameBuffer := make([]byte, btfInfo.NameLen)
	btfInfo.Name, btfInfo.NameLen = sys.NewSlicePointerLen(nameBuffer)
	if err := sys.ObjInfo(fd, &btfInfo); err != nil {
		return nil, err
	}

	return &HandleInfo{
		ID:       ID(btfInfo.Id),
		Name:     unix.ByteSliceToString(nameBuffer),
		IsKernel: btfInfo.KernelBtf != 0,
		size:     btfSize,
	}, nil
}

// IsModule returns true if the BTF is for the kernel itself.
func (i *HandleInfo) IsVmlinux() bool {
	return i.IsKernel && i.Name == "vmlinux"
}

// IsModule returns true if the BTF is for a kernel module.
func (i *HandleInfo) IsModule() bool {
	return i.IsKernel && i.Name != "vmlinux"
}

// HandleIterator allows enumerating BTF blobs loaded into the kernel.
type HandleIterator struct {
	// The ID of the current handle. Only valid after a call to Next.
	ID ID
	// The current Handle. Only valid until a call to Next.
	// See Take if you want to retain the handle.
	Handle *Handle
	err    error
}

// Next retrieves a handle for the next BTF object.
//
// Returns true if another BTF object was found. Call [HandleIterator.Err] after
// the function returns false.
func (it *HandleIterator) Next() bool {
	id := it.ID
	for {
		attr := &sys.BtfGetNextIdAttr{Id: id}
		err := sys.BtfGetNextId(attr)
		if errors.Is(err, os.ErrNotExist) {
			// There are no more BTF objects.
			break
		} else if err != nil {
			it.err = fmt.Errorf("get next BTF ID: %w", err)
			break
		}

		id = attr.NextId
		handle, err := NewHandleFromID(id)
		if errors.Is(err, os.ErrNotExist) {
			// Try again with the next ID.
			continue
		} else if err != nil {
			it.err = fmt.Errorf("retrieve handle for ID %d: %w", id, err)
			break
		}

		it.Handle.Close()
		it.ID, it.Handle = id, handle
		return true
	}

	// No more handles or we encountered an error.
	it.Handle.Close()
	it.Handle = nil
	return false
}

// Take the ownership of the current handle.
//
// It's the callers responsibility to close the handle.
func (it *HandleIterator) Take() *Handle {
	handle := it.Handle
	it.Handle = nil
	return handle
}

// Err returns an error if iteration failed for some reason.
func (it *HandleIterator) Err() error {
	return it.err
}

// FindHandle returns the first handle for which predicate returns true.
//
// Requires CAP_SYS_ADMIN.
//
// Returns an error wrapping ErrNotFound if predicate never returns true or if
// there is no BTF loaded into the kernel.
func FindHandle(predicate func(info *HandleInfo) bool) (*Handle, error) {
	it := new(HandleIterator)
	defer it.Handle.Close()

	for it.Next() {
		info, err := it.Handle.Info()
		if err != nil {
			return nil, fmt.Errorf("info for ID %d: %w", it.ID, err)
		}

		if predicate(info) {
			return it.Take(), nil
		}
	}
	if err := it.Err(); err != nil {
		return nil, fmt.Errorf("iterate handles: %w", err)
	}

	return nil, fmt.Errorf("find handle: %w", ErrNotFound)
}
