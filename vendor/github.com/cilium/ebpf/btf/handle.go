package btf

import (
	"errors"
	"fmt"
	"os"

	"github.com/cilium/ebpf/internal/sys"
	"github.com/cilium/ebpf/internal/unix"
)

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
	// The ID of the last retrieved handle. Only valid after a call to Next.
	ID  ID
	err error
}

// Next retrieves a handle for the next BTF blob.
//
// [Handle.Close] is called if *handle is non-nil to avoid leaking fds.
//
// Returns true if another BTF blob was found. Call [HandleIterator.Err] after
// the function returns false.
func (it *HandleIterator) Next(handle **Handle) bool {
	if *handle != nil {
		(*handle).Close()
		*handle = nil
	}

	id := it.ID
	for {
		attr := &sys.BtfGetNextIdAttr{Id: id}
		err := sys.BtfGetNextId(attr)
		if errors.Is(err, os.ErrNotExist) {
			// There are no more BTF objects.
			return false
		} else if err != nil {
			it.err = fmt.Errorf("get next BTF ID: %w", err)
			return false
		}

		id = attr.NextId
		*handle, err = NewHandleFromID(id)
		if errors.Is(err, os.ErrNotExist) {
			// Try again with the next ID.
			continue
		} else if err != nil {
			it.err = fmt.Errorf("retrieve handle for ID %d: %w", id, err)
			return false
		}

		it.ID = id
		return true
	}
}

// Err returns an error if iteration failed for some reason.
func (it *HandleIterator) Err() error {
	return it.err
}
