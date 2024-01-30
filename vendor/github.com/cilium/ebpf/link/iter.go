package link

import (
	"fmt"
	"io"
	"unsafe"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/internal/sys"
)

type IterOptions struct {
	// Program must be of type Tracing with attach type
	// AttachTraceIter. The kind of iterator to attach to is
	// determined at load time via the AttachTo field.
	//
	// AttachTo requires the kernel to include BTF of itself,
	// and it to be compiled with a recent pahole (>= 1.16).
	Program *ebpf.Program

	// Map specifies the target map for bpf_map_elem and sockmap iterators.
	// It may be nil.
	Map *ebpf.Map
}

// AttachIter attaches a BPF seq_file iterator.
func AttachIter(opts IterOptions) (*Iter, error) {
	if err := haveBPFLink(); err != nil {
		return nil, err
	}

	progFd := opts.Program.FD()
	if progFd < 0 {
		return nil, fmt.Errorf("invalid program: %s", sys.ErrClosedFd)
	}

	var info bpfIterLinkInfoMap
	if opts.Map != nil {
		mapFd := opts.Map.FD()
		if mapFd < 0 {
			return nil, fmt.Errorf("invalid map: %w", sys.ErrClosedFd)
		}
		info.map_fd = uint32(mapFd)
	}

	attr := sys.LinkCreateIterAttr{
		ProgFd:      uint32(progFd),
		AttachType:  sys.AttachType(ebpf.AttachTraceIter),
		IterInfo:    sys.NewPointer(unsafe.Pointer(&info)),
		IterInfoLen: uint32(unsafe.Sizeof(info)),
	}

	fd, err := sys.LinkCreateIter(&attr)
	if err != nil {
		return nil, fmt.Errorf("can't link iterator: %w", err)
	}

	return &Iter{RawLink{fd, ""}}, err
}

// Iter represents an attached bpf_iter.
type Iter struct {
	RawLink
}

// Open creates a new instance of the iterator.
//
// Reading from the returned reader triggers the BPF program.
func (it *Iter) Open() (io.ReadCloser, error) {
	attr := &sys.IterCreateAttr{
		LinkFd: it.fd.Uint(),
	}

	fd, err := sys.IterCreate(attr)
	if err != nil {
		return nil, fmt.Errorf("can't create iterator: %w", err)
	}

	return fd.File("bpf_iter"), nil
}

// union bpf_iter_link_info.map
type bpfIterLinkInfoMap struct {
	map_fd uint32
}
