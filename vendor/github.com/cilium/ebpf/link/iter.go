package link

import (
	"fmt"
	"io"
	"unsafe"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/internal"
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
		return nil, fmt.Errorf("invalid program: %s", internal.ErrClosedFd)
	}

	var info bpfIterLinkInfoMap
	if opts.Map != nil {
		mapFd := opts.Map.FD()
		if mapFd < 0 {
			return nil, fmt.Errorf("invalid map: %w", internal.ErrClosedFd)
		}
		info.map_fd = uint32(mapFd)
	}

	attr := bpfLinkCreateIterAttr{
		prog_fd:       uint32(progFd),
		attach_type:   ebpf.AttachTraceIter,
		iter_info:     internal.NewPointer(unsafe.Pointer(&info)),
		iter_info_len: uint32(unsafe.Sizeof(info)),
	}

	fd, err := bpfLinkCreateIter(&attr)
	if err != nil {
		return nil, fmt.Errorf("can't link iterator: %w", err)
	}

	return &Iter{RawLink{fd, ""}}, err
}

// LoadPinnedIter loads a pinned iterator from a bpffs.
func LoadPinnedIter(fileName string, opts *ebpf.LoadPinOptions) (*Iter, error) {
	link, err := LoadPinnedRawLink(fileName, IterType, opts)
	if err != nil {
		return nil, err
	}

	return &Iter{*link}, err
}

// Iter represents an attached bpf_iter.
type Iter struct {
	RawLink
}

// Open creates a new instance of the iterator.
//
// Reading from the returned reader triggers the BPF program.
func (it *Iter) Open() (io.ReadCloser, error) {
	linkFd, err := it.fd.Value()
	if err != nil {
		return nil, err
	}

	attr := &bpfIterCreateAttr{
		linkFd: linkFd,
	}

	fd, err := bpfIterCreate(attr)
	if err != nil {
		return nil, fmt.Errorf("can't create iterator: %w", err)
	}

	return fd.File("bpf_iter"), nil
}

// union bpf_iter_link_info.map
type bpfIterLinkInfoMap struct {
	map_fd uint32
}
