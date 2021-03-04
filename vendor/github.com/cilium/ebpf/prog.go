package ebpf

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/cilium/ebpf/asm"
	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/btf"
	"github.com/cilium/ebpf/internal/unix"
)

// ErrNotSupported is returned whenever the kernel doesn't support a feature.
var ErrNotSupported = internal.ErrNotSupported

// ProgramID represents the unique ID of an eBPF program
type ProgramID uint32

const (
	// Number of bytes to pad the output buffer for BPF_PROG_TEST_RUN.
	// This is currently the maximum of spare space allocated for SKB
	// and XDP programs, and equal to XDP_PACKET_HEADROOM + NET_IP_ALIGN.
	outputPad = 256 + 2
)

// DefaultVerifierLogSize is the default number of bytes allocated for the
// verifier log.
const DefaultVerifierLogSize = 64 * 1024

// ProgramOptions control loading a program into the kernel.
type ProgramOptions struct {
	// Controls the detail emitted by the kernel verifier. Set to non-zero
	// to enable logging.
	LogLevel uint32
	// Controls the output buffer size for the verifier. Defaults to
	// DefaultVerifierLogSize.
	LogSize int
}

// ProgramSpec defines a Program
type ProgramSpec struct {
	// Name is passed to the kernel as a debug aid. Must only contain
	// alpha numeric and '_' characters.
	Name string
	// Type determines at which hook in the kernel a program will run.
	Type       ProgramType
	AttachType AttachType
	// Name of a kernel data structure to attach to. It's interpretation
	// depends on Type and AttachType.
	AttachTo     string
	Instructions asm.Instructions

	// License of the program. Some helpers are only available if
	// the license is deemed compatible with the GPL.
	//
	// See https://www.kernel.org/doc/html/latest/process/license-rules.html#id1
	License string

	// Version used by tracing programs.
	//
	// Deprecated: superseded by BTF.
	KernelVersion uint32

	// The BTF associated with this program. Changing Instructions
	// will most likely invalidate the contained data, and may
	// result in errors when attempting to load it into the kernel.
	BTF *btf.Program

	// The byte order this program was compiled for, may be nil.
	ByteOrder binary.ByteOrder
}

// Copy returns a copy of the spec.
func (ps *ProgramSpec) Copy() *ProgramSpec {
	if ps == nil {
		return nil
	}

	cpy := *ps
	cpy.Instructions = make(asm.Instructions, len(ps.Instructions))
	copy(cpy.Instructions, ps.Instructions)
	return &cpy
}

// Program represents BPF program loaded into the kernel.
//
// It is not safe to close a Program which is used by other goroutines.
type Program struct {
	// Contains the output of the kernel verifier if enabled,
	// otherwise it is empty.
	VerifierLog string

	fd         *internal.FD
	name       string
	abi        ProgramABI
	attachType AttachType
}

// NewProgram creates a new Program.
//
// Loading a program for the first time will perform
// feature detection by loading small, temporary programs.
func NewProgram(spec *ProgramSpec) (*Program, error) {
	return NewProgramWithOptions(spec, ProgramOptions{})
}

// NewProgramWithOptions creates a new Program.
//
// Loading a program for the first time will perform
// feature detection by loading small, temporary programs.
func NewProgramWithOptions(spec *ProgramSpec, opts ProgramOptions) (*Program, error) {
	if spec.BTF == nil {
		return newProgramWithBTF(spec, nil, opts)
	}

	handle, err := btf.NewHandle(btf.ProgramSpec(spec.BTF))
	if err != nil && !errors.Is(err, btf.ErrNotSupported) {
		return nil, fmt.Errorf("can't load BTF: %w", err)
	}

	return newProgramWithBTF(spec, handle, opts)
}

func newProgramWithBTF(spec *ProgramSpec, btf *btf.Handle, opts ProgramOptions) (*Program, error) {
	attr, err := convertProgramSpec(spec, btf)
	if err != nil {
		return nil, err
	}

	logSize := DefaultVerifierLogSize
	if opts.LogSize > 0 {
		logSize = opts.LogSize
	}

	var logBuf []byte
	if opts.LogLevel > 0 {
		logBuf = make([]byte, logSize)
		attr.logLevel = opts.LogLevel
		attr.logSize = uint32(len(logBuf))
		attr.logBuf = internal.NewSlicePointer(logBuf)
	}

	fd, err := bpfProgLoad(attr)
	if err == nil {
		prog := newProgram(fd, spec.Name, &ProgramABI{spec.Type})
		prog.VerifierLog = internal.CString(logBuf)
		return prog, nil
	}

	logErr := err
	if opts.LogLevel == 0 {
		// Re-run with the verifier enabled to get better error messages.
		logBuf = make([]byte, logSize)
		attr.logLevel = 1
		attr.logSize = uint32(len(logBuf))
		attr.logBuf = internal.NewSlicePointer(logBuf)

		_, logErr = bpfProgLoad(attr)
	}

	err = internal.ErrorWithLog(err, logBuf, logErr)
	return nil, fmt.Errorf("can't load program: %w", err)
}

// NewProgramFromFD creates a program from a raw fd.
//
// You should not use fd after calling this function.
//
// Requires at least Linux 4.11.
func NewProgramFromFD(fd int) (*Program, error) {
	if fd < 0 {
		return nil, errors.New("invalid fd")
	}
	bpfFd := internal.NewFD(uint32(fd))

	name, abi, err := newProgramABIFromFd(bpfFd)
	if err != nil {
		bpfFd.Forget()
		return nil, err
	}

	return newProgram(bpfFd, name, abi), nil
}

func newProgram(fd *internal.FD, name string, abi *ProgramABI) *Program {
	return &Program{
		name: name,
		fd:   fd,
		abi:  *abi,
	}
}

func convertProgramSpec(spec *ProgramSpec, handle *btf.Handle) (*bpfProgLoadAttr, error) {
	if len(spec.Instructions) == 0 {
		return nil, errors.New("Instructions cannot be empty")
	}

	if len(spec.License) == 0 {
		return nil, errors.New("License cannot be empty")
	}

	if spec.ByteOrder != nil && spec.ByteOrder != internal.NativeEndian {
		return nil, fmt.Errorf("can't load %s program on %s", spec.ByteOrder, internal.NativeEndian)
	}

	buf := bytes.NewBuffer(make([]byte, 0, len(spec.Instructions)*asm.InstructionSize))
	err := spec.Instructions.Marshal(buf, internal.NativeEndian)
	if err != nil {
		return nil, err
	}

	bytecode := buf.Bytes()
	insCount := uint32(len(bytecode) / asm.InstructionSize)
	attr := &bpfProgLoadAttr{
		progType:           spec.Type,
		expectedAttachType: spec.AttachType,
		insCount:           insCount,
		instructions:       internal.NewSlicePointer(bytecode),
		license:            internal.NewStringPointer(spec.License),
		kernelVersion:      spec.KernelVersion,
	}

	if haveObjName() == nil {
		attr.progName = newBPFObjName(spec.Name)
	}

	if handle != nil && spec.BTF != nil {
		attr.progBTFFd = uint32(handle.FD())

		recSize, bytes, err := btf.ProgramLineInfos(spec.BTF)
		if err != nil {
			return nil, fmt.Errorf("can't get BTF line infos: %w", err)
		}
		attr.lineInfoRecSize = recSize
		attr.lineInfoCnt = uint32(uint64(len(bytes)) / uint64(recSize))
		attr.lineInfo = internal.NewSlicePointer(bytes)

		recSize, bytes, err = btf.ProgramFuncInfos(spec.BTF)
		if err != nil {
			return nil, fmt.Errorf("can't get BTF function infos: %w", err)
		}
		attr.funcInfoRecSize = recSize
		attr.funcInfoCnt = uint32(uint64(len(bytes)) / uint64(recSize))
		attr.funcInfo = internal.NewSlicePointer(bytes)
	}

	if spec.AttachTo != "" {
		target, err := resolveBTFType(spec.AttachTo, spec.Type, spec.AttachType)
		if err != nil {
			return nil, err
		}
		if target != nil {
			attr.attachBTFID = target.ID()
		}
	}

	return attr, nil
}

func (p *Program) String() string {
	if p.name != "" {
		return fmt.Sprintf("%s(%s)#%v", p.abi.Type, p.name, p.fd)
	}
	return fmt.Sprintf("%s#%v", p.abi.Type, p.fd)
}

// ABI gets the ABI of the Program
func (p *Program) ABI() ProgramABI {
	return p.abi
}

// FD gets the file descriptor of the Program.
//
// It is invalid to call this function after Close has been called.
func (p *Program) FD() int {
	fd, err := p.fd.Value()
	if err != nil {
		// Best effort: -1 is the number most likely to be an
		// invalid file descriptor.
		return -1
	}

	return int(fd)
}

// Clone creates a duplicate of the Program.
//
// Closing the duplicate does not affect the original, and vice versa.
//
// Cloning a nil Program returns nil.
func (p *Program) Clone() (*Program, error) {
	if p == nil {
		return nil, nil
	}

	dup, err := p.fd.Dup()
	if err != nil {
		return nil, fmt.Errorf("can't clone program: %w", err)
	}

	return newProgram(dup, p.name, &p.abi), nil
}

// Pin persists the Program past the lifetime of the process that created it
//
// This requires bpffs to be mounted above fileName. See http://cilium.readthedocs.io/en/doc-1.0/kubernetes/install/#mounting-the-bpf-fs-optional
func (p *Program) Pin(fileName string) error {
	if err := internal.BPFObjPin(fileName, p.fd); err != nil {
		return fmt.Errorf("can't pin program: %w", err)
	}
	return nil
}

// Close unloads the program from the kernel.
func (p *Program) Close() error {
	if p == nil {
		return nil
	}

	return p.fd.Close()
}

// Test runs the Program in the kernel with the given input and returns the
// value returned by the eBPF program. outLen may be zero.
//
// Note: the kernel expects at least 14 bytes input for an ethernet header for
// XDP and SKB programs.
//
// This function requires at least Linux 4.12.
func (p *Program) Test(in []byte) (uint32, []byte, error) {
	ret, out, _, err := p.testRun(in, 1, nil)
	if err != nil {
		return ret, nil, fmt.Errorf("can't test program: %w", err)
	}
	return ret, out, nil
}

// Benchmark runs the Program with the given input for a number of times
// and returns the time taken per iteration.
//
// Returns the result of the last execution of the program and the time per
// run or an error. reset is called whenever the benchmark syscall is
// interrupted, and should be set to testing.B.ResetTimer or similar.
//
// Note: profiling a call to this function will skew it's results, see
// https://github.com/cilium/ebpf/issues/24
//
// This function requires at least Linux 4.12.
func (p *Program) Benchmark(in []byte, repeat int, reset func()) (uint32, time.Duration, error) {
	ret, _, total, err := p.testRun(in, repeat, reset)
	if err != nil {
		return ret, total, fmt.Errorf("can't benchmark program: %w", err)
	}
	return ret, total, nil
}

var haveProgTestRun = internal.FeatureTest("BPF_PROG_TEST_RUN", "4.12", func() (bool, error) {
	prog, err := NewProgram(&ProgramSpec{
		Type: SocketFilter,
		Instructions: asm.Instructions{
			asm.LoadImm(asm.R0, 0, asm.DWord),
			asm.Return(),
		},
		License: "MIT",
	})
	if err != nil {
		// This may be because we lack sufficient permissions, etc.
		return false, err
	}
	defer prog.Close()

	// Programs require at least 14 bytes input
	in := make([]byte, 14)
	attr := bpfProgTestRunAttr{
		fd:         uint32(prog.FD()),
		dataSizeIn: uint32(len(in)),
		dataIn:     internal.NewSlicePointer(in),
	}

	err = bpfProgTestRun(&attr)

	// Check for EINVAL specifically, rather than err != nil since we
	// otherwise misdetect due to insufficient permissions.
	return !errors.Is(err, unix.EINVAL), nil
})

func (p *Program) testRun(in []byte, repeat int, reset func()) (uint32, []byte, time.Duration, error) {
	if uint(repeat) > math.MaxUint32 {
		return 0, nil, 0, fmt.Errorf("repeat is too high")
	}

	if len(in) == 0 {
		return 0, nil, 0, fmt.Errorf("missing input")
	}

	if uint(len(in)) > math.MaxUint32 {
		return 0, nil, 0, fmt.Errorf("input is too long")
	}

	if err := haveProgTestRun(); err != nil {
		return 0, nil, 0, err
	}

	// Older kernels ignore the dataSizeOut argument when copying to user space.
	// Combined with things like bpf_xdp_adjust_head() we don't really know what the final
	// size will be. Hence we allocate an output buffer which we hope will always be large
	// enough, and panic if the kernel wrote past the end of the allocation.
	// See https://patchwork.ozlabs.org/cover/1006822/
	out := make([]byte, len(in)+outputPad)

	fd, err := p.fd.Value()
	if err != nil {
		return 0, nil, 0, err
	}

	attr := bpfProgTestRunAttr{
		fd:          fd,
		dataSizeIn:  uint32(len(in)),
		dataSizeOut: uint32(len(out)),
		dataIn:      internal.NewSlicePointer(in),
		dataOut:     internal.NewSlicePointer(out),
		repeat:      uint32(repeat),
	}

	for {
		err = bpfProgTestRun(&attr)
		if err == nil {
			break
		}

		if errors.Is(err, unix.EINTR) {
			if reset != nil {
				reset()
			}
			continue
		}

		return 0, nil, 0, fmt.Errorf("can't run test: %w", err)
	}

	if int(attr.dataSizeOut) > cap(out) {
		// Houston, we have a problem. The program created more data than we allocated,
		// and the kernel wrote past the end of our buffer.
		panic("kernel wrote past end of output buffer")
	}
	out = out[:int(attr.dataSizeOut)]

	total := time.Duration(attr.duration) * time.Nanosecond
	return attr.retval, out, total, nil
}

func unmarshalProgram(buf []byte) (*Program, error) {
	if len(buf) != 4 {
		return nil, errors.New("program id requires 4 byte value")
	}

	// Looking up an entry in a nested map or prog array returns an id,
	// not an fd.
	id := internal.NativeEndian.Uint32(buf)
	return NewProgramFromID(ProgramID(id))
}

// MarshalBinary implements BinaryMarshaler.
func (p *Program) MarshalBinary() ([]byte, error) {
	value, err := p.fd.Value()
	if err != nil {
		return nil, err
	}

	buf := make([]byte, 4)
	internal.NativeEndian.PutUint32(buf, value)
	return buf, nil
}

// Attach a Program.
//
// Deprecated: use link.RawAttachProgram instead.
func (p *Program) Attach(fd int, typ AttachType, flags AttachFlags) error {
	if fd < 0 {
		return errors.New("invalid fd")
	}

	pfd, err := p.fd.Value()
	if err != nil {
		return err
	}

	attr := internal.BPFProgAttachAttr{
		TargetFd:    uint32(fd),
		AttachBpfFd: pfd,
		AttachType:  uint32(typ),
		AttachFlags: uint32(flags),
	}

	return internal.BPFProgAttach(&attr)
}

// Detach a Program.
//
// Deprecated: use link.RawDetachProgram instead.
func (p *Program) Detach(fd int, typ AttachType, flags AttachFlags) error {
	if fd < 0 {
		return errors.New("invalid fd")
	}

	if flags != 0 {
		return errors.New("flags must be zero")
	}

	pfd, err := p.fd.Value()
	if err != nil {
		return err
	}

	attr := internal.BPFProgDetachAttr{
		TargetFd:    uint32(fd),
		AttachBpfFd: pfd,
		AttachType:  uint32(typ),
	}

	return internal.BPFProgDetach(&attr)
}

// LoadPinnedProgram loads a Program from a BPF file.
//
// Requires at least Linux 4.11.
func LoadPinnedProgram(fileName string) (*Program, error) {
	fd, err := internal.BPFObjGet(fileName)
	if err != nil {
		return nil, err
	}

	name, abi, err := newProgramABIFromFd(fd)
	if err != nil {
		_ = fd.Close()
		return nil, fmt.Errorf("can't get ABI for %s: %w", fileName, err)
	}

	return newProgram(fd, name, abi), nil
}

// SanitizeName replaces all invalid characters in name.
//
// Use this to automatically generate valid names for maps and
// programs at run time.
//
// Passing a negative value for replacement will delete characters
// instead of replacing them.
func SanitizeName(name string, replacement rune) string {
	return strings.Map(func(char rune) rune {
		if invalidBPFObjNameChar(char) {
			return replacement
		}
		return char
	}, name)
}

// ProgramGetNextID returns the ID of the next eBPF program.
//
// Returns ErrNotExist, if there is no next eBPF program.
func ProgramGetNextID(startID ProgramID) (ProgramID, error) {
	id, err := objGetNextID(internal.BPF_PROG_GET_NEXT_ID, uint32(startID))
	return ProgramID(id), err
}

// NewProgramFromID returns the program for a given id.
//
// Returns ErrNotExist, if there is no eBPF program with the given id.
func NewProgramFromID(id ProgramID) (*Program, error) {
	fd, err := bpfObjGetFDByID(internal.BPF_PROG_GET_FD_BY_ID, uint32(id))
	if err != nil {
		return nil, err
	}

	name, abi, err := newProgramABIFromFd(fd)
	if err != nil {
		_ = fd.Close()
		return nil, err
	}

	return newProgram(fd, name, abi), nil
}

// ID returns the systemwide unique ID of the program.
func (p *Program) ID() (ProgramID, error) {
	info, err := bpfGetProgInfoByFD(p.fd)
	if err != nil {
		return ProgramID(0), err
	}
	return ProgramID(info.id), nil
}

func resolveBTFType(name string, progType ProgramType, attachType AttachType) (btf.Type, error) {
	kernel, err := btf.LoadKernelSpec()
	if err != nil {
		return nil, fmt.Errorf("can't resolve BTF type %s: %w", name, err)
	}

	type match struct {
		p ProgramType
		a AttachType
	}

	target := match{progType, attachType}
	switch target {
	case match{Tracing, AttachTraceIter}:
		var target btf.Func
		if err := kernel.FindType("bpf_iter_"+name, &target); err != nil {
			return nil, fmt.Errorf("can't resolve BTF for iterator %s: %w", name, err)
		}

		return &target, nil

	default:
		return nil, nil
	}
}
