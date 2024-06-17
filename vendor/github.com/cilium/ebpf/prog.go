package ebpf

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/cilium/ebpf/asm"
	"github.com/cilium/ebpf/btf"
	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/sys"
	"github.com/cilium/ebpf/internal/unix"
)

// ErrNotSupported is returned whenever the kernel doesn't support a feature.
var ErrNotSupported = internal.ErrNotSupported

// ProgramID represents the unique ID of an eBPF program.
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
	// Type information used for CO-RE relocations and when attaching to
	// kernel functions.
	//
	// This is useful in environments where the kernel BTF is not available
	// (containers) or where it is in a non-standard location. Defaults to
	// use the kernel BTF from a well-known location if nil.
	KernelTypes *btf.Spec
}

// ProgramSpec defines a Program.
type ProgramSpec struct {
	// Name is passed to the kernel as a debug aid. Must only contain
	// alpha numeric and '_' characters.
	Name string

	// Type determines at which hook in the kernel a program will run.
	Type ProgramType

	// AttachType of the program, needed to differentiate allowed context
	// accesses in some newer program types like CGroupSockAddr.
	//
	// Available on kernels 4.17 and later.
	AttachType AttachType

	// Name of a kernel data structure or function to attach to. Its
	// interpretation depends on Type and AttachType.
	AttachTo string

	// The program to attach to. Must be provided manually.
	AttachTarget *Program

	// The name of the ELF section this program orininated from.
	SectionName string

	Instructions asm.Instructions

	// Flags is passed to the kernel and specifies additional program
	// load attributes.
	Flags uint32

	// License of the program. Some helpers are only available if
	// the license is deemed compatible with the GPL.
	//
	// See https://www.kernel.org/doc/html/latest/process/license-rules.html#id1
	License string

	// Version used by Kprobe programs.
	//
	// Deprecated on kernels 5.0 and later. Leave empty to let the library
	// detect this value automatically.
	KernelVersion uint32

	// The BTF associated with this program. Changing Instructions
	// will most likely invalidate the contained data, and may
	// result in errors when attempting to load it into the kernel.
	BTF *btf.Spec

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

// Tag calculates the kernel tag for a series of instructions.
//
// Use asm.Instructions.Tag if you need to calculate for non-native endianness.
func (ps *ProgramSpec) Tag() (string, error) {
	return ps.Instructions.Tag(internal.NativeEndian)
}

type VerifierError = internal.VerifierError

// Program represents BPF program loaded into the kernel.
//
// It is not safe to close a Program which is used by other goroutines.
type Program struct {
	// Contains the output of the kernel verifier if enabled,
	// otherwise it is empty.
	VerifierLog string

	fd         *sys.FD
	name       string
	pinnedPath string
	typ        ProgramType
}

// NewProgram creates a new Program.
//
// See NewProgramWithOptions for details.
func NewProgram(spec *ProgramSpec) (*Program, error) {
	return NewProgramWithOptions(spec, ProgramOptions{})
}

// NewProgramWithOptions creates a new Program.
//
// Loading a program for the first time will perform
// feature detection by loading small, temporary programs.
//
// Returns an error wrapping VerifierError if the program or its BTF is rejected
// by the kernel.
func NewProgramWithOptions(spec *ProgramSpec, opts ProgramOptions) (*Program, error) {
	if spec == nil {
		return nil, errors.New("can't load a program from a nil spec")
	}

	handles := newHandleCache()
	defer handles.close()

	prog, err := newProgramWithOptions(spec, opts, handles)
	if errors.Is(err, asm.ErrUnsatisfiedMapReference) {
		return nil, fmt.Errorf("cannot load program without loading its whole collection: %w", err)
	}
	return prog, err
}

func newProgramWithOptions(spec *ProgramSpec, opts ProgramOptions, handles *handleCache) (*Program, error) {
	if len(spec.Instructions) == 0 {
		return nil, errors.New("instructions cannot be empty")
	}

	if spec.Type == UnspecifiedProgram {
		return nil, errors.New("can't load program of unspecified type")
	}

	if spec.ByteOrder != nil && spec.ByteOrder != internal.NativeEndian {
		return nil, fmt.Errorf("can't load %s program on %s", spec.ByteOrder, internal.NativeEndian)
	}

	// Kernels before 5.0 (6c4fc209fcf9 "bpf: remove useless version check for prog load")
	// require the version field to be set to the value of the KERNEL_VERSION
	// macro for kprobe-type programs.
	// Overwrite Kprobe program version if set to zero or the magic version constant.
	kv := spec.KernelVersion
	if spec.Type == Kprobe && (kv == 0 || kv == internal.MagicKernelVersion) {
		v, err := internal.KernelVersion()
		if err != nil {
			return nil, fmt.Errorf("detecting kernel version: %w", err)
		}
		kv = v.Kernel()
	}

	attr := &sys.ProgLoadAttr{
		ProgType:           sys.ProgType(spec.Type),
		ProgFlags:          spec.Flags,
		ExpectedAttachType: sys.AttachType(spec.AttachType),
		License:            sys.NewStringPointer(spec.License),
		KernVersion:        kv,
	}

	if haveObjName() == nil {
		attr.ProgName = sys.NewObjName(spec.Name)
	}

	kernelTypes := opts.KernelTypes

	insns := make(asm.Instructions, len(spec.Instructions))
	copy(insns, spec.Instructions)

	var btfDisabled bool
	if spec.BTF != nil {
		if err := applyRelocations(insns, spec.BTF, kernelTypes); err != nil {
			return nil, fmt.Errorf("apply CO-RE relocations: %w", err)
		}

		handle, err := handles.btfHandle(spec.BTF)
		btfDisabled = errors.Is(err, btf.ErrNotSupported)
		if err != nil && !btfDisabled {
			return nil, fmt.Errorf("load BTF: %w", err)
		}

		if handle != nil {
			attr.ProgBtfFd = uint32(handle.FD())

			fib, lib, err := btf.MarshalExtInfos(insns, spec.BTF.TypeID)
			if err != nil {
				return nil, err
			}

			attr.FuncInfoRecSize = btf.FuncInfoSize
			attr.FuncInfoCnt = uint32(len(fib)) / btf.FuncInfoSize
			attr.FuncInfo = sys.NewSlicePointer(fib)

			attr.LineInfoRecSize = btf.LineInfoSize
			attr.LineInfoCnt = uint32(len(lib)) / btf.LineInfoSize
			attr.LineInfo = sys.NewSlicePointer(lib)
		}
	}

	if err := fixupAndValidate(insns); err != nil {
		return nil, err
	}

	buf := bytes.NewBuffer(make([]byte, 0, insns.Size()))
	err := insns.Marshal(buf, internal.NativeEndian)
	if err != nil {
		return nil, err
	}

	bytecode := buf.Bytes()
	attr.Insns = sys.NewSlicePointer(bytecode)
	attr.InsnCnt = uint32(len(bytecode) / asm.InstructionSize)

	if spec.AttachTarget != nil {
		targetID, err := findTargetInProgram(spec.AttachTarget, spec.AttachTo, spec.Type, spec.AttachType)
		if err != nil {
			return nil, fmt.Errorf("attach %s/%s: %w", spec.Type, spec.AttachType, err)
		}

		attr.AttachBtfId = uint32(targetID)
		attr.AttachProgFd = uint32(spec.AttachTarget.FD())
		defer runtime.KeepAlive(spec.AttachTarget)
	} else if spec.AttachTo != "" {
		targetID, err := findTargetInKernel(kernelTypes, spec.AttachTo, spec.Type, spec.AttachType)
		if err != nil && !errors.Is(err, errUnrecognizedAttachType) {
			// We ignore errUnrecognizedAttachType since AttachTo may be non-empty
			// for programs that don't attach anywhere.
			return nil, fmt.Errorf("attach %s/%s: %w", spec.Type, spec.AttachType, err)
		}

		attr.AttachBtfId = uint32(targetID)
	}

	logSize := DefaultVerifierLogSize
	if opts.LogSize > 0 {
		logSize = opts.LogSize
	}

	var logBuf []byte
	if opts.LogLevel > 0 {
		logBuf = make([]byte, logSize)
		attr.LogLevel = opts.LogLevel
		attr.LogSize = uint32(len(logBuf))
		attr.LogBuf = sys.NewSlicePointer(logBuf)
	}

	fd, err := sys.ProgLoad(attr)
	if err == nil {
		return &Program{unix.ByteSliceToString(logBuf), fd, spec.Name, "", spec.Type}, nil
	}

	if opts.LogLevel == 0 && opts.LogSize >= 0 {
		// Re-run with the verifier enabled to get better error messages.
		logBuf = make([]byte, logSize)
		attr.LogLevel = 1
		attr.LogSize = uint32(len(logBuf))
		attr.LogBuf = sys.NewSlicePointer(logBuf)
		_, _ = sys.ProgLoad(attr)
	}

	switch {
	case errors.Is(err, unix.EPERM):
		if len(logBuf) > 0 && logBuf[0] == 0 {
			// EPERM due to RLIMIT_MEMLOCK happens before the verifier, so we can
			// check that the log is empty to reduce false positives.
			return nil, fmt.Errorf("load program: %w (MEMLOCK may be too low, consider rlimit.RemoveMemlock)", err)
		}

		fallthrough

	case errors.Is(err, unix.EINVAL):
		if hasFunctionReferences(spec.Instructions) {
			if err := haveBPFToBPFCalls(); err != nil {
				return nil, fmt.Errorf("load program: %w", err)
			}
		}
	}

	err = internal.ErrorWithLog(err, logBuf)
	if btfDisabled {
		return nil, fmt.Errorf("load program: %w (BTF disabled)", err)
	}
	return nil, fmt.Errorf("load program: %w", err)
}

// NewProgramFromFD creates a program from a raw fd.
//
// You should not use fd after calling this function.
//
// Requires at least Linux 4.10.
func NewProgramFromFD(fd int) (*Program, error) {
	f, err := sys.NewFD(fd)
	if err != nil {
		return nil, err
	}

	return newProgramFromFD(f)
}

// NewProgramFromID returns the program for a given id.
//
// Returns ErrNotExist, if there is no eBPF program with the given id.
func NewProgramFromID(id ProgramID) (*Program, error) {
	fd, err := sys.ProgGetFdById(&sys.ProgGetFdByIdAttr{
		Id: uint32(id),
	})
	if err != nil {
		return nil, fmt.Errorf("get program by id: %w", err)
	}

	return newProgramFromFD(fd)
}

func newProgramFromFD(fd *sys.FD) (*Program, error) {
	info, err := newProgramInfoFromFd(fd)
	if err != nil {
		fd.Close()
		return nil, fmt.Errorf("discover program type: %w", err)
	}

	return &Program{"", fd, "", "", info.Type}, nil
}

func (p *Program) String() string {
	if p.name != "" {
		return fmt.Sprintf("%s(%s)#%v", p.typ, p.name, p.fd)
	}
	return fmt.Sprintf("%s(%v)", p.typ, p.fd)
}

// Type returns the underlying type of the program.
func (p *Program) Type() ProgramType {
	return p.typ
}

// Info returns metadata about the program.
//
// Requires at least 4.10.
func (p *Program) Info() (*ProgramInfo, error) {
	return newProgramInfoFromFd(p.fd)
}

// Handle returns a reference to the program's type information in the kernel.
//
// Returns ErrNotSupported if the kernel has no BTF support, or if there is no
// BTF associated with the program.
func (p *Program) Handle() (*btf.Handle, error) {
	info, err := p.Info()
	if err != nil {
		return nil, err
	}

	id, ok := info.BTFID()
	if !ok {
		return nil, fmt.Errorf("program %s: retrieve BTF ID: %w", p, ErrNotSupported)
	}

	return btf.NewHandleFromID(id)
}

// FD gets the file descriptor of the Program.
//
// It is invalid to call this function after Close has been called.
func (p *Program) FD() int {
	return p.fd.Int()
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

	return &Program{p.VerifierLog, dup, p.name, "", p.typ}, nil
}

// Pin persists the Program on the BPF virtual file system past the lifetime of
// the process that created it
//
// Calling Pin on a previously pinned program will overwrite the path, except when
// the new path already exists. Re-pinning across filesystems is not supported.
//
// This requires bpffs to be mounted above fileName. See https://docs.cilium.io/en/k8s-doc/admin/#admin-mount-bpffs
func (p *Program) Pin(fileName string) error {
	if err := internal.Pin(p.pinnedPath, fileName, p.fd); err != nil {
		return err
	}
	p.pinnedPath = fileName
	return nil
}

// Unpin removes the persisted state for the Program from the BPF virtual filesystem.
//
// Failed calls to Unpin will not alter the state returned by IsPinned.
//
// Unpinning an unpinned Program returns nil.
func (p *Program) Unpin() error {
	if err := internal.Unpin(p.pinnedPath); err != nil {
		return err
	}
	p.pinnedPath = ""
	return nil
}

// IsPinned returns true if the Program has a non-empty pinned path.
func (p *Program) IsPinned() bool {
	return p.pinnedPath != ""
}

// Close the Program's underlying file descriptor, which could unload
// the program from the kernel if it is not pinned or attached to a
// kernel hook.
func (p *Program) Close() error {
	if p == nil {
		return nil
	}

	return p.fd.Close()
}

// Various options for Run'ing a Program
type RunOptions struct {
	// Program's data input. Required field.
	Data []byte
	// Program's data after Program has run. Caller must allocate. Optional field.
	DataOut []byte
	// Program's context input. Optional field.
	Context interface{}
	// Program's context after Program has run. Must be a pointer or slice. Optional field.
	ContextOut interface{}
	// Number of times to run Program. Optional field. Defaults to 1.
	Repeat uint32
	// Optional flags.
	Flags uint32
	// CPU to run Program on. Optional field.
	// Note not all program types support this field.
	CPU uint32
	// Called whenever the syscall is interrupted, and should be set to testing.B.ResetTimer
	// or similar. Typically used during benchmarking. Optional field.
	Reset func()
}

// Test runs the Program in the kernel with the given input and returns the
// value returned by the eBPF program. outLen may be zero.
//
// Note: the kernel expects at least 14 bytes input for an ethernet header for
// XDP and SKB programs.
//
// This function requires at least Linux 4.12.
func (p *Program) Test(in []byte) (uint32, []byte, error) {
	// Older kernels ignore the dataSizeOut argument when copying to user space.
	// Combined with things like bpf_xdp_adjust_head() we don't really know what the final
	// size will be. Hence we allocate an output buffer which we hope will always be large
	// enough, and panic if the kernel wrote past the end of the allocation.
	// See https://patchwork.ozlabs.org/cover/1006822/
	var out []byte
	if len(in) > 0 {
		out = make([]byte, len(in)+outputPad)
	}

	opts := RunOptions{
		Data:    in,
		DataOut: out,
		Repeat:  1,
	}

	ret, _, err := p.testRun(&opts)
	if err != nil {
		return ret, nil, fmt.Errorf("can't test program: %w", err)
	}
	return ret, opts.DataOut, nil
}

// Run runs the Program in kernel with given RunOptions.
//
// Note: the same restrictions from Test apply.
func (p *Program) Run(opts *RunOptions) (uint32, error) {
	ret, _, err := p.testRun(opts)
	if err != nil {
		return ret, fmt.Errorf("can't test program: %w", err)
	}
	return ret, nil
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
	if uint(repeat) > math.MaxUint32 {
		return 0, 0, fmt.Errorf("repeat is too high")
	}

	opts := RunOptions{
		Data:   in,
		Repeat: uint32(repeat),
		Reset:  reset,
	}

	ret, total, err := p.testRun(&opts)
	if err != nil {
		return ret, total, fmt.Errorf("can't benchmark program: %w", err)
	}
	return ret, total, nil
}

var haveProgTestRun = internal.FeatureTest("BPF_PROG_TEST_RUN", "4.12", func() error {
	prog, err := NewProgram(&ProgramSpec{
		// SocketFilter does not require privileges on newer kernels.
		Type: SocketFilter,
		Instructions: asm.Instructions{
			asm.LoadImm(asm.R0, 0, asm.DWord),
			asm.Return(),
		},
		License: "MIT",
	})
	if err != nil {
		// This may be because we lack sufficient permissions, etc.
		return err
	}
	defer prog.Close()

	// Programs require at least 14 bytes input
	in := make([]byte, 14)
	attr := sys.ProgRunAttr{
		ProgFd:     uint32(prog.FD()),
		DataSizeIn: uint32(len(in)),
		DataIn:     sys.NewSlicePointer(in),
	}

	err = sys.ProgRun(&attr)
	switch {
	case errors.Is(err, unix.EINVAL):
		// Check for EINVAL specifically, rather than err != nil since we
		// otherwise misdetect due to insufficient permissions.
		return internal.ErrNotSupported

	case errors.Is(err, unix.EINTR):
		// We know that PROG_TEST_RUN is supported if we get EINTR.
		return nil

	case errors.Is(err, unix.ENOTSUPP):
		// The first PROG_TEST_RUN patches shipped in 4.12 didn't include
		// a test runner for SocketFilter. ENOTSUPP means PROG_TEST_RUN is
		// supported, but not for the program type used in the probe.
		return nil
	}

	return err
})

func (p *Program) testRun(opts *RunOptions) (uint32, time.Duration, error) {
	if uint(len(opts.Data)) > math.MaxUint32 {
		return 0, 0, fmt.Errorf("input is too long")
	}

	if err := haveProgTestRun(); err != nil {
		return 0, 0, err
	}

	var ctxBytes []byte
	if opts.Context != nil {
		ctx := new(bytes.Buffer)
		if err := binary.Write(ctx, internal.NativeEndian, opts.Context); err != nil {
			return 0, 0, fmt.Errorf("cannot serialize context: %v", err)
		}
		ctxBytes = ctx.Bytes()
	}

	var ctxOut []byte
	if opts.ContextOut != nil {
		ctxOut = make([]byte, binary.Size(opts.ContextOut))
	}

	attr := sys.ProgRunAttr{
		ProgFd:      p.fd.Uint(),
		DataSizeIn:  uint32(len(opts.Data)),
		DataSizeOut: uint32(len(opts.DataOut)),
		DataIn:      sys.NewSlicePointer(opts.Data),
		DataOut:     sys.NewSlicePointer(opts.DataOut),
		Repeat:      uint32(opts.Repeat),
		CtxSizeIn:   uint32(len(ctxBytes)),
		CtxSizeOut:  uint32(len(ctxOut)),
		CtxIn:       sys.NewSlicePointer(ctxBytes),
		CtxOut:      sys.NewSlicePointer(ctxOut),
		Flags:       opts.Flags,
		Cpu:         opts.CPU,
	}

	for {
		err := sys.ProgRun(&attr)
		if err == nil {
			break
		}

		if errors.Is(err, unix.EINTR) {
			if opts.Reset != nil {
				opts.Reset()
			}
			continue
		}

		if errors.Is(err, unix.ENOTSUPP) {
			return 0, 0, fmt.Errorf("kernel doesn't support testing program type %s: %w", p.Type(), ErrNotSupported)
		}

		return 0, 0, fmt.Errorf("can't run test: %w", err)
	}

	if opts.DataOut != nil {
		if int(attr.DataSizeOut) > cap(opts.DataOut) {
			// Houston, we have a problem. The program created more data than we allocated,
			// and the kernel wrote past the end of our buffer.
			panic("kernel wrote past end of output buffer")
		}
		opts.DataOut = opts.DataOut[:int(attr.DataSizeOut)]
	}

	if len(ctxOut) != 0 {
		b := bytes.NewReader(ctxOut)
		if err := binary.Read(b, internal.NativeEndian, opts.ContextOut); err != nil {
			return 0, 0, fmt.Errorf("failed to decode ContextOut: %v", err)
		}
	}

	total := time.Duration(attr.Duration) * time.Nanosecond
	return attr.Retval, total, nil
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

func marshalProgram(p *Program, length int) ([]byte, error) {
	if length != 4 {
		return nil, fmt.Errorf("can't marshal program to %d bytes", length)
	}

	buf := make([]byte, 4)
	internal.NativeEndian.PutUint32(buf, p.fd.Uint())
	return buf, nil
}

// LoadPinnedProgram loads a Program from a BPF file.
//
// Requires at least Linux 4.11.
func LoadPinnedProgram(fileName string, opts *LoadPinOptions) (*Program, error) {
	fd, err := sys.ObjGet(&sys.ObjGetAttr{
		Pathname:  sys.NewStringPointer(fileName),
		FileFlags: opts.Marshal(),
	})
	if err != nil {
		return nil, err
	}

	info, err := newProgramInfoFromFd(fd)
	if err != nil {
		_ = fd.Close()
		return nil, fmt.Errorf("info for %s: %w", fileName, err)
	}

	return &Program{"", fd, filepath.Base(fileName), fileName, info.Type}, nil
}

// SanitizeName replaces all invalid characters in name with replacement.
// Passing a negative value for replacement will delete characters instead
// of replacing them. Use this to automatically generate valid names for maps
// and programs at runtime.
//
// The set of allowed characters depends on the running kernel version.
// Dots are only allowed as of kernel 5.2.
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
	attr := &sys.ProgGetNextIdAttr{Id: uint32(startID)}
	return ProgramID(attr.NextId), sys.ProgGetNextId(attr)
}

// BindMap binds map to the program and is only released once program is released.
//
// This may be used in cases where metadata should be associated with the program
// which otherwise does not contain any references to the map.
func (p *Program) BindMap(m *Map) error {
	attr := &sys.ProgBindMapAttr{
		ProgFd: uint32(p.FD()),
		MapFd:  uint32(m.FD()),
	}

	return sys.ProgBindMap(attr)
}

var errUnrecognizedAttachType = errors.New("unrecognized attach type")

// find an attach target type in the kernel.
//
// spec may be nil and defaults to the canonical kernel BTF. name together with
// progType and attachType determine which type we need to attach to.
//
// Returns errUnrecognizedAttachType.
func findTargetInKernel(spec *btf.Spec, name string, progType ProgramType, attachType AttachType) (btf.TypeID, error) {
	type match struct {
		p ProgramType
		a AttachType
	}

	var (
		typeName, featureName string
		isBTFTypeFunc         = true
	)

	switch (match{progType, attachType}) {
	case match{LSM, AttachLSMMac}:
		typeName = "bpf_lsm_" + name
		featureName = name + " LSM hook"
	case match{Tracing, AttachTraceIter}:
		typeName = "bpf_iter_" + name
		featureName = name + " iterator"
	case match{Tracing, AttachTraceFEntry}:
		typeName = name
		featureName = fmt.Sprintf("fentry %s", name)
	case match{Tracing, AttachTraceFExit}:
		typeName = name
		featureName = fmt.Sprintf("fexit %s", name)
	case match{Tracing, AttachModifyReturn}:
		typeName = name
		featureName = fmt.Sprintf("fmod_ret %s", name)
	case match{Tracing, AttachTraceRawTp}:
		typeName = fmt.Sprintf("btf_trace_%s", name)
		featureName = fmt.Sprintf("raw_tp %s", name)
		isBTFTypeFunc = false
	default:
		return 0, errUnrecognizedAttachType
	}

	spec, err := maybeLoadKernelBTF(spec)
	if err != nil {
		return 0, fmt.Errorf("load kernel spec: %w", err)
	}

	var target btf.Type
	if isBTFTypeFunc {
		var targetFunc *btf.Func
		err = spec.TypeByName(typeName, &targetFunc)
		target = targetFunc
	} else {
		var targetTypedef *btf.Typedef
		err = spec.TypeByName(typeName, &targetTypedef)
		target = targetTypedef
	}

	if err != nil {
		if errors.Is(err, btf.ErrNotFound) {
			return 0, &internal.UnsupportedFeatureError{
				Name: featureName,
			}
		}
		return 0, fmt.Errorf("find target for %s: %w", featureName, err)
	}

	return spec.TypeID(target)
}

// find an attach target type in a program.
//
// Returns errUnrecognizedAttachType.
func findTargetInProgram(prog *Program, name string, progType ProgramType, attachType AttachType) (btf.TypeID, error) {
	type match struct {
		p ProgramType
		a AttachType
	}

	var typeName string
	switch (match{progType, attachType}) {
	case match{Extension, AttachNone}:
		typeName = name
	default:
		return 0, errUnrecognizedAttachType
	}

	btfHandle, err := prog.Handle()
	if err != nil {
		return 0, fmt.Errorf("load target BTF: %w", err)
	}
	defer btfHandle.Close()

	spec, err := btfHandle.Spec(nil)
	if err != nil {
		return 0, err
	}

	var targetFunc *btf.Func
	err = spec.TypeByName(typeName, &targetFunc)
	if err != nil {
		return 0, fmt.Errorf("find target %s: %w", typeName, err)
	}

	return spec.TypeID(targetFunc)
}
