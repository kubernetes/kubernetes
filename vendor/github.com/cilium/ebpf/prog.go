package ebpf

import (
	"bytes"
	"fmt"
	"math"
	"path/filepath"
	"strings"
	"time"
	"unsafe"

	"github.com/cilium/ebpf/asm"
	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/unix"

	"github.com/pkg/errors"
)

var (
	errNotSupported = errors.New("ebpf: not supported by kernel")
)

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
	Name          string
	Type          ProgramType
	AttachType    AttachType
	Instructions  asm.Instructions
	License       string
	KernelVersion uint32
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

	fd   *bpfFD
	name string
	abi  ProgramABI
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
	attr, err := convertProgramSpec(spec, haveObjName.Result())
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
		attr.logBuf = newPtr(unsafe.Pointer(&logBuf[0]))
	}

	fd, err := bpfProgLoad(attr)
	if err == nil {
		prog := newProgram(fd, spec.Name, &ProgramABI{spec.Type})
		prog.VerifierLog = convertCString(logBuf)
		return prog, nil
	}

	truncated := errors.Cause(err) == unix.ENOSPC
	if opts.LogLevel == 0 {
		// Re-run with the verifier enabled to get better error messages.
		logBuf = make([]byte, logSize)
		attr.logLevel = 1
		attr.logSize = uint32(len(logBuf))
		attr.logBuf = newPtr(unsafe.Pointer(&logBuf[0]))

		_, nerr := bpfProgLoad(attr)
		truncated = errors.Cause(nerr) == unix.ENOSPC
	}

	logs := convertCString(logBuf)
	if truncated {
		logs += "\n(truncated...)"
	}

	return nil, &loadError{err, logs}
}

// NewProgramFromFD creates a program from a raw fd.
//
// You should not use fd after calling this function.
func NewProgramFromFD(fd int) (*Program, error) {
	if fd < 0 {
		return nil, errors.New("invalid fd")
	}
	bpfFd := newBPFFD(uint32(fd))

	info, err := bpfGetProgInfoByFD(bpfFd)
	if err != nil {
		bpfFd.forget()
		return nil, err
	}

	var name string
	if bpfName := convertCString(info.name[:]); bpfName != "" {
		name = bpfName
	} else {
		name = convertCString(info.tag[:])
	}

	return newProgram(bpfFd, name, newProgramABIFromInfo(info)), nil
}

func newProgram(fd *bpfFD, name string, abi *ProgramABI) *Program {
	return &Program{
		name: name,
		fd:   fd,
		abi:  *abi,
	}
}

func convertProgramSpec(spec *ProgramSpec, includeName bool) (*bpfProgLoadAttr, error) {
	if len(spec.Instructions) == 0 {
		return nil, errors.New("Instructions cannot be empty")
	}

	if len(spec.License) == 0 {
		return nil, errors.New("License cannot be empty")
	}

	buf := bytes.NewBuffer(make([]byte, 0, len(spec.Instructions)*asm.InstructionSize))
	err := spec.Instructions.Marshal(buf, internal.NativeEndian)
	if err != nil {
		return nil, err
	}

	bytecode := buf.Bytes()
	insCount := uint32(len(bytecode) / asm.InstructionSize)
	lic := []byte(spec.License)
	attr := &bpfProgLoadAttr{
		progType:           spec.Type,
		expectedAttachType: spec.AttachType,
		insCount:           insCount,
		instructions:       newPtr(unsafe.Pointer(&bytecode[0])),
		license:            newPtr(unsafe.Pointer(&lic[0])),
	}

	name, err := newBPFObjName(spec.Name)
	if err != nil {
		return nil, err
	}

	if includeName {
		attr.progName = name
	}

	return attr, nil
}

func (p *Program) String() string {
	if p.name != "" {
		return fmt.Sprintf("%s(%s)#%s", p.abi.Type, p.name, p.fd)
	}
	return fmt.Sprintf("%s#%s", p.abi.Type, p.fd)
}

// ABI gets the ABI of the Program
func (p *Program) ABI() ProgramABI {
	return p.abi
}

// FD gets the file descriptor of the Program.
//
// It is invalid to call this function after Close has been called.
func (p *Program) FD() int {
	fd, err := p.fd.value()
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

	dup, err := p.fd.dup()
	if err != nil {
		return nil, errors.Wrap(err, "can't clone program")
	}

	return newProgram(dup, p.name, &p.abi), nil
}

// Pin persists the Program past the lifetime of the process that created it
//
// This requires bpffs to be mounted above fileName. See http://cilium.readthedocs.io/en/doc-1.0/kubernetes/install/#mounting-the-bpf-fs-optional
func (p *Program) Pin(fileName string) error {
	return errors.Wrap(bpfPinObject(fileName, p.fd), "can't pin program")
}

// Close unloads the program from the kernel.
func (p *Program) Close() error {
	if p == nil {
		return nil
	}

	return p.fd.close()
}

// Test runs the Program in the kernel with the given input and returns the
// value returned by the eBPF program. outLen may be zero.
//
// Note: the kernel expects at least 14 bytes input for an ethernet header for
// XDP and SKB programs.
//
// This function requires at least Linux 4.12.
func (p *Program) Test(in []byte) (uint32, []byte, error) {
	ret, out, _, err := p.testRun(in, 1)
	return ret, out, err
}

// Benchmark runs the Program with the given input for a number of times
// and returns the time taken per iteration.
//
// The returned value is the return value of the last execution of
// the program.
//
// This function requires at least Linux 4.12.
func (p *Program) Benchmark(in []byte, repeat int) (uint32, time.Duration, error) {
	ret, _, total, err := p.testRun(in, repeat)
	return ret, total, err
}

var noProgTestRun = featureTest{
	Fn: func() bool {
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
			return false
		}
		defer prog.Close()

		fd, err := prog.fd.value()
		if err != nil {
			return false
		}

		// Programs require at least 14 bytes input
		in := make([]byte, 14)
		attr := bpfProgTestRunAttr{
			fd:         fd,
			dataSizeIn: uint32(len(in)),
			dataIn:     newPtr(unsafe.Pointer(&in[0])),
		}

		_, err = bpfCall(_ProgTestRun, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
		return errors.Cause(err) == unix.EINVAL
	},
}

func (p *Program) testRun(in []byte, repeat int) (uint32, []byte, time.Duration, error) {
	if uint(repeat) > math.MaxUint32 {
		return 0, nil, 0, fmt.Errorf("repeat is too high")
	}

	if len(in) == 0 {
		return 0, nil, 0, fmt.Errorf("missing input")
	}

	if uint(len(in)) > math.MaxUint32 {
		return 0, nil, 0, fmt.Errorf("input is too long")
	}

	if noProgTestRun.Result() {
		return 0, nil, 0, errNotSupported
	}

	// Older kernels ignore the dataSizeOut argument when copying to user space.
	// Combined with things like bpf_xdp_adjust_head() we don't really know what the final
	// size will be. Hence we allocate an output buffer which we hope will always be large
	// enough, and panic if the kernel wrote past the end of the allocation.
	// See https://patchwork.ozlabs.org/cover/1006822/
	out := make([]byte, len(in)+outputPad)

	fd, err := p.fd.value()
	if err != nil {
		return 0, nil, 0, err
	}

	attr := bpfProgTestRunAttr{
		fd:          fd,
		dataSizeIn:  uint32(len(in)),
		dataSizeOut: uint32(len(out)),
		dataIn:      newPtr(unsafe.Pointer(&in[0])),
		dataOut:     newPtr(unsafe.Pointer(&out[0])),
		repeat:      uint32(repeat),
	}

	_, err = bpfCall(_ProgTestRun, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	if err != nil {
		return 0, nil, 0, errors.Wrap(err, "can't run test")
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
	fd, err := bpfGetProgramFDByID(id)
	if err != nil {
		return nil, err
	}

	abi, err := newProgramABIFromFd(fd)
	if err != nil {
		_ = fd.close()
		return nil, err
	}

	return newProgram(fd, "", abi), nil
}

// MarshalBinary implements BinaryMarshaler.
func (p *Program) MarshalBinary() ([]byte, error) {
	value, err := p.fd.value()
	if err != nil {
		return nil, err
	}

	buf := make([]byte, 4)
	internal.NativeEndian.PutUint32(buf, value)
	return buf, nil
}

// Attach a Program to a container object fd
func (p *Program) Attach(fd int, typ AttachType, flags AttachFlags) error {
	if fd < 0 {
		return errors.New("invalid fd")
	}

	pfd, err := p.fd.value()
	if err != nil {
		return err
	}

	attr := bpfProgAlterAttr{
		targetFd:    uint32(fd),
		attachBpfFd: pfd,
		attachType:  uint32(typ),
		attachFlags: uint32(flags),
	}

	return bpfProgAlter(_ProgAttach, &attr)
}

// Detach a Program from a container object fd
func (p *Program) Detach(fd int, typ AttachType, flags AttachFlags) error {
	if fd < 0 {
		return errors.New("invalid fd")
	}

	pfd, err := p.fd.value()
	if err != nil {
		return err
	}

	attr := bpfProgAlterAttr{
		targetFd:    uint32(fd),
		attachBpfFd: pfd,
		attachType:  uint32(typ),
		attachFlags: uint32(flags),
	}

	return bpfProgAlter(_ProgDetach, &attr)
}

// LoadPinnedProgram loads a Program from a BPF file.
//
// Requires at least Linux 4.13, use LoadPinnedProgramExplicit on
// earlier versions.
func LoadPinnedProgram(fileName string) (*Program, error) {
	fd, err := bpfGetObject(fileName)
	if err != nil {
		return nil, err
	}

	abi, err := newProgramABIFromFd(fd)
	if err != nil {
		_ = fd.close()
		return nil, err
	}

	return newProgram(fd, filepath.Base(fileName), abi), nil
}

// LoadPinnedProgramExplicit loads a program with explicit parameters.
func LoadPinnedProgramExplicit(fileName string, abi *ProgramABI) (*Program, error) {
	fd, err := bpfGetObject(fileName)
	if err != nil {
		return nil, err
	}

	return newProgram(fd, filepath.Base(fileName), abi), nil
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

type loadError struct {
	cause       error
	verifierLog string
}

func (le *loadError) Error() string {
	if le.verifierLog == "" {
		return fmt.Sprintf("failed to load program: %s", le.cause)
	}
	return fmt.Sprintf("failed to load program: %s: %s", le.cause, le.verifierLog)
}

func (le *loadError) Cause() error {
	return le.cause
}

// IsNotSupported returns true if an error occurred because
// the kernel does not have support for a specific feature.
func IsNotSupported(err error) bool {
	return errors.Cause(err) == errNotSupported
}
