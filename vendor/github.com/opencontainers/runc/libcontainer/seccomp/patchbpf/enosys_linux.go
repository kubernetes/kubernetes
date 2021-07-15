// +build linux,cgo,seccomp

package patchbpf

import (
	"bytes"
	"encoding/binary"
	"io"
	"os"
	"runtime"
	"unsafe"

	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/opencontainers/runc/libcontainer/utils"

	"github.com/pkg/errors"
	libseccomp "github.com/seccomp/libseccomp-golang"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/bpf"
	"golang.org/x/sys/unix"
)

// #cgo pkg-config: libseccomp
/*
#include <errno.h>
#include <stdint.h>
#include <seccomp.h>
#include <linux/seccomp.h>

const uint32_t C_ACT_ERRNO_ENOSYS = SCMP_ACT_ERRNO(ENOSYS);

// Copied from <linux/seccomp.h>.

#ifndef SECCOMP_SET_MODE_FILTER
#	define SECCOMP_SET_MODE_FILTER 1
#endif
const uintptr_t C_SET_MODE_FILTER = SECCOMP_SET_MODE_FILTER;

#ifndef SECCOMP_FILTER_FLAG_LOG
#	define SECCOMP_FILTER_FLAG_LOG (1UL << 1)
#endif
const uintptr_t C_FILTER_FLAG_LOG = SECCOMP_FILTER_FLAG_LOG;

// We use the AUDIT_ARCH_* values because those are the ones used by the kernel
// and SCMP_ARCH_* sometimes has fake values (such as SCMP_ARCH_X32). But we
// use <seccomp.h> so we get libseccomp's fallback definitions of AUDIT_ARCH_*.

const uint32_t C_AUDIT_ARCH_I386         = AUDIT_ARCH_I386;
const uint32_t C_AUDIT_ARCH_X86_64       = AUDIT_ARCH_X86_64;
const uint32_t C_AUDIT_ARCH_ARM          = AUDIT_ARCH_ARM;
const uint32_t C_AUDIT_ARCH_AARCH64      = AUDIT_ARCH_AARCH64;
const uint32_t C_AUDIT_ARCH_MIPS         = AUDIT_ARCH_MIPS;
const uint32_t C_AUDIT_ARCH_MIPS64       = AUDIT_ARCH_MIPS64;
const uint32_t C_AUDIT_ARCH_MIPS64N32    = AUDIT_ARCH_MIPS64N32;
const uint32_t C_AUDIT_ARCH_MIPSEL       = AUDIT_ARCH_MIPSEL;
const uint32_t C_AUDIT_ARCH_MIPSEL64     = AUDIT_ARCH_MIPSEL64;
const uint32_t C_AUDIT_ARCH_MIPSEL64N32  = AUDIT_ARCH_MIPSEL64N32;
const uint32_t C_AUDIT_ARCH_PPC          = AUDIT_ARCH_PPC;
const uint32_t C_AUDIT_ARCH_PPC64        = AUDIT_ARCH_PPC64;
const uint32_t C_AUDIT_ARCH_PPC64LE      = AUDIT_ARCH_PPC64LE;
const uint32_t C_AUDIT_ARCH_S390         = AUDIT_ARCH_S390;
const uint32_t C_AUDIT_ARCH_S390X        = AUDIT_ARCH_S390X;
*/
import "C"

var retErrnoEnosys = uint32(C.C_ACT_ERRNO_ENOSYS)

func isAllowAction(action configs.Action) bool {
	switch action {
	// Trace is considered an "allow" action because a good tracer should
	// support future syscalls (by handling -ENOSYS on its own), and giving
	// -ENOSYS will be disruptive for emulation.
	case configs.Allow, configs.Log, configs.Trace:
		return true
	default:
		return false
	}
}

func parseProgram(rdr io.Reader) ([]bpf.RawInstruction, error) {
	var program []bpf.RawInstruction
loop:
	for {
		// Read the next instruction. We have to use NativeEndian because
		// seccomp_export_bpf outputs the program in *host* endian-ness.
		var insn unix.SockFilter
		if err := binary.Read(rdr, utils.NativeEndian, &insn); err != nil {
			switch err {
			case io.EOF:
				// Parsing complete.
				break loop
			case io.ErrUnexpectedEOF:
				// Parsing stopped mid-instruction.
				return nil, errors.Wrap(err, "program parsing halted mid-instruction")
			default:
				// All other errors.
				return nil, errors.Wrap(err, "parsing instructions")
			}
		}
		program = append(program, bpf.RawInstruction{
			Op: insn.Code,
			Jt: insn.Jt,
			Jf: insn.Jf,
			K:  insn.K,
		})
	}
	return program, nil
}

func disassembleFilter(filter *libseccomp.ScmpFilter) ([]bpf.Instruction, error) {
	rdr, wtr, err := os.Pipe()
	if err != nil {
		return nil, errors.Wrap(err, "creating scratch pipe")
	}
	defer wtr.Close()
	defer rdr.Close()

	readerBuffer := new(bytes.Buffer)
	errChan := make(chan error, 1)
	go func() {
		_, err := io.Copy(readerBuffer, rdr)
		errChan <- err
		close(errChan)
	}()

	if err := filter.ExportBPF(wtr); err != nil {
		return nil, errors.Wrap(err, "exporting BPF")
	}
	// Close so that the reader actually gets EOF.
	_ = wtr.Close()

	if copyErr := <-errChan; copyErr != nil {
		return nil, errors.Wrap(copyErr, "reading from ExportBPF pipe")
	}

	// Parse the instructions.
	rawProgram, err := parseProgram(readerBuffer)
	if err != nil {
		return nil, errors.Wrap(err, "parsing generated BPF filter")
	}
	program, ok := bpf.Disassemble(rawProgram)
	if !ok {
		return nil, errors.Errorf("could not disassemble entire BPF filter")
	}
	return program, nil
}

type nativeArch uint32

const invalidArch nativeArch = 0

func archToNative(arch libseccomp.ScmpArch) (nativeArch, error) {
	switch arch {
	case libseccomp.ArchNative:
		// Convert to actual native architecture.
		arch, err := libseccomp.GetNativeArch()
		if err != nil {
			return invalidArch, errors.Wrap(err, "get native arch")
		}
		return archToNative(arch)
	case libseccomp.ArchX86:
		return nativeArch(C.C_AUDIT_ARCH_I386), nil
	case libseccomp.ArchAMD64, libseccomp.ArchX32:
		// NOTE: x32 is treated like x86_64 except all x32 syscalls have the
		//       30th bit of the syscall number set to indicate that it's not a
		//       normal x86_64 syscall.
		return nativeArch(C.C_AUDIT_ARCH_X86_64), nil
	case libseccomp.ArchARM:
		return nativeArch(C.C_AUDIT_ARCH_ARM), nil
	case libseccomp.ArchARM64:
		return nativeArch(C.C_AUDIT_ARCH_AARCH64), nil
	case libseccomp.ArchMIPS:
		return nativeArch(C.C_AUDIT_ARCH_MIPS), nil
	case libseccomp.ArchMIPS64:
		return nativeArch(C.C_AUDIT_ARCH_MIPS64), nil
	case libseccomp.ArchMIPS64N32:
		return nativeArch(C.C_AUDIT_ARCH_MIPS64N32), nil
	case libseccomp.ArchMIPSEL:
		return nativeArch(C.C_AUDIT_ARCH_MIPSEL), nil
	case libseccomp.ArchMIPSEL64:
		return nativeArch(C.C_AUDIT_ARCH_MIPSEL64), nil
	case libseccomp.ArchMIPSEL64N32:
		return nativeArch(C.C_AUDIT_ARCH_MIPSEL64N32), nil
	case libseccomp.ArchPPC:
		return nativeArch(C.C_AUDIT_ARCH_PPC), nil
	case libseccomp.ArchPPC64:
		return nativeArch(C.C_AUDIT_ARCH_PPC64), nil
	case libseccomp.ArchPPC64LE:
		return nativeArch(C.C_AUDIT_ARCH_PPC64LE), nil
	case libseccomp.ArchS390:
		return nativeArch(C.C_AUDIT_ARCH_S390), nil
	case libseccomp.ArchS390X:
		return nativeArch(C.C_AUDIT_ARCH_S390X), nil
	default:
		return invalidArch, errors.Errorf("unknown architecture: %v", arch)
	}
}

type lastSyscallMap map[nativeArch]map[libseccomp.ScmpArch]libseccomp.ScmpSyscall

// Figure out largest syscall number referenced in the filter for each
// architecture. We will be generating code based on the native architecture
// representation, but SCMP_ARCH_X32 means we have to track cases where the
// same architecture has different largest syscalls based on the mode.
func findLastSyscalls(config *configs.Seccomp) (lastSyscallMap, error) {
	lastSyscalls := make(lastSyscallMap)
	// Only loop over architectures which are present in the filter. Any other
	// architectures will get the libseccomp bad architecture action anyway.
	for _, ociArch := range config.Architectures {
		arch, err := libseccomp.GetArchFromString(ociArch)
		if err != nil {
			return nil, errors.Wrap(err, "validating seccomp architecture")
		}

		// Map native architecture to a real architecture value to avoid
		// doubling-up the lastSyscall mapping.
		if arch == libseccomp.ArchNative {
			nativeArch, err := libseccomp.GetNativeArch()
			if err != nil {
				return nil, errors.Wrap(err, "get native arch")
			}
			arch = nativeArch
		}

		// Figure out native architecture representation of the architecture.
		nativeArch, err := archToNative(arch)
		if err != nil {
			return nil, errors.Wrapf(err, "cannot map architecture %v to AUDIT_ARCH_ constant", arch)
		}

		if _, ok := lastSyscalls[nativeArch]; !ok {
			lastSyscalls[nativeArch] = map[libseccomp.ScmpArch]libseccomp.ScmpSyscall{}
		}
		if _, ok := lastSyscalls[nativeArch][arch]; ok {
			// Because of ArchNative we may hit the same entry multiple times.
			// Just skip it if we've seen this (nativeArch, ScmpArch)
			// combination before.
			continue
		}

		// Find the largest syscall in the filter for this architecture.
		var largestSyscall libseccomp.ScmpSyscall
		for _, rule := range config.Syscalls {
			sysno, err := libseccomp.GetSyscallFromNameByArch(rule.Name, arch)
			if err != nil {
				// Ignore unknown syscalls.
				continue
			}
			if sysno > largestSyscall {
				largestSyscall = sysno
			}
		}
		if largestSyscall != 0 {
			lastSyscalls[nativeArch][arch] = largestSyscall
		} else {
			logrus.Warnf("could not find any syscalls for arch %s", ociArch)
			delete(lastSyscalls[nativeArch], arch)
		}
	}
	return lastSyscalls, nil
}

// FIXME FIXME FIXME
//
// This solution is less than ideal. In the future it would be great to have
// per-arch information about which syscalls were added in which kernel
// versions so we can create far more accurate filter rules (handling holes in
// the syscall table and determining -ENOSYS requirements based on kernel
// minimum version alone.
//
// This implementation can in principle cause issues with syscalls like
// close_range(2) which were added out-of-order in the syscall table between
// kernel releases.
func generateEnosysStub(lastSyscalls lastSyscallMap) ([]bpf.Instruction, error) {
	// A jump-table for each nativeArch used to generate the initial
	// conditional jumps -- measured from the *END* of the program so they
	// remain valid after prepending to the tail.
	archJumpTable := map[nativeArch]uint32{}

	// Generate our own -ENOSYS rules for each architecture. They have to be
	// generated in reverse (prepended to the tail of the program) because the
	// JumpIf jumps need to be computed from the end of the program.
	programTail := []bpf.Instruction{
		// Fall-through rules jump into the filter.
		bpf.Jump{Skip: 1},
		// Rules which jump to here get -ENOSYS.
		bpf.RetConstant{Val: retErrnoEnosys},
	}

	// Generate the syscall -ENOSYS rules.
	for nativeArch, maxSyscalls := range lastSyscalls {
		// The number of instructions from the tail of this section which need
		// to be jumped in order to reach the -ENOSYS return. If the section
		// does not jump, it will fall through to the actual filter.
		baseJumpEnosys := uint32(len(programTail) - 1)
		baseJumpFilter := baseJumpEnosys + 1

		// Add the load instruction for the syscall number -- we jump here
		// directly from the arch code so we need to do it here. Sadly we can't
		// share this code between architecture branches.
		section := []bpf.Instruction{
			// load [0]
			bpf.LoadAbsolute{Off: 0, Size: 4}, // NOTE: We assume sizeof(int) == 4.
		}

		switch len(maxSyscalls) {
		case 0:
			// No syscalls found for this arch -- skip it and move on.
			continue
		case 1:
			// Get the only syscall in the map.
			var sysno libseccomp.ScmpSyscall
			for _, no := range maxSyscalls {
				sysno = no
			}

			// The simplest case just boils down to a single jgt instruction,
			// with special handling if baseJumpEnosys is larger than 255 (and
			// thus a long jump is required).
			var sectionTail []bpf.Instruction
			if baseJumpEnosys+1 <= 255 {
				sectionTail = []bpf.Instruction{
					// jgt [syscall],[baseJumpEnosys+1]
					bpf.JumpIf{
						Cond:     bpf.JumpGreaterThan,
						Val:      uint32(sysno),
						SkipTrue: uint8(baseJumpEnosys + 1),
					},
					// ja [baseJumpFilter]
					bpf.Jump{Skip: baseJumpFilter},
				}
			} else {
				sectionTail = []bpf.Instruction{
					// jle [syscall],1
					bpf.JumpIf{Cond: bpf.JumpLessOrEqual, Val: uint32(sysno), SkipTrue: 1},
					// ja [baseJumpEnosys+1]
					bpf.Jump{Skip: baseJumpEnosys + 1},
					// ja [baseJumpFilter]
					bpf.Jump{Skip: baseJumpFilter},
				}
			}

			// If we're on x86 we need to add a check for x32 and if we're in
			// the wrong mode we jump over the section.
			if uint32(nativeArch) == uint32(C.C_AUDIT_ARCH_X86_64) {
				// Grab the only architecture in the map.
				var scmpArch libseccomp.ScmpArch
				for arch := range maxSyscalls {
					scmpArch = arch
				}

				// Generate a prefix to check the mode.
				switch scmpArch {
				case libseccomp.ArchAMD64:
					sectionTail = append([]bpf.Instruction{
						// jset (1<<30),[len(tail)-1]
						bpf.JumpIf{
							Cond:     bpf.JumpBitsSet,
							Val:      1 << 30,
							SkipTrue: uint8(len(sectionTail) - 1),
						},
					}, sectionTail...)
				case libseccomp.ArchX32:
					sectionTail = append([]bpf.Instruction{
						// jset (1<<30),0,[len(tail)-1]
						bpf.JumpIf{
							Cond:     bpf.JumpBitsNotSet,
							Val:      1 << 30,
							SkipTrue: uint8(len(sectionTail) - 1),
						},
					}, sectionTail...)
				default:
					return nil, errors.Errorf("unknown amd64 native architecture %#x", scmpArch)
				}
			}

			section = append(section, sectionTail...)
		case 2:
			// x32 and x86_64 are a unique case, we can't handle any others.
			if uint32(nativeArch) != uint32(C.C_AUDIT_ARCH_X86_64) {
				return nil, errors.Errorf("unknown architecture overlap on native arch %#x", nativeArch)
			}

			x32sysno, ok := maxSyscalls[libseccomp.ArchX32]
			if !ok {
				return nil, errors.Errorf("missing %v in overlapping x86_64 arch: %v", libseccomp.ArchX32, maxSyscalls)
			}
			x86sysno, ok := maxSyscalls[libseccomp.ArchAMD64]
			if !ok {
				return nil, errors.Errorf("missing %v in overlapping x86_64 arch: %v", libseccomp.ArchAMD64, maxSyscalls)
			}

			// The x32 ABI indicates that a syscall is being made by an x32
			// process by setting the 30th bit of the syscall number, but we
			// need to do some special-casing depending on whether we need to
			// do long jumps.
			if baseJumpEnosys+2 <= 255 {
				// For the simple case we want to have something like:
				//   jset (1<<30),1
				//   jgt [x86 syscall],[baseJumpEnosys+2],1
				//   jgt [x32 syscall],[baseJumpEnosys+1]
				//   ja [baseJumpFilter]
				section = append(section, []bpf.Instruction{
					// jset (1<<30),1
					bpf.JumpIf{Cond: bpf.JumpBitsSet, Val: 1 << 30, SkipTrue: 1},
					// jgt [x86 syscall],[baseJumpEnosys+1],1
					bpf.JumpIf{
						Cond:     bpf.JumpGreaterThan,
						Val:      uint32(x86sysno),
						SkipTrue: uint8(baseJumpEnosys + 2), SkipFalse: 1,
					},
					// jgt [x32 syscall],[baseJumpEnosys]
					bpf.JumpIf{
						Cond:     bpf.JumpGreaterThan,
						Val:      uint32(x32sysno),
						SkipTrue: uint8(baseJumpEnosys + 1),
					},
					// ja [baseJumpFilter]
					bpf.Jump{Skip: baseJumpFilter},
				}...)
			} else {
				// But if the [baseJumpEnosys+2] jump is larger than 255 we
				// need to do a long jump like so:
				//   jset (1<<30),1
				//   jgt [x86 syscall],1,2
				//   jle [x32 syscall],1
				//   ja [baseJumpEnosys+1]
				//   ja [baseJumpFilter]
				section = append(section, []bpf.Instruction{
					// jset (1<<30),1
					bpf.JumpIf{Cond: bpf.JumpBitsSet, Val: 1 << 30, SkipTrue: 1},
					// jgt [x86 syscall],1,2
					bpf.JumpIf{
						Cond:     bpf.JumpGreaterThan,
						Val:      uint32(x86sysno),
						SkipTrue: 1, SkipFalse: 2,
					},
					// jle [x32 syscall],[baseJumpEnosys]
					bpf.JumpIf{
						Cond:     bpf.JumpLessOrEqual,
						Val:      uint32(x32sysno),
						SkipTrue: 1,
					},
					// ja [baseJumpEnosys+1]
					bpf.Jump{Skip: baseJumpEnosys + 1},
					// ja [baseJumpFilter]
					bpf.Jump{Skip: baseJumpFilter},
				}...)
			}
		default:
			return nil, errors.Errorf("invalid number of architecture overlaps: %v", len(maxSyscalls))
		}

		// Prepend this section to the tail.
		programTail = append(section, programTail...)

		// Update jump table.
		archJumpTable[nativeArch] = uint32(len(programTail))
	}

	// Add a dummy "jump to filter" for any architecture we might miss below.
	// Such architectures will probably get the BadArch action of the filter
	// regardless.
	programTail = append([]bpf.Instruction{
		// ja [end of stub and start of filter]
		bpf.Jump{Skip: uint32(len(programTail))},
	}, programTail...)

	// Generate the jump rules for each architecture. This has to be done in
	// reverse as well for the same reason as above. We add to programTail
	// directly because the jumps are impacted by each architecture rule we add
	// as well.
	//
	// TODO: Maybe we want to optimise to avoid long jumps here? So sort the
	//       architectures based on how large the jumps are going to be, or
	//       re-sort the candidate architectures each time to make sure that we
	//       pick the largest jump which is going to be smaller than 255.
	for nativeArch := range lastSyscalls {
		// We jump forwards but the jump table is calculated from the *END*.
		jump := uint32(len(programTail)) - archJumpTable[nativeArch]

		// Same routine as above -- this is a basic jeq check, complicated
		// slightly if it turns out that we need to do a long jump.
		if jump <= 255 {
			programTail = append([]bpf.Instruction{
				// jeq [arch],[jump]
				bpf.JumpIf{
					Cond:     bpf.JumpEqual,
					Val:      uint32(nativeArch),
					SkipTrue: uint8(jump),
				},
			}, programTail...)
		} else {
			programTail = append([]bpf.Instruction{
				// jne [arch],1
				bpf.JumpIf{
					Cond:     bpf.JumpNotEqual,
					Val:      uint32(nativeArch),
					SkipTrue: 1,
				},
				// ja [jump]
				bpf.Jump{Skip: jump},
			}, programTail...)
		}
	}

	// Prepend the load instruction for the architecture.
	programTail = append([]bpf.Instruction{
		// load [4]
		bpf.LoadAbsolute{Off: 4, Size: 4}, // NOTE: We assume sizeof(int) == 4.
	}, programTail...)

	// And that's all folks!
	return programTail, nil
}

func assemble(program []bpf.Instruction) ([]unix.SockFilter, error) {
	rawProgram, err := bpf.Assemble(program)
	if err != nil {
		return nil, errors.Wrap(err, "assembling program")
	}

	// Convert to []unix.SockFilter for unix.SockFilter.
	var filter []unix.SockFilter
	for _, insn := range rawProgram {
		filter = append(filter, unix.SockFilter{
			Code: insn.Op,
			Jt:   insn.Jt,
			Jf:   insn.Jf,
			K:    insn.K,
		})
	}
	return filter, nil
}

func generatePatch(config *configs.Seccomp) ([]bpf.Instruction, error) {
	// Patch the generated cBPF only when there is not a defaultErrnoRet set
	// and it is different from ENOSYS
	if config.DefaultErrnoRet != nil && *config.DefaultErrnoRet == uint(retErrnoEnosys) {
		return nil, nil
	}
	// We only add the stub if the default action is not permissive.
	if isAllowAction(config.DefaultAction) {
		logrus.Debugf("seccomp: skipping -ENOSYS stub filter generation")
		return nil, nil
	}

	lastSyscalls, err := findLastSyscalls(config)
	if err != nil {
		return nil, errors.Wrap(err, "finding last syscalls for -ENOSYS stub")
	}
	stubProgram, err := generateEnosysStub(lastSyscalls)
	if err != nil {
		return nil, errors.Wrap(err, "generating -ENOSYS stub")
	}
	return stubProgram, nil
}

func enosysPatchFilter(config *configs.Seccomp, filter *libseccomp.ScmpFilter) ([]unix.SockFilter, error) {
	program, err := disassembleFilter(filter)
	if err != nil {
		return nil, errors.Wrap(err, "disassembling original filter")
	}

	patch, err := generatePatch(config)
	if err != nil {
		return nil, errors.Wrap(err, "generating patch for filter")
	}
	fullProgram := append(patch, program...)

	logrus.Debugf("seccomp: prepending -ENOSYS stub filter to user filter...")
	for idx, insn := range patch {
		logrus.Debugf("  [%4.1d] %s", idx, insn)
	}
	logrus.Debugf("  [....] --- original filter ---")

	fprog, err := assemble(fullProgram)
	if err != nil {
		return nil, errors.Wrap(err, "assembling modified filter")
	}
	return fprog, nil
}

func filterFlags(filter *libseccomp.ScmpFilter) (flags uint, noNewPrivs bool, err error) {
	// Ignore the error since pre-2.4 libseccomp is treated as API level 0.
	apiLevel, _ := libseccomp.GetApi()

	noNewPrivs, err = filter.GetNoNewPrivsBit()
	if err != nil {
		return 0, false, errors.Wrap(err, "fetch no_new_privs filter bit")
	}

	if apiLevel >= 3 {
		if logBit, err := filter.GetLogBit(); err != nil {
			return 0, false, errors.Wrap(err, "fetch SECCOMP_FILTER_FLAG_LOG bit")
		} else if logBit {
			flags |= uint(C.C_FILTER_FLAG_LOG)
		}
	}

	// TODO: Support seccomp flags not yet added to libseccomp-golang...
	return
}

func sysSeccompSetFilter(flags uint, filter []unix.SockFilter) (err error) {
	fprog := unix.SockFprog{
		Len:    uint16(len(filter)),
		Filter: &filter[0],
	}
	// If no seccomp flags were requested we can use the old-school prctl(2).
	if flags == 0 {
		err = unix.Prctl(unix.PR_SET_SECCOMP,
			unix.SECCOMP_MODE_FILTER,
			uintptr(unsafe.Pointer(&fprog)), 0, 0)
	} else {
		_, _, errno := unix.RawSyscall(unix.SYS_SECCOMP,
			uintptr(C.C_SET_MODE_FILTER),
			uintptr(flags), uintptr(unsafe.Pointer(&fprog)))
		if errno != 0 {
			err = errno
		}
	}
	runtime.KeepAlive(filter)
	runtime.KeepAlive(fprog)
	return
}

// PatchAndLoad takes a seccomp configuration and a libseccomp filter which has
// been pre-configured with the set of rules in the seccomp config. It then
// patches said filter to handle -ENOSYS in a much nicer manner than the
// default libseccomp default action behaviour, and loads the patched filter
// into the kernel for the current process.
func PatchAndLoad(config *configs.Seccomp, filter *libseccomp.ScmpFilter) error {
	// Generate a patched filter.
	fprog, err := enosysPatchFilter(config, filter)
	if err != nil {
		return errors.Wrap(err, "patching filter")
	}

	// Get the set of libseccomp flags set.
	seccompFlags, noNewPrivs, err := filterFlags(filter)
	if err != nil {
		return errors.Wrap(err, "fetch seccomp filter flags")
	}

	// Set no_new_privs if it was requested, though in runc we handle
	// no_new_privs separately so warn if we hit this path.
	if noNewPrivs {
		logrus.Warnf("potentially misconfigured filter -- setting no_new_privs in seccomp path")
		if err := unix.Prctl(unix.PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0); err != nil {
			return errors.Wrap(err, "enable no_new_privs bit")
		}
	}

	// Finally, load the filter.
	if err := sysSeccompSetFilter(seccompFlags, fprog); err != nil {
		return errors.Wrap(err, "loading seccomp filter")
	}
	return nil
}
