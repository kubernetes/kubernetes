// +build linux

// Internal functions for libseccomp Go bindings
// No exported functions

package seccomp

import (
	"fmt"
	"os"
	"syscall"
)

// Unexported C wrapping code - provides the C-Golang interface
// Get the seccomp header in scope
// Need stdlib.h for free() on cstrings

// #cgo LDFLAGS: -lseccomp
/*
#include <stdlib.h>
#include <seccomp.h>

#if SCMP_VER_MAJOR < 2
#error Minimum supported version of Libseccomp is v2.1.0
#elif SCMP_VER_MAJOR == 2 && SCMP_VER_MINOR < 1
#error Minimum supported version of Libseccomp is v2.1.0
#endif

#define ARCH_BAD ~0

const uint32_t C_ARCH_BAD = ARCH_BAD;

#ifndef SCMP_ARCH_AARCH64
#define SCMP_ARCH_AARCH64 ARCH_BAD
#endif

#ifndef SCMP_ARCH_MIPS
#define SCMP_ARCH_MIPS ARCH_BAD
#endif

#ifndef SCMP_ARCH_MIPS64
#define SCMP_ARCH_MIPS64 ARCH_BAD
#endif

#ifndef SCMP_ARCH_MIPS64N32
#define SCMP_ARCH_MIPS64N32 ARCH_BAD
#endif

#ifndef SCMP_ARCH_MIPSEL
#define SCMP_ARCH_MIPSEL ARCH_BAD
#endif

#ifndef SCMP_ARCH_MIPSEL64
#define SCMP_ARCH_MIPSEL64 ARCH_BAD
#endif

#ifndef SCMP_ARCH_MIPSEL64N32
#define SCMP_ARCH_MIPSEL64N32 ARCH_BAD
#endif

const uint32_t C_ARCH_NATIVE       = SCMP_ARCH_NATIVE;
const uint32_t C_ARCH_X86          = SCMP_ARCH_X86;
const uint32_t C_ARCH_X86_64       = SCMP_ARCH_X86_64;
const uint32_t C_ARCH_X32          = SCMP_ARCH_X32;
const uint32_t C_ARCH_ARM          = SCMP_ARCH_ARM;
const uint32_t C_ARCH_AARCH64      = SCMP_ARCH_AARCH64;
const uint32_t C_ARCH_MIPS         = SCMP_ARCH_MIPS;
const uint32_t C_ARCH_MIPS64       = SCMP_ARCH_MIPS64;
const uint32_t C_ARCH_MIPS64N32    = SCMP_ARCH_MIPS64N32;
const uint32_t C_ARCH_MIPSEL       = SCMP_ARCH_MIPSEL;
const uint32_t C_ARCH_MIPSEL64     = SCMP_ARCH_MIPSEL64;
const uint32_t C_ARCH_MIPSEL64N32  = SCMP_ARCH_MIPSEL64N32;

const uint32_t C_ACT_KILL          = SCMP_ACT_KILL;
const uint32_t C_ACT_TRAP          = SCMP_ACT_TRAP;
const uint32_t C_ACT_ERRNO         = SCMP_ACT_ERRNO(0);
const uint32_t C_ACT_TRACE         = SCMP_ACT_TRACE(0);
const uint32_t C_ACT_ALLOW         = SCMP_ACT_ALLOW;

// If TSync is not supported, make sure it doesn't map to a supported filter attribute
// Don't worry about major version < 2, the minimum version checks should catch that case
#if SCMP_VER_MAJOR == 2 && SCMP_VER_MINOR < 2
#define SCMP_FLTATR_CTL_TSYNC _SCMP_CMP_MIN
#endif

const uint32_t C_ATTRIBUTE_DEFAULT = (uint32_t)SCMP_FLTATR_ACT_DEFAULT;
const uint32_t C_ATTRIBUTE_BADARCH = (uint32_t)SCMP_FLTATR_ACT_BADARCH;
const uint32_t C_ATTRIBUTE_NNP     = (uint32_t)SCMP_FLTATR_CTL_NNP;
const uint32_t C_ATTRIBUTE_TSYNC   = (uint32_t)SCMP_FLTATR_CTL_TSYNC;

const int      C_CMP_NE            = (int)SCMP_CMP_NE;
const int      C_CMP_LT            = (int)SCMP_CMP_LT;
const int      C_CMP_LE            = (int)SCMP_CMP_LE;
const int      C_CMP_EQ            = (int)SCMP_CMP_EQ;
const int      C_CMP_GE            = (int)SCMP_CMP_GE;
const int      C_CMP_GT            = (int)SCMP_CMP_GT;
const int      C_CMP_MASKED_EQ     = (int)SCMP_CMP_MASKED_EQ;

const int      C_VERSION_MAJOR     = SCMP_VER_MAJOR;
const int      C_VERSION_MINOR     = SCMP_VER_MINOR;
const int      C_VERSION_MICRO     = SCMP_VER_MICRO;

typedef struct scmp_arg_cmp* scmp_cast_t;

// Wrapper to create an scmp_arg_cmp struct
void*
make_struct_arg_cmp(
                    unsigned int arg,
                    int compare,
                    uint64_t a,
                    uint64_t b
                   )
{
	struct scmp_arg_cmp *s = malloc(sizeof(struct scmp_arg_cmp));

	s->arg = arg;
	s->op = compare;
	s->datum_a = a;
	s->datum_b = b;

	return s;
}
*/
import "C"

// Nonexported types
type scmpFilterAttr uint32

// Nonexported constants

const (
	filterAttrActDefault scmpFilterAttr = iota
	filterAttrActBadArch scmpFilterAttr = iota
	filterAttrNNP        scmpFilterAttr = iota
	filterAttrTsync      scmpFilterAttr = iota
)

const (
	// An error return from certain libseccomp functions
	scmpError C.int = -1
	// Comparison boundaries to check for architecture validity
	archStart ScmpArch = ArchNative
	archEnd   ScmpArch = ArchMIPSEL64N32
	// Comparison boundaries to check for action validity
	actionStart ScmpAction = ActKill
	actionEnd   ScmpAction = ActAllow
	// Comparison boundaries to check for comparison operator validity
	compareOpStart ScmpCompareOp = CompareNotEqual
	compareOpEnd   ScmpCompareOp = CompareMaskedEqual
)

var (
	// Error thrown on bad filter context
	errBadFilter = fmt.Errorf("filter is invalid or uninitialized")
	// Constants representing library major, minor, and micro versions
	verMajor = int(C.C_VERSION_MAJOR)
	verMinor = int(C.C_VERSION_MINOR)
	verMicro = int(C.C_VERSION_MICRO)
)

// Nonexported functions

// Check if library version is greater than or equal to the given one
func checkVersionAbove(major, minor, micro int) bool {
	return (verMajor > major) ||
		(verMajor == major && verMinor > minor) ||
		(verMajor == major && verMinor == minor && verMicro >= micro)
}

// Init function: Verify library version is appropriate
func init() {
	if !checkVersionAbove(2, 1, 0) {
		fmt.Fprintf(os.Stderr, "Libseccomp version too low: minimum supported is 2.1.0, detected %d.%d.%d", C.C_VERSION_MAJOR, C.C_VERSION_MINOR, C.C_VERSION_MICRO)
		os.Exit(-1)
	}
}

// Filter helpers

// Filter finalizer - ensure that kernel context for filters is freed
func filterFinalizer(f *ScmpFilter) {
	f.Release()
}

// Get a raw filter attribute
func (f *ScmpFilter) getFilterAttr(attr scmpFilterAttr) (C.uint32_t, error) {
	f.lock.Lock()
	defer f.lock.Unlock()

	if !f.valid {
		return 0x0, errBadFilter
	}

	if !checkVersionAbove(2, 2, 0) && attr == filterAttrTsync {
		return 0x0, fmt.Errorf("the thread synchronization attribute is not supported in this version of the library")
	}

	var attribute C.uint32_t

	retCode := C.seccomp_attr_get(f.filterCtx, attr.toNative(), &attribute)
	if retCode != 0 {
		return 0x0, syscall.Errno(-1 * retCode)
	}

	return attribute, nil
}

// Set a raw filter attribute
func (f *ScmpFilter) setFilterAttr(attr scmpFilterAttr, value C.uint32_t) error {
	f.lock.Lock()
	defer f.lock.Unlock()

	if !f.valid {
		return errBadFilter
	}

	if !checkVersionAbove(2, 2, 0) && attr == filterAttrTsync {
		return fmt.Errorf("the thread synchronization attribute is not supported in this version of the library")
	}

	retCode := C.seccomp_attr_set(f.filterCtx, attr.toNative(), value)
	if retCode != 0 {
		return syscall.Errno(-1 * retCode)
	}

	return nil
}

// DOES NOT LOCK OR CHECK VALIDITY
// Assumes caller has already done this
// Wrapper for seccomp_rule_add_... functions
func (f *ScmpFilter) addRuleWrapper(call ScmpSyscall, action ScmpAction, exact bool, cond C.scmp_cast_t) error {
	var length C.uint
	if cond != nil {
		length = 1
	} else {
		length = 0
	}

	var retCode C.int
	if exact {
		retCode = C.seccomp_rule_add_exact_array(f.filterCtx, action.toNative(), C.int(call), length, cond)
	} else {
		retCode = C.seccomp_rule_add_array(f.filterCtx, action.toNative(), C.int(call), length, cond)
	}

	if syscall.Errno(-1*retCode) == syscall.EFAULT {
		return fmt.Errorf("unrecognized syscall")
	} else if syscall.Errno(-1*retCode) == syscall.EPERM {
		return fmt.Errorf("requested action matches default action of filter")
	} else if retCode != 0 {
		return syscall.Errno(-1 * retCode)
	}

	return nil
}

// Generic add function for filter rules
func (f *ScmpFilter) addRuleGeneric(call ScmpSyscall, action ScmpAction, exact bool, conds []ScmpCondition) error {
	f.lock.Lock()
	defer f.lock.Unlock()

	if !f.valid {
		return errBadFilter
	}

	if len(conds) == 0 {
		if err := f.addRuleWrapper(call, action, exact, nil); err != nil {
			return err
		}
	} else {
		// We don't support conditional filtering in library version v2.1
		if !checkVersionAbove(2, 2, 1) {
			return fmt.Errorf("conditional filtering requires libseccomp version >= 2.2.1")
		}

		for _, cond := range conds {
			cmpStruct := C.make_struct_arg_cmp(C.uint(cond.Argument), cond.Op.toNative(), C.uint64_t(cond.Operand1), C.uint64_t(cond.Operand2))
			defer C.free(cmpStruct)

			if err := f.addRuleWrapper(call, action, exact, C.scmp_cast_t(cmpStruct)); err != nil {
				return err
			}
		}
	}

	return nil
}

// Generic Helpers

// Helper - Sanitize Arch token input
func sanitizeArch(in ScmpArch) error {
	if in < archStart || in > archEnd {
		return fmt.Errorf("unrecognized architecture")
	}

	if in.toNative() == C.C_ARCH_BAD {
		return fmt.Errorf("architecture is not supported on this version of the library")
	}

	return nil
}

func sanitizeAction(in ScmpAction) error {
	inTmp := in & 0x0000FFFF
	if inTmp < actionStart || inTmp > actionEnd {
		return fmt.Errorf("unrecognized action")
	}

	if inTmp != ActTrace && inTmp != ActErrno && (in&0xFFFF0000) != 0 {
		return fmt.Errorf("highest 16 bits must be zeroed except for Trace and Errno")
	}

	return nil
}

func sanitizeCompareOp(in ScmpCompareOp) error {
	if in < compareOpStart || in > compareOpEnd {
		return fmt.Errorf("unrecognized comparison operator")
	}

	return nil
}

func archFromNative(a C.uint32_t) (ScmpArch, error) {
	switch a {
	case C.C_ARCH_X86:
		return ArchX86, nil
	case C.C_ARCH_X86_64:
		return ArchAMD64, nil
	case C.C_ARCH_X32:
		return ArchX32, nil
	case C.C_ARCH_ARM:
		return ArchARM, nil
	case C.C_ARCH_NATIVE:
		return ArchNative, nil
	case C.C_ARCH_AARCH64:
		return ArchARM64, nil
	case C.C_ARCH_MIPS:
		return ArchMIPS, nil
	case C.C_ARCH_MIPS64:
		return ArchMIPS64, nil
	case C.C_ARCH_MIPS64N32:
		return ArchMIPS64N32, nil
	case C.C_ARCH_MIPSEL:
		return ArchMIPSEL, nil
	case C.C_ARCH_MIPSEL64:
		return ArchMIPSEL64, nil
	case C.C_ARCH_MIPSEL64N32:
		return ArchMIPSEL64N32, nil
	default:
		return 0x0, fmt.Errorf("unrecognized architecture")
	}
}

// Only use with sanitized arches, no error handling
func (a ScmpArch) toNative() C.uint32_t {
	switch a {
	case ArchX86:
		return C.C_ARCH_X86
	case ArchAMD64:
		return C.C_ARCH_X86_64
	case ArchX32:
		return C.C_ARCH_X32
	case ArchARM:
		return C.C_ARCH_ARM
	case ArchARM64:
		return C.C_ARCH_AARCH64
	case ArchMIPS:
		return C.C_ARCH_MIPS
	case ArchMIPS64:
		return C.C_ARCH_MIPS64
	case ArchMIPS64N32:
		return C.C_ARCH_MIPS64N32
	case ArchMIPSEL:
		return C.C_ARCH_MIPSEL
	case ArchMIPSEL64:
		return C.C_ARCH_MIPSEL64
	case ArchMIPSEL64N32:
		return C.C_ARCH_MIPSEL64N32
	case ArchNative:
		return C.C_ARCH_NATIVE
	default:
		return 0x0
	}
}

// Only use with sanitized ops, no error handling
func (a ScmpCompareOp) toNative() C.int {
	switch a {
	case CompareNotEqual:
		return C.C_CMP_NE
	case CompareLess:
		return C.C_CMP_LT
	case CompareLessOrEqual:
		return C.C_CMP_LE
	case CompareEqual:
		return C.C_CMP_EQ
	case CompareGreaterEqual:
		return C.C_CMP_GE
	case CompareGreater:
		return C.C_CMP_GT
	case CompareMaskedEqual:
		return C.C_CMP_MASKED_EQ
	default:
		return 0x0
	}
}

func actionFromNative(a C.uint32_t) (ScmpAction, error) {
	aTmp := a & 0xFFFF
	switch a & 0xFFFF0000 {
	case C.C_ACT_KILL:
		return ActKill, nil
	case C.C_ACT_TRAP:
		return ActTrap, nil
	case C.C_ACT_ERRNO:
		return ActErrno.SetReturnCode(int16(aTmp)), nil
	case C.C_ACT_TRACE:
		return ActTrace.SetReturnCode(int16(aTmp)), nil
	case C.C_ACT_ALLOW:
		return ActAllow, nil
	default:
		return 0x0, fmt.Errorf("unrecognized action")
	}
}

// Only use with sanitized actions, no error handling
func (a ScmpAction) toNative() C.uint32_t {
	switch a & 0xFFFF {
	case ActKill:
		return C.C_ACT_KILL
	case ActTrap:
		return C.C_ACT_TRAP
	case ActErrno:
		return C.C_ACT_ERRNO | (C.uint32_t(a) >> 16)
	case ActTrace:
		return C.C_ACT_TRACE | (C.uint32_t(a) >> 16)
	case ActAllow:
		return C.C_ACT_ALLOW
	default:
		return 0x0
	}
}

// Internal only, assumes safe attribute
func (a scmpFilterAttr) toNative() uint32 {
	switch a {
	case filterAttrActDefault:
		return uint32(C.C_ATTRIBUTE_DEFAULT)
	case filterAttrActBadArch:
		return uint32(C.C_ATTRIBUTE_BADARCH)
	case filterAttrNNP:
		return uint32(C.C_ATTRIBUTE_NNP)
	case filterAttrTsync:
		return uint32(C.C_ATTRIBUTE_TSYNC)
	default:
		return 0x0
	}
}
