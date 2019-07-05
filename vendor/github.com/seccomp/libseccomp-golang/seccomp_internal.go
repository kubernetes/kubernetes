// +build linux

// Internal functions for libseccomp Go bindings
// No exported functions

package seccomp

import (
	"fmt"
	"syscall"
)

// Unexported C wrapping code - provides the C-Golang interface
// Get the seccomp header in scope
// Need stdlib.h for free() on cstrings

// #cgo pkg-config: libseccomp
/*
#include <errno.h>
#include <stdlib.h>
#include <seccomp.h>

#if SCMP_VER_MAJOR < 2
#error Minimum supported version of Libseccomp is v2.2.0
#elif SCMP_VER_MAJOR == 2 && SCMP_VER_MINOR < 2
#error Minimum supported version of Libseccomp is v2.2.0
#endif

#define ARCH_BAD ~0

const uint32_t C_ARCH_BAD = ARCH_BAD;

#ifndef SCMP_ARCH_PPC
#define SCMP_ARCH_PPC ARCH_BAD
#endif

#ifndef SCMP_ARCH_PPC64
#define SCMP_ARCH_PPC64 ARCH_BAD
#endif

#ifndef SCMP_ARCH_PPC64LE
#define SCMP_ARCH_PPC64LE ARCH_BAD
#endif

#ifndef SCMP_ARCH_S390
#define SCMP_ARCH_S390 ARCH_BAD
#endif

#ifndef SCMP_ARCH_S390X
#define SCMP_ARCH_S390X ARCH_BAD
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
const uint32_t C_ARCH_PPC          = SCMP_ARCH_PPC;
const uint32_t C_ARCH_PPC64        = SCMP_ARCH_PPC64;
const uint32_t C_ARCH_PPC64LE      = SCMP_ARCH_PPC64LE;
const uint32_t C_ARCH_S390         = SCMP_ARCH_S390;
const uint32_t C_ARCH_S390X        = SCMP_ARCH_S390X;

#ifndef SCMP_ACT_LOG
#define SCMP_ACT_LOG 0x7ffc0000U
#endif

const uint32_t C_ACT_KILL          = SCMP_ACT_KILL;
const uint32_t C_ACT_TRAP          = SCMP_ACT_TRAP;
const uint32_t C_ACT_ERRNO         = SCMP_ACT_ERRNO(0);
const uint32_t C_ACT_TRACE         = SCMP_ACT_TRACE(0);
const uint32_t C_ACT_LOG           = SCMP_ACT_LOG;
const uint32_t C_ACT_ALLOW         = SCMP_ACT_ALLOW;

// The libseccomp SCMP_FLTATR_CTL_LOG member of the scmp_filter_attr enum was
// added in v2.4.0
#if (SCMP_VER_MAJOR < 2) || \
    (SCMP_VER_MAJOR == 2 && SCMP_VER_MINOR < 4)
#define SCMP_FLTATR_CTL_LOG _SCMP_FLTATR_MIN
#endif

const uint32_t C_ATTRIBUTE_DEFAULT = (uint32_t)SCMP_FLTATR_ACT_DEFAULT;
const uint32_t C_ATTRIBUTE_BADARCH = (uint32_t)SCMP_FLTATR_ACT_BADARCH;
const uint32_t C_ATTRIBUTE_NNP     = (uint32_t)SCMP_FLTATR_CTL_NNP;
const uint32_t C_ATTRIBUTE_TSYNC   = (uint32_t)SCMP_FLTATR_CTL_TSYNC;
const uint32_t C_ATTRIBUTE_LOG     = (uint32_t)SCMP_FLTATR_CTL_LOG;

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

#if SCMP_VER_MAJOR == 2 && SCMP_VER_MINOR >= 3
unsigned int get_major_version()
{
        return seccomp_version()->major;
}

unsigned int get_minor_version()
{
        return seccomp_version()->minor;
}

unsigned int get_micro_version()
{
        return seccomp_version()->micro;
}
#else
unsigned int get_major_version()
{
        return (unsigned int)C_VERSION_MAJOR;
}

unsigned int get_minor_version()
{
        return (unsigned int)C_VERSION_MINOR;
}

unsigned int get_micro_version()
{
        return (unsigned int)C_VERSION_MICRO;
}
#endif

// The libseccomp API level functions were added in v2.4.0
#if (SCMP_VER_MAJOR < 2) || \
    (SCMP_VER_MAJOR == 2 && SCMP_VER_MINOR < 4)
const unsigned int seccomp_api_get(void)
{
	// libseccomp-golang requires libseccomp v2.2.0, at a minimum, which
	// supported API level 2. However, the kernel may not support API level
	// 2 constructs which are the seccomp() system call and the TSYNC
	// filter flag. Return the "reserved" value of 0 here to indicate that
	// proper API level support is not available in libseccomp.
	return 0;
}

int seccomp_api_set(unsigned int level)
{
	return -EOPNOTSUPP;
}
#endif

typedef struct scmp_arg_cmp* scmp_cast_t;

void* make_arg_cmp_array(unsigned int length)
{
        return calloc(length, sizeof(struct scmp_arg_cmp));
}

// Wrapper to add an scmp_arg_cmp struct to an existing arg_cmp array
void add_struct_arg_cmp(
                        struct scmp_arg_cmp* arr,
                        unsigned int pos,
                        unsigned int arg,
                        int compare,
                        uint64_t a,
                        uint64_t b
                       )
{
        arr[pos].arg = arg;
        arr[pos].op = compare;
        arr[pos].datum_a = a;
        arr[pos].datum_b = b;

        return;
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
	filterAttrLog        scmpFilterAttr = iota
)

const (
	// An error return from certain libseccomp functions
	scmpError C.int = -1
	// Comparison boundaries to check for architecture validity
	archStart ScmpArch = ArchNative
	archEnd   ScmpArch = ArchS390X
	// Comparison boundaries to check for action validity
	actionStart ScmpAction = ActKill
	actionEnd   ScmpAction = ActLog
	// Comparison boundaries to check for comparison operator validity
	compareOpStart ScmpCompareOp = CompareNotEqual
	compareOpEnd   ScmpCompareOp = CompareMaskedEqual
)

var (
	// Error thrown on bad filter context
	errBadFilter = fmt.Errorf("filter is invalid or uninitialized")
	// Constants representing library major, minor, and micro versions
	verMajor = uint(C.get_major_version())
	verMinor = uint(C.get_minor_version())
	verMicro = uint(C.get_micro_version())
)

// Nonexported functions

// Check if library version is greater than or equal to the given one
func checkVersionAbove(major, minor, micro uint) bool {
	return (verMajor > major) ||
		(verMajor == major && verMinor > minor) ||
		(verMajor == major && verMinor == minor && verMicro >= micro)
}

// Ensure that the library is supported, i.e. >= 2.2.0.
func ensureSupportedVersion() error {
	if !checkVersionAbove(2, 2, 0) {
		return VersionError{}
	}
	return nil
}

// Get the API level
func getApi() (uint, error) {
	api := C.seccomp_api_get()
	if api == 0 {
		return 0, fmt.Errorf("API level operations are not supported")
	}

	return uint(api), nil
}

// Set the API level
func setApi(api uint) error {
	if retCode := C.seccomp_api_set(C.uint(api)); retCode != 0 {
		if syscall.Errno(-1*retCode) == syscall.EOPNOTSUPP {
			return fmt.Errorf("API level operations are not supported")
		}

		return fmt.Errorf("could not set API level: %v", retCode)
	}

	return nil
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

	retCode := C.seccomp_attr_set(f.filterCtx, attr.toNative(), value)
	if retCode != 0 {
		return syscall.Errno(-1 * retCode)
	}

	return nil
}

// DOES NOT LOCK OR CHECK VALIDITY
// Assumes caller has already done this
// Wrapper for seccomp_rule_add_... functions
func (f *ScmpFilter) addRuleWrapper(call ScmpSyscall, action ScmpAction, exact bool, length C.uint, cond C.scmp_cast_t) error {
	if length != 0 && cond == nil {
		return fmt.Errorf("null conditions list, but length is nonzero")
	}

	var retCode C.int
	if exact {
		retCode = C.seccomp_rule_add_exact_array(f.filterCtx, action.toNative(), C.int(call), length, cond)
	} else {
		retCode = C.seccomp_rule_add_array(f.filterCtx, action.toNative(), C.int(call), length, cond)
	}

	if syscall.Errno(-1*retCode) == syscall.EFAULT {
		return fmt.Errorf("unrecognized syscall %#x", int32(call))
	} else if syscall.Errno(-1*retCode) == syscall.EPERM {
		return fmt.Errorf("requested action matches default action of filter")
	} else if syscall.Errno(-1*retCode) == syscall.EINVAL {
		return fmt.Errorf("two checks on same syscall argument")
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
		if err := f.addRuleWrapper(call, action, exact, 0, nil); err != nil {
			return err
		}
	} else {
		// We don't support conditional filtering in library version v2.1
		if !checkVersionAbove(2, 2, 1) {
			return VersionError{
				message: "conditional filtering is not supported",
				minimum: "2.2.1",
			}
		}

		argsArr := C.make_arg_cmp_array(C.uint(len(conds)))
		if argsArr == nil {
			return fmt.Errorf("error allocating memory for conditions")
		}
		defer C.free(argsArr)

		for i, cond := range conds {
			C.add_struct_arg_cmp(C.scmp_cast_t(argsArr), C.uint(i),
				C.uint(cond.Argument), cond.Op.toNative(),
				C.uint64_t(cond.Operand1), C.uint64_t(cond.Operand2))
		}

		if err := f.addRuleWrapper(call, action, exact, C.uint(len(conds)), C.scmp_cast_t(argsArr)); err != nil {
			return err
		}
	}

	return nil
}

// Generic Helpers

// Helper - Sanitize Arch token input
func sanitizeArch(in ScmpArch) error {
	if in < archStart || in > archEnd {
		return fmt.Errorf("unrecognized architecture %#x", uint(in))
	}

	if in.toNative() == C.C_ARCH_BAD {
		return fmt.Errorf("architecture %v is not supported on this version of the library", in)
	}

	return nil
}

func sanitizeAction(in ScmpAction) error {
	inTmp := in & 0x0000FFFF
	if inTmp < actionStart || inTmp > actionEnd {
		return fmt.Errorf("unrecognized action %#x", uint(inTmp))
	}

	if inTmp != ActTrace && inTmp != ActErrno && (in&0xFFFF0000) != 0 {
		return fmt.Errorf("highest 16 bits must be zeroed except for Trace and Errno")
	}

	return nil
}

func sanitizeCompareOp(in ScmpCompareOp) error {
	if in < compareOpStart || in > compareOpEnd {
		return fmt.Errorf("unrecognized comparison operator %#x", uint(in))
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
	case C.C_ARCH_PPC:
		return ArchPPC, nil
	case C.C_ARCH_PPC64:
		return ArchPPC64, nil
	case C.C_ARCH_PPC64LE:
		return ArchPPC64LE, nil
	case C.C_ARCH_S390:
		return ArchS390, nil
	case C.C_ARCH_S390X:
		return ArchS390X, nil
	default:
		return 0x0, fmt.Errorf("unrecognized architecture %#x", uint32(a))
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
	case ArchPPC:
		return C.C_ARCH_PPC
	case ArchPPC64:
		return C.C_ARCH_PPC64
	case ArchPPC64LE:
		return C.C_ARCH_PPC64LE
	case ArchS390:
		return C.C_ARCH_S390
	case ArchS390X:
		return C.C_ARCH_S390X
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
	case C.C_ACT_LOG:
		return ActLog, nil
	case C.C_ACT_ALLOW:
		return ActAllow, nil
	default:
		return 0x0, fmt.Errorf("unrecognized action %#x", uint32(a))
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
	case ActLog:
		return C.C_ACT_LOG
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
	case filterAttrLog:
		return uint32(C.C_ATTRIBUTE_LOG)
	default:
		return 0x0
	}
}
