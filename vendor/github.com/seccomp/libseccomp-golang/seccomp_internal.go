// Internal functions for libseccomp Go bindings
// No exported functions

package seccomp

import (
	"errors"
	"fmt"
	"syscall"
)

// Unexported C wrapping code - provides the C-Golang interface
// Get the seccomp header in scope
// Need stdlib.h for free() on cstrings

// To compile libseccomp-golang against a specific version of libseccomp:
// cd ../libseccomp && mkdir -p prefix
// ./configure --prefix=$PWD/prefix && make && make install
// cd ../libseccomp-golang
// PKG_CONFIG_PATH=$PWD/../libseccomp/prefix/lib/pkgconfig/ make
// LD_PRELOAD=$PWD/../libseccomp/prefix/lib/libseccomp.so.2.5.0 PKG_CONFIG_PATH=$PWD/../libseccomp/prefix/lib/pkgconfig/ make test

// #cgo pkg-config: libseccomp
/*
#include <errno.h>
#include <stdlib.h>
#include <seccomp.h>

#if (SCMP_VER_MAJOR < 2) || \
    (SCMP_VER_MAJOR == 2 && SCMP_VER_MINOR < 3) || \
    (SCMP_VER_MAJOR == 2 && SCMP_VER_MINOR == 3 && SCMP_VER_MICRO < 1)
#error This package requires libseccomp >= v2.3.1
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

#ifndef SCMP_ARCH_PARISC
#define SCMP_ARCH_PARISC ARCH_BAD
#endif

#ifndef SCMP_ARCH_PARISC64
#define SCMP_ARCH_PARISC64 ARCH_BAD
#endif

#ifndef SCMP_ARCH_RISCV64
#define SCMP_ARCH_RISCV64 ARCH_BAD
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
const uint32_t C_ARCH_PARISC       = SCMP_ARCH_PARISC;
const uint32_t C_ARCH_PARISC64     = SCMP_ARCH_PARISC64;
const uint32_t C_ARCH_RISCV64      = SCMP_ARCH_RISCV64;

#ifndef SCMP_ACT_LOG
#define SCMP_ACT_LOG 0x7ffc0000U
#endif

#ifndef SCMP_ACT_KILL_PROCESS
#define SCMP_ACT_KILL_PROCESS 0x80000000U
#endif

#ifndef SCMP_ACT_KILL_THREAD
#define SCMP_ACT_KILL_THREAD	0x00000000U
#endif

#ifndef SCMP_ACT_NOTIFY
#define SCMP_ACT_NOTIFY 0x7fc00000U
#endif

const uint32_t C_ACT_KILL          = SCMP_ACT_KILL;
const uint32_t C_ACT_KILL_PROCESS  = SCMP_ACT_KILL_PROCESS;
const uint32_t C_ACT_KILL_THREAD   = SCMP_ACT_KILL_THREAD;
const uint32_t C_ACT_TRAP          = SCMP_ACT_TRAP;
const uint32_t C_ACT_ERRNO         = SCMP_ACT_ERRNO(0);
const uint32_t C_ACT_TRACE         = SCMP_ACT_TRACE(0);
const uint32_t C_ACT_LOG           = SCMP_ACT_LOG;
const uint32_t C_ACT_ALLOW         = SCMP_ACT_ALLOW;
const uint32_t C_ACT_NOTIFY        = SCMP_ACT_NOTIFY;

// The libseccomp SCMP_FLTATR_CTL_LOG member of the scmp_filter_attr enum was
// added in v2.4.0
#if SCMP_VER_MAJOR == 2 && SCMP_VER_MINOR < 4
#define SCMP_FLTATR_CTL_LOG _SCMP_FLTATR_MIN
#endif

// The following SCMP_FLTATR_*  were added in libseccomp v2.5.0.
#if SCMP_VER_MAJOR == 2 && SCMP_VER_MINOR < 5
#define SCMP_FLTATR_CTL_SSB      _SCMP_FLTATR_MIN
#define SCMP_FLTATR_CTL_OPTIMIZE _SCMP_FLTATR_MIN
#define SCMP_FLTATR_API_SYSRAWRC _SCMP_FLTATR_MIN
#endif

const uint32_t C_ATTRIBUTE_DEFAULT  = (uint32_t)SCMP_FLTATR_ACT_DEFAULT;
const uint32_t C_ATTRIBUTE_BADARCH  = (uint32_t)SCMP_FLTATR_ACT_BADARCH;
const uint32_t C_ATTRIBUTE_NNP      = (uint32_t)SCMP_FLTATR_CTL_NNP;
const uint32_t C_ATTRIBUTE_TSYNC    = (uint32_t)SCMP_FLTATR_CTL_TSYNC;
const uint32_t C_ATTRIBUTE_LOG      = (uint32_t)SCMP_FLTATR_CTL_LOG;
const uint32_t C_ATTRIBUTE_SSB      = (uint32_t)SCMP_FLTATR_CTL_SSB;
const uint32_t C_ATTRIBUTE_OPTIMIZE = (uint32_t)SCMP_FLTATR_CTL_OPTIMIZE;
const uint32_t C_ATTRIBUTE_SYSRAWRC = (uint32_t)SCMP_FLTATR_API_SYSRAWRC;

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
#if SCMP_VER_MAJOR == 2 && SCMP_VER_MINOR < 4
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

// The seccomp notify API functions were added in v2.5.0
#if SCMP_VER_MAJOR == 2 && SCMP_VER_MINOR < 5

struct seccomp_data {
	int nr;
	__u32 arch;
	__u64 instruction_pointer;
	__u64 args[6];
};

struct seccomp_notif {
	__u64 id;
	__u32 pid;
	__u32 flags;
	struct seccomp_data data;
};

struct seccomp_notif_resp {
	__u64 id;
	__s64 val;
	__s32 error;
	__u32 flags;
};

int seccomp_notify_alloc(struct seccomp_notif **req, struct seccomp_notif_resp **resp) {
	return -EOPNOTSUPP;
}
int seccomp_notify_fd(const scmp_filter_ctx ctx) {
	return -EOPNOTSUPP;
}
void seccomp_notify_free(struct seccomp_notif *req, struct seccomp_notif_resp *resp) {
}
int seccomp_notify_id_valid(int fd, uint64_t id) {
	return -EOPNOTSUPP;
}
int seccomp_notify_receive(int fd, struct seccomp_notif *req) {
	return -EOPNOTSUPP;
}
int seccomp_notify_respond(int fd, struct seccomp_notif_resp *resp) {
	return -EOPNOTSUPP;
}

#endif
*/
import "C"

// Nonexported types
type scmpFilterAttr uint32

// Nonexported constants

const (
	filterAttrActDefault scmpFilterAttr = iota
	filterAttrActBadArch
	filterAttrNNP
	filterAttrTsync
	filterAttrLog
	filterAttrSSB
	filterAttrOptimize
	filterAttrRawRC
)

const (
	// An error return from certain libseccomp functions
	scmpError C.int = -1
	// Comparison boundaries to check for architecture validity
	archStart ScmpArch = ArchNative
	archEnd   ScmpArch = ArchRISCV64
	// Comparison boundaries to check for action validity
	actionStart ScmpAction = ActKillThread
	actionEnd   ScmpAction = ActKillProcess
	// Comparison boundaries to check for comparison operator validity
	compareOpStart ScmpCompareOp = CompareNotEqual
	compareOpEnd   ScmpCompareOp = CompareMaskedEqual
)

var (
	// errBadFilter is thrown on bad filter context.
	errBadFilter = errors.New("filter is invalid or uninitialized")
	errDefAction = errors.New("requested action matches default action of filter")
	// Constants representing library major, minor, and micro versions
	verMajor = uint(C.get_major_version())
	verMinor = uint(C.get_minor_version())
	verMicro = uint(C.get_micro_version())
)

// Nonexported functions

// checkVersion returns an error if the libseccomp version being used
// is less than the one specified by major, minor, and micro arguments.
// Argument op is an arbitrary non-empty operation description, which
// is used as a part of the error message returned.
//
// Most users should use checkAPI instead.
func checkVersion(op string, major, minor, micro uint) error {
	if (verMajor > major) ||
		(verMajor == major && verMinor > minor) ||
		(verMajor == major && verMinor == minor && verMicro >= micro) {
		return nil
	}
	return &VersionError{
		op:    op,
		major: major,
		minor: minor,
		micro: micro,
	}
}

func ensureSupportedVersion() error {
	return checkVersion("seccomp", 2, 3, 1)
}

// Get the API level
func getAPI() (uint, error) {
	api := C.seccomp_api_get()
	if api == 0 {
		return 0, errors.New("API level operations are not supported")
	}

	return uint(api), nil
}

// Set the API level
func setAPI(api uint) error {
	if retCode := C.seccomp_api_set(C.uint(api)); retCode != 0 {
		e := errRc(retCode)
		if e == syscall.EOPNOTSUPP {
			return errors.New("API level operations are not supported")
		}

		return fmt.Errorf("could not set API level: %w", e)
	}

	return nil
}

// Filter helpers

// Filter finalizer - ensure that kernel context for filters is freed
func filterFinalizer(f *ScmpFilter) {
	f.Release()
}

func errRc(rc C.int) error {
	return syscall.Errno(-1 * rc)
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
		return 0x0, errRc(retCode)
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
		return errRc(retCode)
	}

	return nil
}

// DOES NOT LOCK OR CHECK VALIDITY
// Assumes caller has already done this
// Wrapper for seccomp_rule_add_... functions
func (f *ScmpFilter) addRuleWrapper(call ScmpSyscall, action ScmpAction, exact bool, length C.uint, cond C.scmp_cast_t) error {
	if length != 0 && cond == nil {
		return errors.New("null conditions list, but length is nonzero")
	}

	var retCode C.int
	if exact {
		retCode = C.seccomp_rule_add_exact_array(f.filterCtx, action.toNative(), C.int(call), length, cond)
	} else {
		retCode = C.seccomp_rule_add_array(f.filterCtx, action.toNative(), C.int(call), length, cond)
	}

	if retCode != 0 {
		switch e := errRc(retCode); e {
		case syscall.EFAULT:
			return fmt.Errorf("unrecognized syscall %#x", int32(call))
		// libseccomp >= v2.5.0 returns EACCES, older versions return EPERM.
		// TODO: remove EPERM once libseccomp < v2.5.0 is not supported.
		case syscall.EPERM, syscall.EACCES:
			return errDefAction
		case syscall.EINVAL:
			return errors.New("two checks on same syscall argument")
		default:
			return e
		}
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
		argsArr := C.make_arg_cmp_array(C.uint(len(conds)))
		if argsArr == nil {
			return errors.New("error allocating memory for conditions")
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
		return errors.New("highest 16 bits must be zeroed except for Trace and Errno")
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
	case C.C_ARCH_PARISC:
		return ArchPARISC, nil
	case C.C_ARCH_PARISC64:
		return ArchPARISC64, nil
	case C.C_ARCH_RISCV64:
		return ArchRISCV64, nil
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
	case ArchPARISC:
		return C.C_ARCH_PARISC
	case ArchPARISC64:
		return C.C_ARCH_PARISC64
	case ArchRISCV64:
		return C.C_ARCH_RISCV64
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
	case C.C_ACT_KILL_PROCESS:
		return ActKillProcess, nil
	case C.C_ACT_KILL_THREAD:
		return ActKillThread, nil
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
	case C.C_ACT_NOTIFY:
		return ActNotify, nil
	default:
		return 0x0, fmt.Errorf("unrecognized action %#x", uint32(a))
	}
}

// Only use with sanitized actions, no error handling
func (a ScmpAction) toNative() C.uint32_t {
	switch a & 0xFFFF {
	case ActKillProcess:
		return C.C_ACT_KILL_PROCESS
	case ActKillThread:
		return C.C_ACT_KILL_THREAD
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
	case ActNotify:
		return C.C_ACT_NOTIFY
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
	case filterAttrSSB:
		return uint32(C.C_ATTRIBUTE_SSB)
	case filterAttrOptimize:
		return uint32(C.C_ATTRIBUTE_OPTIMIZE)
	case filterAttrRawRC:
		return uint32(C.C_ATTRIBUTE_SYSRAWRC)
	default:
		return 0x0
	}
}

func syscallFromNative(a C.int) ScmpSyscall {
	return ScmpSyscall(a)
}

func notifReqFromNative(req *C.struct_seccomp_notif) (*ScmpNotifReq, error) {
	scmpArgs := make([]uint64, 6)
	for i := 0; i < len(scmpArgs); i++ {
		scmpArgs[i] = uint64(req.data.args[i])
	}

	arch, err := archFromNative(req.data.arch)
	if err != nil {
		return nil, err
	}

	scmpData := ScmpNotifData{
		Syscall:      syscallFromNative(req.data.nr),
		Arch:         arch,
		InstrPointer: uint64(req.data.instruction_pointer),
		Args:         scmpArgs,
	}

	scmpReq := &ScmpNotifReq{
		ID:    uint64(req.id),
		Pid:   uint32(req.pid),
		Flags: uint32(req.flags),
		Data:  scmpData,
	}

	return scmpReq, nil
}

func (scmpResp *ScmpNotifResp) toNative(resp *C.struct_seccomp_notif_resp) {
	resp.id = C.__u64(scmpResp.ID)
	resp.val = C.__s64(scmpResp.Val)
	resp.error = (C.__s32(scmpResp.Error) * -1) // kernel requires a negated value
	resp.flags = C.__u32(scmpResp.Flags)
}

// checkAPI checks that both the API level and the seccomp version is equal to
// or greater than the specified minLevel and major, minor, micro,
// respectively, and returns an error otherwise. Argument op is an arbitrary
// non-empty operation description, used as a part of the error message
// returned.
func checkAPI(op string, minLevel uint, major, minor, micro uint) error {
	// Ignore error from getAPI, as it returns level == 0 in case of error.
	level, _ := getAPI()
	if level >= minLevel {
		return checkVersion(op, major, minor, micro)
	}
	return &VersionError{
		op:     op,
		curAPI: level,
		minAPI: minLevel,
		major:  major,
		minor:  minor,
		micro:  micro,
	}
}

// Userspace Notification API
// Calls to C.seccomp_notify* hidden from seccomp.go

func notifSupported() error {
	return checkAPI("seccomp notification", 6, 2, 5, 0)
}

func (f *ScmpFilter) getNotifFd() (ScmpFd, error) {
	f.lock.Lock()
	defer f.lock.Unlock()

	if !f.valid {
		return -1, errBadFilter
	}
	if err := notifSupported(); err != nil {
		return -1, err
	}

	fd := C.seccomp_notify_fd(f.filterCtx)

	return ScmpFd(fd), nil
}

func notifReceive(fd ScmpFd) (*ScmpNotifReq, error) {
	var req *C.struct_seccomp_notif
	var resp *C.struct_seccomp_notif_resp

	if err := notifSupported(); err != nil {
		return nil, err
	}

	// we only use the request here; the response is unused
	if retCode := C.seccomp_notify_alloc(&req, &resp); retCode != 0 {
		return nil, errRc(retCode)
	}

	defer func() {
		C.seccomp_notify_free(req, resp)
	}()

	for {
		retCode, errno := C.seccomp_notify_receive(C.int(fd), req)
		if retCode == 0 {
			break
		}

		if errno == syscall.EINTR {
			continue
		}

		if errno == syscall.ENOENT {
			return nil, errno
		}

		return nil, errRc(retCode)
	}

	return notifReqFromNative(req)
}

func notifRespond(fd ScmpFd, scmpResp *ScmpNotifResp) error {
	var req *C.struct_seccomp_notif
	var resp *C.struct_seccomp_notif_resp

	if err := notifSupported(); err != nil {
		return err
	}

	// we only use the response here; the request is discarded
	if retCode := C.seccomp_notify_alloc(&req, &resp); retCode != 0 {
		return errRc(retCode)
	}

	defer func() {
		C.seccomp_notify_free(req, resp)
	}()

	scmpResp.toNative(resp)

	for {
		retCode, errno := C.seccomp_notify_respond(C.int(fd), resp)
		if retCode == 0 {
			break
		}

		if errno == syscall.EINTR {
			continue
		}

		if errno == syscall.ENOENT {
			return errno
		}

		return errRc(retCode)
	}

	return nil
}

func notifIDValid(fd ScmpFd, id uint64) error {
	if err := notifSupported(); err != nil {
		return err
	}

	for {
		retCode, errno := C.seccomp_notify_id_valid(C.int(fd), C.uint64_t(id))
		if retCode == 0 {
			break
		}

		if errno == syscall.EINTR {
			continue
		}

		if errno == syscall.ENOENT {
			return errno
		}

		return errRc(retCode)
	}

	return nil
}
