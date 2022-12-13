// Public API specification for libseccomp Go bindings
// Contains public API for the bindings

// Package seccomp provides bindings for libseccomp, a library wrapping the Linux
// seccomp syscall. Seccomp enables an application to restrict system call use
// for itself and its children.
package seccomp

import (
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"unsafe"
)

// #include <stdlib.h>
// #include <seccomp.h>
import "C"

// Exported types

// VersionError represents an error when either the system libseccomp version
// or the kernel version is too old to perform the operation requested.
type VersionError struct {
	op                  string // operation that failed or would fail
	major, minor, micro uint   // minimally required libseccomp version
	curAPI, minAPI      uint   // current and minimally required API versions
}

func init() {
	// This forces the cgo libseccomp to initialize its internal API support state,
	// which is necessary on older versions of libseccomp in order to work
	// correctly.
	_, _ = getAPI()
}

func (e VersionError) Error() string {
	if e.minAPI != 0 {
		return fmt.Sprintf("%s requires libseccomp >= %d.%d.%d and API level >= %d "+
			"(current version: %d.%d.%d, API level: %d)",
			e.op, e.major, e.minor, e.micro, e.minAPI,
			verMajor, verMinor, verMicro, e.curAPI)
	}
	return fmt.Sprintf("%s requires libseccomp >= %d.%d.%d (current version: %d.%d.%d)",
		e.op, e.major, e.minor, e.micro, verMajor, verMinor, verMicro)
}

// ScmpArch represents a CPU architecture. Seccomp can restrict syscalls on a
// per-architecture basis.
type ScmpArch uint

// ScmpAction represents an action to be taken on a filter rule match in
// libseccomp
type ScmpAction uint

// ScmpCompareOp represents a comparison operator which can be used in a filter
// rule
type ScmpCompareOp uint

// ScmpCondition represents a rule in a libseccomp filter context
type ScmpCondition struct {
	Argument uint          `json:"argument,omitempty"`
	Op       ScmpCompareOp `json:"operator,omitempty"`
	Operand1 uint64        `json:"operand_one,omitempty"`
	Operand2 uint64        `json:"operand_two,omitempty"`
}

// Seccomp userspace notification structures associated with filters that use the ActNotify action.

// ScmpSyscall identifies a Linux System Call by its number.
type ScmpSyscall int32

// ScmpFd represents a file-descriptor used for seccomp userspace notifications.
type ScmpFd int32

// ScmpNotifData describes the system call context that triggered a notification.
//
// Syscall:      the syscall number
// Arch:         the filter architecture
// InstrPointer: address of the instruction that triggered a notification
// Args:         arguments (up to 6) for the syscall
//
type ScmpNotifData struct {
	Syscall      ScmpSyscall `json:"syscall,omitempty"`
	Arch         ScmpArch    `json:"arch,omitempty"`
	InstrPointer uint64      `json:"instr_pointer,omitempty"`
	Args         []uint64    `json:"args,omitempty"`
}

// ScmpNotifReq represents a seccomp userspace notification. See NotifReceive() for
// info on how to pull such a notification.
//
// ID:    notification ID
// Pid:   process that triggered the notification event
// Flags: filter flags (see seccomp(2))
// Data:  system call context that triggered the notification
//
type ScmpNotifReq struct {
	ID    uint64        `json:"id,omitempty"`
	Pid   uint32        `json:"pid,omitempty"`
	Flags uint32        `json:"flags,omitempty"`
	Data  ScmpNotifData `json:"data,omitempty"`
}

// ScmpNotifResp represents a seccomp userspace notification response. See NotifRespond()
// for info on how to push such a response.
//
// ID:    notification ID (must match the corresponding ScmpNotifReq ID)
// Error: must be 0 if no error occurred, or an error constant from package
//        syscall (e.g., syscall.EPERM, etc). In the latter case, it's used
//        as an error return from the syscall that created the notification.
// Val:   return value for the syscall that created the notification. Only
//        relevant if Error is 0.
// Flags: userspace notification response flag (e.g., NotifRespFlagContinue)
//
type ScmpNotifResp struct {
	ID    uint64 `json:"id,omitempty"`
	Error int32  `json:"error,omitempty"`
	Val   uint64 `json:"val,omitempty"`
	Flags uint32 `json:"flags,omitempty"`
}

// Exported Constants

const (
	// Valid architectures recognized by libseccomp
	// PowerPC and S390(x) architectures are unavailable below library version
	// v2.3.0 and will returns errors if used with incompatible libraries

	// ArchInvalid is a placeholder to ensure uninitialized ScmpArch
	// variables are invalid
	ArchInvalid ScmpArch = iota
	// ArchNative is the native architecture of the kernel
	ArchNative
	// ArchX86 represents 32-bit x86 syscalls
	ArchX86
	// ArchAMD64 represents 64-bit x86-64 syscalls
	ArchAMD64
	// ArchX32 represents 64-bit x86-64 syscalls (32-bit pointers)
	ArchX32
	// ArchARM represents 32-bit ARM syscalls
	ArchARM
	// ArchARM64 represents 64-bit ARM syscalls
	ArchARM64
	// ArchMIPS represents 32-bit MIPS syscalls
	ArchMIPS
	// ArchMIPS64 represents 64-bit MIPS syscalls
	ArchMIPS64
	// ArchMIPS64N32 represents 64-bit MIPS syscalls (32-bit pointers)
	ArchMIPS64N32
	// ArchMIPSEL represents 32-bit MIPS syscalls (little endian)
	ArchMIPSEL
	// ArchMIPSEL64 represents 64-bit MIPS syscalls (little endian)
	ArchMIPSEL64
	// ArchMIPSEL64N32 represents 64-bit MIPS syscalls (little endian,
	// 32-bit pointers)
	ArchMIPSEL64N32
	// ArchPPC represents 32-bit POWERPC syscalls
	ArchPPC
	// ArchPPC64 represents 64-bit POWER syscalls (big endian)
	ArchPPC64
	// ArchPPC64LE represents 64-bit POWER syscalls (little endian)
	ArchPPC64LE
	// ArchS390 represents 31-bit System z/390 syscalls
	ArchS390
	// ArchS390X represents 64-bit System z/390 syscalls
	ArchS390X
	// ArchPARISC represents 32-bit PA-RISC
	ArchPARISC
	// ArchPARISC64 represents 64-bit PA-RISC
	ArchPARISC64
	// ArchRISCV64 represents RISCV64
	ArchRISCV64
)

const (
	// Supported actions on filter match

	// ActInvalid is a placeholder to ensure uninitialized ScmpAction
	// variables are invalid
	ActInvalid ScmpAction = iota
	// ActKillThread kills the thread that violated the rule.
	// All other threads from the same thread group will continue to execute.
	ActKillThread
	// ActTrap throws SIGSYS
	ActTrap
	// ActNotify triggers a userspace notification. This action is only usable when
	// libseccomp API level 6 or higher is supported.
	ActNotify
	// ActErrno causes the syscall to return a negative error code. This
	// code can be set with the SetReturnCode method
	ActErrno
	// ActTrace causes the syscall to notify tracing processes with the
	// given error code. This code can be set with the SetReturnCode method
	ActTrace
	// ActAllow permits the syscall to continue execution
	ActAllow
	// ActLog permits the syscall to continue execution after logging it.
	// This action is only usable when libseccomp API level 3 or higher is
	// supported.
	ActLog
	// ActKillProcess kills the process that violated the rule.
	// All threads in the thread group are also terminated.
	// This action is only usable when libseccomp API level 3 or higher is
	// supported.
	ActKillProcess
	// ActKill kills the thread that violated the rule.
	// All other threads from the same thread group will continue to execute.
	//
	// Deprecated: use ActKillThread
	ActKill = ActKillThread
)

const (
	// These are comparison operators used in conditional seccomp rules
	// They are used to compare the value of a single argument of a syscall
	// against a user-defined constant

	// CompareInvalid is a placeholder to ensure uninitialized ScmpCompareOp
	// variables are invalid
	CompareInvalid ScmpCompareOp = iota
	// CompareNotEqual returns true if the argument is not equal to the
	// given value
	CompareNotEqual
	// CompareLess returns true if the argument is less than the given value
	CompareLess
	// CompareLessOrEqual returns true if the argument is less than or equal
	// to the given value
	CompareLessOrEqual
	// CompareEqual returns true if the argument is equal to the given value
	CompareEqual
	// CompareGreaterEqual returns true if the argument is greater than or
	// equal to the given value
	CompareGreaterEqual
	// CompareGreater returns true if the argument is greater than the given
	// value
	CompareGreater
	// CompareMaskedEqual returns true if the masked argument value is
	// equal to the masked datum value. Mask is the first argument, and
	// datum is the second one.
	CompareMaskedEqual
)

// ErrSyscallDoesNotExist represents an error condition where
// libseccomp is unable to resolve the syscall
var ErrSyscallDoesNotExist = fmt.Errorf("could not resolve syscall name")

const (
	// Userspace notification response flags

	// NotifRespFlagContinue tells the kernel to continue executing the system
	// call that triggered the notification. Must only be used when the notification
	// response's error is 0.
	NotifRespFlagContinue uint32 = 1
)

// Helpers for types

// GetArchFromString returns an ScmpArch constant from a string representing an
// architecture
func GetArchFromString(arch string) (ScmpArch, error) {
	if err := ensureSupportedVersion(); err != nil {
		return ArchInvalid, err
	}

	switch strings.ToLower(arch) {
	case "x86":
		return ArchX86, nil
	case "amd64", "x86-64", "x86_64", "x64":
		return ArchAMD64, nil
	case "x32":
		return ArchX32, nil
	case "arm":
		return ArchARM, nil
	case "arm64", "aarch64":
		return ArchARM64, nil
	case "mips":
		return ArchMIPS, nil
	case "mips64":
		return ArchMIPS64, nil
	case "mips64n32":
		return ArchMIPS64N32, nil
	case "mipsel":
		return ArchMIPSEL, nil
	case "mipsel64":
		return ArchMIPSEL64, nil
	case "mipsel64n32":
		return ArchMIPSEL64N32, nil
	case "ppc":
		return ArchPPC, nil
	case "ppc64":
		return ArchPPC64, nil
	case "ppc64le":
		return ArchPPC64LE, nil
	case "s390":
		return ArchS390, nil
	case "s390x":
		return ArchS390X, nil
	case "parisc":
		return ArchPARISC, nil
	case "parisc64":
		return ArchPARISC64, nil
	case "riscv64":
		return ArchRISCV64, nil
	default:
		return ArchInvalid, fmt.Errorf("cannot convert unrecognized string %q", arch)
	}
}

// String returns a string representation of an architecture constant
func (a ScmpArch) String() string {
	switch a {
	case ArchX86:
		return "x86"
	case ArchAMD64:
		return "amd64"
	case ArchX32:
		return "x32"
	case ArchARM:
		return "arm"
	case ArchARM64:
		return "arm64"
	case ArchMIPS:
		return "mips"
	case ArchMIPS64:
		return "mips64"
	case ArchMIPS64N32:
		return "mips64n32"
	case ArchMIPSEL:
		return "mipsel"
	case ArchMIPSEL64:
		return "mipsel64"
	case ArchMIPSEL64N32:
		return "mipsel64n32"
	case ArchPPC:
		return "ppc"
	case ArchPPC64:
		return "ppc64"
	case ArchPPC64LE:
		return "ppc64le"
	case ArchS390:
		return "s390"
	case ArchS390X:
		return "s390x"
	case ArchPARISC:
		return "parisc"
	case ArchPARISC64:
		return "parisc64"
	case ArchRISCV64:
		return "riscv64"
	case ArchNative:
		return "native"
	case ArchInvalid:
		return "Invalid architecture"
	default:
		return fmt.Sprintf("Unknown architecture %#x", uint(a))
	}
}

// String returns a string representation of a comparison operator constant
func (a ScmpCompareOp) String() string {
	switch a {
	case CompareNotEqual:
		return "Not equal"
	case CompareLess:
		return "Less than"
	case CompareLessOrEqual:
		return "Less than or equal to"
	case CompareEqual:
		return "Equal"
	case CompareGreaterEqual:
		return "Greater than or equal to"
	case CompareGreater:
		return "Greater than"
	case CompareMaskedEqual:
		return "Masked equality"
	case CompareInvalid:
		return "Invalid comparison operator"
	default:
		return fmt.Sprintf("Unrecognized comparison operator %#x", uint(a))
	}
}

// String returns a string representation of a seccomp match action
func (a ScmpAction) String() string {
	switch a & 0xFFFF {
	case ActKillThread:
		return "Action: Kill thread"
	case ActKillProcess:
		return "Action: Kill process"
	case ActTrap:
		return "Action: Send SIGSYS"
	case ActErrno:
		return fmt.Sprintf("Action: Return error code %d", (a >> 16))
	case ActTrace:
		return fmt.Sprintf("Action: Notify tracing processes with code %d",
			(a >> 16))
	case ActNotify:
		return "Action: Notify userspace"
	case ActLog:
		return "Action: Log system call"
	case ActAllow:
		return "Action: Allow system call"
	default:
		return fmt.Sprintf("Unrecognized Action %#x", uint(a))
	}
}

// SetReturnCode adds a return code to a supporting ScmpAction, clearing any
// existing code Only valid on ActErrno and ActTrace. Takes no action otherwise.
// Accepts 16-bit return code as argument.
// Returns a valid ScmpAction of the original type with the new error code set.
func (a ScmpAction) SetReturnCode(code int16) ScmpAction {
	aTmp := a & 0x0000FFFF
	if aTmp == ActErrno || aTmp == ActTrace {
		return (aTmp | (ScmpAction(code)&0xFFFF)<<16)
	}
	return a
}

// GetReturnCode returns the return code of an ScmpAction
func (a ScmpAction) GetReturnCode() int16 {
	return int16(a >> 16)
}

// General utility functions

// GetLibraryVersion returns the version of the library the bindings are built
// against.
// The version is formatted as follows: Major.Minor.Micro
func GetLibraryVersion() (major, minor, micro uint) {
	return verMajor, verMinor, verMicro
}

// GetAPI returns the API level supported by the system.
// Returns a positive int containing the API level, or 0 with an error if the
// API level could not be detected due to the library being older than v2.4.0.
// See the seccomp_api_get(3) man page for details on available API levels:
// https://github.com/seccomp/libseccomp/blob/main/doc/man/man3/seccomp_api_get.3
func GetAPI() (uint, error) {
	return getAPI()
}

// SetAPI forcibly sets the API level. General use of this function is strongly
// discouraged.
// Returns an error if the API level could not be set. An error is always
// returned if the library is older than v2.4.0
// See the seccomp_api_get(3) man page for details on available API levels:
// https://github.com/seccomp/libseccomp/blob/main/doc/man/man3/seccomp_api_get.3
func SetAPI(api uint) error {
	return setAPI(api)
}

// Syscall functions

// GetName retrieves the name of a syscall from its number.
// Acts on any syscall number.
// Returns either a string containing the name of the syscall, or an error.
func (s ScmpSyscall) GetName() (string, error) {
	return s.GetNameByArch(ArchNative)
}

// GetNameByArch retrieves the name of a syscall from its number for a given
// architecture.
// Acts on any syscall number.
// Accepts a valid architecture constant.
// Returns either a string containing the name of the syscall, or an error.
// if the syscall is unrecognized or an issue occurred.
func (s ScmpSyscall) GetNameByArch(arch ScmpArch) (string, error) {
	if err := sanitizeArch(arch); err != nil {
		return "", err
	}

	cString := C.seccomp_syscall_resolve_num_arch(arch.toNative(), C.int(s))
	if cString == nil {
		return "", ErrSyscallDoesNotExist
	}
	defer C.free(unsafe.Pointer(cString))

	finalStr := C.GoString(cString)
	return finalStr, nil
}

// GetSyscallFromName returns the number of a syscall by name on the kernel's
// native architecture.
// Accepts a string containing the name of a syscall.
// Returns the number of the syscall, or an error if no syscall with that name
// was found.
func GetSyscallFromName(name string) (ScmpSyscall, error) {
	if err := ensureSupportedVersion(); err != nil {
		return 0, err
	}

	cString := C.CString(name)
	defer C.free(unsafe.Pointer(cString))

	result := C.seccomp_syscall_resolve_name(cString)
	if result == scmpError {
		return 0, ErrSyscallDoesNotExist
	}

	return ScmpSyscall(result), nil
}

// GetSyscallFromNameByArch returns the number of a syscall by name for a given
// architecture's ABI.
// Accepts the name of a syscall and an architecture constant.
// Returns the number of the syscall, or an error if an invalid architecture is
// passed or a syscall with that name was not found.
func GetSyscallFromNameByArch(name string, arch ScmpArch) (ScmpSyscall, error) {
	if err := ensureSupportedVersion(); err != nil {
		return 0, err
	}
	if err := sanitizeArch(arch); err != nil {
		return 0, err
	}

	cString := C.CString(name)
	defer C.free(unsafe.Pointer(cString))

	result := C.seccomp_syscall_resolve_name_arch(arch.toNative(), cString)
	if result == scmpError {
		return 0, ErrSyscallDoesNotExist
	}

	return ScmpSyscall(result), nil
}

// MakeCondition creates and returns a new condition to attach to a filter rule.
// Associated rules will only match if this condition is true.
// Accepts the number the argument we are checking, and a comparison operator
// and value to compare to.
// The rule will match if argument $arg (zero-indexed) of the syscall is
// $COMPARE_OP the provided comparison value.
// Some comparison operators accept two values. Masked equals, for example,
// will mask $arg of the syscall with the second value provided (via bitwise
// AND) and then compare against the first value provided.
// For example, in the less than or equal case, if the syscall argument was
// 0 and the value provided was 1, the condition would match, as 0 is less
// than or equal to 1.
// Return either an error on bad argument or a valid ScmpCondition struct.
func MakeCondition(arg uint, comparison ScmpCompareOp, values ...uint64) (ScmpCondition, error) {
	var condStruct ScmpCondition

	if err := ensureSupportedVersion(); err != nil {
		return condStruct, err
	}

	if err := sanitizeCompareOp(comparison); err != nil {
		return condStruct, err
	} else if arg > 5 {
		return condStruct, fmt.Errorf("syscalls only have up to 6 arguments (%d given)", arg)
	} else if len(values) > 2 {
		return condStruct, fmt.Errorf("conditions can have at most 2 arguments (%d given)", len(values))
	} else if len(values) == 0 {
		return condStruct, fmt.Errorf("must provide at least one value to compare against")
	}

	condStruct.Argument = arg
	condStruct.Op = comparison
	condStruct.Operand1 = values[0]
	if len(values) == 2 {
		condStruct.Operand2 = values[1]
	} else {
		condStruct.Operand2 = 0 // Unused
	}

	return condStruct, nil
}

// Utility Functions

// GetNativeArch returns architecture token representing the native kernel
// architecture
func GetNativeArch() (ScmpArch, error) {
	if err := ensureSupportedVersion(); err != nil {
		return ArchInvalid, err
	}

	arch := C.seccomp_arch_native()

	return archFromNative(arch)
}

// Public Filter API

// ScmpFilter represents a filter context in libseccomp.
// A filter context is initially empty. Rules can be added to it, and it can
// then be loaded into the kernel.
type ScmpFilter struct {
	filterCtx C.scmp_filter_ctx
	valid     bool
	lock      sync.Mutex
}

// NewFilter creates and returns a new filter context.  Accepts a default action to be
// taken for syscalls which match no rules in the filter.
// Returns a reference to a valid filter context, or nil and an error
// if the filter context could not be created or an invalid default action was given.
func NewFilter(defaultAction ScmpAction) (*ScmpFilter, error) {
	if err := ensureSupportedVersion(); err != nil {
		return nil, err
	}

	if err := sanitizeAction(defaultAction); err != nil {
		return nil, err
	}

	fPtr := C.seccomp_init(defaultAction.toNative())
	if fPtr == nil {
		return nil, fmt.Errorf("could not create filter")
	}

	filter := new(ScmpFilter)
	filter.filterCtx = fPtr
	filter.valid = true
	runtime.SetFinalizer(filter, filterFinalizer)

	// Enable TSync so all goroutines will receive the same rules.
	// If the kernel does not support TSYNC, allow us to continue without error.
	if err := filter.setFilterAttr(filterAttrTsync, 0x1); err != nil && err != syscall.ENOTSUP {
		filter.Release()
		return nil, fmt.Errorf("could not create filter - error setting tsync bit: %v", err)
	}

	return filter, nil
}

// IsValid determines whether a filter context is valid to use.
// Some operations (Release and Merge) render filter contexts invalid and
// consequently prevent further use.
func (f *ScmpFilter) IsValid() bool {
	f.lock.Lock()
	defer f.lock.Unlock()

	return f.valid
}

// Reset resets a filter context, removing all its existing state.
// Accepts a new default action to be taken for syscalls which do not match.
// Returns an error if the filter or action provided are invalid.
func (f *ScmpFilter) Reset(defaultAction ScmpAction) error {
	f.lock.Lock()
	defer f.lock.Unlock()

	if err := sanitizeAction(defaultAction); err != nil {
		return err
	} else if !f.valid {
		return errBadFilter
	}

	if retCode := C.seccomp_reset(f.filterCtx, defaultAction.toNative()); retCode != 0 {
		return errRc(retCode)
	}

	return nil
}

// Release releases a filter context, freeing its memory. Should be called after
// loading into the kernel, when the filter is no longer needed.
// After calling this function, the given filter is no longer valid and cannot
// be used.
// Release() will be invoked automatically when a filter context is garbage
// collected, but can also be called manually to free memory.
func (f *ScmpFilter) Release() {
	f.lock.Lock()
	defer f.lock.Unlock()

	if !f.valid {
		return
	}

	f.valid = false
	C.seccomp_release(f.filterCtx)
}

// Merge merges two filter contexts.
// The source filter src will be released as part of the process, and will no
// longer be usable or valid after this call.
// To be merged, filters must NOT share any architectures, and all their
// attributes (Default Action, Bad Arch Action, and No New Privs bools)
// must match.
// The filter src will be merged into the filter this is called on.
// The architectures of the src filter not present in the destination, and all
// associated rules, will be added to the destination.
// Returns an error if merging the filters failed.
func (f *ScmpFilter) Merge(src *ScmpFilter) error {
	f.lock.Lock()
	defer f.lock.Unlock()

	src.lock.Lock()
	defer src.lock.Unlock()

	if !src.valid || !f.valid {
		return fmt.Errorf("one or more of the filter contexts is invalid or uninitialized")
	}

	// Merge the filters
	if retCode := C.seccomp_merge(f.filterCtx, src.filterCtx); retCode != 0 {
		e := errRc(retCode)
		if e == syscall.EINVAL {
			return fmt.Errorf("filters could not be merged due to a mismatch in attributes or invalid filter")
		}
		return e
	}

	src.valid = false

	return nil
}

// IsArchPresent checks if an architecture is present in a filter.
// If a filter contains an architecture, it uses its default action for
// syscalls which do not match rules in it, and its rules can match syscalls
// for that ABI.
// If a filter does not contain an architecture, all syscalls made to that
// kernel ABI will fail with the filter's default Bad Architecture Action
// (by default, killing the process).
// Accepts an architecture constant.
// Returns true if the architecture is present in the filter, false otherwise,
// and an error on an invalid filter context, architecture constant, or an
// issue with the call to libseccomp.
func (f *ScmpFilter) IsArchPresent(arch ScmpArch) (bool, error) {
	f.lock.Lock()
	defer f.lock.Unlock()

	if err := sanitizeArch(arch); err != nil {
		return false, err
	} else if !f.valid {
		return false, errBadFilter
	}

	if retCode := C.seccomp_arch_exist(f.filterCtx, arch.toNative()); retCode != 0 {
		e := errRc(retCode)
		if e == syscall.EEXIST {
			// -EEXIST is "arch not present"
			return false, nil
		}
		return false, e
	}

	return true, nil
}

// AddArch adds an architecture to the filter.
// Accepts an architecture constant.
// Returns an error on invalid filter context or architecture token, or an
// issue with the call to libseccomp.
func (f *ScmpFilter) AddArch(arch ScmpArch) error {
	f.lock.Lock()
	defer f.lock.Unlock()

	if err := sanitizeArch(arch); err != nil {
		return err
	} else if !f.valid {
		return errBadFilter
	}

	// Libseccomp returns -EEXIST if the specified architecture is already
	// present. Succeed silently in this case, as it's not fatal, and the
	// architecture is present already.
	if retCode := C.seccomp_arch_add(f.filterCtx, arch.toNative()); retCode != 0 {
		if e := errRc(retCode); e != syscall.EEXIST {
			return e
		}
	}

	return nil
}

// RemoveArch removes an architecture from the filter.
// Accepts an architecture constant.
// Returns an error on invalid filter context or architecture token, or an
// issue with the call to libseccomp.
func (f *ScmpFilter) RemoveArch(arch ScmpArch) error {
	f.lock.Lock()
	defer f.lock.Unlock()

	if err := sanitizeArch(arch); err != nil {
		return err
	} else if !f.valid {
		return errBadFilter
	}

	// Similar to AddArch, -EEXIST is returned if the arch is not present
	// Succeed silently in that case, this is not fatal and the architecture
	// is not present in the filter after RemoveArch
	if retCode := C.seccomp_arch_remove(f.filterCtx, arch.toNative()); retCode != 0 {
		if e := errRc(retCode); e != syscall.EEXIST {
			return e
		}
	}

	return nil
}

// Load loads a filter context into the kernel.
// Returns an error if the filter context is invalid or the syscall failed.
func (f *ScmpFilter) Load() error {
	f.lock.Lock()
	defer f.lock.Unlock()

	if !f.valid {
		return errBadFilter
	}

	if retCode := C.seccomp_load(f.filterCtx); retCode != 0 {
		return errRc(retCode)
	}

	return nil
}

// GetDefaultAction returns the default action taken on a syscall which does not
// match a rule in the filter, or an error if an issue was encountered
// retrieving the value.
func (f *ScmpFilter) GetDefaultAction() (ScmpAction, error) {
	action, err := f.getFilterAttr(filterAttrActDefault)
	if err != nil {
		return 0x0, err
	}

	return actionFromNative(action)
}

// GetBadArchAction returns the default action taken on a syscall for an
// architecture not in the filter, or an error if an issue was encountered
// retrieving the value.
func (f *ScmpFilter) GetBadArchAction() (ScmpAction, error) {
	action, err := f.getFilterAttr(filterAttrActBadArch)
	if err != nil {
		return 0x0, err
	}

	return actionFromNative(action)
}

// GetNoNewPrivsBit returns the current state the No New Privileges bit will be set
// to on the filter being loaded, or an error if an issue was encountered
// retrieving the value.
// The No New Privileges bit tells the kernel that new processes run with exec()
// cannot gain more privileges than the process that ran exec().
// For example, a process with No New Privileges set would be unable to exec
// setuid/setgid executables.
func (f *ScmpFilter) GetNoNewPrivsBit() (bool, error) {
	noNewPrivs, err := f.getFilterAttr(filterAttrNNP)
	if err != nil {
		return false, err
	}

	if noNewPrivs == 0 {
		return false, nil
	}

	return true, nil
}

// GetLogBit returns the current state the Log bit will be set to on the filter
// being loaded, or an error if an issue was encountered retrieving the value.
// The Log bit tells the kernel that all actions taken by the filter, with the
// exception of ActAllow, should be logged.
// The Log bit is only usable when libseccomp API level 3 or higher is
// supported.
func (f *ScmpFilter) GetLogBit() (bool, error) {
	log, err := f.getFilterAttr(filterAttrLog)
	if err != nil {
		if e := checkAPI("GetLogBit", 3, 2, 4, 0); e != nil {
			err = e
		}

		return false, err
	}

	if log == 0 {
		return false, nil
	}

	return true, nil
}

// GetSSB returns the current state the SSB bit will be set to on the filter
// being loaded, or an error if an issue was encountered retrieving the value.
// The SSB bit tells the kernel that a seccomp user is not interested in enabling
// Speculative Store Bypass mitigation.
// The SSB bit is only usable when libseccomp API level 4 or higher is
// supported.
func (f *ScmpFilter) GetSSB() (bool, error) {
	ssb, err := f.getFilterAttr(filterAttrSSB)
	if err != nil {
		if e := checkAPI("GetSSB", 4, 2, 5, 0); e != nil {
			err = e
		}

		return false, err
	}

	if ssb == 0 {
		return false, nil
	}

	return true, nil
}

// GetOptimize returns the current optimization level of the filter,
// or an error if an issue was encountered retrieving the value.
// See SetOptimize for more details.
func (f *ScmpFilter) GetOptimize() (int, error) {
	level, err := f.getFilterAttr(filterAttrOptimize)
	if err != nil {
		if e := checkAPI("GetOptimize", 4, 2, 5, 0); e != nil {
			err = e
		}

		return 0, err
	}

	return int(level), nil
}

// GetRawRC returns the current state of RawRC flag, or an error
// if an issue was encountered retrieving the value.
// See SetRawRC for more details.
func (f *ScmpFilter) GetRawRC() (bool, error) {
	rawrc, err := f.getFilterAttr(filterAttrRawRC)
	if err != nil {
		if e := checkAPI("GetRawRC", 4, 2, 5, 0); e != nil {
			err = e
		}

		return false, err
	}

	if rawrc == 0 {
		return false, nil
	}

	return true, nil
}

// SetBadArchAction sets the default action taken on a syscall for an
// architecture not in the filter, or an error if an issue was encountered
// setting the value.
func (f *ScmpFilter) SetBadArchAction(action ScmpAction) error {
	if err := sanitizeAction(action); err != nil {
		return err
	}

	return f.setFilterAttr(filterAttrActBadArch, action.toNative())
}

// SetNoNewPrivsBit sets the state of the No New Privileges bit, which will be
// applied on filter load, or an error if an issue was encountered setting the
// value.
// Filters with No New Privileges set to 0 can only be loaded if the process
// has the CAP_SYS_ADMIN capability.
func (f *ScmpFilter) SetNoNewPrivsBit(state bool) error {
	var toSet C.uint32_t = 0x0

	if state {
		toSet = 0x1
	}

	return f.setFilterAttr(filterAttrNNP, toSet)
}

// SetLogBit sets the state of the Log bit, which will be applied on filter
// load, or an error if an issue was encountered setting the value.
// The Log bit is only usable when libseccomp API level 3 or higher is
// supported.
func (f *ScmpFilter) SetLogBit(state bool) error {
	var toSet C.uint32_t = 0x0

	if state {
		toSet = 0x1
	}

	err := f.setFilterAttr(filterAttrLog, toSet)
	if err != nil {
		if e := checkAPI("SetLogBit", 3, 2, 4, 0); e != nil {
			err = e
		}
	}

	return err
}

// SetSSB sets the state of the SSB bit, which will be applied on filter
// load, or an error if an issue was encountered setting the value.
// The SSB bit is only usable when libseccomp API level 4 or higher is
// supported.
func (f *ScmpFilter) SetSSB(state bool) error {
	var toSet C.uint32_t = 0x0

	if state {
		toSet = 0x1
	}

	err := f.setFilterAttr(filterAttrSSB, toSet)
	if err != nil {
		if e := checkAPI("SetSSB", 4, 2, 5, 0); e != nil {
			err = e
		}
	}

	return err
}

// SetOptimize sets optimization level of the seccomp filter. By default
// libseccomp generates a set of sequential "if" statements for each rule in
// the filter. SetSyscallPriority can be used to prioritize the order for the
// default cause. The binary tree optimization sorts by syscall numbers and
// generates consistent O(log n) filter traversal for every rule in the filter.
// The binary tree may be advantageous for large filters. Note that
// SetSyscallPriority is ignored when level == 2.
//
// The different optimization levels are:
// 0: Reserved value, not currently used.
// 1: Rules sorted by priority and complexity (DEFAULT).
// 2: Binary tree sorted by syscall number.
func (f *ScmpFilter) SetOptimize(level int) error {
	cLevel := C.uint32_t(level)

	err := f.setFilterAttr(filterAttrOptimize, cLevel)
	if err != nil {
		if e := checkAPI("SetOptimize", 4, 2, 5, 0); e != nil {
			err = e
		}
	}

	return err
}

// SetRawRC sets whether libseccomp should pass system error codes back to the
// caller, instead of the default ECANCELED. Defaults to false.
func (f *ScmpFilter) SetRawRC(state bool) error {
	var toSet C.uint32_t = 0x0

	if state {
		toSet = 0x1
	}

	err := f.setFilterAttr(filterAttrRawRC, toSet)
	if err != nil {
		if e := checkAPI("SetRawRC", 4, 2, 5, 0); e != nil {
			err = e
		}
	}

	return err
}

// SetSyscallPriority sets a syscall's priority.
// This provides a hint to the filter generator in libseccomp about the
// importance of this syscall. High-priority syscalls are placed
// first in the filter code, and incur less overhead (at the expense of
// lower-priority syscalls).
func (f *ScmpFilter) SetSyscallPriority(call ScmpSyscall, priority uint8) error {
	f.lock.Lock()
	defer f.lock.Unlock()

	if !f.valid {
		return errBadFilter
	}

	if retCode := C.seccomp_syscall_priority(f.filterCtx, C.int(call),
		C.uint8_t(priority)); retCode != 0 {
		return errRc(retCode)
	}

	return nil
}

// AddRule adds a single rule for an unconditional action on a syscall.
// Accepts the number of the syscall and the action to be taken on the call
// being made.
// Returns an error if an issue was encountered adding the rule.
func (f *ScmpFilter) AddRule(call ScmpSyscall, action ScmpAction) error {
	return f.addRuleGeneric(call, action, false, nil)
}

// AddRuleExact adds a single rule for an unconditional action on a syscall.
// Accepts the number of the syscall and the action to be taken on the call
// being made.
// No modifications will be made to the rule, and it will fail to add if it
// cannot be applied to the current architecture without modification.
// The rule will function exactly as described, but it may not function identically
// (or be able to be applied to) all architectures.
// Returns an error if an issue was encountered adding the rule.
func (f *ScmpFilter) AddRuleExact(call ScmpSyscall, action ScmpAction) error {
	return f.addRuleGeneric(call, action, true, nil)
}

// AddRuleConditional adds a single rule for a conditional action on a syscall.
// Returns an error if an issue was encountered adding the rule.
// All conditions must match for the rule to match.
func (f *ScmpFilter) AddRuleConditional(call ScmpSyscall, action ScmpAction, conds []ScmpCondition) error {
	return f.addRuleGeneric(call, action, false, conds)
}

// AddRuleConditionalExact adds a single rule for a conditional action on a
// syscall.
// No modifications will be made to the rule, and it will fail to add if it
// cannot be applied to the current architecture without modification.
// The rule will function exactly as described, but it may not function identically
// (or be able to be applied to) all architectures.
// Returns an error if an issue was encountered adding the rule.
func (f *ScmpFilter) AddRuleConditionalExact(call ScmpSyscall, action ScmpAction, conds []ScmpCondition) error {
	return f.addRuleGeneric(call, action, true, conds)
}

// ExportPFC output PFC-formatted, human-readable dump of a filter context's
// rules to a file.
// Accepts file to write to (must be open for writing).
// Returns an error if writing to the file fails.
func (f *ScmpFilter) ExportPFC(file *os.File) error {
	f.lock.Lock()
	defer f.lock.Unlock()

	fd := file.Fd()

	if !f.valid {
		return errBadFilter
	}

	if retCode := C.seccomp_export_pfc(f.filterCtx, C.int(fd)); retCode != 0 {
		return errRc(retCode)
	}

	return nil
}

// ExportBPF outputs Berkeley Packet Filter-formatted, kernel-readable dump of a
// filter context's rules to a file.
// Accepts file to write to (must be open for writing).
// Returns an error if writing to the file fails.
func (f *ScmpFilter) ExportBPF(file *os.File) error {
	f.lock.Lock()
	defer f.lock.Unlock()

	fd := file.Fd()

	if !f.valid {
		return errBadFilter
	}

	if retCode := C.seccomp_export_bpf(f.filterCtx, C.int(fd)); retCode != 0 {
		return errRc(retCode)
	}

	return nil
}

// Userspace Notification API

// GetNotifFd returns the userspace notification file descriptor associated with the given
// filter context. Such a file descriptor is only valid after the filter has been loaded
// and only when the filter uses the ActNotify action. The file descriptor can be used to
// retrieve and respond to notifications associated with the filter (see NotifReceive(),
// NotifRespond(), and NotifIDValid()).
func (f *ScmpFilter) GetNotifFd() (ScmpFd, error) {
	return f.getNotifFd()
}

// NotifReceive retrieves a seccomp userspace notification from a filter whose ActNotify
// action has triggered. The caller is expected to process the notification and return a
// response via NotifRespond(). Each invocation of this function returns one
// notification. As multiple notifications may be pending at any time, this function is
// normally called within a polling loop.
func NotifReceive(fd ScmpFd) (*ScmpNotifReq, error) {
	return notifReceive(fd)
}

// NotifRespond responds to a notification retrieved via NotifReceive(). The response Id
// must match that of the corresponding notification retrieved via NotifReceive().
func NotifRespond(fd ScmpFd, scmpResp *ScmpNotifResp) error {
	return notifRespond(fd, scmpResp)
}

// NotifIDValid checks if a notification is still valid. An return value of nil means the
// notification is still valid. Otherwise the notification is not valid. This can be used
// to mitigate time-of-check-time-of-use (TOCTOU) attacks as described in seccomp_notify_id_valid(2).
func NotifIDValid(fd ScmpFd, id uint64) error {
	return notifIDValid(fd, id)
}
