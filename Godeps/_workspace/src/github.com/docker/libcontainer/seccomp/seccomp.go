// Package seccomp provides native seccomp ( https://www.kernel.org/doc/Documentation/prctl/seccomp_filter.txt ) support for go.
package seccomp

import (
	"syscall"
	"unsafe"
)

// Operator that is used for argument comparison.
type Operator int

const (
	EqualTo Operator = iota
	NotEqualTo
	GreatherThan
	LessThan
	MaskEqualTo
)

const (
	jumpJT  = 0xff
	jumpJF  = 0xff
	labelJT = 0xfe
	labelJF = 0xfe
)

const (
	pfLD                 = 0x0
	retKill              = 0x00000000
	retTrap              = 0x00030000
	retAllow             = 0x7fff0000
	modeFilter           = 0x2
	prSetNoNewPrivileges = 0x26
)

func actionErrno(errno uint32) uint32 {
	return 0x00050000 | (errno & 0x0000ffff)
}

var (
	secData = struct {
		nr         int32
		arch       uint32
		insPointer uint64
		args       [6]uint64
	}{0, 0, 0, [6]uint64{0, 0, 0, 0, 0, 0}}
)

var isLittle = func() bool {
	var (
		x  = 0x1234
		p  = unsafe.Pointer(&x)
		p2 = (*[unsafe.Sizeof(0)]byte)(p)
	)
	if p2[0] == 0 {
		return false
	}
	return true
}()

var endian endianSupport

type endianSupport struct {
}

func (e endianSupport) hi(i uint32) uint32 {
	if isLittle {
		return e.little(i)
	}
	return e.big(i)
}

func (e endianSupport) low(i uint32) uint32 {
	if isLittle {
		return e.big(i)
	}
	return e.little(i)
}

func (endianSupport) big(idx uint32) uint32 {
	if idx >= 6 {
		return 0
	}
	return uint32(unsafe.Offsetof(secData.args)) + 8*idx
}

func (endianSupport) little(idx uint32) uint32 {
	if idx < 0 || idx >= 6 {
		return 0
	}
	return uint32(unsafe.Offsetof(secData.args)) +
		uint32(unsafe.Alignof(secData.args[0]))*idx + uint32(unsafe.Sizeof(secData.arch))
}

func prctl(option int, arg2, arg3, arg4, arg5 uintptr) error {
	_, _, err := syscall.Syscall6(syscall.SYS_PRCTL, uintptr(option), arg2, arg3, arg4, arg5, 0)
	if err != 0 {
		return err
	}
	return nil
}

func newSockFprog(filter []sockFilter) *sockFprog {
	return &sockFprog{
		len:  uint16(len(filter)),
		filt: filter,
	}
}

type sockFprog struct {
	len  uint16
	filt []sockFilter
}

func (s *sockFprog) set() error {
	_, _, err := syscall.Syscall(syscall.SYS_PRCTL, uintptr(syscall.PR_SET_SECCOMP),
		uintptr(modeFilter), uintptr(unsafe.Pointer(s)))
	if err != 0 {
		return err
	}
	return nil
}
