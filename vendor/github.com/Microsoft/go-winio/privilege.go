package winio

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"runtime"
	"syscall"
	"unicode/utf16"
)

//sys adjustTokenPrivileges(token syscall.Handle, releaseAll bool, input *byte, outputSize uint32, output *byte, requiredSize *uint32) (success bool, err error) [true] = advapi32.AdjustTokenPrivileges
//sys impersonateSelf(level uint32) (err error) = advapi32.ImpersonateSelf
//sys revertToSelf() (err error) = advapi32.RevertToSelf
//sys openThreadToken(thread syscall.Handle, accessMask uint32, openAsSelf bool, token *syscall.Handle) (err error) = advapi32.OpenThreadToken
//sys getCurrentThread() (h syscall.Handle) = GetCurrentThread
//sys lookupPrivilegeValue(systemName string, name string, luid *uint64) (err error) = advapi32.LookupPrivilegeValueW
//sys lookupPrivilegeName(systemName string, luid *uint64, buffer *uint16, size *uint32) (err error) = advapi32.LookupPrivilegeNameW
//sys lookupPrivilegeDisplayName(systemName string, name *uint16, buffer *uint16, size *uint32, languageId *uint32) (err error) = advapi32.LookupPrivilegeDisplayNameW

const (
	SE_PRIVILEGE_ENABLED = 2

	ERROR_NOT_ALL_ASSIGNED syscall.Errno = 1300

	SeBackupPrivilege  = "SeBackupPrivilege"
	SeRestorePrivilege = "SeRestorePrivilege"
)

const (
	securityAnonymous = iota
	securityIdentification
	securityImpersonation
	securityDelegation
)

type PrivilegeError struct {
	privileges []uint64
}

func (e *PrivilegeError) Error() string {
	s := ""
	if len(e.privileges) > 1 {
		s = "Could not enable privileges "
	} else {
		s = "Could not enable privilege "
	}
	for i, p := range e.privileges {
		if i != 0 {
			s += ", "
		}
		s += `"`
		s += getPrivilegeName(p)
		s += `"`
	}
	return s
}

func RunWithPrivilege(name string, fn func() error) error {
	return RunWithPrivileges([]string{name}, fn)
}

func RunWithPrivileges(names []string, fn func() error) error {
	var privileges []uint64
	for _, name := range names {
		p := uint64(0)
		err := lookupPrivilegeValue("", name, &p)
		if err != nil {
			return err
		}
		privileges = append(privileges, p)
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	token, err := newThreadToken()
	if err != nil {
		return err
	}
	defer releaseThreadToken(token)
	err = adjustPrivileges(token, privileges)
	if err != nil {
		return err
	}
	return fn()
}

func adjustPrivileges(token syscall.Handle, privileges []uint64) error {
	var b bytes.Buffer
	binary.Write(&b, binary.LittleEndian, uint32(len(privileges)))
	for _, p := range privileges {
		binary.Write(&b, binary.LittleEndian, p)
		binary.Write(&b, binary.LittleEndian, uint32(SE_PRIVILEGE_ENABLED))
	}
	prevState := make([]byte, b.Len())
	reqSize := uint32(0)
	success, err := adjustTokenPrivileges(token, false, &b.Bytes()[0], uint32(len(prevState)), &prevState[0], &reqSize)
	if !success {
		return err
	}
	if err == ERROR_NOT_ALL_ASSIGNED {
		return &PrivilegeError{privileges}
	}
	return nil
}

func getPrivilegeName(luid uint64) string {
	var nameBuffer [256]uint16
	bufSize := uint32(len(nameBuffer))
	err := lookupPrivilegeName("", &luid, &nameBuffer[0], &bufSize)
	if err != nil {
		return fmt.Sprintf("<unknown privilege %d>", luid)
	}

	var displayNameBuffer [256]uint16
	displayBufSize := uint32(len(displayNameBuffer))
	var langId uint32
	err = lookupPrivilegeDisplayName("", &nameBuffer[0], &displayNameBuffer[0], &displayBufSize, &langId)
	if err != nil {
		return fmt.Sprintf("<unknown privilege %s>", utf16.Decode(nameBuffer[:bufSize]))
	}

	return string(utf16.Decode(displayNameBuffer[:displayBufSize]))
}

func newThreadToken() (syscall.Handle, error) {
	err := impersonateSelf(securityImpersonation)
	if err != nil {
		panic(err)
		return 0, err
	}

	var token syscall.Handle
	err = openThreadToken(getCurrentThread(), syscall.TOKEN_ADJUST_PRIVILEGES|syscall.TOKEN_QUERY, false, &token)
	if err != nil {
		rerr := revertToSelf()
		if rerr != nil {
			panic(rerr)
		}
		return 0, err
	}
	return token, nil
}

func releaseThreadToken(h syscall.Handle) {
	err := revertToSelf()
	if err != nil {
		panic(err)
	}
	syscall.Close(h)
}
