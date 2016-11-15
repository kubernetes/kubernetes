// +build linux

package keys

import (
	"fmt"
	"strconv"
	"strings"
	"syscall"
	"unsafe"
)

const KEYCTL_JOIN_SESSION_KEYRING = 1
const KEYCTL_SETPERM = 5
const KEYCTL_DESCRIBE = 6

type KeySerial uint32

func JoinSessionKeyring(name string) (KeySerial, error) {
	var _name *byte
	var err error

	if len(name) > 0 {
		_name, err = syscall.BytePtrFromString(name)
		if err != nil {
			return KeySerial(0), err
		}
	}

	sessKeyId, _, errn := syscall.Syscall(syscall.SYS_KEYCTL, KEYCTL_JOIN_SESSION_KEYRING, uintptr(unsafe.Pointer(_name)), 0)
	if errn != 0 {
		return 0, fmt.Errorf("could not create session key: %v", errn)
	}
	return KeySerial(sessKeyId), nil
}

// ModKeyringPerm modifies permissions on a keyring by reading the current permissions,
// anding the bits with the given mask (clearing permissions) and setting
// additional permission bits
func ModKeyringPerm(ringId KeySerial, mask, setbits uint32) error {
	dest := make([]byte, 1024)
	destBytes := unsafe.Pointer(&dest[0])

	if _, _, err := syscall.Syscall6(syscall.SYS_KEYCTL, uintptr(KEYCTL_DESCRIBE), uintptr(ringId), uintptr(destBytes), uintptr(len(dest)), 0, 0); err != 0 {
		return err
	}

	res := strings.Split(string(dest), ";")
	if len(res) < 5 {
		return fmt.Errorf("Destination buffer for key description is too small")
	}

	// parse permissions
	perm64, err := strconv.ParseUint(res[3], 16, 32)
	if err != nil {
		return err
	}

	perm := (uint32(perm64) & mask) | setbits

	if _, _, err := syscall.Syscall(syscall.SYS_KEYCTL, uintptr(KEYCTL_SETPERM), uintptr(ringId), uintptr(perm)); err != 0 {
		return err
	}

	return nil
}
