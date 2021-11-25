// +build linux
// +build mips64le

package remote

import (
        "golang.org/x/sys/unix"
)

func interceptorDupx(oldfd int, newfd int) {
	unix.Dup3(oldfd, newfd, 0)
}
