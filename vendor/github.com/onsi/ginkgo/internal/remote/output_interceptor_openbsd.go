// +build openbsd

package remote

import (
        "golang.org/x/sys/unix"
)

func interceptorDupx(oldfd int, newfd int) {
	unix.Dup2(oldfd, newfd)
}
