// +build solaris

package remote

import "golang.org/x/sys/unix"

func syscallDup(oldfd int, newfd int) (err error) {
	return unix.Dup2(oldfd, newfd)
}