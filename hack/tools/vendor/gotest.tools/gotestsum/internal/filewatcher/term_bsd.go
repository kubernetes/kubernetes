// +build darwin dragonfly freebsd netbsd openbsd

package filewatcher

import "golang.org/x/sys/unix"

const tcGet = unix.TIOCGETA
const tcSet = unix.TIOCSETA
