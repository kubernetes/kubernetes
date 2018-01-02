// +build darwin freebsd

package sys

import (
	"syscall"
)

// StatAtime returns the access time from a stat struct
func StatAtime(st *syscall.Stat_t) syscall.Timespec {
	return st.Atimespec
}

// StatCtime returns the created time from a stat struct
func StatCtime(st *syscall.Stat_t) syscall.Timespec {
	return st.Ctimespec
}

// StatMtime returns the modified time from a stat struct
func StatMtime(st *syscall.Stat_t) syscall.Timespec {
	return st.Mtimespec
}
