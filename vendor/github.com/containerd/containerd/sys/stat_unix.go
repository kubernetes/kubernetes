// +build linux solaris

package sys

import (
	"syscall"
)

// StatAtime returns the Atim
func StatAtime(st *syscall.Stat_t) syscall.Timespec {
	return st.Atim
}

// StatCtime returns the Ctim
func StatCtime(st *syscall.Stat_t) syscall.Timespec {
	return st.Ctim
}

// StatMtime returns the Mtim
func StatMtime(st *syscall.Stat_t) syscall.Timespec {
	return st.Mtim
}
