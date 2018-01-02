package commands

import (
	"syscall"

	"golang.org/x/sys/windows"
)

var signalMap = map[string]syscall.Signal{
	"HUP":    syscall.Signal(windows.SIGHUP),
	"INT":    syscall.Signal(windows.SIGINT),
	"QUIT":   syscall.Signal(windows.SIGQUIT),
	"SIGILL": syscall.Signal(windows.SIGILL),
	"TRAP":   syscall.Signal(windows.SIGTRAP),
	"ABRT":   syscall.Signal(windows.SIGABRT),
	"BUS":    syscall.Signal(windows.SIGBUS),
	"FPE":    syscall.Signal(windows.SIGFPE),
	"KILL":   syscall.Signal(windows.SIGKILL),
	"SEGV":   syscall.Signal(windows.SIGSEGV),
	"PIPE":   syscall.Signal(windows.SIGPIPE),
	"ALRM":   syscall.Signal(windows.SIGALRM),
	"TERM":   syscall.Signal(windows.SIGTERM),
}
