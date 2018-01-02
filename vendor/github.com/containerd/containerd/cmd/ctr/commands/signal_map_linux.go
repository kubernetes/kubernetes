package commands

import (
	"syscall"

	"golang.org/x/sys/unix"
)

var signalMap = map[string]syscall.Signal{
	"ABRT":   unix.SIGABRT,
	"ALRM":   unix.SIGALRM,
	"BUS":    unix.SIGBUS,
	"CHLD":   unix.SIGCHLD,
	"CLD":    unix.SIGCLD,
	"CONT":   unix.SIGCONT,
	"FPE":    unix.SIGFPE,
	"HUP":    unix.SIGHUP,
	"ILL":    unix.SIGILL,
	"INT":    unix.SIGINT,
	"IO":     unix.SIGIO,
	"IOT":    unix.SIGIOT,
	"KILL":   unix.SIGKILL,
	"PIPE":   unix.SIGPIPE,
	"POLL":   unix.SIGPOLL,
	"PROF":   unix.SIGPROF,
	"PWR":    unix.SIGPWR,
	"QUIT":   unix.SIGQUIT,
	"SEGV":   unix.SIGSEGV,
	"STKFLT": unix.SIGSTKFLT,
	"STOP":   unix.SIGSTOP,
	"SYS":    unix.SIGSYS,
	"TERM":   unix.SIGTERM,
	"TRAP":   unix.SIGTRAP,
	"TSTP":   unix.SIGTSTP,
	"TTIN":   unix.SIGTTIN,
	"TTOU":   unix.SIGTTOU,
	"URG":    unix.SIGURG,
	"USR1":   unix.SIGUSR1,
	"USR2":   unix.SIGUSR2,
	"VTALRM": unix.SIGVTALRM,
	"WINCH":  unix.SIGWINCH,
	"XCPU":   unix.SIGXCPU,
	"XFSZ":   unix.SIGXFSZ,
}
