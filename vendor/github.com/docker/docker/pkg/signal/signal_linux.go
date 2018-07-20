package signal

import (
	"syscall"

	"golang.org/x/sys/unix"
)

const (
	sigrtmin = 34
	sigrtmax = 64
)

// SignalMap is a map of Linux signals.
var SignalMap = map[string]syscall.Signal{
	"ABRT":     unix.SIGABRT,
	"ALRM":     unix.SIGALRM,
	"BUS":      unix.SIGBUS,
	"CHLD":     unix.SIGCHLD,
	"CLD":      unix.SIGCLD,
	"CONT":     unix.SIGCONT,
	"FPE":      unix.SIGFPE,
	"HUP":      unix.SIGHUP,
	"ILL":      unix.SIGILL,
	"INT":      unix.SIGINT,
	"IO":       unix.SIGIO,
	"IOT":      unix.SIGIOT,
	"KILL":     unix.SIGKILL,
	"PIPE":     unix.SIGPIPE,
	"POLL":     unix.SIGPOLL,
	"PROF":     unix.SIGPROF,
	"PWR":      unix.SIGPWR,
	"QUIT":     unix.SIGQUIT,
	"SEGV":     unix.SIGSEGV,
	"STKFLT":   unix.SIGSTKFLT,
	"STOP":     unix.SIGSTOP,
	"SYS":      unix.SIGSYS,
	"TERM":     unix.SIGTERM,
	"TRAP":     unix.SIGTRAP,
	"TSTP":     unix.SIGTSTP,
	"TTIN":     unix.SIGTTIN,
	"TTOU":     unix.SIGTTOU,
	"URG":      unix.SIGURG,
	"USR1":     unix.SIGUSR1,
	"USR2":     unix.SIGUSR2,
	"VTALRM":   unix.SIGVTALRM,
	"WINCH":    unix.SIGWINCH,
	"XCPU":     unix.SIGXCPU,
	"XFSZ":     unix.SIGXFSZ,
	"RTMIN":    sigrtmin,
	"RTMIN+1":  sigrtmin + 1,
	"RTMIN+2":  sigrtmin + 2,
	"RTMIN+3":  sigrtmin + 3,
	"RTMIN+4":  sigrtmin + 4,
	"RTMIN+5":  sigrtmin + 5,
	"RTMIN+6":  sigrtmin + 6,
	"RTMIN+7":  sigrtmin + 7,
	"RTMIN+8":  sigrtmin + 8,
	"RTMIN+9":  sigrtmin + 9,
	"RTMIN+10": sigrtmin + 10,
	"RTMIN+11": sigrtmin + 11,
	"RTMIN+12": sigrtmin + 12,
	"RTMIN+13": sigrtmin + 13,
	"RTMIN+14": sigrtmin + 14,
	"RTMIN+15": sigrtmin + 15,
	"RTMAX-14": sigrtmax - 14,
	"RTMAX-13": sigrtmax - 13,
	"RTMAX-12": sigrtmax - 12,
	"RTMAX-11": sigrtmax - 11,
	"RTMAX-10": sigrtmax - 10,
	"RTMAX-9":  sigrtmax - 9,
	"RTMAX-8":  sigrtmax - 8,
	"RTMAX-7":  sigrtmax - 7,
	"RTMAX-6":  sigrtmax - 6,
	"RTMAX-5":  sigrtmax - 5,
	"RTMAX-4":  sigrtmax - 4,
	"RTMAX-3":  sigrtmax - 3,
	"RTMAX-2":  sigrtmax - 2,
	"RTMAX-1":  sigrtmax - 1,
	"RTMAX":    sigrtmax,
}
