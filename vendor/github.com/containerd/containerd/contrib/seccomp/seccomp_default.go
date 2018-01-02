// +build linux

package seccomp

import (
	"runtime"
	"syscall"

	"github.com/opencontainers/runtime-spec/specs-go"
)

func arches() []specs.Arch {
	switch runtime.GOARCH {
	case "amd64":
		return []specs.Arch{specs.ArchX86_64, specs.ArchX86, specs.ArchX32}
	case "arm64":
		return []specs.Arch{specs.ArchARM, specs.ArchAARCH64}
	case "mips64":
		return []specs.Arch{specs.ArchMIPS, specs.ArchMIPS64, specs.ArchMIPS64N32}
	case "mips64n32":
		return []specs.Arch{specs.ArchMIPS, specs.ArchMIPS64, specs.ArchMIPS64N32}
	case "mipsel64":
		return []specs.Arch{specs.ArchMIPSEL, specs.ArchMIPSEL64, specs.ArchMIPSEL64N32}
	case "mipsel64n32":
		return []specs.Arch{specs.ArchMIPSEL, specs.ArchMIPSEL64, specs.ArchMIPSEL64N32}
	case "s390x":
		return []specs.Arch{specs.ArchS390, specs.ArchS390X}
	default:
		return []specs.Arch{}
	}
}

// DefaultProfile defines the whitelist for the default seccomp profile.
func DefaultProfile(sp *specs.Spec) *specs.LinuxSeccomp {
	syscalls := []specs.LinuxSyscall{
		{
			Names: []string{
				"accept",
				"accept4",
				"access",
				"alarm",
				"alarm",
				"bind",
				"brk",
				"capget",
				"capset",
				"chdir",
				"chmod",
				"chown",
				"chown32",
				"clock_getres",
				"clock_gettime",
				"clock_nanosleep",
				"close",
				"connect",
				"copy_file_range",
				"creat",
				"dup",
				"dup2",
				"dup3",
				"epoll_create",
				"epoll_create1",
				"epoll_ctl",
				"epoll_ctl_old",
				"epoll_pwait",
				"epoll_wait",
				"epoll_wait_old",
				"eventfd",
				"eventfd2",
				"execve",
				"execveat",
				"exit",
				"exit_group",
				"faccessat",
				"fadvise64",
				"fadvise64_64",
				"fallocate",
				"fanotify_mark",
				"fchdir",
				"fchmod",
				"fchmodat",
				"fchown",
				"fchown32",
				"fchownat",
				"fcntl",
				"fcntl64",
				"fdatasync",
				"fgetxattr",
				"flistxattr",
				"flock",
				"fork",
				"fremovexattr",
				"fsetxattr",
				"fstat",
				"fstat64",
				"fstatat64",
				"fstatfs",
				"fstatfs64",
				"fsync",
				"ftruncate",
				"ftruncate64",
				"futex",
				"futimesat",
				"getcpu",
				"getcwd",
				"getdents",
				"getdents64",
				"getegid",
				"getegid32",
				"geteuid",
				"geteuid32",
				"getgid",
				"getgid32",
				"getgroups",
				"getgroups32",
				"getitimer",
				"getpeername",
				"getpgid",
				"getpgrp",
				"getpid",
				"getppid",
				"getpriority",
				"getrandom",
				"getresgid",
				"getresgid32",
				"getresuid",
				"getresuid32",
				"getrlimit",
				"get_robust_list",
				"getrusage",
				"getsid",
				"getsockname",
				"getsockopt",
				"get_thread_area",
				"gettid",
				"gettimeofday",
				"getuid",
				"getuid32",
				"getxattr",
				"inotify_add_watch",
				"inotify_init",
				"inotify_init1",
				"inotify_rm_watch",
				"io_cancel",
				"ioctl",
				"io_destroy",
				"io_getevents",
				"ioprio_get",
				"ioprio_set",
				"io_setup",
				"io_submit",
				"ipc",
				"kill",
				"lchown",
				"lchown32",
				"lgetxattr",
				"link",
				"linkat",
				"listen",
				"listxattr",
				"llistxattr",
				"_llseek",
				"lremovexattr",
				"lseek",
				"lsetxattr",
				"lstat",
				"lstat64",
				"madvise",
				"memfd_create",
				"mincore",
				"mkdir",
				"mkdirat",
				"mknod",
				"mknodat",
				"mlock",
				"mlock2",
				"mlockall",
				"mmap",
				"mmap2",
				"mprotect",
				"mq_getsetattr",
				"mq_notify",
				"mq_open",
				"mq_timedreceive",
				"mq_timedsend",
				"mq_unlink",
				"mremap",
				"msgctl",
				"msgget",
				"msgrcv",
				"msgsnd",
				"msync",
				"munlock",
				"munlockall",
				"munmap",
				"nanosleep",
				"newfstatat",
				"_newselect",
				"open",
				"openat",
				"pause",
				"pipe",
				"pipe2",
				"poll",
				"ppoll",
				"prctl",
				"pread64",
				"preadv",
				"prlimit64",
				"pselect6",
				"pwrite64",
				"pwritev",
				"read",
				"readahead",
				"readlink",
				"readlinkat",
				"readv",
				"recv",
				"recvfrom",
				"recvmmsg",
				"recvmsg",
				"remap_file_pages",
				"removexattr",
				"rename",
				"renameat",
				"renameat2",
				"restart_syscall",
				"rmdir",
				"rt_sigaction",
				"rt_sigpending",
				"rt_sigprocmask",
				"rt_sigqueueinfo",
				"rt_sigreturn",
				"rt_sigsuspend",
				"rt_sigtimedwait",
				"rt_tgsigqueueinfo",
				"sched_getaffinity",
				"sched_getattr",
				"sched_getparam",
				"sched_get_priority_max",
				"sched_get_priority_min",
				"sched_getscheduler",
				"sched_rr_get_interval",
				"sched_setaffinity",
				"sched_setattr",
				"sched_setparam",
				"sched_setscheduler",
				"sched_yield",
				"seccomp",
				"select",
				"semctl",
				"semget",
				"semop",
				"semtimedop",
				"send",
				"sendfile",
				"sendfile64",
				"sendmmsg",
				"sendmsg",
				"sendto",
				"setfsgid",
				"setfsgid32",
				"setfsuid",
				"setfsuid32",
				"setgid",
				"setgid32",
				"setgroups",
				"setgroups32",
				"setitimer",
				"setpgid",
				"setpriority",
				"setregid",
				"setregid32",
				"setresgid",
				"setresgid32",
				"setresuid",
				"setresuid32",
				"setreuid",
				"setreuid32",
				"setrlimit",
				"set_robust_list",
				"setsid",
				"setsockopt",
				"set_thread_area",
				"set_tid_address",
				"setuid",
				"setuid32",
				"setxattr",
				"shmat",
				"shmctl",
				"shmdt",
				"shmget",
				"shutdown",
				"sigaltstack",
				"signalfd",
				"signalfd4",
				"sigreturn",
				"socket",
				"socketcall",
				"socketpair",
				"splice",
				"stat",
				"stat64",
				"statfs",
				"statfs64",
				"symlink",
				"symlinkat",
				"sync",
				"sync_file_range",
				"syncfs",
				"sysinfo",
				"syslog",
				"tee",
				"tgkill",
				"time",
				"timer_create",
				"timer_delete",
				"timerfd_create",
				"timerfd_gettime",
				"timerfd_settime",
				"timer_getoverrun",
				"timer_gettime",
				"timer_settime",
				"times",
				"tkill",
				"truncate",
				"truncate64",
				"ugetrlimit",
				"umask",
				"uname",
				"unlink",
				"unlinkat",
				"utime",
				"utimensat",
				"utimes",
				"vfork",
				"vmsplice",
				"wait4",
				"waitid",
				"waitpid",
				"write",
				"writev",
			},
			Action: specs.ActAllow,
			Args:   []specs.LinuxSeccompArg{},
		},
		{
			Names:  []string{"personality"},
			Action: specs.ActAllow,
			Args: []specs.LinuxSeccompArg{
				{
					Index: 0,
					Value: 0x0,
					Op:    specs.OpEqualTo,
				},
			},
		},
		{
			Names:  []string{"personality"},
			Action: specs.ActAllow,
			Args: []specs.LinuxSeccompArg{
				{
					Index: 0,
					Value: 0x0008,
					Op:    specs.OpEqualTo,
				},
			},
		},
		{
			Names:  []string{"personality"},
			Action: specs.ActAllow,
			Args: []specs.LinuxSeccompArg{
				{
					Index: 0,
					Value: 0xffffffff,
					Op:    specs.OpEqualTo,
				},
			},
		},
	}

	s := &specs.LinuxSeccomp{
		DefaultAction: specs.ActErrno,
		Architectures: arches(),
		Syscalls:      syscalls,
	}

	// include by arch
	switch runtime.GOARCH {
	case "arm", "arm64":
		s.Syscalls = append(s.Syscalls, specs.LinuxSyscall{
			Names: []string{
				"arm_fadvise64_64",
				"arm_sync_file_range",
				"breakpoint",
				"cacheflush",
				"set_tls",
			},
			Action: specs.ActAllow,
			Args:   []specs.LinuxSeccompArg{},
		})
	case "amd64":
		s.Syscalls = append(s.Syscalls, specs.LinuxSyscall{
			Names: []string{
				"arch_prctl",
			},
			Action: specs.ActAllow,
			Args:   []specs.LinuxSeccompArg{},
		})
		fallthrough
	case "386":
		s.Syscalls = append(s.Syscalls, specs.LinuxSyscall{
			Names: []string{
				"modify_ldt",
			},
			Action: specs.ActAllow,
			Args:   []specs.LinuxSeccompArg{},
		})
	case "s390", "s390x":
		s.Syscalls = append(s.Syscalls, specs.LinuxSyscall{
			Names: []string{
				"s390_pci_mmio_read",
				"s390_pci_mmio_write",
				"s390_runtime_instr",
			},
			Action: specs.ActAllow,
			Args:   []specs.LinuxSeccompArg{},
		})
	}

	// make a map of enabled capabilities
	caps := make(map[string]bool)
	for _, c := range sp.Process.Capabilities.Bounding {
		caps[c] = true
	}
	for _, c := range sp.Process.Capabilities.Effective {
		caps[c] = true
	}
	for _, c := range sp.Process.Capabilities.Inheritable {
		caps[c] = true
	}
	for _, c := range sp.Process.Capabilities.Permitted {
		caps[c] = true
	}
	for _, c := range sp.Process.Capabilities.Ambient {
		caps[c] = true
	}

	for c := range caps {
		switch c {
		case "CAP_DAC_READ_SEARCH":
			s.Syscalls = append(s.Syscalls, specs.LinuxSyscall{
				Names:  []string{"open_by_handle_at"},
				Action: specs.ActAllow,
				Args:   []specs.LinuxSeccompArg{},
			})
		case "CAP_SYS_ADMIN":
			s.Syscalls = append(s.Syscalls, specs.LinuxSyscall{
				Names: []string{
					"bpf",
					"clone",
					"fanotify_init",
					"lookup_dcookie",
					"mount",
					"name_to_handle_at",
					"perf_event_open",
					"setdomainname",
					"sethostname",
					"setns",
					"umount",
					"umount2",
					"unshare",
				},
				Action: specs.ActAllow,
				Args:   []specs.LinuxSeccompArg{},
			})
		case "CAP_SYS_BOOT":
			s.Syscalls = append(s.Syscalls, specs.LinuxSyscall{
				Names:  []string{"reboot"},
				Action: specs.ActAllow,
				Args:   []specs.LinuxSeccompArg{},
			})
		case "CAP_SYS_CHROOT":
			s.Syscalls = append(s.Syscalls, specs.LinuxSyscall{
				Names:  []string{"chroot"},
				Action: specs.ActAllow,
				Args:   []specs.LinuxSeccompArg{},
			})
		case "CAP_SYS_MODULE":
			s.Syscalls = append(s.Syscalls, specs.LinuxSyscall{
				Names: []string{
					"delete_module",
					"init_module",
					"finit_module",
					"query_module",
				},
				Action: specs.ActAllow,
				Args:   []specs.LinuxSeccompArg{},
			})
		case "CAP_SYS_PACCT":
			s.Syscalls = append(s.Syscalls, specs.LinuxSyscall{
				Names:  []string{"acct"},
				Action: specs.ActAllow,
				Args:   []specs.LinuxSeccompArg{},
			})
		case "CAP_SYS_PTRACE":
			s.Syscalls = append(s.Syscalls, specs.LinuxSyscall{
				Names: []string{
					"kcmp",
					"process_vm_readv",
					"process_vm_writev",
					"ptrace",
				},
				Action: specs.ActAllow,
				Args:   []specs.LinuxSeccompArg{},
			})
		case "CAP_SYS_RAWIO":
			s.Syscalls = append(s.Syscalls, specs.LinuxSyscall{
				Names: []string{
					"iopl",
					"ioperm",
				},
				Action: specs.ActAllow,
				Args:   []specs.LinuxSeccompArg{},
			})
		case "CAP_SYS_TIME":
			s.Syscalls = append(s.Syscalls, specs.LinuxSyscall{
				Names: []string{
					"settimeofday",
					"stime",
					"adjtimex",
				},
				Action: specs.ActAllow,
				Args:   []specs.LinuxSeccompArg{},
			})
		case "CAP_SYS_TTY_CONFIG":
			s.Syscalls = append(s.Syscalls, specs.LinuxSyscall{
				Names:  []string{"vhangup"},
				Action: specs.ActAllow,
				Args:   []specs.LinuxSeccompArg{},
			})
		}
	}

	if !caps["CAP_SYS_ADMIN"] {
		switch runtime.GOARCH {
		case "s390", "s390x":
			s.Syscalls = append(s.Syscalls, specs.LinuxSyscall{
				Names: []string{
					"clone",
				},
				Action: specs.ActAllow,
				Args: []specs.LinuxSeccompArg{
					{
						Index:    1,
						Value:    syscall.CLONE_NEWNS | syscall.CLONE_NEWUTS | syscall.CLONE_NEWIPC | syscall.CLONE_NEWUSER | syscall.CLONE_NEWPID | syscall.CLONE_NEWNET,
						ValueTwo: 0,
						Op:       specs.OpMaskedEqual,
					},
				},
			})
		default:
			s.Syscalls = append(s.Syscalls, specs.LinuxSyscall{
				Names: []string{
					"clone",
				},
				Action: specs.ActAllow,
				Args: []specs.LinuxSeccompArg{
					{
						Index:    0,
						Value:    syscall.CLONE_NEWNS | syscall.CLONE_NEWUTS | syscall.CLONE_NEWIPC | syscall.CLONE_NEWUSER | syscall.CLONE_NEWPID | syscall.CLONE_NEWNET,
						ValueTwo: 0,
						Op:       specs.OpMaskedEqual,
					},
				},
			})
		}
	}

	return s
}
