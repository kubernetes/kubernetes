package seccomp

import (
	"runtime"

	"github.com/opencontainers/runtime-spec/specs-go"
	rspec "github.com/opencontainers/runtime-spec/specs-go"
)

func arches() []rspec.Arch {
	native := runtime.GOARCH

	switch native {
	case "amd64":
		return []rspec.Arch{rspec.ArchX86_64, rspec.ArchX86, rspec.ArchX32}
	case "arm64":
		return []rspec.Arch{rspec.ArchARM, rspec.ArchAARCH64}
	case "mips64":
		return []rspec.Arch{rspec.ArchMIPS, rspec.ArchMIPS64, rspec.ArchMIPS64N32}
	case "mips64n32":
		return []rspec.Arch{rspec.ArchMIPS, rspec.ArchMIPS64, rspec.ArchMIPS64N32}
	case "mipsel64":
		return []rspec.Arch{rspec.ArchMIPSEL, rspec.ArchMIPSEL64, rspec.ArchMIPSEL64N32}
	case "mipsel64n32":
		return []rspec.Arch{rspec.ArchMIPSEL, rspec.ArchMIPSEL64, rspec.ArchMIPSEL64N32}
	case "s390x":
		return []rspec.Arch{rspec.ArchS390, rspec.ArchS390X}
	default:
		return []rspec.Arch{}
	}
}

// DefaultProfile defines the whitelist for the default seccomp profile.
func DefaultProfile(rs *specs.Spec) *rspec.LinuxSeccomp {

	syscalls := []rspec.LinuxSyscall{
		{
			Names: []string{
				"accept",
				"accept4",
				"access",
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
			Action: rspec.ActAllow,
			Args:   []rspec.LinuxSeccompArg{},
		},
		{
			Names:  []string{"personality"},
			Action: rspec.ActAllow,
			Args: []rspec.LinuxSeccompArg{
				{
					Index: 0,
					Value: 0x0,
					Op:    rspec.OpEqualTo,
				},
				{
					Index: 0,
					Value: 0x0008,
					Op:    rspec.OpEqualTo,
				},
				{
					Index: 0,
					Value: 0xffffffff,
					Op:    rspec.OpEqualTo,
				},
			},
		},
	}
	var sysCloneFlagsIndex uint

	capSysAdmin := false
	caps := make(map[string]bool)

	for _, cap := range rs.Process.Capabilities.Bounding {
		caps[cap] = true
	}
	for _, cap := range rs.Process.Capabilities.Effective {
		caps[cap] = true
	}
	for _, cap := range rs.Process.Capabilities.Inheritable {
		caps[cap] = true
	}
	for _, cap := range rs.Process.Capabilities.Permitted {
		caps[cap] = true
	}
	for _, cap := range rs.Process.Capabilities.Ambient {
		caps[cap] = true
	}

	for cap := range caps {
		switch cap {
		case "CAP_DAC_READ_SEARCH":
			syscalls = append(syscalls, []rspec.LinuxSyscall{
				{
					Names:  []string{"open_by_handle_at"},
					Action: rspec.ActAllow,
					Args:   []rspec.LinuxSeccompArg{},
				},
			}...)
		case "CAP_SYS_ADMIN":
			capSysAdmin = true
			syscalls = append(syscalls, []rspec.LinuxSyscall{
				{
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
					Action: rspec.ActAllow,
					Args:   []rspec.LinuxSeccompArg{},
				},
			}...)
		case "CAP_SYS_BOOT":
			syscalls = append(syscalls, []rspec.LinuxSyscall{
				{
					Names:  []string{"reboot"},
					Action: rspec.ActAllow,
					Args:   []rspec.LinuxSeccompArg{},
				},
			}...)
		case "CAP_SYS_CHROOT":
			syscalls = append(syscalls, []rspec.LinuxSyscall{
				{
					Names:  []string{"chroot"},
					Action: rspec.ActAllow,
					Args:   []rspec.LinuxSeccompArg{},
				},
			}...)
		case "CAP_SYS_MODULE":
			syscalls = append(syscalls, []rspec.LinuxSyscall{
				{
					Names: []string{
						"delete_module",
						"init_module",
						"finit_module",
						"query_module",
					},
					Action: rspec.ActAllow,
					Args:   []rspec.LinuxSeccompArg{},
				},
			}...)
		case "CAP_SYS_PACCT":
			syscalls = append(syscalls, []rspec.LinuxSyscall{
				{
					Names:  []string{"acct"},
					Action: rspec.ActAllow,
					Args:   []rspec.LinuxSeccompArg{},
				},
			}...)
		case "CAP_SYS_PTRACE":
			syscalls = append(syscalls, []rspec.LinuxSyscall{
				{
					Names: []string{
						"kcmp",
						"process_vm_readv",
						"process_vm_writev",
						"ptrace",
					},
					Action: rspec.ActAllow,
					Args:   []rspec.LinuxSeccompArg{},
				},
			}...)
		case "CAP_SYS_RAWIO":
			syscalls = append(syscalls, []rspec.LinuxSyscall{
				{
					Names: []string{
						"iopl",
						"ioperm",
					},
					Action: rspec.ActAllow,
					Args:   []rspec.LinuxSeccompArg{},
				},
			}...)
		case "CAP_SYS_TIME":
			syscalls = append(syscalls, []rspec.LinuxSyscall{
				{
					Names: []string{
						"settimeofday",
						"stime",
						"adjtimex",
					},
					Action: rspec.ActAllow,
					Args:   []rspec.LinuxSeccompArg{},
				},
			}...)
		case "CAP_SYS_TTY_CONFIG":
			syscalls = append(syscalls, []rspec.LinuxSyscall{
				{
					Names:  []string{"vhangup"},
					Action: rspec.ActAllow,
					Args:   []rspec.LinuxSeccompArg{},
				},
			}...)
		}
	}

	if !capSysAdmin {
		syscalls = append(syscalls, []rspec.LinuxSyscall{
			{
				Names:  []string{"clone"},
				Action: rspec.ActAllow,
				Args: []rspec.LinuxSeccompArg{
					{
						Index:    sysCloneFlagsIndex,
						Value:    CloneNewNS | CloneNewUTS | CloneNewIPC | CloneNewUser | CloneNewPID | CloneNewNet,
						ValueTwo: 0,
						Op:       rspec.OpMaskedEqual,
					},
				},
			},
		}...)

	}

	arch := runtime.GOARCH
	switch arch {
	case "arm", "arm64":
		syscalls = append(syscalls, []rspec.LinuxSyscall{
			{
				Names: []string{
					"breakpoint",
					"cacheflush",
					"set_tls",
				},
				Action: rspec.ActAllow,
				Args:   []rspec.LinuxSeccompArg{},
			},
		}...)
	case "amd64", "x32":
		syscalls = append(syscalls, []rspec.LinuxSyscall{
			{
				Names:  []string{"arch_prctl"},
				Action: rspec.ActAllow,
				Args:   []rspec.LinuxSeccompArg{},
			},
		}...)
		fallthrough
	case "x86":
		syscalls = append(syscalls, []rspec.LinuxSyscall{
			{
				Names:  []string{"modify_ldt"},
				Action: rspec.ActAllow,
				Args:   []rspec.LinuxSeccompArg{},
			},
		}...)
	case "s390", "s390x":
		syscalls = append(syscalls, []rspec.LinuxSyscall{
			{
				Names: []string{
					"s390_pci_mmio_read",
					"s390_pci_mmio_write",
					"s390_runtime_instr",
				},
				Action: rspec.ActAllow,
				Args:   []rspec.LinuxSeccompArg{},
			},
		}...)
		/* Flags parameter of the clone syscall is the 2nd on s390 */
	}

	return &rspec.LinuxSeccomp{
		DefaultAction: rspec.ActErrno,
		Architectures: arches(),
		Syscalls:      syscalls,
	}
}
