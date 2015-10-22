package main

import (
	"syscall"

	"github.com/docker/libcontainer/configs"
	"github.com/docker/libcontainer/system"
)

var profiles = map[string]*securityProfile{
	"high":   highProfile,
	"medium": mediumProfile,
	"low":    lowProfile,
}

type securityProfile struct {
	Capabilities    []string         `json:"capabilities"`
	ApparmorProfile string           `json:"apparmor_profile"`
	MountLabel      string           `json:"mount_label"`
	ProcessLabel    string           `json:"process_label"`
	Rlimits         []configs.Rlimit `json:"rlimits"`
	Seccomp         *configs.Seccomp `json:"seccomp"`
}

// this should be a runtime config that is not able to do things like apt-get or yum install.
var highProfile = &securityProfile{
	Capabilities: []string{
		"NET_BIND_SERVICE",
		"KILL",
		"AUDIT_WRITE",
	},
	Rlimits: []configs.Rlimit{
		{
			Type: syscall.RLIMIT_NOFILE,
			Hard: 1024,
			Soft: 1024,
		},
	},
	// http://man7.org/linux/man-pages/man2/syscalls.2.html
	Seccomp: &configs.Seccomp{
		Syscalls: []*configs.Syscall{
			{
				Value:  syscall.SYS_CAPSET, // http://man7.org/linux/man-pages/man2/capset.2.html
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  syscall.SYS_UNSHARE, // http://man7.org/linux/man-pages/man2/unshare.2.html
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  int(system.SysSetns()),
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  syscall.SYS_MOUNT, // http://man7.org/linux/man-pages/man2/mount.2.html
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  syscall.SYS_UMOUNT2, // http://man7.org/linux/man-pages/man2/umount.2.html
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  syscall.SYS_CREATE_MODULE, // http://man7.org/linux/man-pages/man2/create_module.2.html
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  syscall.SYS_DELETE_MODULE, // http://man7.org/linux/man-pages/man2/delete_module.2.html
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  syscall.SYS_CHMOD, // http://man7.org/linux/man-pages/man2/chmod.2.html
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  syscall.SYS_CHOWN, // http://man7.org/linux/man-pages/man2/chown.2.html
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  syscall.SYS_LINK, // http://man7.org/linux/man-pages/man2/link.2.html
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  syscall.SYS_LINKAT, // http://man7.org/linux/man-pages/man2/linkat.2.html
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  syscall.SYS_UNLINK, // http://man7.org/linux/man-pages/man2/unlink.2.html
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  syscall.SYS_UNLINKAT, // http://man7.org/linux/man-pages/man2/unlinkat.2.html
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  syscall.SYS_CHROOT, // http://man7.org/linux/man-pages/man2/chroot.2.html
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  syscall.SYS_KEXEC_LOAD, // http://man7.org/linux/man-pages/man2/kexec_load.2.html
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  syscall.SYS_SETDOMAINNAME, // http://man7.org/linux/man-pages/man2/setdomainname.2.html
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  syscall.SYS_SETHOSTNAME, // http://man7.org/linux/man-pages/man2/sethostname.2.html
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  syscall.SYS_CLONE, // http://man7.org/linux/man-pages/man2/clone.2.html
				Action: configs.Action(syscall.EPERM),
				Args: []*configs.Arg{
					{
						Index: 0, // the glibc wrapper has the flags at arg2 but the raw syscall has flags at arg0
						Value: syscall.CLONE_NEWUSER,
						Op:    configs.MaskEqualTo,
					},
				},
			},
		},
	},
}

// This is a medium level profile that should be able to do things like installing from
// apt-get or yum.
var mediumProfile = &securityProfile{
	Capabilities: []string{
		"CHOWN",
		"DAC_OVERRIDE",
		"FSETID",
		"FOWNER",
		"SETGID",
		"SETUID",
		"SETFCAP",
		"SETPCAP",
		"NET_BIND_SERVICE",
		"KILL",
		"AUDIT_WRITE",
	},
	Rlimits: []configs.Rlimit{
		{
			Type: syscall.RLIMIT_NOFILE,
			Hard: 1024,
			Soft: 1024,
		},
	},
	// http://man7.org/linux/man-pages/man2/syscalls.2.html
	Seccomp: &configs.Seccomp{
		Syscalls: []*configs.Syscall{
			{
				Value:  syscall.SYS_UNSHARE, // http://man7.org/linux/man-pages/man2/unshare.2.html
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  int(system.SysSetns()),
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  syscall.SYS_MOUNT, // http://man7.org/linux/man-pages/man2/mount.2.html
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  syscall.SYS_UMOUNT2, // http://man7.org/linux/man-pages/man2/umount.2.html
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  syscall.SYS_CHROOT, // http://man7.org/linux/man-pages/man2/chroot.2.html
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  syscall.SYS_CREATE_MODULE, // http://man7.org/linux/man-pages/man2/create_module.2.html
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  syscall.SYS_DELETE_MODULE, // http://man7.org/linux/man-pages/man2/delete_module.2.html
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  syscall.SYS_KEXEC_LOAD, // http://man7.org/linux/man-pages/man2/kexec_load.2.html
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  syscall.SYS_SETDOMAINNAME, // http://man7.org/linux/man-pages/man2/setdomainname.2.html
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  syscall.SYS_SETHOSTNAME, // http://man7.org/linux/man-pages/man2/sethostname.2.html
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  syscall.SYS_CLONE, // http://man7.org/linux/man-pages/man2/clone.2.html
				Action: configs.Action(syscall.EPERM),
				Args: []*configs.Arg{
					{
						Index: 0, // the glibc wrapper has the flags at arg2 but the raw syscall has flags at arg0
						Value: syscall.CLONE_NEWUSER,
						Op:    configs.MaskEqualTo,
					},
				},
			},
		},
	},
}

var lowProfile = &securityProfile{
	Capabilities: []string{
		"CHOWN",
		"DAC_OVERRIDE",
		"FSETID",
		"FOWNER",
		"SETGID",
		"SETUID",
		"SYS_CHROOT",
		"SETFCAP",
		"SETPCAP",
		"NET_BIND_SERVICE",
		"KILL",
		"AUDIT_WRITE",
	},
	Rlimits: []configs.Rlimit{
		{
			Type: syscall.RLIMIT_NOFILE,
			Hard: 1024,
			Soft: 1024,
		},
	},
	// http://man7.org/linux/man-pages/man2/syscalls.2.html
	Seccomp: &configs.Seccomp{
		Syscalls: []*configs.Syscall{
			{
				Value:  syscall.SYS_UNSHARE, // http://man7.org/linux/man-pages/man2/unshare.2.html
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  int(system.SysSetns()),
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  syscall.SYS_MOUNT, // http://man7.org/linux/man-pages/man2/mount.2.html
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  syscall.SYS_UMOUNT2, // http://man7.org/linux/man-pages/man2/umount.2.html
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  syscall.SYS_CREATE_MODULE, // http://man7.org/linux/man-pages/man2/create_module.2.html
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  syscall.SYS_DELETE_MODULE, // http://man7.org/linux/man-pages/man2/delete_module.2.html
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  syscall.SYS_KEXEC_LOAD, // http://man7.org/linux/man-pages/man2/kexec_load.2.html
				Action: configs.Action(syscall.EPERM),
			},
			{
				Value:  syscall.SYS_CLONE, // http://man7.org/linux/man-pages/man2/clone.2.html
				Action: configs.Action(syscall.EPERM),
				Args: []*configs.Arg{
					{
						Index: 0, // the glibc wrapper has the flags at arg2 but the raw syscall has flags at arg0
						Value: syscall.CLONE_NEWUSER,
						Op:    configs.MaskEqualTo,
					},
				},
			},
		},
	},
}
