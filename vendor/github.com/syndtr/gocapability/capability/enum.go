// Copyright (c) 2013, Suryandaru Triandana <syndtr@gmail.com>
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package capability

type CapType uint

func (c CapType) String() string {
	switch c {
	case EFFECTIVE:
		return "effective"
	case PERMITTED:
		return "permitted"
	case INHERITABLE:
		return "inheritable"
	case BOUNDING:
		return "bounding"
	case CAPS:
		return "caps"
	case AMBIENT:
		return "ambient"
	}
	return "unknown"
}

const (
	EFFECTIVE CapType = 1 << iota
	PERMITTED
	INHERITABLE
	BOUNDING
	AMBIENT

	CAPS   = EFFECTIVE | PERMITTED | INHERITABLE
	BOUNDS = BOUNDING
	AMBS   = AMBIENT
)

//go:generate go run enumgen/gen.go
type Cap int

// POSIX-draft defined capabilities and Linux extensions.
//
// Defined in https://github.com/torvalds/linux/blob/master/include/uapi/linux/capability.h
const (
	// In a system with the [_POSIX_CHOWN_RESTRICTED] option defined, this
	// overrides the restriction of changing file ownership and group
	// ownership.
	CAP_CHOWN = Cap(0)

	// Override all DAC access, including ACL execute access if
	// [_POSIX_ACL] is defined. Excluding DAC access covered by
	// CAP_LINUX_IMMUTABLE.
	CAP_DAC_OVERRIDE = Cap(1)

	// Overrides all DAC restrictions regarding read and search on files
	// and directories, including ACL restrictions if [_POSIX_ACL] is
	// defined. Excluding DAC access covered by CAP_LINUX_IMMUTABLE.
	CAP_DAC_READ_SEARCH = Cap(2)

	// Overrides all restrictions about allowed operations on files, where
	// file owner ID must be equal to the user ID, except where CAP_FSETID
	// is applicable. It doesn't override MAC and DAC restrictions.
	CAP_FOWNER = Cap(3)

	// Overrides the following restrictions that the effective user ID
	// shall match the file owner ID when setting the S_ISUID and S_ISGID
	// bits on that file; that the effective group ID (or one of the
	// supplementary group IDs) shall match the file owner ID when setting
	// the S_ISGID bit on that file; that the S_ISUID and S_ISGID bits are
	// cleared on successful return from chown(2) (not implemented).
	CAP_FSETID = Cap(4)

	// Overrides the restriction that the real or effective user ID of a
	// process sending a signal must match the real or effective user ID
	// of the process receiving the signal.
	CAP_KILL = Cap(5)

	// Allows setgid(2) manipulation
	// Allows setgroups(2)
	// Allows forged gids on socket credentials passing.
	CAP_SETGID = Cap(6)

	// Allows set*uid(2) manipulation (including fsuid).
	// Allows forged pids on socket credentials passing.
	CAP_SETUID = Cap(7)

	// Linux-specific capabilities

	// Without VFS support for capabilities:
	//   Transfer any capability in your permitted set to any pid,
	//   remove any capability in your permitted set from any pid
	// With VFS support for capabilities (neither of above, but)
	//   Add any capability from current's capability bounding set
	//     to the current process' inheritable set
	//   Allow taking bits out of capability bounding set
	//   Allow modification of the securebits for a process
	CAP_SETPCAP = Cap(8)

	// Allow modification of S_IMMUTABLE and S_APPEND file attributes
	CAP_LINUX_IMMUTABLE = Cap(9)

	// Allows binding to TCP/UDP sockets below 1024
	// Allows binding to ATM VCIs below 32
	CAP_NET_BIND_SERVICE = Cap(10)

	// Allow broadcasting, listen to multicast
	CAP_NET_BROADCAST = Cap(11)

	// Allow interface configuration
	// Allow administration of IP firewall, masquerading and accounting
	// Allow setting debug option on sockets
	// Allow modification of routing tables
	// Allow setting arbitrary process / process group ownership on
	// sockets
	// Allow binding to any address for transparent proxying (also via NET_RAW)
	// Allow setting TOS (type of service)
	// Allow setting promiscuous mode
	// Allow clearing driver statistics
	// Allow multicasting
	// Allow read/write of device-specific registers
	// Allow activation of ATM control sockets
	CAP_NET_ADMIN = Cap(12)

	// Allow use of RAW sockets
	// Allow use of PACKET sockets
	// Allow binding to any address for transparent proxying (also via NET_ADMIN)
	CAP_NET_RAW = Cap(13)

	// Allow locking of shared memory segments
	// Allow mlock and mlockall (which doesn't really have anything to do
	// with IPC)
	CAP_IPC_LOCK = Cap(14)

	// Override IPC ownership checks
	CAP_IPC_OWNER = Cap(15)

	// Insert and remove kernel modules - modify kernel without limit
	CAP_SYS_MODULE = Cap(16)

	// Allow ioperm/iopl access
	// Allow sending USB messages to any device via /proc/bus/usb
	CAP_SYS_RAWIO = Cap(17)

	// Allow use of chroot()
	CAP_SYS_CHROOT = Cap(18)

	// Allow ptrace() of any process
	CAP_SYS_PTRACE = Cap(19)

	// Allow configuration of process accounting
	CAP_SYS_PACCT = Cap(20)

	// Allow configuration of the secure attention key
	// Allow administration of the random device
	// Allow examination and configuration of disk quotas
	// Allow setting the domainname
	// Allow setting the hostname
	// Allow calling bdflush()
	// Allow mount() and umount(), setting up new smb connection
	// Allow some autofs root ioctls
	// Allow nfsservctl
	// Allow VM86_REQUEST_IRQ
	// Allow to read/write pci config on alpha
	// Allow irix_prctl on mips (setstacksize)
	// Allow flushing all cache on m68k (sys_cacheflush)
	// Allow removing semaphores
	// Used instead of CAP_CHOWN to "chown" IPC message queues, semaphores
	// and shared memory
	// Allow locking/unlocking of shared memory segment
	// Allow turning swap on/off
	// Allow forged pids on socket credentials passing
	// Allow setting readahead and flushing buffers on block devices
	// Allow setting geometry in floppy driver
	// Allow turning DMA on/off in xd driver
	// Allow administration of md devices (mostly the above, but some
	// extra ioctls)
	// Allow tuning the ide driver
	// Allow access to the nvram device
	// Allow administration of apm_bios, serial and bttv (TV) device
	// Allow manufacturer commands in isdn CAPI support driver
	// Allow reading non-standardized portions of pci configuration space
	// Allow DDI debug ioctl on sbpcd driver
	// Allow setting up serial ports
	// Allow sending raw qic-117 commands
	// Allow enabling/disabling tagged queuing on SCSI controllers and sending
	// arbitrary SCSI commands
	// Allow setting encryption key on loopback filesystem
	// Allow setting zone reclaim policy
	// Allow everything under CAP_BPF and CAP_PERFMON for backward compatibility
	CAP_SYS_ADMIN = Cap(21)

	// Allow use of reboot()
	CAP_SYS_BOOT = Cap(22)

	// Allow raising priority and setting priority on other (different
	// UID) processes
	// Allow use of FIFO and round-robin (realtime) scheduling on own
	// processes and setting the scheduling algorithm used by another
	// process.
	// Allow setting cpu affinity on other processes
	CAP_SYS_NICE = Cap(23)

	// Override resource limits. Set resource limits.
	// Override quota limits.
	// Override reserved space on ext2 filesystem
	// Modify data journaling mode on ext3 filesystem (uses journaling
	// resources)
	// NOTE: ext2 honors fsuid when checking for resource overrides, so
	// you can override using fsuid too
	// Override size restrictions on IPC message queues
	// Allow more than 64hz interrupts from the real-time clock
	// Override max number of consoles on console allocation
	// Override max number of keymaps
	// Control memory reclaim behavior
	CAP_SYS_RESOURCE = Cap(24)

	// Allow manipulation of system clock
	// Allow irix_stime on mips
	// Allow setting the real-time clock
	CAP_SYS_TIME = Cap(25)

	// Allow configuration of tty devices
	// Allow vhangup() of tty
	CAP_SYS_TTY_CONFIG = Cap(26)

	// Allow the privileged aspects of mknod()
	CAP_MKNOD = Cap(27)

	// Allow taking of leases on files
	CAP_LEASE = Cap(28)

	CAP_AUDIT_WRITE   = Cap(29)
	CAP_AUDIT_CONTROL = Cap(30)
	CAP_SETFCAP       = Cap(31)

	// Override MAC access.
	// The base kernel enforces no MAC policy.
	// An LSM may enforce a MAC policy, and if it does and it chooses
	// to implement capability based overrides of that policy, this is
	// the capability it should use to do so.
	CAP_MAC_OVERRIDE = Cap(32)

	// Allow MAC configuration or state changes.
	// The base kernel requires no MAC configuration.
	// An LSM may enforce a MAC policy, and if it does and it chooses
	// to implement capability based checks on modifications to that
	// policy or the data required to maintain it, this is the
	// capability it should use to do so.
	CAP_MAC_ADMIN = Cap(33)

	// Allow configuring the kernel's syslog (printk behaviour)
	CAP_SYSLOG = Cap(34)

	// Allow triggering something that will wake the system
	CAP_WAKE_ALARM = Cap(35)

	// Allow preventing system suspends
	CAP_BLOCK_SUSPEND = Cap(36)

	// Allow reading the audit log via multicast netlink socket
	CAP_AUDIT_READ = Cap(37)

	// Allow system performance and observability privileged operations
	// using perf_events, i915_perf and other kernel subsystems
	CAP_PERFMON = Cap(38)

	// CAP_BPF allows the following BPF operations:
	// - Creating all types of BPF maps
	// - Advanced verifier features
	//   - Indirect variable access
	//   - Bounded loops
	//   - BPF to BPF function calls
	//   - Scalar precision tracking
	//   - Larger complexity limits
	//   - Dead code elimination
	//   - And potentially other features
	// - Loading BPF Type Format (BTF) data
	// - Retrieve xlated and JITed code of BPF programs
	// - Use bpf_spin_lock() helper
	//
	// CAP_PERFMON relaxes the verifier checks further:
	// - BPF progs can use of pointer-to-integer conversions
	// - speculation attack hardening measures are bypassed
	// - bpf_probe_read to read arbitrary kernel memory is allowed
	// - bpf_trace_printk to print kernel memory is allowed
	//
	// CAP_SYS_ADMIN is required to use bpf_probe_write_user.
	//
	// CAP_SYS_ADMIN is required to iterate system wide loaded
	// programs, maps, links, BTFs and convert their IDs to file descriptors.
	//
	// CAP_PERFMON and CAP_BPF are required to load tracing programs.
	// CAP_NET_ADMIN and CAP_BPF are required to load networking programs.
	CAP_BPF = Cap(39)

	// Allow checkpoint/restore related operations.
	// Introduced in kernel 5.9
	CAP_CHECKPOINT_RESTORE = Cap(40)
)

var (
	// Highest valid capability of the running kernel.
	CAP_LAST_CAP = Cap(63)

	capUpperMask = ^uint32(0)
)
