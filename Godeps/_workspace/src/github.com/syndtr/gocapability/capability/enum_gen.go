// generated file; DO NOT EDIT - use go generate in directory with source

package capability

func (c Cap) String() string {
	switch c {
	case CAP_CHOWN:
		return "chown"
	case CAP_DAC_OVERRIDE:
		return "dac_override"
	case CAP_DAC_READ_SEARCH:
		return "dac_read_search"
	case CAP_FOWNER:
		return "fowner"
	case CAP_FSETID:
		return "fsetid"
	case CAP_KILL:
		return "kill"
	case CAP_SETGID:
		return "setgid"
	case CAP_SETUID:
		return "setuid"
	case CAP_SETPCAP:
		return "setpcap"
	case CAP_LINUX_IMMUTABLE:
		return "linux_immutable"
	case CAP_NET_BIND_SERVICE:
		return "net_bind_service"
	case CAP_NET_BROADCAST:
		return "net_broadcast"
	case CAP_NET_ADMIN:
		return "net_admin"
	case CAP_NET_RAW:
		return "net_raw"
	case CAP_IPC_LOCK:
		return "ipc_lock"
	case CAP_IPC_OWNER:
		return "ipc_owner"
	case CAP_SYS_MODULE:
		return "sys_module"
	case CAP_SYS_RAWIO:
		return "sys_rawio"
	case CAP_SYS_CHROOT:
		return "sys_chroot"
	case CAP_SYS_PTRACE:
		return "sys_ptrace"
	case CAP_SYS_PACCT:
		return "sys_pacct"
	case CAP_SYS_ADMIN:
		return "sys_admin"
	case CAP_SYS_BOOT:
		return "sys_boot"
	case CAP_SYS_NICE:
		return "sys_nice"
	case CAP_SYS_RESOURCE:
		return "sys_resource"
	case CAP_SYS_TIME:
		return "sys_time"
	case CAP_SYS_TTY_CONFIG:
		return "sys_tty_config"
	case CAP_MKNOD:
		return "mknod"
	case CAP_LEASE:
		return "lease"
	case CAP_AUDIT_WRITE:
		return "audit_write"
	case CAP_AUDIT_CONTROL:
		return "audit_control"
	case CAP_SETFCAP:
		return "setfcap"
	case CAP_MAC_OVERRIDE:
		return "mac_override"
	case CAP_MAC_ADMIN:
		return "mac_admin"
	case CAP_SYSLOG:
		return "syslog"
	case CAP_WAKE_ALARM:
		return "wake_alarm"
	case CAP_BLOCK_SUSPEND:
		return "block_suspend"
	case CAP_AUDIT_READ:
		return "audit_read"
	}
	return "unknown"
}

// List returns list of all supported capabilities
func List() []Cap {
	return []Cap{
		CAP_CHOWN,
		CAP_DAC_OVERRIDE,
		CAP_DAC_READ_SEARCH,
		CAP_FOWNER,
		CAP_FSETID,
		CAP_KILL,
		CAP_SETGID,
		CAP_SETUID,
		CAP_SETPCAP,
		CAP_LINUX_IMMUTABLE,
		CAP_NET_BIND_SERVICE,
		CAP_NET_BROADCAST,
		CAP_NET_ADMIN,
		CAP_NET_RAW,
		CAP_IPC_LOCK,
		CAP_IPC_OWNER,
		CAP_SYS_MODULE,
		CAP_SYS_RAWIO,
		CAP_SYS_CHROOT,
		CAP_SYS_PTRACE,
		CAP_SYS_PACCT,
		CAP_SYS_ADMIN,
		CAP_SYS_BOOT,
		CAP_SYS_NICE,
		CAP_SYS_RESOURCE,
		CAP_SYS_TIME,
		CAP_SYS_TTY_CONFIG,
		CAP_MKNOD,
		CAP_LEASE,
		CAP_AUDIT_WRITE,
		CAP_AUDIT_CONTROL,
		CAP_SETFCAP,
		CAP_MAC_OVERRIDE,
		CAP_MAC_ADMIN,
		CAP_SYSLOG,
		CAP_WAKE_ALARM,
		CAP_BLOCK_SUSPEND,
		CAP_AUDIT_READ,
	}
}
