// +build darwin dragonfly freebsd netbsd openbsd

package pty

// from <sys/ioccom.h>
const (
	_IOC_VOID    uintptr = 0x20000000
	_IOC_OUT     uintptr = 0x40000000
	_IOC_IN      uintptr = 0x80000000
	_IOC_IN_OUT  uintptr = _IOC_OUT | _IOC_IN
	_IOC_DIRMASK         = _IOC_VOID | _IOC_OUT | _IOC_IN

	_IOC_PARAM_SHIFT = 13
	_IOC_PARAM_MASK  = (1 << _IOC_PARAM_SHIFT) - 1
)

func _IOC_PARM_LEN(ioctl uintptr) uintptr {
	return (ioctl >> 16) & _IOC_PARAM_MASK
}

func _IOC(inout uintptr, group byte, ioctl_num uintptr, param_len uintptr) uintptr {
	return inout | (param_len&_IOC_PARAM_MASK)<<16 | uintptr(group)<<8 | ioctl_num
}

func _IO(group byte, ioctl_num uintptr) uintptr {
	return _IOC(_IOC_VOID, group, ioctl_num, 0)
}

func _IOR(group byte, ioctl_num uintptr, param_len uintptr) uintptr {
	return _IOC(_IOC_OUT, group, ioctl_num, param_len)
}

func _IOW(group byte, ioctl_num uintptr, param_len uintptr) uintptr {
	return _IOC(_IOC_IN, group, ioctl_num, param_len)
}

func _IOWR(group byte, ioctl_num uintptr, param_len uintptr) uintptr {
	return _IOC(_IOC_IN_OUT, group, ioctl_num, param_len)
}
