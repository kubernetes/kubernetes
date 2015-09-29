// +build linux

package libcontainer

import (
	"fmt"
	"os"

	"github.com/syndtr/gocapability/capability"
)

const allCapabilityTypes = capability.CAPS | capability.BOUNDS

var capabilityList = map[string]capability.Cap{
	"SETPCAP":          capability.CAP_SETPCAP,
	"SYS_MODULE":       capability.CAP_SYS_MODULE,
	"SYS_RAWIO":        capability.CAP_SYS_RAWIO,
	"SYS_PACCT":        capability.CAP_SYS_PACCT,
	"SYS_ADMIN":        capability.CAP_SYS_ADMIN,
	"SYS_NICE":         capability.CAP_SYS_NICE,
	"SYS_RESOURCE":     capability.CAP_SYS_RESOURCE,
	"SYS_TIME":         capability.CAP_SYS_TIME,
	"SYS_TTY_CONFIG":   capability.CAP_SYS_TTY_CONFIG,
	"MKNOD":            capability.CAP_MKNOD,
	"AUDIT_WRITE":      capability.CAP_AUDIT_WRITE,
	"AUDIT_CONTROL":    capability.CAP_AUDIT_CONTROL,
	"MAC_OVERRIDE":     capability.CAP_MAC_OVERRIDE,
	"MAC_ADMIN":        capability.CAP_MAC_ADMIN,
	"NET_ADMIN":        capability.CAP_NET_ADMIN,
	"SYSLOG":           capability.CAP_SYSLOG,
	"CHOWN":            capability.CAP_CHOWN,
	"NET_RAW":          capability.CAP_NET_RAW,
	"DAC_OVERRIDE":     capability.CAP_DAC_OVERRIDE,
	"FOWNER":           capability.CAP_FOWNER,
	"DAC_READ_SEARCH":  capability.CAP_DAC_READ_SEARCH,
	"FSETID":           capability.CAP_FSETID,
	"KILL":             capability.CAP_KILL,
	"SETGID":           capability.CAP_SETGID,
	"SETUID":           capability.CAP_SETUID,
	"LINUX_IMMUTABLE":  capability.CAP_LINUX_IMMUTABLE,
	"NET_BIND_SERVICE": capability.CAP_NET_BIND_SERVICE,
	"NET_BROADCAST":    capability.CAP_NET_BROADCAST,
	"IPC_LOCK":         capability.CAP_IPC_LOCK,
	"IPC_OWNER":        capability.CAP_IPC_OWNER,
	"SYS_CHROOT":       capability.CAP_SYS_CHROOT,
	"SYS_PTRACE":       capability.CAP_SYS_PTRACE,
	"SYS_BOOT":         capability.CAP_SYS_BOOT,
	"LEASE":            capability.CAP_LEASE,
	"SETFCAP":          capability.CAP_SETFCAP,
	"WAKE_ALARM":       capability.CAP_WAKE_ALARM,
	"BLOCK_SUSPEND":    capability.CAP_BLOCK_SUSPEND,
	"AUDIT_READ":       capability.CAP_AUDIT_READ,
}

func newCapWhitelist(caps []string) (*whitelist, error) {
	l := []capability.Cap{}
	for _, c := range caps {
		v, ok := capabilityList[c]
		if !ok {
			return nil, fmt.Errorf("unknown capability %q", c)
		}
		l = append(l, v)
	}
	pid, err := capability.NewPid(os.Getpid())
	if err != nil {
		return nil, err
	}
	return &whitelist{
		keep: l,
		pid:  pid,
	}, nil
}

type whitelist struct {
	pid  capability.Capabilities
	keep []capability.Cap
}

// dropBoundingSet drops the capability bounding set to those specified in the whitelist.
func (w *whitelist) dropBoundingSet() error {
	w.pid.Clear(capability.BOUNDS)
	w.pid.Set(capability.BOUNDS, w.keep...)
	return w.pid.Apply(capability.BOUNDS)
}

// drop drops all capabilities for the current process except those specified in the whitelist.
func (w *whitelist) drop() error {
	w.pid.Clear(allCapabilityTypes)
	w.pid.Set(allCapabilityTypes, w.keep...)
	return w.pid.Apply(allCapabilityTypes)
}
