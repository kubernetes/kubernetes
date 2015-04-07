package integration

import (
	"syscall"

	"github.com/docker/libcontainer"
	"github.com/docker/libcontainer/cgroups"
	"github.com/docker/libcontainer/devices"
)

// newTemplateConfig returns a base template for running a container
//
// it uses a network strategy of just setting a loopback interface
// and the default setup for devices
func newTemplateConfig(rootfs string) *libcontainer.Config {
	return &libcontainer.Config{
		RootFs: rootfs,
		Tty:    false,
		Capabilities: []string{
			"CHOWN",
			"DAC_OVERRIDE",
			"FSETID",
			"FOWNER",
			"MKNOD",
			"NET_RAW",
			"SETGID",
			"SETUID",
			"SETFCAP",
			"SETPCAP",
			"NET_BIND_SERVICE",
			"SYS_CHROOT",
			"KILL",
			"AUDIT_WRITE",
		},
		Namespaces: libcontainer.Namespaces([]libcontainer.Namespace{
			{Type: libcontainer.NEWNS},
			{Type: libcontainer.NEWUTS},
			{Type: libcontainer.NEWIPC},
			{Type: libcontainer.NEWPID},
			{Type: libcontainer.NEWNET},
		}),
		Cgroups: &cgroups.Cgroup{
			Parent:          "integration",
			AllowAllDevices: false,
			AllowedDevices:  devices.DefaultAllowedDevices,
		},

		MountConfig: &libcontainer.MountConfig{
			DeviceNodes: devices.DefaultAutoCreatedDevices,
		},
		Hostname: "integration",
		Env: []string{
			"HOME=/root",
			"PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
			"HOSTNAME=integration",
			"TERM=xterm",
		},
		Networks: []*libcontainer.Network{
			{
				Type:    "loopback",
				Address: "127.0.0.1/0",
				Gateway: "localhost",
			},
		},
		Rlimits: []libcontainer.Rlimit{
			{
				Type: syscall.RLIMIT_NOFILE,
				Hard: uint64(1024),
				Soft: uint64(1024),
			},
		},
	}
}
