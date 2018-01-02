package specconv

import (
	"os"
	"strings"

	"github.com/opencontainers/runtime-spec/specs-go"
)

func sPtr(s string) *string { return &s }

// Example returns an example spec file, with many options set so a user can
// see what a standard spec file looks like.
func Example() *specs.Spec {
	return &specs.Spec{
		Version: specs.Version,
		Root: &specs.Root{
			Path:     "rootfs",
			Readonly: true,
		},
		Process: &specs.Process{
			Terminal: true,
			User:     specs.User{},
			Args: []string{
				"sh",
			},
			Env: []string{
				"PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
				"TERM=xterm",
			},
			Cwd:             "/",
			NoNewPrivileges: true,
			Capabilities: &specs.LinuxCapabilities{
				Bounding: []string{
					"CAP_AUDIT_WRITE",
					"CAP_KILL",
					"CAP_NET_BIND_SERVICE",
				},
				Permitted: []string{
					"CAP_AUDIT_WRITE",
					"CAP_KILL",
					"CAP_NET_BIND_SERVICE",
				},
				Inheritable: []string{
					"CAP_AUDIT_WRITE",
					"CAP_KILL",
					"CAP_NET_BIND_SERVICE",
				},
				Ambient: []string{
					"CAP_AUDIT_WRITE",
					"CAP_KILL",
					"CAP_NET_BIND_SERVICE",
				},
				Effective: []string{
					"CAP_AUDIT_WRITE",
					"CAP_KILL",
					"CAP_NET_BIND_SERVICE",
				},
			},
			Rlimits: []specs.POSIXRlimit{
				{
					Type: "RLIMIT_NOFILE",
					Hard: uint64(1024),
					Soft: uint64(1024),
				},
			},
		},
		Hostname: "runc",
		Mounts: []specs.Mount{
			{
				Destination: "/proc",
				Type:        "proc",
				Source:      "proc",
				Options:     nil,
			},
			{
				Destination: "/dev",
				Type:        "tmpfs",
				Source:      "tmpfs",
				Options:     []string{"nosuid", "strictatime", "mode=755", "size=65536k"},
			},
			{
				Destination: "/dev/pts",
				Type:        "devpts",
				Source:      "devpts",
				Options:     []string{"nosuid", "noexec", "newinstance", "ptmxmode=0666", "mode=0620", "gid=5"},
			},
			{
				Destination: "/dev/shm",
				Type:        "tmpfs",
				Source:      "shm",
				Options:     []string{"nosuid", "noexec", "nodev", "mode=1777", "size=65536k"},
			},
			{
				Destination: "/dev/mqueue",
				Type:        "mqueue",
				Source:      "mqueue",
				Options:     []string{"nosuid", "noexec", "nodev"},
			},
			{
				Destination: "/sys",
				Type:        "sysfs",
				Source:      "sysfs",
				Options:     []string{"nosuid", "noexec", "nodev", "ro"},
			},
			{
				Destination: "/sys/fs/cgroup",
				Type:        "cgroup",
				Source:      "cgroup",
				Options:     []string{"nosuid", "noexec", "nodev", "relatime", "ro"},
			},
		},
		Linux: &specs.Linux{
			MaskedPaths: []string{
				"/proc/kcore",
				"/proc/latency_stats",
				"/proc/timer_list",
				"/proc/timer_stats",
				"/proc/sched_debug",
				"/sys/firmware",
			},
			ReadonlyPaths: []string{
				"/proc/asound",
				"/proc/bus",
				"/proc/fs",
				"/proc/irq",
				"/proc/sys",
				"/proc/sysrq-trigger",
			},
			Resources: &specs.LinuxResources{
				Devices: []specs.LinuxDeviceCgroup{
					{
						Allow:  false,
						Access: "rwm",
					},
				},
			},
			Namespaces: []specs.LinuxNamespace{
				{
					Type: "pid",
				},
				{
					Type: "network",
				},
				{
					Type: "ipc",
				},
				{
					Type: "uts",
				},
				{
					Type: "mount",
				},
			},
		},
	}
}

// ExampleRootless returns an example spec file that works with rootless
// containers. It's essentially a modified version of the specfile from
// Example().
func ToRootless(spec *specs.Spec) {
	var namespaces []specs.LinuxNamespace

	// Remove networkns from the spec.
	for _, ns := range spec.Linux.Namespaces {
		switch ns.Type {
		case specs.NetworkNamespace, specs.UserNamespace:
			// Do nothing.
		default:
			namespaces = append(namespaces, ns)
		}
	}
	// Add userns to the spec.
	namespaces = append(namespaces, specs.LinuxNamespace{
		Type: specs.UserNamespace,
	})
	spec.Linux.Namespaces = namespaces

	// Add mappings for the current user.
	spec.Linux.UIDMappings = []specs.LinuxIDMapping{{
		HostID:      uint32(os.Geteuid()),
		ContainerID: 0,
		Size:        1,
	}}
	spec.Linux.GIDMappings = []specs.LinuxIDMapping{{
		HostID:      uint32(os.Getegid()),
		ContainerID: 0,
		Size:        1,
	}}

	// Fix up mounts.
	var mounts []specs.Mount
	for _, mount := range spec.Mounts {
		// Ignore all mounts that are under /sys.
		if strings.HasPrefix(mount.Destination, "/sys") {
			continue
		}

		// Remove all gid= and uid= mappings.
		var options []string
		for _, option := range mount.Options {
			if !strings.HasPrefix(option, "gid=") && !strings.HasPrefix(option, "uid=") {
				options = append(options, option)
			}
		}

		mount.Options = options
		mounts = append(mounts, mount)
	}
	// Add the sysfs mount as an rbind.
	mounts = append(mounts, specs.Mount{
		Source:      "/sys",
		Destination: "/sys",
		Type:        "none",
		Options:     []string{"rbind", "nosuid", "noexec", "nodev", "ro"},
	})
	spec.Mounts = mounts

	// Remove cgroup settings.
	spec.Linux.Resources = nil
}
