// +build !windows

package containerd

import (
	"context"
	"io/ioutil"
	"os"
	"path/filepath"
	"syscall"

	"golang.org/x/sys/unix"

	"github.com/containerd/containerd/mount"
	"github.com/containerd/containerd/namespaces"
	specs "github.com/opencontainers/runtime-spec/specs-go"
)

const (
	rwm               = "rwm"
	defaultRootfsPath = "rootfs"
)

var (
	defaultEnv = []string{
		"PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
	}
)

func defaultCaps() []string {
	return []string{
		"CAP_CHOWN",
		"CAP_DAC_OVERRIDE",
		"CAP_FSETID",
		"CAP_FOWNER",
		"CAP_MKNOD",
		"CAP_NET_RAW",
		"CAP_SETGID",
		"CAP_SETUID",
		"CAP_SETFCAP",
		"CAP_SETPCAP",
		"CAP_NET_BIND_SERVICE",
		"CAP_SYS_CHROOT",
		"CAP_KILL",
		"CAP_AUDIT_WRITE",
	}
}

func defaultNamespaces() []specs.LinuxNamespace {
	return []specs.LinuxNamespace{
		{
			Type: specs.PIDNamespace,
		},
		{
			Type: specs.IPCNamespace,
		},
		{
			Type: specs.UTSNamespace,
		},
		{
			Type: specs.MountNamespace,
		},
		{
			Type: specs.NetworkNamespace,
		},
	}
}

func createDefaultSpec(ctx context.Context, id string) (*specs.Spec, error) {
	ns, err := namespaces.NamespaceRequired(ctx)
	if err != nil {
		return nil, err
	}
	s := &specs.Spec{
		Version: specs.Version,
		Root: &specs.Root{
			Path: defaultRootfsPath,
		},
		Process: &specs.Process{
			Env:             defaultEnv,
			Cwd:             "/",
			NoNewPrivileges: true,
			User: specs.User{
				UID: 0,
				GID: 0,
			},
			Capabilities: &specs.LinuxCapabilities{
				Bounding:    defaultCaps(),
				Permitted:   defaultCaps(),
				Inheritable: defaultCaps(),
				Effective:   defaultCaps(),
			},
			Rlimits: []specs.POSIXRlimit{
				{
					Type: "RLIMIT_NOFILE",
					Hard: uint64(1024),
					Soft: uint64(1024),
				},
			},
		},
		Mounts: []specs.Mount{
			{
				Destination: "/proc",
				Type:        "proc",
				Source:      "proc",
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
				Destination: "/run",
				Type:        "tmpfs",
				Source:      "tmpfs",
				Options:     []string{"nosuid", "strictatime", "mode=755", "size=65536k"},
			},
		},
		Linux: &specs.Linux{
			// TODO (AkihiroSuda): unmask /sys/firmware on Windows daemon for LCOW support?
			// https://github.com/moby/moby/pull/33241/files#diff-a1f5051ce84e711a2ee688ab9ded5e74R215
			MaskedPaths: []string{
				"/proc/kcore",
				"/proc/latency_stats",
				"/proc/timer_list",
				"/proc/timer_stats",
				"/proc/sched_debug",
				"/sys/firmware",
				"/proc/scsi",
			},
			ReadonlyPaths: []string{
				"/proc/asound",
				"/proc/bus",
				"/proc/fs",
				"/proc/irq",
				"/proc/sys",
				"/proc/sysrq-trigger",
			},
			CgroupsPath: filepath.Join("/", ns, id),
			Resources: &specs.LinuxResources{
				Devices: []specs.LinuxDeviceCgroup{
					{
						Allow:  false,
						Access: rwm,
					},
				},
			},
			Namespaces: defaultNamespaces(),
		},
	}
	return s, nil
}

func remapRootFS(mounts []mount.Mount, uid, gid uint32) error {
	root, err := ioutil.TempDir("", "ctd-remap")
	if err != nil {
		return err
	}
	defer os.RemoveAll(root)
	for _, m := range mounts {
		if err := m.Mount(root); err != nil {
			return err
		}
	}
	defer unix.Unmount(root, 0)
	return filepath.Walk(root, incrementFS(root, uid, gid))
}

func incrementFS(root string, uidInc, gidInc uint32) filepath.WalkFunc {
	return func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		var (
			stat = info.Sys().(*syscall.Stat_t)
			u, g = int(stat.Uid + uidInc), int(stat.Gid + gidInc)
		)
		// be sure the lchown the path as to not de-reference the symlink to a host file
		return os.Lchown(path, u, g)
	}
}
