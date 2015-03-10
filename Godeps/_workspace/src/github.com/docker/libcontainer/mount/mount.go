package mount

import (
	"fmt"
	"os"
	"path/filepath"
	"syscall"

	"github.com/docker/docker/pkg/symlink"
	"github.com/docker/libcontainer/label"
)

type Mount struct {
	Type        string `json:"type,omitempty"`
	Source      string `json:"source,omitempty"`      // Source path, in the host namespace
	Destination string `json:"destination,omitempty"` // Destination path, in the container
	Writable    bool   `json:"writable,omitempty"`
	Relabel     string `json:"relabel,omitempty"` // Relabel source if set, "z" indicates shared, "Z" indicates unshared
	Private     bool   `json:"private,omitempty"`
	Slave       bool   `json:"slave,omitempty"`
}

func (m *Mount) Mount(rootfs, mountLabel string) error {
	switch m.Type {
	case "bind":
		return m.bindMount(rootfs, mountLabel)
	case "tmpfs":
		return m.tmpfsMount(rootfs, mountLabel)
	default:
		return fmt.Errorf("unsupported mount type %s for %s", m.Type, m.Destination)
	}
}

func (m *Mount) bindMount(rootfs, mountLabel string) error {
	var (
		flags = syscall.MS_BIND | syscall.MS_REC
		dest  = filepath.Join(rootfs, m.Destination)
	)

	if !m.Writable {
		flags = flags | syscall.MS_RDONLY
	}

	if m.Slave {
		flags = flags | syscall.MS_SLAVE
	}

	stat, err := os.Stat(m.Source)
	if err != nil {
		return err
	}

	// FIXME: (crosbymichael) This does not belong here and should be done a layer above
	dest, err = symlink.FollowSymlinkInScope(dest, rootfs)
	if err != nil {
		return err
	}

	if err := createIfNotExists(dest, stat.IsDir()); err != nil {
		return fmt.Errorf("creating new bind mount target %s", err)
	}

	if err := syscall.Mount(m.Source, dest, "bind", uintptr(flags), ""); err != nil {
		return fmt.Errorf("mounting %s into %s %s", m.Source, dest, err)
	}

	if !m.Writable {
		if err := syscall.Mount(m.Source, dest, "bind", uintptr(flags|syscall.MS_REMOUNT), ""); err != nil {
			return fmt.Errorf("remounting %s into %s %s", m.Source, dest, err)
		}
	}

	if m.Relabel != "" {
		if err := label.Relabel(m.Source, mountLabel, m.Relabel); err != nil {
			return fmt.Errorf("relabeling %s to %s %s", m.Source, mountLabel, err)
		}
	}

	if m.Private {
		if err := syscall.Mount("", dest, "none", uintptr(syscall.MS_PRIVATE), ""); err != nil {
			return fmt.Errorf("mounting %s private %s", dest, err)
		}
	}

	return nil
}

func (m *Mount) tmpfsMount(rootfs, mountLabel string) error {
	var (
		err  error
		l    = label.FormatMountLabel("", mountLabel)
		dest = filepath.Join(rootfs, m.Destination)
	)

	// FIXME: (crosbymichael) This does not belong here and should be done a layer above
	if dest, err = symlink.FollowSymlinkInScope(dest, rootfs); err != nil {
		return err
	}

	if err := createIfNotExists(dest, true); err != nil {
		return fmt.Errorf("creating new tmpfs mount target %s", err)
	}

	if err := syscall.Mount("tmpfs", dest, "tmpfs", uintptr(defaultMountFlags), l); err != nil {
		return fmt.Errorf("%s mounting %s in tmpfs", err, dest)
	}

	return nil
}
