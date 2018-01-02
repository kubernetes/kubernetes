// +build linux

package client

import (
	"os/exec"
	"syscall"

	"github.com/containerd/cgroups"
	"github.com/pkg/errors"
)

func getSysProcAttr(nonewns bool) *syscall.SysProcAttr {
	attr := syscall.SysProcAttr{
		Setpgid: true,
	}
	if !nonewns {
		attr.Cloneflags = syscall.CLONE_NEWNS
	}
	return &attr
}

func setCgroup(cgroupPath string, cmd *exec.Cmd) error {
	cg, err := cgroups.Load(cgroups.V1, cgroups.StaticPath(cgroupPath))
	if err != nil {
		return errors.Wrapf(err, "failed to load cgroup %s", cgroupPath)
	}
	if err := cg.Add(cgroups.Process{
		Pid: cmd.Process.Pid,
	}); err != nil {
		return errors.Wrapf(err, "failed to join cgroup %s", cgroupPath)
	}
	return nil
}
