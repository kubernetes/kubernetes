package ebpf

import (
	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/asm"
	"github.com/cilium/ebpf/link"
	"github.com/pkg/errors"
	"golang.org/x/sys/unix"
)

// LoadAttachCgroupDeviceFilter installs eBPF device filter program to /sys/fs/cgroup/<foo> directory.
//
// Requires the system to be running in cgroup2 unified-mode with kernel >= 4.15 .
//
// https://github.com/torvalds/linux/commit/ebc614f687369f9df99828572b1d85a7c2de3d92
func LoadAttachCgroupDeviceFilter(insts asm.Instructions, license string, dirFD int) (func() error, error) {
	nilCloser := func() error {
		return nil
	}
	// Increase `ulimit -l` limit to avoid BPF_PROG_LOAD error (#2167).
	// This limit is not inherited into the container.
	memlockLimit := &unix.Rlimit{
		Cur: unix.RLIM_INFINITY,
		Max: unix.RLIM_INFINITY,
	}
	_ = unix.Setrlimit(unix.RLIMIT_MEMLOCK, memlockLimit)
	spec := &ebpf.ProgramSpec{
		Type:         ebpf.CGroupDevice,
		Instructions: insts,
		License:      license,
	}
	prog, err := ebpf.NewProgram(spec)
	if err != nil {
		return nilCloser, err
	}
	err = link.RawAttachProgram(link.RawAttachProgramOptions{
		Target:  dirFD,
		Program: prog,
		Attach:  ebpf.AttachCGroupDevice,
		Flags:   unix.BPF_F_ALLOW_MULTI,
	})
	if err != nil {
		return nilCloser, errors.Wrap(err, "failed to call BPF_PROG_ATTACH (BPF_CGROUP_DEVICE, BPF_F_ALLOW_MULTI)")
	}
	closer := func() error {
		err = link.RawDetachProgram(link.RawDetachProgramOptions{
			Target:  dirFD,
			Program: prog,
			Attach:  ebpf.AttachCGroupDevice,
		})
		if err != nil {
			return errors.Wrap(err, "failed to call BPF_PROG_DETACH (BPF_CGROUP_DEVICE)")
		}
		return nil
	}
	return closer, nil
}
