package ebpf

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"sync"
	"unsafe"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/asm"
	"github.com/cilium/ebpf/link"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
)

func nilCloser() error {
	return nil
}

func findAttachedCgroupDeviceFilters(dirFd int) ([]*ebpf.Program, error) {
	type bpfAttrQuery struct {
		TargetFd    uint32
		AttachType  uint32
		QueryType   uint32
		AttachFlags uint32
		ProgIds     uint64 // __aligned_u64
		ProgCnt     uint32
	}

	// Currently you can only have 64 eBPF programs attached to a cgroup.
	size := 64
	retries := 0
	for retries < 10 {
		progIds := make([]uint32, size)
		query := bpfAttrQuery{
			TargetFd:   uint32(dirFd),
			AttachType: uint32(unix.BPF_CGROUP_DEVICE),
			ProgIds:    uint64(uintptr(unsafe.Pointer(&progIds[0]))),
			ProgCnt:    uint32(len(progIds)),
		}

		// Fetch the list of program ids.
		_, _, errno := unix.Syscall(unix.SYS_BPF,
			uintptr(unix.BPF_PROG_QUERY),
			uintptr(unsafe.Pointer(&query)),
			unsafe.Sizeof(query))
		size = int(query.ProgCnt)
		runtime.KeepAlive(query)
		if errno != 0 {
			// On ENOSPC we get the correct number of programs.
			if errno == unix.ENOSPC {
				retries++
				continue
			}
			return nil, fmt.Errorf("bpf_prog_query(BPF_CGROUP_DEVICE) failed: %w", errno)
		}

		// Convert the ids to program handles.
		progIds = progIds[:size]
		programs := make([]*ebpf.Program, 0, len(progIds))
		for _, progId := range progIds {
			program, err := ebpf.NewProgramFromID(ebpf.ProgramID(progId))
			if err != nil {
				// We skip over programs that give us -EACCES or -EPERM. This
				// is necessary because there may be BPF programs that have
				// been attached (such as with --systemd-cgroup) which have an
				// LSM label that blocks us from interacting with the program.
				//
				// Because additional BPF_CGROUP_DEVICE programs only can add
				// restrictions, there's no real issue with just ignoring these
				// programs (and stops runc from breaking on distributions with
				// very strict SELinux policies).
				if errors.Is(err, os.ErrPermission) {
					logrus.Debugf("ignoring existing CGROUP_DEVICE program (prog_id=%v) which cannot be accessed by runc -- likely due to LSM policy: %v", progId, err)
					continue
				}
				return nil, fmt.Errorf("cannot fetch program from id: %w", err)
			}
			programs = append(programs, program)
		}
		runtime.KeepAlive(progIds)
		return programs, nil
	}

	return nil, errors.New("could not get complete list of CGROUP_DEVICE programs")
}

var (
	haveBpfProgReplaceBool bool
	haveBpfProgReplaceOnce sync.Once
)

// Loosely based on the BPF_F_REPLACE support check in
//   <https://github.com/cilium/ebpf/blob/v0.6.0/link/syscalls.go>.
//
// TODO: move this logic to cilium/ebpf
func haveBpfProgReplace() bool {
	haveBpfProgReplaceOnce.Do(func() {
		prog, err := ebpf.NewProgram(&ebpf.ProgramSpec{
			Type:    ebpf.CGroupDevice,
			License: "MIT",
			Instructions: asm.Instructions{
				asm.Mov.Imm(asm.R0, 0),
				asm.Return(),
			},
		})
		if err != nil {
			logrus.Debugf("checking for BPF_F_REPLACE support: ebpf.NewProgram failed: %v", err)
			return
		}
		defer prog.Close()

		devnull, err := os.Open("/dev/null")
		if err != nil {
			logrus.Debugf("checking for BPF_F_REPLACE support: open dummy target fd: %v", err)
			return
		}
		defer devnull.Close()

		// We know that we have BPF_PROG_ATTACH since we can load
		// BPF_CGROUP_DEVICE programs. If passing BPF_F_REPLACE gives us EINVAL
		// we know that the feature isn't present.
		err = link.RawAttachProgram(link.RawAttachProgramOptions{
			// We rely on this fd being checked after attachFlags.
			Target: int(devnull.Fd()),
			// Attempt to "replace" bad fds with this program.
			Program: prog,
			Attach:  ebpf.AttachCGroupDevice,
			Flags:   unix.BPF_F_ALLOW_MULTI | unix.BPF_F_REPLACE,
		})
		if errors.Is(err, unix.EINVAL) {
			// not supported
			return
		}
		// attach_flags test succeeded.
		if !errors.Is(err, unix.EBADF) {
			logrus.Debugf("checking for BPF_F_REPLACE: got unexpected (not EBADF or EINVAL) error: %v", err)
		}
		haveBpfProgReplaceBool = true
	})
	return haveBpfProgReplaceBool
}

// LoadAttachCgroupDeviceFilter installs eBPF device filter program to /sys/fs/cgroup/<foo> directory.
//
// Requires the system to be running in cgroup2 unified-mode with kernel >= 4.15 .
//
// https://github.com/torvalds/linux/commit/ebc614f687369f9df99828572b1d85a7c2de3d92
func LoadAttachCgroupDeviceFilter(insts asm.Instructions, license string, dirFd int) (func() error, error) {
	// Increase `ulimit -l` limit to avoid BPF_PROG_LOAD error (#2167).
	// This limit is not inherited into the container.
	memlockLimit := &unix.Rlimit{
		Cur: unix.RLIM_INFINITY,
		Max: unix.RLIM_INFINITY,
	}
	_ = unix.Setrlimit(unix.RLIMIT_MEMLOCK, memlockLimit)

	// Get the list of existing programs.
	oldProgs, err := findAttachedCgroupDeviceFilters(dirFd)
	if err != nil {
		return nilCloser, err
	}
	useReplaceProg := haveBpfProgReplace() && len(oldProgs) == 1

	// Generate new program.
	spec := &ebpf.ProgramSpec{
		Type:         ebpf.CGroupDevice,
		Instructions: insts,
		License:      license,
	}
	prog, err := ebpf.NewProgram(spec)
	if err != nil {
		return nilCloser, err
	}

	// If there is only one old program, we can just replace it directly.
	var (
		replaceProg *ebpf.Program
		attachFlags uint32 = unix.BPF_F_ALLOW_MULTI
	)
	if useReplaceProg {
		replaceProg = oldProgs[0]
		attachFlags |= unix.BPF_F_REPLACE
	}
	err = link.RawAttachProgram(link.RawAttachProgramOptions{
		Target:  dirFd,
		Program: prog,
		Replace: replaceProg,
		Attach:  ebpf.AttachCGroupDevice,
		Flags:   attachFlags,
	})
	if err != nil {
		return nilCloser, fmt.Errorf("failed to call BPF_PROG_ATTACH (BPF_CGROUP_DEVICE, BPF_F_ALLOW_MULTI): %w", err)
	}
	closer := func() error {
		err = link.RawDetachProgram(link.RawDetachProgramOptions{
			Target:  dirFd,
			Program: prog,
			Attach:  ebpf.AttachCGroupDevice,
		})
		if err != nil {
			return fmt.Errorf("failed to call BPF_PROG_DETACH (BPF_CGROUP_DEVICE): %w", err)
		}
		// TODO: Should we attach the old filters back in this case? Otherwise
		//       we fail-open on a security feature, which is a bit scary.
		return nil
	}
	if !useReplaceProg {
		logLevel := logrus.DebugLevel
		// If there was more than one old program, give a warning (since this
		// really shouldn't happen with runc-managed cgroups) and then detach
		// all the old programs.
		if len(oldProgs) > 1 {
			// NOTE: Ideally this should be a warning but it turns out that
			//       systemd-managed cgroups trigger this warning (apparently
			//       systemd doesn't delete old non-systemd programs when
			//       setting properties).
			logrus.Infof("found more than one filter (%d) attached to a cgroup -- removing extra filters!", len(oldProgs))
			logLevel = logrus.InfoLevel
		}
		for idx, oldProg := range oldProgs {
			// Output some extra debug info.
			if info, err := oldProg.Info(); err == nil {
				fields := logrus.Fields{
					"type": info.Type.String(),
					"tag":  info.Tag,
					"name": info.Name,
				}
				if id, ok := info.ID(); ok {
					fields["id"] = id
				}
				if runCount, ok := info.RunCount(); ok {
					fields["run_count"] = runCount
				}
				if runtime, ok := info.Runtime(); ok {
					fields["runtime"] = runtime.String()
				}
				logrus.WithFields(fields).Logf(logLevel, "removing old filter %d from cgroup", idx)
			}
			err = link.RawDetachProgram(link.RawDetachProgramOptions{
				Target:  dirFd,
				Program: oldProg,
				Attach:  ebpf.AttachCGroupDevice,
			})
			if err != nil {
				return closer, fmt.Errorf("failed to call BPF_PROG_DETACH (BPF_CGROUP_DEVICE) on old filter program: %w", err)
			}
		}
	}
	return closer, nil
}
