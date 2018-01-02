package libcontainerd

import (
	"syscall"

	containerd "github.com/containerd/containerd/api/grpc/types"
	"github.com/opencontainers/runtime-spec/specs-go"
	"golang.org/x/sys/unix"
)

func getRootIDs(s specs.Spec) (int, int, error) {
	var hasUserns bool
	for _, ns := range s.Linux.Namespaces {
		if ns.Type == specs.UserNamespace {
			hasUserns = true
			break
		}
	}
	if !hasUserns {
		return 0, 0, nil
	}
	uid := hostIDFromMap(0, s.Linux.UIDMappings)
	gid := hostIDFromMap(0, s.Linux.GIDMappings)
	return uid, gid, nil
}

func hostIDFromMap(id uint32, mp []specs.LinuxIDMapping) int {
	for _, m := range mp {
		if id >= m.ContainerID && id <= m.ContainerID+m.Size-1 {
			return int(m.HostID + id - m.ContainerID)
		}
	}
	return 0
}

func systemPid(ctr *containerd.Container) uint32 {
	var pid uint32
	for _, p := range ctr.Processes {
		if p.Pid == InitFriendlyName {
			pid = p.SystemPid
		}
	}
	return pid
}

func convertRlimits(sr []specs.LinuxRlimit) (cr []*containerd.Rlimit) {
	for _, r := range sr {
		cr = append(cr, &containerd.Rlimit{
			Type: r.Type,
			Hard: r.Hard,
			Soft: r.Soft,
		})
	}
	return
}

// setPDeathSig sets the parent death signal to SIGKILL
func setSysProcAttr(sid bool) *syscall.SysProcAttr {
	return &syscall.SysProcAttr{
		Setsid:    sid,
		Pdeathsig: unix.SIGKILL,
	}
}
