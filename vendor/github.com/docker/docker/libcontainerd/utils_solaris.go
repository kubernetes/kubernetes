package libcontainerd

import (
	"syscall"

	containerd "github.com/containerd/containerd/api/grpc/types"
	"github.com/opencontainers/runtime-spec/specs-go"
)

func getRootIDs(s specs.Spec) (int, int, error) {
	return 0, 0, nil
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

// setPDeathSig sets the parent death signal to SIGKILL
func setSysProcAttr(sid bool) *syscall.SysProcAttr {
	return nil
}
