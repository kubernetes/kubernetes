package libcontainer

import "github.com/opencontainers/runc/libcontainer/cgroups"
import "github.com/opencontainers/runc/libcontainer/intelrdt"

type Stats struct {
	Interfaces    []*NetworkInterface
	CgroupStats   *cgroups.Stats
	IntelRdtStats *intelrdt.Stats
}
