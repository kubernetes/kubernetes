package libcontainer

import "github.com/docker/libcontainer/cgroups"

type Stats struct {
	Interfaces  []*NetworkInterface
	CgroupStats *cgroups.Stats
}
