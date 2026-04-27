package fs

import (
	"strconv"

	"github.com/opencontainers/cgroups"
)

type NetClsGroup struct{}

func (s *NetClsGroup) Name() string {
	return "net_cls"
}

func (s *NetClsGroup) Apply(path string, _ *cgroups.Resources, pid int) error {
	return apply(path, pid)
}

func (s *NetClsGroup) Set(path string, r *cgroups.Resources) error {
	if r.NetClsClassid != 0 {
		if err := cgroups.WriteFile(path, "net_cls.classid", strconv.FormatUint(uint64(r.NetClsClassid), 10)); err != nil {
			return err
		}
	}

	return nil
}

func (s *NetClsGroup) GetStats(path string, stats *cgroups.Stats) error {
	return nil
}
