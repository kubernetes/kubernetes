package fs

import (
	"github.com/docker/libcontainer/cgroups"
	"github.com/docker/libcontainer/configs"
)

type HugetlbGroup struct {
}

func (s *HugetlbGroup) Apply(d *data) error {
	// we just want to join this group even though we don't set anything
	if _, err := d.join("hugetlb"); err != nil && !cgroups.IsNotFound(err) {
		return err
	}
	return nil
}

func (s *HugetlbGroup) Set(path string, cgroup *configs.Cgroup) error {
	return nil
}

func (s *HugetlbGroup) Remove(d *data) error {
	return removePath(d.path("hugetlb"))
}

func (s *HugetlbGroup) GetStats(path string, stats *cgroups.Stats) error {
	return nil
}
