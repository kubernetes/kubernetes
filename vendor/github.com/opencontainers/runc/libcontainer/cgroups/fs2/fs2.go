// +build linux

package fs2

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/pkg/errors"
	"golang.org/x/sys/unix"
)

type manager struct {
	config *configs.Cgroup
	// dirPath is like "/sys/fs/cgroup/user.slice/user-1001.slice/session-1.scope"
	dirPath string
	// controllers is content of "cgroup.controllers" file.
	// excludes pseudo-controllers ("devices" and "freezer").
	controllers map[string]struct{}
	rootless    bool
}

// NewManager creates a manager for cgroup v2 unified hierarchy.
// dirPath is like "/sys/fs/cgroup/user.slice/user-1001.slice/session-1.scope".
// If dirPath is empty, it is automatically set using config.
func NewManager(config *configs.Cgroup, dirPath string, rootless bool) (cgroups.Manager, error) {
	if config == nil {
		config = &configs.Cgroup{}
	}
	if dirPath == "" {
		var err error
		dirPath, err = defaultDirPath(config)
		if err != nil {
			return nil, err
		}
	}

	m := &manager{
		config:   config,
		dirPath:  dirPath,
		rootless: rootless,
	}
	return m, nil
}

func (m *manager) getControllers() error {
	if m.controllers != nil {
		return nil
	}

	file := filepath.Join(m.dirPath, "cgroup.controllers")
	data, err := ioutil.ReadFile(file)
	if err != nil {
		if m.rootless && m.config.Path == "" {
			return nil
		}
		return err
	}
	fields := strings.Fields(string(data))
	m.controllers = make(map[string]struct{}, len(fields))
	for _, c := range fields {
		m.controllers[c] = struct{}{}
	}

	return nil
}

func (m *manager) Apply(pid int) error {
	if err := CreateCgroupPath(m.dirPath, m.config); err != nil {
		// Related tests:
		// - "runc create (no limits + no cgrouppath + no permission) succeeds"
		// - "runc create (rootless + no limits + cgrouppath + no permission) fails with permission error"
		// - "runc create (rootless + limits + no cgrouppath + no permission) fails with informative error"
		if m.rootless {
			if m.config.Path == "" {
				if blNeed, nErr := needAnyControllers(m.config); nErr == nil && !blNeed {
					return nil
				}
				return errors.Wrap(err, "rootless needs no limits + no cgrouppath when no permission is granted for cgroups")
			}
		}
		return err
	}
	if err := cgroups.WriteCgroupProc(m.dirPath, pid); err != nil {
		return err
	}
	return nil
}

func (m *manager) GetPids() ([]int, error) {
	return cgroups.GetPids(m.dirPath)
}

func (m *manager) GetAllPids() ([]int, error) {
	return cgroups.GetAllPids(m.dirPath)
}

func (m *manager) GetStats() (*cgroups.Stats, error) {
	var (
		errs []error
	)

	st := cgroups.NewStats()
	if err := m.getControllers(); err != nil {
		return st, err
	}

	// pids (since kernel 4.5)
	if _, ok := m.controllers["pids"]; ok {
		if err := statPids(m.dirPath, st); err != nil {
			errs = append(errs, err)
		}
	} else {
		if err := statPidsWithoutController(m.dirPath, st); err != nil {
			errs = append(errs, err)
		}
	}
	// memory (since kernel 4.5)
	if _, ok := m.controllers["memory"]; ok {
		if err := statMemory(m.dirPath, st); err != nil {
			errs = append(errs, err)
		}
	}
	// io (since kernel 4.5)
	if _, ok := m.controllers["io"]; ok {
		if err := statIo(m.dirPath, st); err != nil {
			errs = append(errs, err)
		}
	}
	// cpu (since kernel 4.15)
	if _, ok := m.controllers["cpu"]; ok {
		if err := statCpu(m.dirPath, st); err != nil {
			errs = append(errs, err)
		}
	}
	// hugetlb (since kernel 5.6)
	if _, ok := m.controllers["hugetlb"]; ok {
		if err := statHugeTlb(m.dirPath, st); err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) > 0 && !m.rootless {
		return st, errors.Errorf("error while statting cgroup v2: %+v", errs)
	}
	return st, nil
}

func (m *manager) Freeze(state configs.FreezerState) error {
	if err := setFreezer(m.dirPath, state); err != nil {
		return err
	}
	m.config.Resources.Freezer = state
	return nil
}

func rmdir(path string) error {
	err := unix.Rmdir(path)
	if err == nil || err == unix.ENOENT {
		return nil
	}
	return &os.PathError{Op: "rmdir", Path: path, Err: err}
}

// removeCgroupPath aims to remove cgroup path recursively
// Because there may be subcgroups in it.
func removeCgroupPath(path string) error {
	// try the fast path first
	if err := rmdir(path); err == nil {
		return nil
	}

	infos, err := ioutil.ReadDir(path)
	if err != nil {
		if os.IsNotExist(err) {
			err = nil
		}
		return err
	}
	for _, info := range infos {
		if info.IsDir() {
			// We should remove subcgroups dir first
			if err = removeCgroupPath(filepath.Join(path, info.Name())); err != nil {
				break
			}
		}
	}
	if err == nil {
		err = rmdir(path)
	}
	return err
}

func (m *manager) Destroy() error {
	return removeCgroupPath(m.dirPath)
}

func (m *manager) Path(_ string) string {
	return m.dirPath
}

func (m *manager) Set(container *configs.Config) error {
	if container == nil || container.Cgroups == nil {
		return nil
	}
	if err := m.getControllers(); err != nil {
		return err
	}
	// pids (since kernel 4.5)
	if err := setPids(m.dirPath, container.Cgroups); err != nil {
		return err
	}
	// memory (since kernel 4.5)
	if err := setMemory(m.dirPath, container.Cgroups); err != nil {
		return err
	}
	// io (since kernel 4.5)
	if err := setIo(m.dirPath, container.Cgroups); err != nil {
		return err
	}
	// cpu (since kernel 4.15)
	if err := setCpu(m.dirPath, container.Cgroups); err != nil {
		return err
	}
	// devices (since kernel 4.15, pseudo-controller)
	//
	// When m.Rootless is true, errors from the device subsystem are ignored because it is really not expected to work.
	// However, errors from other subsystems are not ignored.
	// see @test "runc create (rootless + limits + no cgrouppath + no permission) fails with informative error"
	if err := setDevices(m.dirPath, container.Cgroups); err != nil && !m.rootless {
		return err
	}
	// cpuset (since kernel 5.0)
	if err := setCpuset(m.dirPath, container.Cgroups); err != nil {
		return err
	}
	// hugetlb (since kernel 5.6)
	if err := setHugeTlb(m.dirPath, container.Cgroups); err != nil {
		return err
	}
	// freezer (since kernel 5.2, pseudo-controller)
	if err := setFreezer(m.dirPath, container.Cgroups.Freezer); err != nil {
		return err
	}
	m.config = container.Cgroups
	return nil
}

func (m *manager) GetPaths() map[string]string {
	paths := make(map[string]string, 1)
	paths[""] = m.dirPath
	return paths
}

func (m *manager) GetCgroups() (*configs.Cgroup, error) {
	return m.config, nil
}

func (m *manager) GetFreezerState() (configs.FreezerState, error) {
	return getFreezer(m.dirPath)
}

func (m *manager) Exists() bool {
	return cgroups.PathExists(m.dirPath)
}
