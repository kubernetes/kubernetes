// +build linux

package fs2

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	securejoin "github.com/cyphar/filepath-securejoin"
	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/pkg/errors"
)

// NewManager creates a manager for cgroup v2 unified hierarchy.
// dirPath is like "/sys/fs/cgroup/user.slice/user-1001.slice/session-1.scope".
// If dirPath is empty, it is automatically set using config.
func NewManager(config *configs.Cgroup, dirPath string, rootless bool) (cgroups.Manager, error) {
	if config == nil {
		config = &configs.Cgroup{}
	}
	if dirPath != "" {
		if filepath.Clean(dirPath) != dirPath || !filepath.IsAbs(dirPath) {
			return nil, errors.Errorf("invalid dir path %q", dirPath)
		}
	} else {
		var err error
		dirPath, err = defaultDirPath(config)
		if err != nil {
			return nil, err
		}
	}
	controllers, err := detectControllers(dirPath)
	if err != nil && !rootless {
		return nil, err
	}

	m := &manager{
		config:      config,
		dirPath:     dirPath,
		controllers: controllers,
		rootless:    rootless,
	}
	return m, nil
}

func detectControllers(dirPath string) (map[string]struct{}, error) {
	if err := os.MkdirAll(dirPath, 0755); err != nil {
		return nil, err
	}
	controllersPath, err := securejoin.SecureJoin(dirPath, "cgroup.controllers")
	if err != nil {
		return nil, err
	}
	controllersData, err := ioutil.ReadFile(controllersPath)
	if err != nil {
		return nil, err
	}
	controllersFields := strings.Fields(string(controllersData))
	controllers := make(map[string]struct{}, len(controllersFields))
	for _, c := range controllersFields {
		controllers[c] = struct{}{}
	}
	return controllers, nil
}

type manager struct {
	config *configs.Cgroup
	// dirPath is like "/sys/fs/cgroup/user.slice/user-1001.slice/session-1.scope"
	dirPath string
	// controllers is content of "cgroup.controllers" file.
	// excludes pseudo-controllers ("devices" and "freezer").
	controllers map[string]struct{}
	rootless    bool
}

func (m *manager) Apply(pid int) error {
	if err := cgroups.WriteCgroupProc(m.dirPath, pid); err != nil && !m.rootless {
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
		st   cgroups.Stats
		errs []error
	)
	// pids (since kernel 4.5)
	if _, ok := m.controllers["pids"]; ok {
		if err := statPids(m.dirPath, &st); err != nil {
			errs = append(errs, err)
		}
	} else {
		if err := statPidsWithoutController(m.dirPath, &st); err != nil {
			errs = append(errs, err)
		}
	}
	// memory (since kenrel 4.5)
	if _, ok := m.controllers["memory"]; ok {
		if err := statMemory(m.dirPath, &st); err != nil {
			errs = append(errs, err)
		}
	}
	// io (since kernel 4.5)
	if _, ok := m.controllers["io"]; ok {
		if err := statIo(m.dirPath, &st); err != nil {
			errs = append(errs, err)
		}
	}
	// cpu (since kernel 4.15)
	if _, ok := m.controllers["cpu"]; ok {
		if err := statCpu(m.dirPath, &st); err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) > 0 && !m.rootless {
		return &st, errors.Errorf("error while statting cgroup v2: %+v", errs)
	}
	return &st, nil
}

func (m *manager) Freeze(state configs.FreezerState) error {
	if err := setFreezer(m.dirPath, state); err != nil {
		return err
	}
	m.config.Resources.Freezer = state
	return nil
}

func (m *manager) Destroy() error {
	return os.RemoveAll(m.dirPath)
}

// GetPaths is for compatibility purpose and should be removed in future
func (m *manager) GetPaths() map[string]string {
	paths := map[string]string{
		// pseudo-controller for compatibility
		"devices": m.dirPath,
		"freezer": m.dirPath,
	}
	for c := range m.controllers {
		paths[c] = m.dirPath
	}
	return paths
}

func (m *manager) GetUnifiedPath() (string, error) {
	return m.dirPath, nil
}

func (m *manager) Set(container *configs.Config) error {
	if container == nil || container.Cgroups == nil {
		return nil
	}
	var errs []error
	// pids (since kernel 4.5)
	if _, ok := m.controllers["pids"]; ok {
		if err := setPids(m.dirPath, container.Cgroups); err != nil {
			errs = append(errs, err)
		}
	}
	// memory (since kernel 4.5)
	if _, ok := m.controllers["memory"]; ok {
		if err := setMemory(m.dirPath, container.Cgroups); err != nil {
			errs = append(errs, err)
		}
	}
	// io (since kernel 4.5)
	if _, ok := m.controllers["io"]; ok {
		if err := setIo(m.dirPath, container.Cgroups); err != nil {
			errs = append(errs, err)
		}
	}
	// cpu (since kernel 4.15)
	if _, ok := m.controllers["cpu"]; ok {
		if err := setCpu(m.dirPath, container.Cgroups); err != nil {
			errs = append(errs, err)
		}
	}
	// devices (since kernel 4.15, pseudo-controller)
	if err := setDevices(m.dirPath, container.Cgroups); err != nil {
		errs = append(errs, err)
	}
	// cpuset (since kernel 5.0)
	if _, ok := m.controllers["cpuset"]; ok {
		if err := setCpuset(m.dirPath, container.Cgroups); err != nil {
			errs = append(errs, err)
		}
	}
	// freezer (since kernel 5.2, pseudo-controller)
	if err := setFreezer(m.dirPath, container.Cgroups.Freezer); err != nil {
		errs = append(errs, err)
	}
	if len(errs) > 0 && !m.rootless {
		return errors.Errorf("error while setting cgroup v2: %+v", errs)
	}
	m.config = container.Cgroups
	return nil
}

func (m *manager) GetCgroups() (*configs.Cgroup, error) {
	return m.config, nil
}
