package fs2

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/opencontainers/cgroups"
	"github.com/opencontainers/cgroups/fscommon"
)

type parseError = fscommon.ParseError

type Manager struct {
	config *cgroups.Cgroup
	// dirPath is like "/sys/fs/cgroup/user.slice/user-1001.slice/session-1.scope"
	dirPath string
	// controllers is content of "cgroup.controllers" file.
	// excludes pseudo-controllers ("devices" and "freezer").
	controllers map[string]struct{}
}

// NewManager creates a manager for cgroup v2 unified hierarchy.
// dirPath is like "/sys/fs/cgroup/user.slice/user-1001.slice/session-1.scope".
// If dirPath is empty, it is automatically set using config.
func NewManager(config *cgroups.Cgroup, dirPath string) (*Manager, error) {
	if dirPath == "" {
		var err error
		dirPath, err = defaultDirPath(config)
		if err != nil {
			return nil, err
		}
	}

	m := &Manager{
		config:  config,
		dirPath: dirPath,
	}
	return m, nil
}

func (m *Manager) getControllers() error {
	if m.controllers != nil {
		return nil
	}

	data, err := cgroups.ReadFile(m.dirPath, "cgroup.controllers")
	if err != nil {
		if m.config.Rootless && m.config.Path == "" {
			return nil
		}
		return err
	}
	fields := strings.Fields(data)
	m.controllers = make(map[string]struct{}, len(fields))
	for _, c := range fields {
		m.controllers[c] = struct{}{}
	}

	return nil
}

func (m *Manager) Apply(pid int) error {
	if err := CreateCgroupPath(m.dirPath, m.config); err != nil {
		// Related tests:
		// - "runc create (no limits + no cgrouppath + no permission) succeeds"
		// - "runc create (rootless + no limits + cgrouppath + no permission) fails with permission error"
		// - "runc create (rootless + limits + no cgrouppath + no permission) fails with informative error"
		if m.config.Rootless {
			if m.config.Path == "" {
				if blNeed, nErr := needAnyControllers(m.config.Resources); nErr == nil && !blNeed {
					return cgroups.ErrRootless
				}
				return fmt.Errorf("rootless needs no limits + no cgrouppath when no permission is granted for cgroups: %w", err)
			}
		}
		return err
	}
	if err := cgroups.WriteCgroupProc(m.dirPath, pid); err != nil {
		return err
	}
	return nil
}

// AddPid adds a process with a given pid to an existing cgroup.
// The subcgroup argument is either empty, or a path relative to
// a cgroup under under the manager's cgroup.
func (m *Manager) AddPid(subcgroup string, pid int) error {
	path := filepath.Join(m.dirPath, subcgroup)
	if !strings.HasPrefix(path, m.dirPath) {
		return fmt.Errorf("bad sub cgroup path: %s", subcgroup)
	}

	return cgroups.WriteCgroupProc(path, pid)
}

func (m *Manager) GetPids() ([]int, error) {
	return cgroups.GetPids(m.dirPath)
}

func (m *Manager) GetAllPids() ([]int, error) {
	return cgroups.GetAllPids(m.dirPath)
}

func (m *Manager) GetStats() (*cgroups.Stats, error) {
	var errs []error

	st := cgroups.NewStats()

	// pids (since kernel 4.5)
	if err := statPids(m.dirPath, st); err != nil {
		errs = append(errs, err)
	}
	// memory (since kernel 4.5)
	if err := statMemory(m.dirPath, st); err != nil && !os.IsNotExist(err) {
		errs = append(errs, err)
	}
	// io (since kernel 4.5)
	if err := statIo(m.dirPath, st); err != nil && !os.IsNotExist(err) {
		errs = append(errs, err)
	}
	// cpu (since kernel 4.15)
	// Note cpu.stat is available even if the controller is not enabled.
	if err := statCpu(m.dirPath, st); err != nil && !os.IsNotExist(err) {
		errs = append(errs, err)
	}
	// PSI (since kernel 4.20).
	var err error
	if st.CpuStats.PSI, err = statPSI(m.dirPath, "cpu.pressure"); err != nil {
		errs = append(errs, err)
	}
	if st.MemoryStats.PSI, err = statPSI(m.dirPath, "memory.pressure"); err != nil {
		errs = append(errs, err)
	}
	if st.BlkioStats.PSI, err = statPSI(m.dirPath, "io.pressure"); err != nil {
		errs = append(errs, err)
	}
	// hugetlb (since kernel 5.6)
	if err := statHugeTlb(m.dirPath, st); err != nil && !os.IsNotExist(err) {
		errs = append(errs, err)
	}
	// rdma (since kernel 4.11)
	if err := fscommon.RdmaGetStats(m.dirPath, st); err != nil && !os.IsNotExist(err) {
		errs = append(errs, err)
	}
	// misc (since kernel 5.13)
	if err := statMisc(m.dirPath, st); err != nil && !os.IsNotExist(err) {
		errs = append(errs, err)
	}
	if len(errs) > 0 && !m.config.Rootless {
		return st, fmt.Errorf("error while statting cgroup v2: %+v", errs)
	}
	return st, nil
}

func (m *Manager) Freeze(state cgroups.FreezerState) error {
	if m.config.Resources == nil {
		return errors.New("cannot toggle freezer: cgroups not configured for container")
	}
	if err := setFreezer(m.dirPath, state); err != nil {
		return err
	}
	m.config.Resources.Freezer = state
	return nil
}

func (m *Manager) Destroy() error {
	return cgroups.RemovePath(m.dirPath)
}

func (m *Manager) Path(_ string) string {
	return m.dirPath
}

func (m *Manager) Set(r *cgroups.Resources) error {
	if r == nil {
		return nil
	}
	if err := m.getControllers(); err != nil {
		return err
	}
	// pids (since kernel 4.5)
	if err := setPids(m.dirPath, r); err != nil {
		return err
	}
	// memory (since kernel 4.5)
	if err := setMemory(m.dirPath, r); err != nil {
		return err
	}
	// io (since kernel 4.5)
	if err := setIo(m.dirPath, r); err != nil {
		return err
	}
	// cpu (since kernel 4.15)
	if err := setCPU(m.dirPath, r); err != nil {
		return err
	}
	// devices (since kernel 4.15, pseudo-controller)
	//
	// When rootless is true, errors from the device subsystem are ignored because it is really not expected to work.
	// However, errors from other subsystems are not ignored.
	// see @test "runc create (rootless + limits + no cgrouppath + no permission) fails with informative error"
	if err := setDevices(m.dirPath, r); err != nil {
		if !m.config.Rootless || errors.Is(err, cgroups.ErrDevicesUnsupported) {
			return err
		}
	}
	// cpuset (since kernel 5.0)
	if err := setCpuset(m.dirPath, r); err != nil {
		return err
	}
	// hugetlb (since kernel 5.6)
	if err := setHugeTlb(m.dirPath, r); err != nil {
		return err
	}
	// rdma (since kernel 4.11)
	if err := fscommon.RdmaSet(m.dirPath, r); err != nil {
		return err
	}
	// freezer (since kernel 5.2, pseudo-controller)
	if err := setFreezer(m.dirPath, r.Freezer); err != nil {
		return err
	}
	if err := m.setUnified(r.Unified); err != nil {
		return err
	}
	m.config.Resources = r
	return nil
}

func setDevices(dirPath string, r *cgroups.Resources) error {
	if cgroups.DevicesSetV2 == nil {
		if len(r.Devices) > 0 {
			return cgroups.ErrDevicesUnsupported
		}
		return nil
	}
	return cgroups.DevicesSetV2(dirPath, r)
}

func (m *Manager) setUnified(res map[string]string) error {
	for k, v := range res {
		if strings.Contains(k, "/") {
			return fmt.Errorf("unified resource %q must be a file name (no slashes)", k)
		}
		if err := cgroups.WriteFileByLine(m.dirPath, k, v); err != nil {
			// Check for both EPERM and ENOENT since O_CREAT is used by WriteFile.
			if errors.Is(err, os.ErrPermission) || errors.Is(err, os.ErrNotExist) {
				// Check if a controller is available,
				// to give more specific error if not.
				c, _, ok := strings.Cut(k, ".")
				if !ok {
					return fmt.Errorf("unified resource %q must be in the form CONTROLLER.PARAMETER", k)
				}
				if _, ok := m.controllers[c]; !ok && c != "cgroup" {
					return fmt.Errorf("unified resource %q can't be set: controller %q not available", k, c)
				}
			}
			return fmt.Errorf("unable to set unified resource %q: %w", k, err)
		}
	}

	return nil
}

func (m *Manager) GetPaths() map[string]string {
	paths := make(map[string]string, 1)
	paths[""] = m.dirPath
	return paths
}

func (m *Manager) GetCgroups() (*cgroups.Cgroup, error) {
	return m.config, nil
}

func (m *Manager) GetFreezerState() (cgroups.FreezerState, error) {
	return getFreezer(m.dirPath)
}

func (m *Manager) Exists() bool {
	return cgroups.PathExists(m.dirPath)
}

func OOMKillCount(path string) (uint64, error) {
	return fscommon.GetValueByKey(path, "memory.events", "oom_kill")
}

func (m *Manager) OOMKillCount() (uint64, error) {
	c, err := OOMKillCount(m.dirPath)
	if err != nil && m.config.Rootless && os.IsNotExist(err) {
		err = nil
	}

	return c, err
}

func CheckMemoryUsage(dirPath string, r *cgroups.Resources) error {
	if !r.MemoryCheckBeforeUpdate {
		return nil
	}

	if r.Memory <= 0 && r.MemorySwap <= 0 {
		return nil
	}

	usage, err := fscommon.GetCgroupParamUint(dirPath, "memory.current")
	if err != nil {
		// This check is on best-effort basis, so if we can't read the
		// current usage (cgroup not yet created, or any other error),
		// we should not fail.
		return nil
	}

	if r.MemorySwap > 0 {
		if uint64(r.MemorySwap) <= usage {
			return fmt.Errorf("rejecting memory+swap limit %d <= usage %d", r.MemorySwap, usage)
		}
	}

	if r.Memory > 0 {
		if uint64(r.Memory) <= usage {
			return fmt.Errorf("rejecting memory limit %d <= usage %d", r.Memory, usage)
		}
	}

	return nil
}
