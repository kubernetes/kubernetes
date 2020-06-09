// +build !linux

package systemd

import (
	"fmt"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/configs"
)

type Manager struct {
	Cgroups *configs.Cgroup
	Paths   map[string]string
}

func UseSystemd() bool {
	return false
}

func NewSystemdCgroupsManager() (func(config *configs.Cgroup, paths map[string]string) cgroups.Manager, error) {
	return nil, fmt.Errorf("Systemd not supported")
}

func (m *Manager) Apply(pid int) error {
	return fmt.Errorf("Systemd not supported")
}

func (m *Manager) GetPids() ([]int, error) {
	return nil, fmt.Errorf("Systemd not supported")
}

func (m *Manager) GetAllPids() ([]int, error) {
	return nil, fmt.Errorf("Systemd not supported")
}

func (m *Manager) Destroy() error {
	return fmt.Errorf("Systemd not supported")
}

func (m *Manager) GetPaths() map[string]string {
	return nil
}

func (m *Manager) GetUnifiedPath() (string, error) {
	return "", fmt.Errorf("Systemd not supported")
}

func (m *Manager) GetStats() (*cgroups.Stats, error) {
	return nil, fmt.Errorf("Systemd not supported")
}

func (m *Manager) Set(container *configs.Config) error {
	return fmt.Errorf("Systemd not supported")
}

func (m *Manager) Freeze(state configs.FreezerState) error {
	return fmt.Errorf("Systemd not supported")
}

func Freeze(c *configs.Cgroup, state configs.FreezerState) error {
	return fmt.Errorf("Systemd not supported")
}

func (m *Manager) GetCgroups() (*configs.Cgroup, error) {
	return nil, fmt.Errorf("Systemd not supported")
}
