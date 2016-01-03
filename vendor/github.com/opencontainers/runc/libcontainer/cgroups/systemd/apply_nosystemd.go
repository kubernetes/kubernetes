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

func (m *Manager) Apply(pid int) error {
	return fmt.Errorf("Systemd not supported")
}

func (m *Manager) GetPids() ([]int, error) {
	return nil, fmt.Errorf("Systemd not supported")
}

func (m *Manager) Destroy() error {
	return fmt.Errorf("Systemd not supported")
}

func (m *Manager) GetPaths() map[string]string {
	return nil
}

func (m *Manager) GetStats() (*cgroups.Stats, error) {
	return nil, fmt.Errorf("Systemd not supported")
}

func (m *Manager) Set(container *configs.Config) error {
	return nil, fmt.Errorf("Systemd not supported")
}

func (m *Manager) Freeze(state configs.FreezerState) error {
	return fmt.Errorf("Systemd not supported")
}

func Freeze(c *configs.Cgroup, state configs.FreezerState) error {
	return fmt.Errorf("Systemd not supported")
}
