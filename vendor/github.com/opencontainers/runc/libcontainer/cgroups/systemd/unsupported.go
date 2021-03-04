// +build !linux

package systemd

import (
	"errors"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/configs"
)

type Manager struct {
	Cgroups *configs.Cgroup
	Paths   map[string]string
}

func IsRunningSystemd() bool {
	return false
}

func NewSystemdCgroupsManager() (func(config *configs.Cgroup, paths map[string]string) cgroups.Manager, error) {
	return nil, errors.New("Systemd not supported")
}

func (m *Manager) Apply(pid int) error {
	return errors.New("Systemd not supported")
}

func (m *Manager) GetPids() ([]int, error) {
	return nil, errors.New("Systemd not supported")
}

func (m *Manager) GetAllPids() ([]int, error) {
	return nil, errors.New("Systemd not supported")
}

func (m *Manager) Destroy() error {
	return errors.New("Systemd not supported")
}

func (m *Manager) GetPaths() map[string]string {
	return nil
}

func (m *Manager) Path(_ string) string {
	return ""
}

func (m *Manager) GetStats() (*cgroups.Stats, error) {
	return nil, errors.New("Systemd not supported")
}

func (m *Manager) Set(container *configs.Config) error {
	return errors.New("Systemd not supported")
}

func (m *Manager) Freeze(state configs.FreezerState) error {
	return errors.New("Systemd not supported")
}

func Freeze(c *configs.Cgroup, state configs.FreezerState) error {
	return errors.New("Systemd not supported")
}

func (m *Manager) GetCgroups() (*configs.Cgroup, error) {
	return nil, errors.New("Systemd not supported")
}

func (m *Manager) Exists() bool {
	return false
}
