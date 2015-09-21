// +build linux

package libcontainer

import (
	"fmt"
	"os"
	"testing"

	"github.com/docker/libcontainer/cgroups"
	"github.com/docker/libcontainer/configs"
)

type mockCgroupManager struct {
	pids  []int
	stats *cgroups.Stats
	paths map[string]string
}

func (m *mockCgroupManager) GetPids() ([]int, error) {
	return m.pids, nil
}

func (m *mockCgroupManager) GetStats() (*cgroups.Stats, error) {
	return m.stats, nil
}

func (m *mockCgroupManager) Apply(pid int) error {
	return nil
}

func (m *mockCgroupManager) Set(container *configs.Config) error {
	return nil
}

func (m *mockCgroupManager) Destroy() error {
	return nil
}

func (m *mockCgroupManager) GetPaths() map[string]string {
	return m.paths
}

func (m *mockCgroupManager) Freeze(state configs.FreezerState) error {
	return nil
}

type mockProcess struct {
	_pid    int
	started string
}

func (m *mockProcess) terminate() error {
	return nil
}

func (m *mockProcess) pid() int {
	return m._pid
}

func (m *mockProcess) startTime() (string, error) {
	return m.started, nil
}

func (m *mockProcess) start() error {
	return nil
}

func (m *mockProcess) wait() (*os.ProcessState, error) {
	return nil, nil
}

func (m *mockProcess) signal(_ os.Signal) error {
	return nil
}

func TestGetContainerPids(t *testing.T) {
	container := &linuxContainer{
		id:            "myid",
		config:        &configs.Config{},
		cgroupManager: &mockCgroupManager{pids: []int{1, 2, 3}},
	}
	pids, err := container.Processes()
	if err != nil {
		t.Fatal(err)
	}
	for i, expected := range []int{1, 2, 3} {
		if pids[i] != expected {
			t.Fatalf("expected pid %d but received %d", expected, pids[i])
		}
	}
}

func TestGetContainerStats(t *testing.T) {
	container := &linuxContainer{
		id:     "myid",
		config: &configs.Config{},
		cgroupManager: &mockCgroupManager{
			pids: []int{1, 2, 3},
			stats: &cgroups.Stats{
				MemoryStats: cgroups.MemoryStats{
					Usage: 1024,
				},
			},
		},
	}
	stats, err := container.Stats()
	if err != nil {
		t.Fatal(err)
	}
	if stats.CgroupStats == nil {
		t.Fatal("cgroup stats are nil")
	}
	if stats.CgroupStats.MemoryStats.Usage != 1024 {
		t.Fatalf("expected memory usage 1024 but recevied %d", stats.CgroupStats.MemoryStats.Usage)
	}
}

func TestGetContainerState(t *testing.T) {
	var (
		pid                 = os.Getpid()
		expectedMemoryPath  = "/sys/fs/cgroup/memory/myid"
		expectedNetworkPath = "/networks/fd"
	)
	container := &linuxContainer{
		id: "myid",
		config: &configs.Config{
			Namespaces: []configs.Namespace{
				{Type: configs.NEWPID},
				{Type: configs.NEWNS},
				{Type: configs.NEWNET, Path: expectedNetworkPath},
				{Type: configs.NEWUTS},
				// emulate host for IPC
				//{Type: configs.NEWIPC},
			},
		},
		initProcess: &mockProcess{
			_pid:    pid,
			started: "010",
		},
		cgroupManager: &mockCgroupManager{
			pids: []int{1, 2, 3},
			stats: &cgroups.Stats{
				MemoryStats: cgroups.MemoryStats{
					Usage: 1024,
				},
			},
			paths: map[string]string{
				"memory": expectedMemoryPath,
			},
		},
	}
	state, err := container.State()
	if err != nil {
		t.Fatal(err)
	}
	if state.InitProcessPid != pid {
		t.Fatalf("expected pid %d but received %d", pid, state.InitProcessPid)
	}
	if state.InitProcessStartTime != "010" {
		t.Fatalf("expected process start time 010 but received %s", state.InitProcessStartTime)
	}
	paths := state.CgroupPaths
	if paths == nil {
		t.Fatal("cgroup paths should not be nil")
	}
	if memPath := paths["memory"]; memPath != expectedMemoryPath {
		t.Fatalf("expected memory path %q but received %q", expectedMemoryPath, memPath)
	}
	for _, ns := range container.config.Namespaces {
		path := state.NamespacePaths[ns.Type]
		if path == "" {
			t.Fatalf("expected non nil namespace path for %s", ns.Type)
		}
		if ns.Type == configs.NEWNET {
			if path != expectedNetworkPath {
				t.Fatalf("expected path %q but received %q", expectedNetworkPath, path)
			}
		} else {
			file := ""
			switch ns.Type {
			case configs.NEWNET:
				file = "net"
			case configs.NEWNS:
				file = "mnt"
			case configs.NEWPID:
				file = "pid"
			case configs.NEWIPC:
				file = "ipc"
			case configs.NEWUSER:
				file = "user"
			case configs.NEWUTS:
				file = "uts"
			}
			expected := fmt.Sprintf("/proc/%d/ns/%s", pid, file)
			if expected != path {
				t.Fatalf("expected path %q but received %q", expected, path)
			}
		}
	}
}
