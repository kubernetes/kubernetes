// +build linux

package libcontainer

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"

	"golang.org/x/sys/unix"
)

const oomCgroupName = "memory"

type PressureLevel uint

const (
	LowPressure PressureLevel = iota
	MediumPressure
	CriticalPressure
)

func registerMemoryEvent(cgDir string, evName string, arg string) (<-chan struct{}, error) {
	evFile, err := os.Open(filepath.Join(cgDir, evName))
	if err != nil {
		return nil, err
	}
	fd, err := unix.Eventfd(0, unix.EFD_CLOEXEC)
	if err != nil {
		evFile.Close()
		return nil, err
	}

	eventfd := os.NewFile(uintptr(fd), "eventfd")

	eventControlPath := filepath.Join(cgDir, "cgroup.event_control")
	data := fmt.Sprintf("%d %d %s", eventfd.Fd(), evFile.Fd(), arg)
	if err := ioutil.WriteFile(eventControlPath, []byte(data), 0700); err != nil {
		eventfd.Close()
		evFile.Close()
		return nil, err
	}
	ch := make(chan struct{})
	go func() {
		defer func() {
			eventfd.Close()
			evFile.Close()
			close(ch)
		}()
		buf := make([]byte, 8)
		for {
			if _, err := eventfd.Read(buf); err != nil {
				return
			}
			// When a cgroup is destroyed, an event is sent to eventfd.
			// So if the control path is gone, return instead of notifying.
			if _, err := os.Lstat(eventControlPath); os.IsNotExist(err) {
				return
			}
			ch <- struct{}{}
		}
	}()
	return ch, nil
}

// notifyOnOOM returns channel on which you can expect event about OOM,
// if process died without OOM this channel will be closed.
func notifyOnOOM(paths map[string]string) (<-chan struct{}, error) {
	dir := paths[oomCgroupName]
	if dir == "" {
		return nil, fmt.Errorf("path %q missing", oomCgroupName)
	}

	return registerMemoryEvent(dir, "memory.oom_control", "")
}

func notifyMemoryPressure(paths map[string]string, level PressureLevel) (<-chan struct{}, error) {
	dir := paths[oomCgroupName]
	if dir == "" {
		return nil, fmt.Errorf("path %q missing", oomCgroupName)
	}

	if level > CriticalPressure {
		return nil, fmt.Errorf("invalid pressure level %d", level)
	}

	levelStr := []string{"low", "medium", "critical"}[level]
	return registerMemoryEvent(dir, "memory.pressure_level", levelStr)
}
