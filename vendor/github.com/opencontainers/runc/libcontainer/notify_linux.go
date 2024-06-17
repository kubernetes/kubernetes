package libcontainer

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"golang.org/x/sys/unix"
)

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
	if err := os.WriteFile(eventControlPath, []byte(data), 0o700); err != nil {
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
func notifyOnOOM(dir string) (<-chan struct{}, error) {
	if dir == "" {
		return nil, errors.New("memory controller missing")
	}

	return registerMemoryEvent(dir, "memory.oom_control", "")
}

func notifyMemoryPressure(dir string, level PressureLevel) (<-chan struct{}, error) {
	if dir == "" {
		return nil, errors.New("memory controller missing")
	}

	if level > CriticalPressure {
		return nil, fmt.Errorf("invalid pressure level %d", level)
	}

	levelStr := []string{"low", "medium", "critical"}[level]
	return registerMemoryEvent(dir, "memory.pressure_level", levelStr)
}
