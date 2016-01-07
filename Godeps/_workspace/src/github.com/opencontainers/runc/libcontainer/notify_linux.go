// +build linux

package libcontainer

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"syscall"
)

const oomCgroupName = "memory"

// notifyOnOOM returns channel on which you can expect event about OOM,
// if process died without OOM this channel will be closed.
// s is current *libcontainer.State for container.
func notifyOnOOM(paths map[string]string) (<-chan struct{}, error) {
	dir := paths[oomCgroupName]
	if dir == "" {
		return nil, fmt.Errorf("There is no path for %q in state", oomCgroupName)
	}
	oomControl, err := os.Open(filepath.Join(dir, "memory.oom_control"))
	if err != nil {
		return nil, err
	}
	fd, _, syserr := syscall.RawSyscall(syscall.SYS_EVENTFD2, 0, syscall.FD_CLOEXEC, 0)
	if syserr != 0 {
		oomControl.Close()
		return nil, syserr
	}

	eventfd := os.NewFile(fd, "eventfd")

	eventControlPath := filepath.Join(dir, "cgroup.event_control")
	data := fmt.Sprintf("%d %d", eventfd.Fd(), oomControl.Fd())
	if err := ioutil.WriteFile(eventControlPath, []byte(data), 0700); err != nil {
		eventfd.Close()
		oomControl.Close()
		return nil, err
	}
	ch := make(chan struct{})
	go func() {
		defer func() {
			close(ch)
			eventfd.Close()
			oomControl.Close()
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
