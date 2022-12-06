package libcontainer

import (
	"fmt"
	"path/filepath"
	"unsafe"

	"github.com/opencontainers/runc/libcontainer/cgroups/fscommon"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
)

func registerMemoryEventV2(cgDir, evName, cgEvName string) (<-chan struct{}, error) {
	fd, err := unix.InotifyInit()
	if err != nil {
		return nil, fmt.Errorf("unable to init inotify: %w", err)
	}
	// watching oom kill
	evFd, err := unix.InotifyAddWatch(fd, filepath.Join(cgDir, evName), unix.IN_MODIFY)
	if err != nil {
		unix.Close(fd)
		return nil, fmt.Errorf("unable to add inotify watch: %w", err)
	}
	// Because no `unix.IN_DELETE|unix.IN_DELETE_SELF` event for cgroup file system, so watching all process exited
	cgFd, err := unix.InotifyAddWatch(fd, filepath.Join(cgDir, cgEvName), unix.IN_MODIFY)
	if err != nil {
		unix.Close(fd)
		return nil, fmt.Errorf("unable to add inotify watch: %w", err)
	}
	ch := make(chan struct{})
	go func() {
		var (
			buffer [unix.SizeofInotifyEvent + unix.PathMax + 1]byte
			offset uint32
		)
		defer func() {
			unix.Close(fd)
			close(ch)
		}()

		for {
			n, err := unix.Read(fd, buffer[:])
			if err != nil {
				logrus.Warnf("unable to read event data from inotify, got error: %v", err)
				return
			}
			if n < unix.SizeofInotifyEvent {
				logrus.Warnf("we should read at least %d bytes from inotify, but got %d bytes.", unix.SizeofInotifyEvent, n)
				return
			}
			offset = 0
			for offset <= uint32(n-unix.SizeofInotifyEvent) {
				rawEvent := (*unix.InotifyEvent)(unsafe.Pointer(&buffer[offset]))
				offset += unix.SizeofInotifyEvent + rawEvent.Len
				if rawEvent.Mask&unix.IN_MODIFY != unix.IN_MODIFY {
					continue
				}
				switch int(rawEvent.Wd) {
				case evFd:
					oom, err := fscommon.GetValueByKey(cgDir, evName, "oom_kill")
					if err != nil || oom > 0 {
						ch <- struct{}{}
					}
				case cgFd:
					pids, err := fscommon.GetValueByKey(cgDir, cgEvName, "populated")
					if err != nil || pids == 0 {
						return
					}
				}
			}
		}
	}()
	return ch, nil
}

// notifyOnOOMV2 returns channel on which you can expect event about OOM,
// if process died without OOM this channel will be closed.
func notifyOnOOMV2(path string) (<-chan struct{}, error) {
	return registerMemoryEventV2(path, "memory.events", "cgroup.events")
}
