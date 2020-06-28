// +build linux

package libcontainer

import (
	"io/ioutil"
	"path/filepath"
	"strconv"
	"strings"
	"unsafe"

	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
)

func getValueFromCgroup(path, key string) (int, error) {
	content, err := ioutil.ReadFile(path)
	if err != nil {
		return 0, err
	}

	lines := strings.Split(string(content), "\n")
	for _, line := range lines {
		arr := strings.Split(line, " ")
		if len(arr) == 2 && arr[0] == key {
			return strconv.Atoi(arr[1])
		}
	}
	return 0, nil
}

func registerMemoryEventV2(cgDir, evName, cgEvName string) (<-chan struct{}, error) {
	eventControlPath := filepath.Join(cgDir, evName)
	cgEvPath := filepath.Join(cgDir, cgEvName)
	fd, err := unix.InotifyInit()
	if err != nil {
		return nil, errors.Wrap(err, "unable to init inotify")
	}
	// watching oom kill
	evFd, err := unix.InotifyAddWatch(fd, eventControlPath, unix.IN_MODIFY)
	if err != nil {
		unix.Close(fd)
		return nil, errors.Wrap(err, "unable to add inotify watch")
	}
	// Because no `unix.IN_DELETE|unix.IN_DELETE_SELF` event for cgroup file system, so watching all process exited
	cgFd, err := unix.InotifyAddWatch(fd, cgEvPath, unix.IN_MODIFY)
	if err != nil {
		unix.Close(fd)
		return nil, errors.Wrap(err, "unable to add inotify watch")
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
				offset += unix.SizeofInotifyEvent + uint32(rawEvent.Len)
				if rawEvent.Mask&unix.IN_MODIFY != unix.IN_MODIFY {
					continue
				}
				switch int(rawEvent.Wd) {
				case evFd:
					oom, err := getValueFromCgroup(eventControlPath, "oom_kill")
					if err != nil || oom > 0 {
						ch <- struct{}{}
					}
				case cgFd:
					pids, err := getValueFromCgroup(cgEvPath, "populated")
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
