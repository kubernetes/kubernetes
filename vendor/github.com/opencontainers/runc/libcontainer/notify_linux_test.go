// +build linux

package libcontainer

import (
	"encoding/binary"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

type notifyFunc func(paths map[string]string) (<-chan struct{}, error)

func testMemoryNotification(t *testing.T, evName string, notify notifyFunc, targ string) {
	memoryPath, err := ioutil.TempDir("", "testmemnotification-"+evName)
	if err != nil {
		t.Fatal(err)
	}
	evFile := filepath.Join(memoryPath, evName)
	eventPath := filepath.Join(memoryPath, "cgroup.event_control")
	if err := ioutil.WriteFile(evFile, []byte{}, 0700); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(eventPath, []byte{}, 0700); err != nil {
		t.Fatal(err)
	}
	paths := map[string]string{
		"memory": memoryPath,
	}
	ch, err := notify(paths)
	if err != nil {
		t.Fatal("expected no error, got:", err)
	}

	data, err := ioutil.ReadFile(eventPath)
	if err != nil {
		t.Fatal("couldn't read event control file:", err)
	}

	var eventFd, evFd int
	var arg string
	if targ != "" {
		_, err = fmt.Sscanf(string(data), "%d %d %s", &eventFd, &evFd, &arg)
	} else {
		_, err = fmt.Sscanf(string(data), "%d %d", &eventFd, &evFd)
	}
	if err != nil || arg != targ {
		t.Fatalf("invalid control data %q: %s", data, err)
	}

	// re-open the eventfd
	efd, err := syscall.Dup(eventFd)
	if err != nil {
		t.Fatal("unable to reopen eventfd:", err)
	}
	defer syscall.Close(efd)

	if err != nil {
		t.Fatal("unable to dup event fd:", err)
	}

	buf := make([]byte, 8)
	binary.LittleEndian.PutUint64(buf, 1)

	if _, err := syscall.Write(efd, buf); err != nil {
		t.Fatal("unable to write to eventfd:", err)
	}

	select {
	case <-ch:
	case <-time.After(100 * time.Millisecond):
		t.Fatal("no notification on channel after 100ms")
	}

	// simulate what happens when a cgroup is destroyed by cleaning up and then
	// writing to the eventfd.
	if err := os.RemoveAll(memoryPath); err != nil {
		t.Fatal(err)
	}
	if _, err := syscall.Write(efd, buf); err != nil {
		t.Fatal("unable to write to eventfd:", err)
	}

	// give things a moment to shut down
	select {
	case _, ok := <-ch:
		if ok {
			t.Fatal("expected no notification to be triggered")
		}
	case <-time.After(100 * time.Millisecond):
	}

	if _, _, err := syscall.Syscall(syscall.SYS_FCNTL, uintptr(evFd), syscall.F_GETFD, 0); err != syscall.EBADF {
		t.Error("expected event control to be closed")
	}

	if _, _, err := syscall.Syscall(syscall.SYS_FCNTL, uintptr(eventFd), syscall.F_GETFD, 0); err != syscall.EBADF {
		t.Error("expected event fd to be closed")
	}
}

func TestNotifyOnOOM(t *testing.T) {
	f := func(paths map[string]string) (<-chan struct{}, error) {
		return notifyOnOOM(paths)
	}

	testMemoryNotification(t, "memory.oom_control", f, "")
}

func TestNotifyMemoryPressure(t *testing.T) {
	tests := map[PressureLevel]string{
		LowPressure:      "low",
		MediumPressure:   "medium",
		CriticalPressure: "critical",
	}

	for level, arg := range tests {
		f := func(paths map[string]string) (<-chan struct{}, error) {
			return notifyMemoryPressure(paths, level)
		}

		testMemoryNotification(t, "memory.pressure_level", f, arg)
	}
}
