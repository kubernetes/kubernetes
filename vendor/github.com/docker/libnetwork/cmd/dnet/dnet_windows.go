package main

import (
	"fmt"
	"os"
	"syscall"

	"github.com/docker/docker/pkg/signal"
	"github.com/docker/docker/pkg/system"
	"github.com/sirupsen/logrus"
)

// Copied over from docker/daemon/debugtrap_windows.go
func setupDumpStackTrap() {
	go func() {
		sa := syscall.SecurityAttributes{
			Length: 0,
		}
		ev := "Global\\docker-daemon-" + fmt.Sprint(os.Getpid())
		if h, _ := system.CreateEvent(&sa, false, false, ev); h != 0 {
			logrus.Debugf("Stackdump - waiting signal at %s", ev)
			for {
				syscall.WaitForSingleObject(h, syscall.INFINITE)
				signal.DumpStacks("")
			}
		}
	}()
}
