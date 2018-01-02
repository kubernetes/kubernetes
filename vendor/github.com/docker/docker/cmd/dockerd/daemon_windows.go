package main

import (
	"fmt"
	"net"
	"os"
	"path/filepath"

	"github.com/docker/docker/libcontainerd"
	"github.com/docker/docker/pkg/system"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/windows"
)

var defaultDaemonConfigFile = ""

// currentUserIsOwner checks whether the current user is the owner of the given
// file.
func currentUserIsOwner(f string) bool {
	return false
}

// setDefaultUmask doesn't do anything on windows
func setDefaultUmask() error {
	return nil
}

func getDaemonConfDir(root string) string {
	return filepath.Join(root, `\config`)
}

// preNotifySystem sends a message to the host when the API is active, but before the daemon is
func preNotifySystem() {
	// start the service now to prevent timeouts waiting for daemon to start
	// but still (eventually) complete all requests that are sent after this
	if service != nil {
		err := service.started()
		if err != nil {
			logrus.Fatal(err)
		}
	}
}

// notifySystem sends a message to the host when the server is ready to be used
func notifySystem() {
}

// notifyShutdown is called after the daemon shuts down but before the process exits.
func notifyShutdown(err error) {
	if service != nil {
		if err != nil {
			logrus.Fatal(err)
		}
		service.stopped(err)
	}
}

// setupConfigReloadTrap configures a Win32 event to reload the configuration.
func (cli *DaemonCli) setupConfigReloadTrap() {
	go func() {
		sa := windows.SecurityAttributes{
			Length: 0,
		}
		ev := "Global\\docker-daemon-config-" + fmt.Sprint(os.Getpid())
		if h, _ := system.CreateEvent(&sa, false, false, ev); h != 0 {
			logrus.Debugf("Config reload - waiting signal at %s", ev)
			for {
				windows.WaitForSingleObject(h, windows.INFINITE)
				cli.reloadConfig()
			}
		}
	}()
}

func (cli *DaemonCli) getPlatformRemoteOptions() []libcontainerd.RemoteOption {
	return nil
}

// getLibcontainerdRoot gets the root directory for libcontainerd to store its
// state. The Windows libcontainerd implementation does not need to write a spec
// or state to disk, so this is a no-op.
func (cli *DaemonCli) getLibcontainerdRoot() string {
	return ""
}

// getSwarmRunRoot gets the root directory for swarm to store runtime state
// For example, the control socket
func (cli *DaemonCli) getSwarmRunRoot() string {
	return ""
}

func allocateDaemonPort(addr string) error {
	return nil
}

func wrapListeners(proto string, ls []net.Listener) []net.Listener {
	return ls
}
