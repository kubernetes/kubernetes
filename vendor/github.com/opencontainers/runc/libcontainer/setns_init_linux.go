// +build linux

package libcontainer

import (
	"os"
	"runtime"

	"github.com/opencontainers/runc/libcontainer/apparmor"
	"github.com/opencontainers/runc/libcontainer/keys"
	"github.com/opencontainers/runc/libcontainer/seccomp"
	"github.com/opencontainers/runc/libcontainer/system"
	"github.com/opencontainers/selinux/go-selinux"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
)

// linuxSetnsInit performs the container's initialization for running a new process
// inside an existing container.
type linuxSetnsInit struct {
	pipe          *os.File
	consoleSocket *os.File
	config        *initConfig
	logFd         int
}

func (l *linuxSetnsInit) getSessionRingName() string {
	return "_ses." + l.config.ContainerId
}

func (l *linuxSetnsInit) Init() error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if !l.config.Config.NoNewKeyring {
		if err := selinux.SetKeyLabel(l.config.ProcessLabel); err != nil {
			return err
		}
		defer selinux.SetKeyLabel("") //nolint: errcheck
		// Do not inherit the parent's session keyring.
		if _, err := keys.JoinSessionKeyring(l.getSessionRingName()); err != nil {
			// Same justification as in standart_init_linux.go as to why we
			// don't bail on ENOSYS.
			//
			// TODO(cyphar): And we should have logging here too.
			if errors.Cause(err) != unix.ENOSYS {
				return errors.Wrap(err, "join session keyring")
			}
		}
	}
	if l.config.CreateConsole {
		if err := setupConsole(l.consoleSocket, l.config, false); err != nil {
			return err
		}
		if err := system.Setctty(); err != nil {
			return err
		}
	}
	if l.config.NoNewPrivileges {
		if err := unix.Prctl(unix.PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0); err != nil {
			return err
		}
	}
	if err := selinux.SetExecLabel(l.config.ProcessLabel); err != nil {
		return err
	}
	defer selinux.SetExecLabel("") //nolint: errcheck
	// Without NoNewPrivileges seccomp is a privileged operation, so we need to
	// do this before dropping capabilities; otherwise do it as late as possible
	// just before execve so as few syscalls take place after it as possible.
	if l.config.Config.Seccomp != nil && !l.config.NoNewPrivileges {
		if err := seccomp.InitSeccomp(l.config.Config.Seccomp); err != nil {
			return err
		}
	}
	if err := finalizeNamespace(l.config); err != nil {
		return err
	}
	if err := apparmor.ApplyProfile(l.config.AppArmorProfile); err != nil {
		return err
	}
	// Set seccomp as close to execve as possible, so as few syscalls take
	// place afterward (reducing the amount of syscalls that users need to
	// enable in their seccomp profiles).
	if l.config.Config.Seccomp != nil && l.config.NoNewPrivileges {
		if err := seccomp.InitSeccomp(l.config.Config.Seccomp); err != nil {
			return newSystemErrorWithCause(err, "init seccomp")
		}
	}
	logrus.Debugf("setns_init: about to exec")
	// Close the log pipe fd so the parent's ForwardLogs can exit.
	if err := unix.Close(l.logFd); err != nil {
		return newSystemErrorWithCause(err, "closing log pipe fd")
	}

	return system.Execv(l.config.Args[0], l.config.Args[0:], os.Environ())
}
