package libcontainer

import (
	"errors"
	"fmt"
	"os"
	"os/exec"

	"github.com/opencontainers/selinux/go-selinux"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"

	"github.com/opencontainers/runc/libcontainer/apparmor"
	"github.com/opencontainers/runc/libcontainer/keys"
	"github.com/opencontainers/runc/libcontainer/seccomp"
	"github.com/opencontainers/runc/libcontainer/system"
	"github.com/opencontainers/runc/libcontainer/utils"
)

// linuxSetnsInit performs the container's initialization for running a new process
// inside an existing container.
type linuxSetnsInit struct {
	pipe          *syncSocket
	consoleSocket *os.File
	pidfdSocket   *os.File
	config        *initConfig
	logPipe       *os.File
	dmzExe        *os.File
}

func (l *linuxSetnsInit) getSessionRingName() string {
	return "_ses." + l.config.ContainerID
}

func (l *linuxSetnsInit) Init() error {
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
			if !errors.Is(err, unix.ENOSYS) {
				return fmt.Errorf("unable to join session keyring: %w", err)
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
	if l.pidfdSocket != nil {
		if err := setupPidfd(l.pidfdSocket, "setns"); err != nil {
			return fmt.Errorf("failed to setup pidfd: %w", err)
		}
	}
	if l.config.NoNewPrivileges {
		if err := unix.Prctl(unix.PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0); err != nil {
			return err
		}
	}
	if l.config.Config.Umask != nil {
		unix.Umask(int(*l.config.Config.Umask))
	}

	if l.config.Config.Scheduler != nil {
		if err := setupScheduler(l.config.Config); err != nil {
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
		seccompFd, err := seccomp.InitSeccomp(l.config.Config.Seccomp)
		if err != nil {
			return err
		}
		if err := syncParentSeccomp(l.pipe, seccompFd); err != nil {
			return err
		}
	}
	if err := finalizeNamespace(l.config); err != nil {
		return err
	}
	if err := apparmor.ApplyProfile(l.config.AppArmorProfile); err != nil {
		return err
	}
	if l.config.Config.Personality != nil {
		if err := setupPersonality(l.config.Config); err != nil {
			return err
		}
	}
	// Check for the arg early to make sure it exists.
	name, err := exec.LookPath(l.config.Args[0])
	if err != nil {
		return err
	}
	// exec.LookPath in Go < 1.20 might return no error for an executable
	// residing on a file system mounted with noexec flag, so perform this
	// extra check now while we can still return a proper error.
	// TODO: remove this once go < 1.20 is not supported.
	if err := eaccess(name); err != nil {
		return &os.PathError{Op: "eaccess", Path: name, Err: err}
	}
	// Set seccomp as close to execve as possible, so as few syscalls take
	// place afterward (reducing the amount of syscalls that users need to
	// enable in their seccomp profiles).
	if l.config.Config.Seccomp != nil && l.config.NoNewPrivileges {
		seccompFd, err := seccomp.InitSeccomp(l.config.Config.Seccomp)
		if err != nil {
			return fmt.Errorf("unable to init seccomp: %w", err)
		}
		if err := syncParentSeccomp(l.pipe, seccompFd); err != nil {
			return err
		}
	}

	// Close the pipe to signal that we have completed our init.
	// Please keep this because we don't want to get a pipe write error if
	// there is an error from `execve` after all fds closed.
	_ = l.pipe.Close()

	// Close the log pipe fd so the parent's ForwardLogs can exit.
	logrus.Debugf("setns_init: about to exec")
	if err := l.logPipe.Close(); err != nil {
		return fmt.Errorf("close log pipe: %w", err)
	}

	if l.dmzExe != nil {
		l.config.Args[0] = name
		return system.Fexecve(l.dmzExe.Fd(), l.config.Args, os.Environ())
	}
	// Close all file descriptors we are not passing to the container. This is
	// necessary because the execve target could use internal runc fds as the
	// execve path, potentially giving access to binary files from the host
	// (which can then be opened by container processes, leading to container
	// escapes). Note that because this operation will close any open file
	// descriptors that are referenced by (*os.File) handles from underneath
	// the Go runtime, we must not do any file operations after this point
	// (otherwise the (*os.File) finaliser could close the wrong file). See
	// CVE-2024-21626 for more information as to why this protection is
	// necessary.
	//
	// This is not needed for runc-dmz, because the extra execve(2) step means
	// that all O_CLOEXEC file descriptors have already been closed and thus
	// the second execve(2) from runc-dmz cannot access internal file
	// descriptors from runc.
	if err := utils.UnsafeCloseFrom(l.config.PassedFilesCount + 3); err != nil {
		return err
	}
	return system.Exec(name, l.config.Args, os.Environ())
}
