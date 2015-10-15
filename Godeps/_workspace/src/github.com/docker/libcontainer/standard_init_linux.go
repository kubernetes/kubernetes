// +build linux

package libcontainer

import (
	"os"
	"syscall"

	"github.com/docker/libcontainer/apparmor"
	"github.com/docker/libcontainer/configs"
	"github.com/docker/libcontainer/label"
	"github.com/docker/libcontainer/system"
)

type linuxStandardInit struct {
	parentPid int
	config    *initConfig
}

func (l *linuxStandardInit) Init() error {
	// join any namespaces via a path to the namespace fd if provided
	if err := joinExistingNamespaces(l.config.Config.Namespaces); err != nil {
		return err
	}
	var console *linuxConsole
	if l.config.Console != "" {
		console = newConsoleFromPath(l.config.Console)
		if err := console.dupStdio(); err != nil {
			return err
		}
	}
	if _, err := syscall.Setsid(); err != nil {
		return err
	}
	if console != nil {
		if err := system.Setctty(); err != nil {
			return err
		}
	}
	if err := setupNetwork(l.config); err != nil {
		return err
	}
	if err := setupRoute(l.config.Config); err != nil {
		return err
	}
	if err := setupRlimits(l.config.Config); err != nil {
		return err
	}
	label.Init()
	// InitializeMountNamespace() can be executed only for a new mount namespace
	if l.config.Config.Namespaces.Contains(configs.NEWNS) {
		if err := setupRootfs(l.config.Config, console); err != nil {
			return err
		}
	}
	if hostname := l.config.Config.Hostname; hostname != "" {
		if err := syscall.Sethostname([]byte(hostname)); err != nil {
			return err
		}
	}
	if err := apparmor.ApplyProfile(l.config.Config.AppArmorProfile); err != nil {
		return err
	}
	if err := label.SetProcessLabel(l.config.Config.ProcessLabel); err != nil {
		return err
	}

	for key, value := range l.config.Config.SystemProperties {
		if err := writeSystemProperty(key, value); err != nil {
			return err
		}
	}

	for _, path := range l.config.Config.ReadonlyPaths {
		if err := remountReadonly(path); err != nil {
			return err
		}
	}
	for _, path := range l.config.Config.MaskPaths {
		if err := maskFile(path); err != nil {
			return err
		}
	}
	pdeath, err := system.GetParentDeathSignal()
	if err != nil {
		return err
	}
	if err := finalizeNamespace(l.config); err != nil {
		return err
	}
	// finalizeNamespace can change user/group which clears the parent death
	// signal, so we restore it here.
	if err := pdeath.Restore(); err != nil {
		return err
	}
	// compare the parent from the inital start of the init process and make sure that it did not change.
	// if the parent changes that means it died and we were reparened to something else so we should
	// just kill ourself and not cause problems for someone else.
	if syscall.Getppid() != l.parentPid {
		return syscall.Kill(syscall.Getpid(), syscall.SIGKILL)
	}
	return system.Execv(l.config.Args[0], l.config.Args[0:], os.Environ())
}
