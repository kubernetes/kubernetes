// +build linux

package namespaces

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"strings"
	"syscall"

	"github.com/docker/libcontainer"
	"github.com/docker/libcontainer/apparmor"
	"github.com/docker/libcontainer/console"
	"github.com/docker/libcontainer/label"
	"github.com/docker/libcontainer/mount"
	"github.com/docker/libcontainer/netlink"
	"github.com/docker/libcontainer/network"
	"github.com/docker/libcontainer/security/capabilities"
	"github.com/docker/libcontainer/security/restrict"
	"github.com/docker/libcontainer/system"
	"github.com/docker/libcontainer/user"
	"github.com/docker/libcontainer/utils"
)

// TODO(vishh): This is part of the libcontainer API and it does much more than just namespaces related work.
// Move this to libcontainer package.
// Init is the init process that first runs inside a new namespace to setup mounts, users, networking,
// and other options required for the new container.
// The caller of Init function has to ensure that the go runtime is locked to an OS thread
// (using runtime.LockOSThread) else system calls like setns called within Init may not work as intended.
func Init(container *libcontainer.Config, uncleanRootfs, consolePath string, pipe *os.File, args []string) (err error) {
	defer func() {
		// if we have an error during the initialization of the container's init then send it back to the
		// parent process in the form of an initError.
		if err != nil {
			// ensure that any data sent from the parent is consumed so it doesn't
			// receive ECONNRESET when the child writes to the pipe.
			ioutil.ReadAll(pipe)
			if err := json.NewEncoder(pipe).Encode(initError{
				Message: err.Error(),
			}); err != nil {
				panic(err)
			}
		}
		// ensure that this pipe is always closed
		pipe.Close()
	}()

	rootfs, err := utils.ResolveRootfs(uncleanRootfs)
	if err != nil {
		return err
	}

	// clear the current processes env and replace it with the environment
	// defined on the container
	if err := LoadContainerEnvironment(container); err != nil {
		return err
	}

	// We always read this as it is a way to sync with the parent as well
	var networkState *network.NetworkState
	if err := json.NewDecoder(pipe).Decode(&networkState); err != nil {
		return err
	}
	// join any namespaces via a path to the namespace fd if provided
	if err := joinExistingNamespaces(container.Namespaces); err != nil {
		return err
	}
	if consolePath != "" {
		if err := console.OpenAndDup(consolePath); err != nil {
			return err
		}
	}
	if _, err := syscall.Setsid(); err != nil {
		return fmt.Errorf("setsid %s", err)
	}
	if consolePath != "" {
		if err := system.Setctty(); err != nil {
			return fmt.Errorf("setctty %s", err)
		}
	}

	if err := setupNetwork(container, networkState); err != nil {
		return fmt.Errorf("setup networking %s", err)
	}
	if err := setupRoute(container); err != nil {
		return fmt.Errorf("setup route %s", err)
	}

	if err := setupRlimits(container); err != nil {
		return fmt.Errorf("setup rlimits %s", err)
	}

	label.Init()

	if err := mount.InitializeMountNamespace(rootfs,
		consolePath,
		container.RestrictSys,
		(*mount.MountConfig)(container.MountConfig)); err != nil {
		return fmt.Errorf("setup mount namespace %s", err)
	}

	if container.Hostname != "" {
		if err := syscall.Sethostname([]byte(container.Hostname)); err != nil {
			return fmt.Errorf("unable to sethostname %q: %s", container.Hostname, err)
		}
	}

	if err := apparmor.ApplyProfile(container.AppArmorProfile); err != nil {
		return fmt.Errorf("set apparmor profile %s: %s", container.AppArmorProfile, err)
	}

	if err := label.SetProcessLabel(container.ProcessLabel); err != nil {
		return fmt.Errorf("set process label %s", err)
	}

	// TODO: (crosbymichael) make this configurable at the Config level
	if container.RestrictSys {
		if err := restrict.Restrict("proc/sys", "proc/sysrq-trigger", "proc/irq", "proc/bus"); err != nil {
			return err
		}
	}

	pdeathSignal, err := system.GetParentDeathSignal()
	if err != nil {
		return fmt.Errorf("get parent death signal %s", err)
	}

	if err := FinalizeNamespace(container); err != nil {
		return fmt.Errorf("finalize namespace %s", err)
	}

	// FinalizeNamespace can change user/group which clears the parent death
	// signal, so we restore it here.
	if err := RestoreParentDeathSignal(pdeathSignal); err != nil {
		return fmt.Errorf("restore parent death signal %s", err)
	}

	return system.Execv(args[0], args[0:], os.Environ())
}

// RestoreParentDeathSignal sets the parent death signal to old.
func RestoreParentDeathSignal(old int) error {
	if old == 0 {
		return nil
	}

	current, err := system.GetParentDeathSignal()
	if err != nil {
		return fmt.Errorf("get parent death signal %s", err)
	}

	if old == current {
		return nil
	}

	if err := system.ParentDeathSignal(uintptr(old)); err != nil {
		return fmt.Errorf("set parent death signal %s", err)
	}

	// Signal self if parent is already dead. Does nothing if running in a new
	// PID namespace, as Getppid will always return 0.
	if syscall.Getppid() == 1 {
		return syscall.Kill(syscall.Getpid(), syscall.SIGKILL)
	}

	return nil
}

// SetupUser changes the groups, gid, and uid for the user inside the container
func SetupUser(u string) error {
	// Set up defaults.
	defaultExecUser := user.ExecUser{
		Uid:  syscall.Getuid(),
		Gid:  syscall.Getgid(),
		Home: "/",
	}

	passwdPath, err := user.GetPasswdPath()
	if err != nil {
		return err
	}

	groupPath, err := user.GetGroupPath()
	if err != nil {
		return err
	}

	execUser, err := user.GetExecUserPath(u, &defaultExecUser, passwdPath, groupPath)
	if err != nil {
		return fmt.Errorf("get supplementary groups %s", err)
	}

	if err := syscall.Setgroups(execUser.Sgids); err != nil {
		return fmt.Errorf("setgroups %s", err)
	}

	if err := system.Setgid(execUser.Gid); err != nil {
		return fmt.Errorf("setgid %s", err)
	}

	if err := system.Setuid(execUser.Uid); err != nil {
		return fmt.Errorf("setuid %s", err)
	}

	// if we didn't get HOME already, set it based on the user's HOME
	if envHome := os.Getenv("HOME"); envHome == "" {
		if err := os.Setenv("HOME", execUser.Home); err != nil {
			return fmt.Errorf("set HOME %s", err)
		}
	}

	return nil
}

// setupVethNetwork uses the Network config if it is not nil to initialize
// the new veth interface inside the container for use by changing the name to eth0
// setting the MTU and IP address along with the default gateway
func setupNetwork(container *libcontainer.Config, networkState *network.NetworkState) error {
	for _, config := range container.Networks {
		strategy, err := network.GetStrategy(config.Type)
		if err != nil {
			return err
		}

		err1 := strategy.Initialize((*network.Network)(config), networkState)
		if err1 != nil {
			return err1
		}
	}
	return nil
}

func setupRoute(container *libcontainer.Config) error {
	for _, config := range container.Routes {
		if err := netlink.AddRoute(config.Destination, config.Source, config.Gateway, config.InterfaceName); err != nil {
			return err
		}
	}
	return nil
}

func setupRlimits(container *libcontainer.Config) error {
	for _, rlimit := range container.Rlimits {
		l := &syscall.Rlimit{Max: rlimit.Hard, Cur: rlimit.Soft}
		if err := syscall.Setrlimit(rlimit.Type, l); err != nil {
			return fmt.Errorf("error setting rlimit type %v: %v", rlimit.Type, err)
		}
	}
	return nil
}

// FinalizeNamespace drops the caps, sets the correct user
// and working dir, and closes any leaky file descriptors
// before execing the command inside the namespace
func FinalizeNamespace(container *libcontainer.Config) error {
	// Ensure that all non-standard fds we may have accidentally
	// inherited are marked close-on-exec so they stay out of the
	// container
	if err := utils.CloseExecFrom(3); err != nil {
		return fmt.Errorf("close open file descriptors %s", err)
	}

	// drop capabilities in bounding set before changing user
	if err := capabilities.DropBoundingSet(container.Capabilities); err != nil {
		return fmt.Errorf("drop bounding set %s", err)
	}

	// preserve existing capabilities while we change users
	if err := system.SetKeepCaps(); err != nil {
		return fmt.Errorf("set keep caps %s", err)
	}

	if err := SetupUser(container.User); err != nil {
		return fmt.Errorf("setup user %s", err)
	}

	if err := system.ClearKeepCaps(); err != nil {
		return fmt.Errorf("clear keep caps %s", err)
	}

	// drop all other capabilities
	if err := capabilities.DropCapabilities(container.Capabilities); err != nil {
		return fmt.Errorf("drop capabilities %s", err)
	}

	if container.WorkingDir != "" {
		if err := syscall.Chdir(container.WorkingDir); err != nil {
			return fmt.Errorf("chdir to %s %s", container.WorkingDir, err)
		}
	}

	return nil
}

func LoadContainerEnvironment(container *libcontainer.Config) error {
	os.Clearenv()
	for _, pair := range container.Env {
		p := strings.SplitN(pair, "=", 2)
		if len(p) < 2 {
			return fmt.Errorf("invalid environment '%v'", pair)
		}
		if err := os.Setenv(p[0], p[1]); err != nil {
			return err
		}
	}
	return nil
}

// joinExistingNamespaces gets all the namespace paths specified for the container and
// does a setns on the namespace fd so that the current process joins the namespace.
func joinExistingNamespaces(namespaces []libcontainer.Namespace) error {
	for _, ns := range namespaces {
		if ns.Path != "" {
			f, err := os.OpenFile(ns.Path, os.O_RDONLY, 0)
			if err != nil {
				return err
			}
			err = system.Setns(f.Fd(), uintptr(namespaceInfo[ns.Type]))
			f.Close()
			if err != nil {
				return err
			}
		}
	}
	return nil
}
