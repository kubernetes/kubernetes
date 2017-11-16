// +build linux

package libcontainer

import (
	"encoding/json"
	"fmt"
	"io"
	"net"
	"os"
	"strings"
	"syscall" // only for Errno
	"unsafe"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/opencontainers/runc/libcontainer/system"
	"github.com/opencontainers/runc/libcontainer/user"
	"github.com/opencontainers/runc/libcontainer/utils"

	"github.com/sirupsen/logrus"
	"github.com/vishvananda/netlink"
	"golang.org/x/sys/unix"
)

type initType string

const (
	initSetns    initType = "setns"
	initStandard initType = "standard"
)

type pid struct {
	Pid           int `json:"pid"`
	PidFirstChild int `json:"pid_first"`
}

// network is an internal struct used to setup container networks.
type network struct {
	configs.Network

	// TempVethPeerName is a unique temporary veth peer name that was placed into
	// the container's namespace.
	TempVethPeerName string `json:"temp_veth_peer_name"`
}

// initConfig is used for transferring parameters from Exec() to Init()
type initConfig struct {
	Args             []string              `json:"args"`
	Env              []string              `json:"env"`
	Cwd              string                `json:"cwd"`
	Capabilities     *configs.Capabilities `json:"capabilities"`
	ProcessLabel     string                `json:"process_label"`
	AppArmorProfile  string                `json:"apparmor_profile"`
	NoNewPrivileges  bool                  `json:"no_new_privileges"`
	User             string                `json:"user"`
	AdditionalGroups []string              `json:"additional_groups"`
	Config           *configs.Config       `json:"config"`
	Networks         []*network            `json:"network"`
	PassedFilesCount int                   `json:"passed_files_count"`
	ContainerId      string                `json:"containerid"`
	Rlimits          []configs.Rlimit      `json:"rlimits"`
	CreateConsole    bool                  `json:"create_console"`
	Rootless         bool                  `json:"rootless"`
}

type initer interface {
	Init() error
}

func newContainerInit(t initType, pipe *os.File, consoleSocket *os.File, fifoFd int) (initer, error) {
	var config *initConfig
	if err := json.NewDecoder(pipe).Decode(&config); err != nil {
		return nil, err
	}
	if err := populateProcessEnvironment(config.Env); err != nil {
		return nil, err
	}
	switch t {
	case initSetns:
		return &linuxSetnsInit{
			pipe:          pipe,
			consoleSocket: consoleSocket,
			config:        config,
		}, nil
	case initStandard:
		return &linuxStandardInit{
			pipe:          pipe,
			consoleSocket: consoleSocket,
			parentPid:     unix.Getppid(),
			config:        config,
			fifoFd:        fifoFd,
		}, nil
	}
	return nil, fmt.Errorf("unknown init type %q", t)
}

// populateProcessEnvironment loads the provided environment variables into the
// current processes's environment.
func populateProcessEnvironment(env []string) error {
	for _, pair := range env {
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

// finalizeNamespace drops the caps, sets the correct user
// and working dir, and closes any leaked file descriptors
// before executing the command inside the namespace
func finalizeNamespace(config *initConfig) error {
	// Ensure that all unwanted fds we may have accidentally
	// inherited are marked close-on-exec so they stay out of the
	// container
	if err := utils.CloseExecFrom(config.PassedFilesCount + 3); err != nil {
		return err
	}

	capabilities := &configs.Capabilities{}
	if config.Capabilities != nil {
		capabilities = config.Capabilities
	} else if config.Config.Capabilities != nil {
		capabilities = config.Config.Capabilities
	}
	w, err := newContainerCapList(capabilities)
	if err != nil {
		return err
	}
	// drop capabilities in bounding set before changing user
	if err := w.ApplyBoundingSet(); err != nil {
		return err
	}
	// preserve existing capabilities while we change users
	if err := system.SetKeepCaps(); err != nil {
		return err
	}
	if err := setupUser(config); err != nil {
		return err
	}
	if err := system.ClearKeepCaps(); err != nil {
		return err
	}
	if err := w.ApplyCaps(); err != nil {
		return err
	}
	if config.Cwd != "" {
		if err := unix.Chdir(config.Cwd); err != nil {
			return fmt.Errorf("chdir to cwd (%q) set in config.json failed: %v", config.Cwd, err)
		}
	}
	return nil
}

// setupConsole sets up the console from inside the container, and sends the
// master pty fd to the config.Pipe (using cmsg). This is done to ensure that
// consoles are scoped to a container properly (see runc#814 and the many
// issues related to that). This has to be run *after* we've pivoted to the new
// rootfs (and the users' configuration is entirely set up).
func setupConsole(socket *os.File, config *initConfig, mount bool) error {
	defer socket.Close()
	// At this point, /dev/ptmx points to something that we would expect. We
	// used to change the owner of the slave path, but since the /dev/pts mount
	// can have gid=X set (at the users' option). So touching the owner of the
	// slave PTY is not necessary, as the kernel will handle that for us. Note
	// however, that setupUser (specifically fixStdioPermissions) *will* change
	// the UID owner of the console to be the user the process will run as (so
	// they can actually control their console).
	console, err := newConsole()
	if err != nil {
		return err
	}
	// After we return from here, we don't need the console anymore.
	defer console.Close()

	linuxConsole, ok := console.(*linuxConsole)
	if !ok {
		return fmt.Errorf("failed to cast console to *linuxConsole")
	}
	// Mount the console inside our rootfs.
	if mount {
		if err := linuxConsole.mount(); err != nil {
			return err
		}
	}
	// While we can access console.master, using the API is a good idea.
	if err := utils.SendFd(socket, linuxConsole.File()); err != nil {
		return err
	}
	// Now, dup over all the things.
	return linuxConsole.dupStdio()
}

// syncParentReady sends to the given pipe a JSON payload which indicates that
// the init is ready to Exec the child process. It then waits for the parent to
// indicate that it is cleared to Exec.
func syncParentReady(pipe io.ReadWriter) error {
	// Tell parent.
	if err := writeSync(pipe, procReady); err != nil {
		return err
	}

	// Wait for parent to give the all-clear.
	if err := readSync(pipe, procRun); err != nil {
		return err
	}

	return nil
}

// syncParentHooks sends to the given pipe a JSON payload which indicates that
// the parent should execute pre-start hooks. It then waits for the parent to
// indicate that it is cleared to resume.
func syncParentHooks(pipe io.ReadWriter) error {
	// Tell parent.
	if err := writeSync(pipe, procHooks); err != nil {
		return err
	}

	// Wait for parent to give the all-clear.
	if err := readSync(pipe, procResume); err != nil {
		return err
	}

	return nil
}

// setupUser changes the groups, gid, and uid for the user inside the container
func setupUser(config *initConfig) error {
	// Set up defaults.
	defaultExecUser := user.ExecUser{
		Uid:  0,
		Gid:  0,
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

	execUser, err := user.GetExecUserPath(config.User, &defaultExecUser, passwdPath, groupPath)
	if err != nil {
		return err
	}

	var addGroups []int
	if len(config.AdditionalGroups) > 0 {
		addGroups, err = user.GetAdditionalGroupsPath(config.AdditionalGroups, groupPath)
		if err != nil {
			return err
		}
	}

	if config.Rootless {
		if execUser.Uid != 0 {
			return fmt.Errorf("cannot run as a non-root user in a rootless container")
		}

		if execUser.Gid != 0 {
			return fmt.Errorf("cannot run as a non-root group in a rootless container")
		}

		// We cannot set any additional groups in a rootless container and thus we
		// bail if the user asked us to do so. TODO: We currently can't do this
		// earlier, but if libcontainer.Process.User was typesafe this might work.
		if len(addGroups) > 0 {
			return fmt.Errorf("cannot set any additional groups in a rootless container")
		}
	}

	// before we change to the container's user make sure that the processes STDIO
	// is correctly owned by the user that we are switching to.
	if err := fixStdioPermissions(config, execUser); err != nil {
		return err
	}

	// This isn't allowed in an unprivileged user namespace since Linux 3.19.
	// There's nothing we can do about /etc/group entries, so we silently
	// ignore setting groups here (since the user didn't explicitly ask us to
	// set the group).
	if !config.Rootless {
		suppGroups := append(execUser.Sgids, addGroups...)
		if err := unix.Setgroups(suppGroups); err != nil {
			return err
		}
	}

	if err := system.Setgid(execUser.Gid); err != nil {
		return err
	}

	if err := system.Setuid(execUser.Uid); err != nil {
		return err
	}

	// if we didn't get HOME already, set it based on the user's HOME
	if envHome := os.Getenv("HOME"); envHome == "" {
		if err := os.Setenv("HOME", execUser.Home); err != nil {
			return err
		}
	}
	return nil
}

// fixStdioPermissions fixes the permissions of PID 1's STDIO within the container to the specified user.
// The ownership needs to match because it is created outside of the container and needs to be
// localized.
func fixStdioPermissions(config *initConfig, u *user.ExecUser) error {
	var null unix.Stat_t
	if err := unix.Stat("/dev/null", &null); err != nil {
		return err
	}
	for _, fd := range []uintptr{
		os.Stdin.Fd(),
		os.Stderr.Fd(),
		os.Stdout.Fd(),
	} {
		var s unix.Stat_t
		if err := unix.Fstat(int(fd), &s); err != nil {
			return err
		}

		// Skip chown of /dev/null if it was used as one of the STDIO fds.
		if s.Rdev == null.Rdev {
			continue
		}

		// Skip chown if s.Gid is actually an unmapped gid in the host. While
		// this is a bit dodgy if it just so happens that the console _is_
		// owned by overflow_gid, there's no way for us to disambiguate this as
		// a userspace program.
		if _, err := config.Config.HostGID(int(s.Gid)); err != nil {
			continue
		}

		// We only change the uid owner (as it is possible for the mount to
		// prefer a different gid, and there's no reason for us to change it).
		// The reason why we don't just leave the default uid=X mount setup is
		// that users expect to be able to actually use their console. Without
		// this code, you couldn't effectively run as a non-root user inside a
		// container and also have a console set up.
		if err := unix.Fchown(int(fd), u.Uid, int(s.Gid)); err != nil {
			return err
		}
	}
	return nil
}

// setupNetwork sets up and initializes any network interface inside the container.
func setupNetwork(config *initConfig) error {
	for _, config := range config.Networks {
		strategy, err := getStrategy(config.Type)
		if err != nil {
			return err
		}
		if err := strategy.initialize(config); err != nil {
			return err
		}
	}
	return nil
}

func setupRoute(config *configs.Config) error {
	for _, config := range config.Routes {
		_, dst, err := net.ParseCIDR(config.Destination)
		if err != nil {
			return err
		}
		src := net.ParseIP(config.Source)
		if src == nil {
			return fmt.Errorf("Invalid source for route: %s", config.Source)
		}
		gw := net.ParseIP(config.Gateway)
		if gw == nil {
			return fmt.Errorf("Invalid gateway for route: %s", config.Gateway)
		}
		l, err := netlink.LinkByName(config.InterfaceName)
		if err != nil {
			return err
		}
		route := &netlink.Route{
			Scope:     netlink.SCOPE_UNIVERSE,
			Dst:       dst,
			Src:       src,
			Gw:        gw,
			LinkIndex: l.Attrs().Index,
		}
		if err := netlink.RouteAdd(route); err != nil {
			return err
		}
	}
	return nil
}

func setupRlimits(limits []configs.Rlimit, pid int) error {
	for _, rlimit := range limits {
		if err := system.Prlimit(pid, rlimit.Type, unix.Rlimit{Max: rlimit.Hard, Cur: rlimit.Soft}); err != nil {
			return fmt.Errorf("error setting rlimit type %v: %v", rlimit.Type, err)
		}
	}
	return nil
}

const _P_PID = 1

type siginfo struct {
	si_signo int32
	si_errno int32
	si_code  int32
	// below here is a union; si_pid is the only field we use
	si_pid int32
	// Pad to 128 bytes as detailed in blockUntilWaitable
	pad [96]byte
}

// isWaitable returns true if the process has exited false otherwise.
// Its based off blockUntilWaitable in src/os/wait_waitid.go
func isWaitable(pid int) (bool, error) {
	si := &siginfo{}
	_, _, e := unix.Syscall6(unix.SYS_WAITID, _P_PID, uintptr(pid), uintptr(unsafe.Pointer(si)), unix.WEXITED|unix.WNOWAIT|unix.WNOHANG, 0, 0)
	if e != 0 {
		return false, os.NewSyscallError("waitid", e)
	}

	return si.si_pid != 0, nil
}

// isNoChildren returns true if err represents a unix.ECHILD (formerly syscall.ECHILD) false otherwise
func isNoChildren(err error) bool {
	switch err := err.(type) {
	case syscall.Errno:
		if err == unix.ECHILD {
			return true
		}
	case *os.SyscallError:
		if err.Err == unix.ECHILD {
			return true
		}
	}
	return false
}

// signalAllProcesses freezes then iterates over all the processes inside the
// manager's cgroups sending the signal s to them.
// If s is SIGKILL then it will wait for each process to exit.
// For all other signals it will check if the process is ready to report its
// exit status and only if it is will a wait be performed.
func signalAllProcesses(m cgroups.Manager, s os.Signal) error {
	var procs []*os.Process
	if err := m.Freeze(configs.Frozen); err != nil {
		logrus.Warn(err)
	}
	pids, err := m.GetAllPids()
	if err != nil {
		m.Freeze(configs.Thawed)
		return err
	}
	for _, pid := range pids {
		p, err := os.FindProcess(pid)
		if err != nil {
			logrus.Warn(err)
			continue
		}
		procs = append(procs, p)
		if err := p.Signal(s); err != nil {
			logrus.Warn(err)
		}
	}
	if err := m.Freeze(configs.Thawed); err != nil {
		logrus.Warn(err)
	}

	for _, p := range procs {
		if s != unix.SIGKILL {
			if ok, err := isWaitable(p.Pid); err != nil {
				if !isNoChildren(err) {
					logrus.Warn("signalAllProcesses: ", p.Pid, err)
				}
				continue
			} else if !ok {
				// Not ready to report so don't wait
				continue
			}
		}

		if _, err := p.Wait(); err != nil {
			if !isNoChildren(err) {
				logrus.Warn("wait: ", err)
			}
		}
	}
	return nil
}
