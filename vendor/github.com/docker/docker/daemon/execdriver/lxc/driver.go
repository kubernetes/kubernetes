// +build linux

package lxc

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/Sirupsen/logrus"
	"github.com/docker/docker/daemon/execdriver"
	"github.com/docker/docker/pkg/stringutils"
	sysinfo "github.com/docker/docker/pkg/system"
	"github.com/docker/docker/pkg/term"
	"github.com/docker/docker/pkg/version"
	"github.com/kr/pty"
	"github.com/opencontainers/runc/libcontainer"
	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/opencontainers/runc/libcontainer/system"
	"github.com/opencontainers/runc/libcontainer/user"
	"github.com/vishvananda/netns"
)

const DriverName = "lxc"

var ErrExec = errors.New("Unsupported: Exec is not supported by the lxc driver")

type driver struct {
	root             string // root path for the driver to use
	libPath          string
	initPath         string
	apparmor         bool
	sharedRoot       bool
	activeContainers map[string]*activeContainer
	machineMemory    int64
	sync.Mutex
}

type activeContainer struct {
	container *configs.Config
	cmd       *exec.Cmd
}

func NewDriver(root, libPath, initPath string, apparmor bool) (*driver, error) {
	if err := os.MkdirAll(root, 0700); err != nil {
		return nil, err
	}
	// setup unconfined symlink
	if err := linkLxcStart(root); err != nil {
		return nil, err
	}
	meminfo, err := sysinfo.ReadMemInfo()
	if err != nil {
		return nil, err
	}
	return &driver{
		apparmor:         apparmor,
		root:             root,
		libPath:          libPath,
		initPath:         initPath,
		sharedRoot:       rootIsShared(),
		activeContainers: make(map[string]*activeContainer),
		machineMemory:    meminfo.MemTotal,
	}, nil
}

func (d *driver) Name() string {
	version := d.version()
	return fmt.Sprintf("%s-%s", DriverName, version)
}

func setupNetNs(nsPath string) (*os.Process, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	origns, err := netns.Get()
	if err != nil {
		return nil, err
	}
	defer origns.Close()

	f, err := os.OpenFile(nsPath, os.O_RDONLY, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to get network namespace %q: %v", nsPath, err)
	}
	defer f.Close()

	nsFD := f.Fd()
	if err := netns.Set(netns.NsHandle(nsFD)); err != nil {
		return nil, fmt.Errorf("failed to set network namespace %q: %v", nsPath, err)
	}
	defer netns.Set(origns)

	cmd := exec.Command("/bin/sh", "-c", "while true; do sleep 1; done")
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start netns process: %v", err)
	}

	return cmd.Process, nil
}

func killNetNsProc(proc *os.Process) {
	proc.Kill()
	proc.Wait()
}

func (d *driver) Run(c *execdriver.Command, pipes *execdriver.Pipes, startCallback execdriver.StartCallback) (execdriver.ExitStatus, error) {
	var (
		term     execdriver.Terminal
		err      error
		dataPath = d.containerDir(c.ID)
	)

	if c.Network == nil || (c.Network.NamespacePath == "" && c.Network.ContainerID == "") {
		return execdriver.ExitStatus{ExitCode: -1}, fmt.Errorf("empty namespace path for non-container network")
	}

	container, err := d.createContainer(c)
	if err != nil {
		return execdriver.ExitStatus{ExitCode: -1}, err
	}

	if c.ProcessConfig.Tty {
		term, err = NewTtyConsole(&c.ProcessConfig, pipes)
	} else {
		term, err = execdriver.NewStdConsole(&c.ProcessConfig, pipes)
	}
	if err != nil {
		return execdriver.ExitStatus{ExitCode: -1}, err
	}
	c.ProcessConfig.Terminal = term

	d.Lock()
	d.activeContainers[c.ID] = &activeContainer{
		container: container,
		cmd:       &c.ProcessConfig.Cmd,
	}
	d.Unlock()

	c.Mounts = append(c.Mounts, execdriver.Mount{
		Source:      d.initPath,
		Destination: c.InitPath,
		Writable:    false,
		Private:     true,
	})

	if err := d.generateEnvConfig(c); err != nil {
		return execdriver.ExitStatus{ExitCode: -1}, err
	}
	configPath, err := d.generateLXCConfig(c)
	if err != nil {
		return execdriver.ExitStatus{ExitCode: -1}, err
	}
	params := []string{
		"lxc-start",
		"-n", c.ID,
		"-f", configPath,
		"-q",
	}

	// From lxc>=1.1 the default behavior is to daemonize containers after start
	lxcVersion := version.Version(d.version())
	if lxcVersion.GreaterThanOrEqualTo(version.Version("1.1")) {
		params = append(params, "-F")
	}

	proc := &os.Process{}
	if c.Network.ContainerID != "" {
		params = append(params,
			"--share-net", c.Network.ContainerID,
		)
	} else {
		proc, err = setupNetNs(c.Network.NamespacePath)
		if err != nil {
			return execdriver.ExitStatus{ExitCode: -1}, err
		}

		pidStr := fmt.Sprintf("%d", proc.Pid)
		params = append(params,
			"--share-net", pidStr)
	}
	if c.Ipc != nil {
		if c.Ipc.ContainerID != "" {
			params = append(params,
				"--share-ipc", c.Ipc.ContainerID,
			)
		} else if c.Ipc.HostIpc {
			params = append(params,
				"--share-ipc", "1",
			)
		}
	}

	params = append(params,
		"--",
		c.InitPath,
	)

	if c.ProcessConfig.User != "" {
		params = append(params, "-u", c.ProcessConfig.User)
	}

	if c.ProcessConfig.Privileged {
		if d.apparmor {
			params[0] = path.Join(d.root, "lxc-start-unconfined")

		}
		params = append(params, "-privileged")
	}

	if c.WorkingDir != "" {
		params = append(params, "-w", c.WorkingDir)
	}

	params = append(params, "--", c.ProcessConfig.Entrypoint)
	params = append(params, c.ProcessConfig.Arguments...)

	if d.sharedRoot {
		// lxc-start really needs / to be non-shared, or all kinds of stuff break
		// when lxc-start unmount things and those unmounts propagate to the main
		// mount namespace.
		// What we really want is to clone into a new namespace and then
		// mount / MS_REC|MS_SLAVE, but since we can't really clone or fork
		// without exec in go we have to do this horrible shell hack...
		shellString :=
			"mount --make-rslave /; exec " +
				stringutils.ShellQuoteArguments(params)

		params = []string{
			"unshare", "-m", "--", "/bin/sh", "-c", shellString,
		}
	}
	logrus.Debugf("lxc params %s", params)
	var (
		name = params[0]
		arg  = params[1:]
	)
	aname, err := exec.LookPath(name)
	if err != nil {
		aname = name
	}
	c.ProcessConfig.Path = aname
	c.ProcessConfig.Args = append([]string{name}, arg...)

	if err := createDeviceNodes(c.Rootfs, c.AutoCreatedDevices); err != nil {
		killNetNsProc(proc)
		return execdriver.ExitStatus{ExitCode: -1}, err
	}

	if err := c.ProcessConfig.Start(); err != nil {
		killNetNsProc(proc)
		return execdriver.ExitStatus{ExitCode: -1}, err
	}

	var (
		waitErr  error
		waitLock = make(chan struct{})
	)

	go func() {
		if err := c.ProcessConfig.Wait(); err != nil {
			if _, ok := err.(*exec.ExitError); !ok { // Do not propagate the error if it's simply a status code != 0
				waitErr = err
			}
		}
		close(waitLock)
	}()

	terminate := func(terr error) (execdriver.ExitStatus, error) {
		if c.ProcessConfig.Process != nil {
			c.ProcessConfig.Process.Kill()
			c.ProcessConfig.Wait()
		}
		return execdriver.ExitStatus{ExitCode: -1}, terr
	}
	// Poll lxc for RUNNING status
	pid, err := d.waitForStart(c, waitLock)
	if err != nil {
		killNetNsProc(proc)
		return terminate(err)
	}
	killNetNsProc(proc)

	cgroupPaths, err := cgroupPaths(c.ID)
	if err != nil {
		return terminate(err)
	}

	state := &libcontainer.State{
		InitProcessPid: pid,
		CgroupPaths:    cgroupPaths,
	}

	f, err := os.Create(filepath.Join(dataPath, "state.json"))
	if err != nil {
		return terminate(err)
	}
	defer f.Close()

	if err := json.NewEncoder(f).Encode(state); err != nil {
		return terminate(err)
	}

	c.ContainerPid = pid

	if startCallback != nil {
		logrus.Debugf("Invoking startCallback")
		startCallback(&c.ProcessConfig, pid)
	}

	oomKill := false
	oomKillNotification, err := notifyOnOOM(cgroupPaths)

	<-waitLock
	exitCode := getExitCode(c)

	if err == nil {
		_, oomKill = <-oomKillNotification
		logrus.Debugf("oomKill error: %v, waitErr: %v", oomKill, waitErr)
	} else {
		logrus.Warnf("Your kernel does not support OOM notifications: %s", err)
	}

	// check oom error
	if oomKill {
		exitCode = 137
	}

	return execdriver.ExitStatus{ExitCode: exitCode, OOMKilled: oomKill}, waitErr
}

// copy from libcontainer
func notifyOnOOM(paths map[string]string) (<-chan struct{}, error) {
	dir := paths["memory"]
	if dir == "" {
		return nil, fmt.Errorf("There is no path for %q in state", "memory")
	}
	oomControl, err := os.Open(filepath.Join(dir, "memory.oom_control"))
	if err != nil {
		return nil, err
	}
	fd, _, syserr := syscall.RawSyscall(syscall.SYS_EVENTFD2, 0, syscall.FD_CLOEXEC, 0)
	if syserr != 0 {
		oomControl.Close()
		return nil, syserr
	}

	eventfd := os.NewFile(fd, "eventfd")

	eventControlPath := filepath.Join(dir, "cgroup.event_control")
	data := fmt.Sprintf("%d %d", eventfd.Fd(), oomControl.Fd())
	if err := ioutil.WriteFile(eventControlPath, []byte(data), 0700); err != nil {
		eventfd.Close()
		oomControl.Close()
		return nil, err
	}
	ch := make(chan struct{})
	go func() {
		defer func() {
			close(ch)
			eventfd.Close()
			oomControl.Close()
		}()
		buf := make([]byte, 8)
		for {
			if _, err := eventfd.Read(buf); err != nil {
				return
			}
			// When a cgroup is destroyed, an event is sent to eventfd.
			// So if the control path is gone, return instead of notifying.
			if _, err := os.Lstat(eventControlPath); os.IsNotExist(err) {
				return
			}
			ch <- struct{}{}
		}
	}()
	return ch, nil
}

// createContainer populates and configures the container type with the
// data provided by the execdriver.Command
func (d *driver) createContainer(c *execdriver.Command) (*configs.Config, error) {
	container := execdriver.InitContainer(c)
	if err := execdriver.SetupCgroups(container, c); err != nil {
		return nil, err
	}
	return container, nil
}

// Return an map of susbystem -> container cgroup
func cgroupPaths(containerId string) (map[string]string, error) {
	subsystems, err := cgroups.GetAllSubsystems()
	if err != nil {
		return nil, err
	}
	logrus.Debugf("subsystems: %s", subsystems)
	paths := make(map[string]string)
	for _, subsystem := range subsystems {
		cgroupRoot, cgroupDir, err := findCgroupRootAndDir(subsystem)
		logrus.Debugf("cgroup path %s %s", cgroupRoot, cgroupDir)
		if err != nil {
			//unsupported subystem
			continue
		}
		path := filepath.Join(cgroupRoot, cgroupDir, "lxc", containerId)
		paths[subsystem] = path
	}

	return paths, nil
}

// this is copy from old libcontainer nodes.go
func createDeviceNodes(rootfs string, nodesToCreate []*configs.Device) error {
	oldMask := syscall.Umask(0000)
	defer syscall.Umask(oldMask)

	for _, node := range nodesToCreate {
		if err := createDeviceNode(rootfs, node); err != nil {
			return err
		}
	}
	return nil
}

// Creates the device node in the rootfs of the container.
func createDeviceNode(rootfs string, node *configs.Device) error {
	var (
		dest   = filepath.Join(rootfs, node.Path)
		parent = filepath.Dir(dest)
	)

	if err := os.MkdirAll(parent, 0755); err != nil {
		return err
	}

	fileMode := node.FileMode
	switch node.Type {
	case 'c':
		fileMode |= syscall.S_IFCHR
	case 'b':
		fileMode |= syscall.S_IFBLK
	default:
		return fmt.Errorf("%c is not a valid device type for device %s", node.Type, node.Path)
	}

	if err := syscall.Mknod(dest, uint32(fileMode), node.Mkdev()); err != nil && !os.IsExist(err) {
		return fmt.Errorf("mknod %s %s", node.Path, err)
	}

	if err := syscall.Chown(dest, int(node.Uid), int(node.Gid)); err != nil {
		return fmt.Errorf("chown %s to %d:%d", node.Path, node.Uid, node.Gid)
	}

	return nil
}

// setupUser changes the groups, gid, and uid for the user inside the container
// copy from libcontainer, cause not it's private
func setupUser(userSpec string) error {
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
	execUser, err := user.GetExecUserPath(userSpec, &defaultExecUser, passwdPath, groupPath)
	if err != nil {
		return err
	}
	if err := syscall.Setgroups(execUser.Sgids); err != nil {
		return err
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

/// Return the exit code of the process
// if the process has not exited -1 will be returned
func getExitCode(c *execdriver.Command) int {
	if c.ProcessConfig.ProcessState == nil {
		return -1
	}
	return c.ProcessConfig.ProcessState.Sys().(syscall.WaitStatus).ExitStatus()
}

func (d *driver) Kill(c *execdriver.Command, sig int) error {
	if sig == 9 || c.ProcessConfig.Process == nil {
		return KillLxc(c.ID, sig)
	}

	return c.ProcessConfig.Process.Signal(syscall.Signal(sig))
}

func (d *driver) Pause(c *execdriver.Command) error {
	_, err := exec.LookPath("lxc-freeze")
	if err == nil {
		output, errExec := exec.Command("lxc-freeze", "-n", c.ID).CombinedOutput()
		if errExec != nil {
			return fmt.Errorf("Err: %s Output: %s", errExec, output)
		}
	}

	return err
}

func (d *driver) Unpause(c *execdriver.Command) error {
	_, err := exec.LookPath("lxc-unfreeze")
	if err == nil {
		output, errExec := exec.Command("lxc-unfreeze", "-n", c.ID).CombinedOutput()
		if errExec != nil {
			return fmt.Errorf("Err: %s Output: %s", errExec, output)
		}
	}

	return err
}

func (d *driver) Terminate(c *execdriver.Command) error {
	return KillLxc(c.ID, 9)
}

func (d *driver) version() string {
	var (
		version string
		output  []byte
		err     error
	)
	if _, errPath := exec.LookPath("lxc-version"); errPath == nil {
		output, err = exec.Command("lxc-version").CombinedOutput()
	} else {
		output, err = exec.Command("lxc-start", "--version").CombinedOutput()
	}
	if err == nil {
		version = strings.TrimSpace(string(output))
		if parts := strings.SplitN(version, ":", 2); len(parts) == 2 {
			version = strings.TrimSpace(parts[1])
		}
	}
	return version
}

func KillLxc(id string, sig int) error {
	var (
		err    error
		output []byte
	)
	_, err = exec.LookPath("lxc-kill")
	if err == nil {
		output, err = exec.Command("lxc-kill", "-n", id, strconv.Itoa(sig)).CombinedOutput()
	} else {
		// lxc-stop does not take arbitrary signals like lxc-kill does
		output, err = exec.Command("lxc-stop", "-k", "-n", id).CombinedOutput()
	}
	if err != nil {
		return fmt.Errorf("Err: %s Output: %s", err, output)
	}
	return nil
}

// wait for the process to start and return the pid for the process
func (d *driver) waitForStart(c *execdriver.Command, waitLock chan struct{}) (int, error) {
	var (
		err    error
		output []byte
	)
	// We wait for the container to be fully running.
	// Timeout after 5 seconds. In case of broken pipe, just retry.
	// Note: The container can run and finish correctly before
	// the end of this loop
	for now := time.Now(); time.Since(now) < 5*time.Second; {
		select {
		case <-waitLock:
			// If the process dies while waiting for it, just return
			return -1, nil
		default:
		}

		output, err = d.getInfo(c.ID)
		if err == nil {
			info, err := parseLxcInfo(string(output))
			if err != nil {
				return -1, err
			}
			if info.Running {
				return info.Pid, nil
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	return -1, execdriver.ErrNotRunning
}

func (d *driver) getInfo(id string) ([]byte, error) {
	return exec.Command("lxc-info", "-n", id).CombinedOutput()
}

type info struct {
	ID     string
	driver *driver
}

func (i *info) IsRunning() bool {
	var running bool

	output, err := i.driver.getInfo(i.ID)
	if err != nil {
		logrus.Errorf("Error getting info for lxc container %s: %s (%s)", i.ID, err, output)
		return false
	}
	if strings.Contains(string(output), "RUNNING") {
		running = true
	}
	return running
}

func (d *driver) Info(id string) execdriver.Info {
	return &info{
		ID:     id,
		driver: d,
	}
}

func findCgroupRootAndDir(subsystem string) (string, string, error) {
	cgroupRoot, err := cgroups.FindCgroupMountpoint(subsystem)
	if err != nil {
		return "", "", err
	}

	cgroupDir, err := cgroups.GetThisCgroupDir(subsystem)
	if err != nil {
		return "", "", err
	}
	return cgroupRoot, cgroupDir, nil
}

func (d *driver) GetPidsForContainer(id string) ([]int, error) {
	pids := []int{}

	// cpu is chosen because it is the only non optional subsystem in cgroups
	subsystem := "cpu"
	cgroupRoot, cgroupDir, err := findCgroupRootAndDir(subsystem)
	if err != nil {
		return pids, err
	}

	filename := filepath.Join(cgroupRoot, cgroupDir, id, "tasks")
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		// With more recent lxc versions use, cgroup will be in lxc/
		filename = filepath.Join(cgroupRoot, cgroupDir, "lxc", id, "tasks")
	}

	output, err := ioutil.ReadFile(filename)
	if err != nil {
		return pids, err
	}
	for _, p := range strings.Split(string(output), "\n") {
		if len(p) == 0 {
			continue
		}
		pid, err := strconv.Atoi(p)
		if err != nil {
			return pids, fmt.Errorf("Invalid pid '%s': %s", p, err)
		}
		pids = append(pids, pid)
	}
	return pids, nil
}

func linkLxcStart(root string) error {
	sourcePath, err := exec.LookPath("lxc-start")
	if err != nil {
		return err
	}
	targetPath := path.Join(root, "lxc-start-unconfined")

	if _, err := os.Lstat(targetPath); err != nil && !os.IsNotExist(err) {
		return err
	} else if err == nil {
		if err := os.Remove(targetPath); err != nil {
			return err
		}
	}
	return os.Symlink(sourcePath, targetPath)
}

// TODO: This can be moved to the mountinfo reader in the mount pkg
func rootIsShared() bool {
	if data, err := ioutil.ReadFile("/proc/self/mountinfo"); err == nil {
		for _, line := range strings.Split(string(data), "\n") {
			cols := strings.Split(line, " ")
			if len(cols) >= 6 && cols[4] == "/" {
				return strings.HasPrefix(cols[6], "shared")
			}
		}
	}

	// No idea, probably safe to assume so
	return true
}

func (d *driver) containerDir(containerId string) string {
	return path.Join(d.libPath, "containers", containerId)
}

func (d *driver) generateLXCConfig(c *execdriver.Command) (string, error) {
	root := path.Join(d.containerDir(c.ID), "config.lxc")

	fo, err := os.Create(root)
	if err != nil {
		return "", err
	}
	defer fo.Close()

	if err := LxcTemplateCompiled.Execute(fo, struct {
		*execdriver.Command
		AppArmor bool
	}{
		Command:  c,
		AppArmor: d.apparmor,
	}); err != nil {
		return "", err
	}

	return root, nil
}

func (d *driver) generateEnvConfig(c *execdriver.Command) error {
	data, err := json.Marshal(c.ProcessConfig.Env)
	if err != nil {
		return err
	}
	p := path.Join(d.libPath, "containers", c.ID, "config.env")
	c.Mounts = append(c.Mounts, execdriver.Mount{
		Source:      p,
		Destination: "/.dockerenv",
		Writable:    false,
		Private:     true,
	})

	return ioutil.WriteFile(p, data, 0600)
}

// Clean not implemented for lxc
func (d *driver) Clean(id string) error {
	return nil
}

type TtyConsole struct {
	MasterPty *os.File
	SlavePty  *os.File
}

func NewTtyConsole(processConfig *execdriver.ProcessConfig, pipes *execdriver.Pipes) (*TtyConsole, error) {
	// lxc is special in that we cannot create the master outside of the container without
	// opening the slave because we have nothing to provide to the cmd.  We have to open both then do
	// the crazy setup on command right now instead of passing the console path to lxc and telling it
	// to open up that console.  we save a couple of openfiles in the native driver because we can do
	// this.
	ptyMaster, ptySlave, err := pty.Open()
	if err != nil {
		return nil, err
	}

	tty := &TtyConsole{
		MasterPty: ptyMaster,
		SlavePty:  ptySlave,
	}

	if err := tty.AttachPipes(&processConfig.Cmd, pipes); err != nil {
		tty.Close()
		return nil, err
	}

	processConfig.Console = tty.SlavePty.Name()

	return tty, nil
}

func (t *TtyConsole) Resize(h, w int) error {
	return term.SetWinsize(t.MasterPty.Fd(), &term.Winsize{Height: uint16(h), Width: uint16(w)})
}

func (t *TtyConsole) AttachPipes(command *exec.Cmd, pipes *execdriver.Pipes) error {
	command.Stdout = t.SlavePty
	command.Stderr = t.SlavePty

	go func() {
		if wb, ok := pipes.Stdout.(interface {
			CloseWriters() error
		}); ok {
			defer wb.CloseWriters()
		}

		io.Copy(pipes.Stdout, t.MasterPty)
	}()

	if pipes.Stdin != nil {
		command.Stdin = t.SlavePty
		command.SysProcAttr.Setctty = true

		go func() {
			io.Copy(t.MasterPty, pipes.Stdin)

			pipes.Stdin.Close()
		}()
	}
	return nil
}

func (t *TtyConsole) Close() error {
	t.SlavePty.Close()
	return t.MasterPty.Close()
}

func (d *driver) Exec(c *execdriver.Command, processConfig *execdriver.ProcessConfig, pipes *execdriver.Pipes, startCallback execdriver.StartCallback) (int, error) {
	return -1, ErrExec
}

func (d *driver) Stats(id string) (*execdriver.ResourceStats, error) {
	if _, ok := d.activeContainers[id]; !ok {
		return nil, fmt.Errorf("%s is not a key in active containers", id)
	}
	return execdriver.Stats(d.containerDir(id), d.activeContainers[id].container.Cgroups.Memory, d.machineMemory)
}
