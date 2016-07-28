// +build linux

package libcontainer

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/Sirupsen/logrus"
	"github.com/golang/protobuf/proto"
	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/opencontainers/runc/libcontainer/criurpc"
	"github.com/opencontainers/runc/libcontainer/utils"
	"github.com/syndtr/gocapability/capability"
	"github.com/vishvananda/netlink/nl"
)

const stdioFdCount = 3

type linuxContainer struct {
	id            string
	root          string
	config        *configs.Config
	cgroupManager cgroups.Manager
	initPath      string
	initArgs      []string
	initProcess   parentProcess
	criuPath      string
	m             sync.Mutex
	criuVersion   int
	state         containerState
	created       time.Time
}

// State represents a running container's state
type State struct {
	BaseState

	// Platform specific fields below here

	// Path to all the cgroups setup for a container. Key is cgroup subsystem name
	// with the value as the path.
	CgroupPaths map[string]string `json:"cgroup_paths"`

	// NamespacePaths are filepaths to the container's namespaces. Key is the namespace type
	// with the value as the path.
	NamespacePaths map[configs.NamespaceType]string `json:"namespace_paths"`

	// Container's standard descriptors (std{in,out,err}), needed for checkpoint and restore
	ExternalDescriptors []string `json:"external_descriptors,omitempty"`
}

// A libcontainer container object.
//
// Each container is thread-safe within the same process. Since a container can
// be destroyed by a separate process, any function may return that the container
// was not found.
type Container interface {
	BaseContainer

	// Methods below here are platform specific

	// Checkpoint checkpoints the running container's state to disk using the criu(8) utility.
	//
	// errors:
	// Systemerror - System error.
	Checkpoint(criuOpts *CriuOpts) error

	// Restore restores the checkpointed container to a running state using the criu(8) utility.
	//
	// errors:
	// Systemerror - System error.
	Restore(process *Process, criuOpts *CriuOpts) error

	// If the Container state is RUNNING or PAUSING, sets the Container state to PAUSING and pauses
	// the execution of any user processes. Asynchronously, when the container finished being paused the
	// state is changed to PAUSED.
	// If the Container state is PAUSED, do nothing.
	//
	// errors:
	// ContainerDestroyed - Container no longer exists,
	// Systemerror - System error.
	Pause() error

	// If the Container state is PAUSED, resumes the execution of any user processes in the
	// Container before setting the Container state to RUNNING.
	// If the Container state is RUNNING, do nothing.
	//
	// errors:
	// ContainerDestroyed - Container no longer exists,
	// Systemerror - System error.
	Resume() error

	// NotifyOOM returns a read-only channel signaling when the container receives an OOM notification.
	//
	// errors:
	// Systemerror - System error.
	NotifyOOM() (<-chan struct{}, error)

	// NotifyMemoryPressure returns a read-only channel signaling when the container reaches a given pressure level
	//
	// errors:
	// Systemerror - System error.
	NotifyMemoryPressure(level PressureLevel) (<-chan struct{}, error)
}

// ID returns the container's unique ID
func (c *linuxContainer) ID() string {
	return c.id
}

// Config returns the container's configuration
func (c *linuxContainer) Config() configs.Config {
	return *c.config
}

func (c *linuxContainer) Status() (Status, error) {
	c.m.Lock()
	defer c.m.Unlock()
	return c.currentStatus()
}

func (c *linuxContainer) State() (*State, error) {
	c.m.Lock()
	defer c.m.Unlock()
	return c.currentState()
}

func (c *linuxContainer) Processes() ([]int, error) {
	pids, err := c.cgroupManager.GetAllPids()
	if err != nil {
		return nil, newSystemError(err)
	}
	return pids, nil
}

func (c *linuxContainer) Stats() (*Stats, error) {
	var (
		err   error
		stats = &Stats{}
	)
	if stats.CgroupStats, err = c.cgroupManager.GetStats(); err != nil {
		return stats, newSystemError(err)
	}
	for _, iface := range c.config.Networks {
		switch iface.Type {
		case "veth":
			istats, err := getNetworkInterfaceStats(iface.HostInterfaceName)
			if err != nil {
				return stats, newSystemError(err)
			}
			stats.Interfaces = append(stats.Interfaces, istats)
		}
	}
	return stats, nil
}

func (c *linuxContainer) Set(config configs.Config) error {
	c.m.Lock()
	defer c.m.Unlock()
	c.config = &config
	return c.cgroupManager.Set(c.config)
}

func (c *linuxContainer) Start(process *Process) error {
	c.m.Lock()
	defer c.m.Unlock()
	status, err := c.currentStatus()
	if err != nil {
		return err
	}
	doInit := status == Destroyed
	parent, err := c.newParentProcess(process, doInit)
	if err != nil {
		return newSystemError(err)
	}
	if err := parent.start(); err != nil {
		// terminate the process to ensure that it properly is reaped.
		if err := parent.terminate(); err != nil {
			logrus.Warn(err)
		}
		return newSystemError(err)
	}
	// generate a timestamp indicating when the container was started
	c.created = time.Now().UTC()

	c.state = &runningState{
		c: c,
	}
	if doInit {
		if err := c.updateState(parent); err != nil {
			return err
		}
		if c.config.Hooks != nil {
			s := configs.HookState{
				Version:    c.config.Version,
				ID:         c.id,
				Pid:        parent.pid(),
				Root:       c.config.Rootfs,
				BundlePath: utils.SearchLabels(c.config.Labels, "bundle"),
			}
			for _, hook := range c.config.Hooks.Poststart {
				if err := hook.Run(s); err != nil {
					if err := parent.terminate(); err != nil {
						logrus.Warn(err)
					}
					return newSystemError(err)
				}
			}
		}
	}
	return nil
}

func (c *linuxContainer) Signal(s os.Signal) error {
	if err := c.initProcess.signal(s); err != nil {
		return newSystemError(err)
	}
	return nil
}

func (c *linuxContainer) newParentProcess(p *Process, doInit bool) (parentProcess, error) {
	parentPipe, childPipe, err := newPipe()
	if err != nil {
		return nil, newSystemError(err)
	}
	cmd, err := c.commandTemplate(p, childPipe)
	if err != nil {
		return nil, newSystemError(err)
	}
	if !doInit {
		return c.newSetnsProcess(p, cmd, parentPipe, childPipe)
	}
	return c.newInitProcess(p, cmd, parentPipe, childPipe)
}

func (c *linuxContainer) commandTemplate(p *Process, childPipe *os.File) (*exec.Cmd, error) {
	cmd := &exec.Cmd{
		Path: c.initPath,
		Args: c.initArgs,
	}
	cmd.Stdin = p.Stdin
	cmd.Stdout = p.Stdout
	cmd.Stderr = p.Stderr
	cmd.Dir = c.config.Rootfs
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.ExtraFiles = append(p.ExtraFiles, childPipe)
	cmd.Env = append(cmd.Env, fmt.Sprintf("_LIBCONTAINER_INITPIPE=%d", stdioFdCount+len(cmd.ExtraFiles)-1))
	// NOTE: when running a container with no PID namespace and the parent process spawning the container is
	// PID1 the pdeathsig is being delivered to the container's init process by the kernel for some reason
	// even with the parent still running.
	if c.config.ParentDeathSignal > 0 {
		cmd.SysProcAttr.Pdeathsig = syscall.Signal(c.config.ParentDeathSignal)
	}
	return cmd, nil
}

func (c *linuxContainer) newInitProcess(p *Process, cmd *exec.Cmd, parentPipe, childPipe *os.File) (*initProcess, error) {
	cmd.Env = append(cmd.Env, "_LIBCONTAINER_INITTYPE="+string(initStandard))
	nsMaps := make(map[configs.NamespaceType]string)
	for _, ns := range c.config.Namespaces {
		if ns.Path != "" {
			nsMaps[ns.Type] = ns.Path
		}
	}
	_, sharePidns := nsMaps[configs.NEWPID]
	data, err := c.bootstrapData(c.config.Namespaces.CloneFlags(), nsMaps, "")
	if err != nil {
		return nil, err
	}
	return &initProcess{
		cmd:           cmd,
		childPipe:     childPipe,
		parentPipe:    parentPipe,
		manager:       c.cgroupManager,
		config:        c.newInitConfig(p),
		container:     c,
		process:       p,
		bootstrapData: data,
		sharePidns:    sharePidns,
	}, nil
}

func (c *linuxContainer) newSetnsProcess(p *Process, cmd *exec.Cmd, parentPipe, childPipe *os.File) (*setnsProcess, error) {
	cmd.Env = append(cmd.Env, "_LIBCONTAINER_INITTYPE="+string(initSetns))
	state, err := c.currentState()
	if err != nil {
		return nil, newSystemError(err)
	}
	// for setns process, we dont have to set cloneflags as the process namespaces
	// will only be set via setns syscall
	data, err := c.bootstrapData(0, state.NamespacePaths, p.consolePath)
	if err != nil {
		return nil, err
	}
	// TODO: set on container for process management
	return &setnsProcess{
		cmd:           cmd,
		cgroupPaths:   c.cgroupManager.GetPaths(),
		childPipe:     childPipe,
		parentPipe:    parentPipe,
		config:        c.newInitConfig(p),
		process:       p,
		bootstrapData: data,
	}, nil
}

func (c *linuxContainer) newInitConfig(process *Process) *initConfig {
	cfg := &initConfig{
		Config:           c.config,
		Args:             process.Args,
		Env:              process.Env,
		User:             process.User,
		Cwd:              process.Cwd,
		Console:          process.consolePath,
		Capabilities:     process.Capabilities,
		PassedFilesCount: len(process.ExtraFiles),
		ContainerId:      c.ID(),
		NoNewPrivileges:  c.config.NoNewPrivileges,
		AppArmorProfile:  c.config.AppArmorProfile,
		ProcessLabel:     c.config.ProcessLabel,
		Rlimits:          c.config.Rlimits,
	}
	if process.NoNewPrivileges != nil {
		cfg.NoNewPrivileges = *process.NoNewPrivileges
	}
	if process.AppArmorProfile != "" {
		cfg.AppArmorProfile = process.AppArmorProfile
	}
	if process.Label != "" {
		cfg.ProcessLabel = process.Label
	}
	if len(process.Rlimits) > 0 {
		cfg.Rlimits = process.Rlimits
	}
	return cfg
}

func newPipe() (parent *os.File, child *os.File, err error) {
	fds, err := syscall.Socketpair(syscall.AF_LOCAL, syscall.SOCK_STREAM|syscall.SOCK_CLOEXEC, 0)
	if err != nil {
		return nil, nil, err
	}
	return os.NewFile(uintptr(fds[1]), "parent"), os.NewFile(uintptr(fds[0]), "child"), nil
}

func (c *linuxContainer) Destroy() error {
	c.m.Lock()
	defer c.m.Unlock()
	return c.state.destroy()
}

func (c *linuxContainer) Pause() error {
	c.m.Lock()
	defer c.m.Unlock()
	status, err := c.currentStatus()
	if err != nil {
		return err
	}
	if status != Running {
		return newGenericError(fmt.Errorf("container not running"), ContainerNotRunning)
	}
	if err := c.cgroupManager.Freeze(configs.Frozen); err != nil {
		return err
	}
	return c.state.transition(&pausedState{
		c: c,
	})
}

func (c *linuxContainer) Resume() error {
	c.m.Lock()
	defer c.m.Unlock()
	status, err := c.currentStatus()
	if err != nil {
		return err
	}
	if status != Paused {
		return newGenericError(fmt.Errorf("container not paused"), ContainerNotPaused)
	}
	if err := c.cgroupManager.Freeze(configs.Thawed); err != nil {
		return err
	}
	return c.state.transition(&runningState{
		c: c,
	})
}

func (c *linuxContainer) NotifyOOM() (<-chan struct{}, error) {
	return notifyOnOOM(c.cgroupManager.GetPaths())
}

func (c *linuxContainer) NotifyMemoryPressure(level PressureLevel) (<-chan struct{}, error) {
	return notifyMemoryPressure(c.cgroupManager.GetPaths(), level)
}

// check Criu version greater than or equal to min_version
func (c *linuxContainer) checkCriuVersion(min_version string) error {
	var x, y, z, versionReq int

	_, err := fmt.Sscanf(min_version, "%d.%d.%d\n", &x, &y, &z) // 1.5.2
	if err != nil {
		_, err = fmt.Sscanf(min_version, "Version: %d.%d\n", &x, &y) // 1.6
	}
	versionReq = x*10000 + y*100 + z

	out, err := exec.Command(c.criuPath, "-V").Output()
	if err != nil {
		return fmt.Errorf("Unable to execute CRIU command: %s", c.criuPath)
	}

	x = 0
	y = 0
	z = 0
	if ep := strings.Index(string(out), "-"); ep >= 0 {
		// criu Git version format
		var version string
		if sp := strings.Index(string(out), "GitID"); sp > 0 {
			version = string(out)[sp:ep]
		} else {
			return fmt.Errorf("Unable to parse the CRIU version: %s", c.criuPath)
		}

		n, err := fmt.Sscanf(string(version), "GitID: v%d.%d.%d", &x, &y, &z) // 1.5.2
		if err != nil {
			n, err = fmt.Sscanf(string(version), "GitID: v%d.%d", &x, &y) // 1.6
			y++
		} else {
			z++
		}
		if n < 2 || err != nil {
			return fmt.Errorf("Unable to parse the CRIU version: %s %d %s", version, n, err)
		}
	} else {
		// criu release version format
		n, err := fmt.Sscanf(string(out), "Version: %d.%d.%d\n", &x, &y, &z) // 1.5.2
		if err != nil {
			n, err = fmt.Sscanf(string(out), "Version: %d.%d\n", &x, &y) // 1.6
		}
		if n < 2 || err != nil {
			return fmt.Errorf("Unable to parse the CRIU version: %s %d %s", out, n, err)
		}
	}

	c.criuVersion = x*10000 + y*100 + z

	if c.criuVersion < versionReq {
		return fmt.Errorf("CRIU version must be %s or higher", min_version)
	}

	return nil
}

const descriptorsFilename = "descriptors.json"

func (c *linuxContainer) addCriuDumpMount(req *criurpc.CriuReq, m *configs.Mount) {
	mountDest := m.Destination
	if strings.HasPrefix(mountDest, c.config.Rootfs) {
		mountDest = mountDest[len(c.config.Rootfs):]
	}

	extMnt := &criurpc.ExtMountMap{
		Key: proto.String(mountDest),
		Val: proto.String(mountDest),
	}
	req.Opts.ExtMnt = append(req.Opts.ExtMnt, extMnt)
}

func (c *linuxContainer) Checkpoint(criuOpts *CriuOpts) error {
	c.m.Lock()
	defer c.m.Unlock()

	if err := c.checkCriuVersion("1.5.2"); err != nil {
		return err
	}

	if criuOpts.ImagesDirectory == "" {
		return fmt.Errorf("invalid directory to save checkpoint")
	}

	// Since a container can be C/R'ed multiple times,
	// the checkpoint directory may already exist.
	if err := os.Mkdir(criuOpts.ImagesDirectory, 0755); err != nil && !os.IsExist(err) {
		return err
	}

	if criuOpts.WorkDirectory == "" {
		criuOpts.WorkDirectory = filepath.Join(c.root, "criu.work")
	}

	if err := os.Mkdir(criuOpts.WorkDirectory, 0755); err != nil && !os.IsExist(err) {
		return err
	}

	workDir, err := os.Open(criuOpts.WorkDirectory)
	if err != nil {
		return err
	}
	defer workDir.Close()

	imageDir, err := os.Open(criuOpts.ImagesDirectory)
	if err != nil {
		return err
	}
	defer imageDir.Close()

	rpcOpts := criurpc.CriuOpts{
		ImagesDirFd:    proto.Int32(int32(imageDir.Fd())),
		WorkDirFd:      proto.Int32(int32(workDir.Fd())),
		LogLevel:       proto.Int32(4),
		LogFile:        proto.String("dump.log"),
		Root:           proto.String(c.config.Rootfs),
		ManageCgroups:  proto.Bool(true),
		NotifyScripts:  proto.Bool(true),
		Pid:            proto.Int32(int32(c.initProcess.pid())),
		ShellJob:       proto.Bool(criuOpts.ShellJob),
		LeaveRunning:   proto.Bool(criuOpts.LeaveRunning),
		TcpEstablished: proto.Bool(criuOpts.TcpEstablished),
		ExtUnixSk:      proto.Bool(criuOpts.ExternalUnixConnections),
		FileLocks:      proto.Bool(criuOpts.FileLocks),
		EmptyNs:        proto.Uint32(criuOpts.EmptyNs),
	}

	// append optional criu opts, e.g., page-server and port
	if criuOpts.PageServer.Address != "" && criuOpts.PageServer.Port != 0 {
		rpcOpts.Ps = &criurpc.CriuPageServerInfo{
			Address: proto.String(criuOpts.PageServer.Address),
			Port:    proto.Int32(criuOpts.PageServer.Port),
		}
	}

	// append optional manage cgroups mode
	if criuOpts.ManageCgroupsMode != 0 {
		if err := c.checkCriuVersion("1.7"); err != nil {
			return err
		}
		mode := criurpc.CriuCgMode(criuOpts.ManageCgroupsMode)
		rpcOpts.ManageCgroupsMode = &mode
	}

	t := criurpc.CriuReqType_DUMP
	req := &criurpc.CriuReq{
		Type: &t,
		Opts: &rpcOpts,
	}

	for _, m := range c.config.Mounts {
		switch m.Device {
		case "bind":
			c.addCriuDumpMount(req, m)
			break
		case "cgroup":
			binds, err := getCgroupMounts(m)
			if err != nil {
				return err
			}
			for _, b := range binds {
				c.addCriuDumpMount(req, b)
			}
			break
		}
	}

	// Write the FD info to a file in the image directory

	fdsJSON, err := json.Marshal(c.initProcess.externalDescriptors())
	if err != nil {
		return err
	}

	err = ioutil.WriteFile(filepath.Join(criuOpts.ImagesDirectory, descriptorsFilename), fdsJSON, 0655)
	if err != nil {
		return err
	}

	err = c.criuSwrk(nil, req, criuOpts, false)
	if err != nil {
		return err
	}
	return nil
}

func (c *linuxContainer) addCriuRestoreMount(req *criurpc.CriuReq, m *configs.Mount) {
	mountDest := m.Destination
	if strings.HasPrefix(mountDest, c.config.Rootfs) {
		mountDest = mountDest[len(c.config.Rootfs):]
	}

	extMnt := &criurpc.ExtMountMap{
		Key: proto.String(mountDest),
		Val: proto.String(m.Source),
	}
	req.Opts.ExtMnt = append(req.Opts.ExtMnt, extMnt)
}

func (c *linuxContainer) Restore(process *Process, criuOpts *CriuOpts) error {
	c.m.Lock()
	defer c.m.Unlock()
	if err := c.checkCriuVersion("1.5.2"); err != nil {
		return err
	}
	if criuOpts.WorkDirectory == "" {
		criuOpts.WorkDirectory = filepath.Join(c.root, "criu.work")
	}
	// Since a container can be C/R'ed multiple times,
	// the work directory may already exist.
	if err := os.Mkdir(criuOpts.WorkDirectory, 0655); err != nil && !os.IsExist(err) {
		return err
	}
	workDir, err := os.Open(criuOpts.WorkDirectory)
	if err != nil {
		return err
	}
	defer workDir.Close()
	if criuOpts.ImagesDirectory == "" {
		return fmt.Errorf("invalid directory to restore checkpoint")
	}
	imageDir, err := os.Open(criuOpts.ImagesDirectory)
	if err != nil {
		return err
	}
	defer imageDir.Close()
	// CRIU has a few requirements for a root directory:
	// * it must be a mount point
	// * its parent must not be overmounted
	// c.config.Rootfs is bind-mounted to a temporary directory
	// to satisfy these requirements.
	root := filepath.Join(c.root, "criu-root")
	if err := os.Mkdir(root, 0755); err != nil {
		return err
	}
	defer os.Remove(root)
	root, err = filepath.EvalSymlinks(root)
	if err != nil {
		return err
	}
	err = syscall.Mount(c.config.Rootfs, root, "", syscall.MS_BIND|syscall.MS_REC, "")
	if err != nil {
		return err
	}
	defer syscall.Unmount(root, syscall.MNT_DETACH)
	t := criurpc.CriuReqType_RESTORE
	req := &criurpc.CriuReq{
		Type: &t,
		Opts: &criurpc.CriuOpts{
			ImagesDirFd:    proto.Int32(int32(imageDir.Fd())),
			WorkDirFd:      proto.Int32(int32(workDir.Fd())),
			EvasiveDevices: proto.Bool(true),
			LogLevel:       proto.Int32(4),
			LogFile:        proto.String("restore.log"),
			RstSibling:     proto.Bool(true),
			Root:           proto.String(root),
			ManageCgroups:  proto.Bool(true),
			NotifyScripts:  proto.Bool(true),
			ShellJob:       proto.Bool(criuOpts.ShellJob),
			ExtUnixSk:      proto.Bool(criuOpts.ExternalUnixConnections),
			TcpEstablished: proto.Bool(criuOpts.TcpEstablished),
			FileLocks:      proto.Bool(criuOpts.FileLocks),
			EmptyNs:        proto.Uint32(criuOpts.EmptyNs),
		},
	}

	for _, m := range c.config.Mounts {
		switch m.Device {
		case "bind":
			c.addCriuRestoreMount(req, m)
			break
		case "cgroup":
			binds, err := getCgroupMounts(m)
			if err != nil {
				return err
			}
			for _, b := range binds {
				c.addCriuRestoreMount(req, b)
			}
			break
		}
	}
	for _, iface := range c.config.Networks {
		switch iface.Type {
		case "veth":
			veth := new(criurpc.CriuVethPair)
			veth.IfOut = proto.String(iface.HostInterfaceName)
			veth.IfIn = proto.String(iface.Name)
			req.Opts.Veths = append(req.Opts.Veths, veth)
			break
		case "loopback":
			break
		}
	}
	for _, i := range criuOpts.VethPairs {
		veth := new(criurpc.CriuVethPair)
		veth.IfOut = proto.String(i.HostInterfaceName)
		veth.IfIn = proto.String(i.ContainerInterfaceName)
		req.Opts.Veths = append(req.Opts.Veths, veth)
	}

	// append optional manage cgroups mode
	if criuOpts.ManageCgroupsMode != 0 {
		if err := c.checkCriuVersion("1.7"); err != nil {
			return err
		}
		mode := criurpc.CriuCgMode(criuOpts.ManageCgroupsMode)
		req.Opts.ManageCgroupsMode = &mode
	}

	var (
		fds    []string
		fdJSON []byte
	)
	if fdJSON, err = ioutil.ReadFile(filepath.Join(criuOpts.ImagesDirectory, descriptorsFilename)); err != nil {
		return err
	}

	if err := json.Unmarshal(fdJSON, &fds); err != nil {
		return err
	}
	for i := range fds {
		if s := fds[i]; strings.Contains(s, "pipe:") {
			inheritFd := new(criurpc.InheritFd)
			inheritFd.Key = proto.String(s)
			inheritFd.Fd = proto.Int32(int32(i))
			req.Opts.InheritFd = append(req.Opts.InheritFd, inheritFd)
		}
	}
	return c.criuSwrk(process, req, criuOpts, true)
}

func (c *linuxContainer) criuApplyCgroups(pid int, req *criurpc.CriuReq) error {
	if err := c.cgroupManager.Apply(pid); err != nil {
		return err
	}

	path := fmt.Sprintf("/proc/%d/cgroup", pid)
	cgroupsPaths, err := cgroups.ParseCgroupFile(path)
	if err != nil {
		return err
	}

	for c, p := range cgroupsPaths {
		cgroupRoot := &criurpc.CgroupRoot{
			Ctrl: proto.String(c),
			Path: proto.String(p),
		}
		req.Opts.CgRoot = append(req.Opts.CgRoot, cgroupRoot)
	}

	return nil
}

func (c *linuxContainer) criuSwrk(process *Process, req *criurpc.CriuReq, opts *CriuOpts, applyCgroups bool) error {
	fds, err := syscall.Socketpair(syscall.AF_LOCAL, syscall.SOCK_SEQPACKET|syscall.SOCK_CLOEXEC, 0)
	if err != nil {
		return err
	}

	logPath := filepath.Join(opts.WorkDirectory, req.GetOpts().GetLogFile())
	criuClient := os.NewFile(uintptr(fds[0]), "criu-transport-client")
	criuServer := os.NewFile(uintptr(fds[1]), "criu-transport-server")
	defer criuClient.Close()
	defer criuServer.Close()

	args := []string{"swrk", "3"}
	logrus.Debugf("Using CRIU %d at: %s", c.criuVersion, c.criuPath)
	logrus.Debugf("Using CRIU with following args: %s", args)
	cmd := exec.Command(c.criuPath, args...)
	if process != nil {
		cmd.Stdin = process.Stdin
		cmd.Stdout = process.Stdout
		cmd.Stderr = process.Stderr
	}
	cmd.ExtraFiles = append(cmd.ExtraFiles, criuServer)

	if err := cmd.Start(); err != nil {
		return err
	}
	criuServer.Close()

	defer func() {
		criuClient.Close()
		_, err := cmd.Process.Wait()
		if err != nil {
			return
		}
	}()

	if applyCgroups {
		err := c.criuApplyCgroups(cmd.Process.Pid, req)
		if err != nil {
			return err
		}
	}

	var extFds []string
	if process != nil {
		extFds, err = getPipeFds(cmd.Process.Pid)
		if err != nil {
			return err
		}
	}

	logrus.Debugf("Using CRIU in %s mode", req.GetType().String())
	val := reflect.ValueOf(req.GetOpts())
	v := reflect.Indirect(val)
	for i := 0; i < v.NumField(); i++ {
		st := v.Type()
		name := st.Field(i).Name
		if strings.HasPrefix(name, "XXX_") {
			continue
		}
		value := val.MethodByName("Get" + name).Call([]reflect.Value{})
		logrus.Debugf("CRIU option %s with value %v", name, value[0])
	}
	data, err := proto.Marshal(req)
	if err != nil {
		return err
	}
	_, err = criuClient.Write(data)
	if err != nil {
		return err
	}

	buf := make([]byte, 10*4096)
	for true {
		n, err := criuClient.Read(buf)
		if err != nil {
			return err
		}
		if n == 0 {
			return fmt.Errorf("unexpected EOF")
		}
		if n == len(buf) {
			return fmt.Errorf("buffer is too small")
		}

		resp := new(criurpc.CriuResp)
		err = proto.Unmarshal(buf[:n], resp)
		if err != nil {
			return err
		}
		if !resp.GetSuccess() {
			typeString := req.GetType().String()
			return fmt.Errorf("criu failed: type %s errno %d\nlog file: %s", typeString, resp.GetCrErrno(), logPath)
		}

		t := resp.GetType()
		switch {
		case t == criurpc.CriuReqType_NOTIFY:
			if err := c.criuNotifications(resp, process, opts, extFds); err != nil {
				return err
			}
			t = criurpc.CriuReqType_NOTIFY
			req = &criurpc.CriuReq{
				Type:          &t,
				NotifySuccess: proto.Bool(true),
			}
			data, err = proto.Marshal(req)
			if err != nil {
				return err
			}
			_, err = criuClient.Write(data)
			if err != nil {
				return err
			}
			continue
		case t == criurpc.CriuReqType_RESTORE:
		case t == criurpc.CriuReqType_DUMP:
			break
		default:
			return fmt.Errorf("unable to parse the response %s", resp.String())
		}

		break
	}

	// cmd.Wait() waits cmd.goroutines which are used for proxying file descriptors.
	// Here we want to wait only the CRIU process.
	st, err := cmd.Process.Wait()
	if err != nil {
		return err
	}
	if !st.Success() {
		return fmt.Errorf("criu failed: %s\nlog file: %s", st.String(), logPath)
	}
	return nil
}

// block any external network activity
func lockNetwork(config *configs.Config) error {
	for _, config := range config.Networks {
		strategy, err := getStrategy(config.Type)
		if err != nil {
			return err
		}

		if err := strategy.detach(config); err != nil {
			return err
		}
	}
	return nil
}

func unlockNetwork(config *configs.Config) error {
	for _, config := range config.Networks {
		strategy, err := getStrategy(config.Type)
		if err != nil {
			return err
		}
		if err = strategy.attach(config); err != nil {
			return err
		}
	}
	return nil
}

func (c *linuxContainer) criuNotifications(resp *criurpc.CriuResp, process *Process, opts *CriuOpts, fds []string) error {
	notify := resp.GetNotify()
	if notify == nil {
		return fmt.Errorf("invalid response: %s", resp.String())
	}
	switch {
	case notify.GetScript() == "post-dump":
		f, err := os.Create(filepath.Join(c.root, "checkpoint"))
		if err != nil {
			return err
		}
		f.Close()
	case notify.GetScript() == "network-unlock":
		if err := unlockNetwork(c.config); err != nil {
			return err
		}
	case notify.GetScript() == "network-lock":
		if err := lockNetwork(c.config); err != nil {
			return err
		}
	case notify.GetScript() == "setup-namespaces":
		if c.config.Hooks != nil {
			s := configs.HookState{
				Version: c.config.Version,
				ID:      c.id,
				Pid:     int(notify.GetPid()),
				Root:    c.config.Rootfs,
			}
			for _, hook := range c.config.Hooks.Prestart {
				if err := hook.Run(s); err != nil {
					return newSystemError(err)
				}
			}
		}
	case notify.GetScript() == "post-restore":
		pid := notify.GetPid()
		r, err := newRestoredProcess(int(pid), fds)
		if err != nil {
			return err
		}
		process.ops = r
		if err := c.state.transition(&restoredState{
			imageDir: opts.ImagesDirectory,
			c:        c,
		}); err != nil {
			return err
		}
		if err := c.updateState(r); err != nil {
			return err
		}
		if err := os.Remove(filepath.Join(c.root, "checkpoint")); err != nil {
			if !os.IsNotExist(err) {
				logrus.Error(err)
			}
		}
	}
	return nil
}

func (c *linuxContainer) updateState(process parentProcess) error {
	c.initProcess = process
	state, err := c.currentState()
	if err != nil {
		return err
	}
	return c.saveState(state)
}

func (c *linuxContainer) saveState(s *State) error {
	f, err := os.Create(filepath.Join(c.root, stateFilename))
	if err != nil {
		return err
	}
	defer f.Close()
	return utils.WriteJSON(f, s)
}

func (c *linuxContainer) deleteState() error {
	return os.Remove(filepath.Join(c.root, stateFilename))
}

func (c *linuxContainer) currentStatus() (Status, error) {
	if err := c.refreshState(); err != nil {
		return -1, err
	}
	return c.state.status(), nil
}

// refreshState needs to be called to verify that the current state on the
// container is what is true.  Because consumers of libcontainer can use it
// out of process we need to verify the container's status based on runtime
// information and not rely on our in process info.
func (c *linuxContainer) refreshState() error {
	paused, err := c.isPaused()
	if err != nil {
		return err
	}
	if paused {
		return c.state.transition(&pausedState{c: c})
	}
	running, err := c.isRunning()
	if err != nil {
		return err
	}
	if running {
		return c.state.transition(&runningState{c: c})
	}
	return c.state.transition(&stoppedState{c: c})
}

func (c *linuxContainer) isRunning() (bool, error) {
	if c.initProcess == nil {
		return false, nil
	}
	// return Running if the init process is alive
	if err := syscall.Kill(c.initProcess.pid(), 0); err != nil {
		if err == syscall.ESRCH {
			return false, nil
		}
		return false, newSystemError(err)
	}
	return true, nil
}

func (c *linuxContainer) isPaused() (bool, error) {
	data, err := ioutil.ReadFile(filepath.Join(c.cgroupManager.GetPaths()["freezer"], "freezer.state"))
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, newSystemError(err)
	}
	return bytes.Equal(bytes.TrimSpace(data), []byte("FROZEN")), nil
}

func (c *linuxContainer) currentState() (*State, error) {
	var (
		startTime           string
		externalDescriptors []string
		pid                 = -1
	)
	if c.initProcess != nil {
		pid = c.initProcess.pid()
		startTime, _ = c.initProcess.startTime()
		externalDescriptors = c.initProcess.externalDescriptors()
	}
	state := &State{
		BaseState: BaseState{
			ID:                   c.ID(),
			Config:               *c.config,
			InitProcessPid:       pid,
			InitProcessStartTime: startTime,
			Created:              c.created,
		},
		CgroupPaths:         c.cgroupManager.GetPaths(),
		NamespacePaths:      make(map[configs.NamespaceType]string),
		ExternalDescriptors: externalDescriptors,
	}
	if pid > 0 {
		for _, ns := range c.config.Namespaces {
			state.NamespacePaths[ns.Type] = ns.GetPath(pid)
		}
		for _, nsType := range configs.NamespaceTypes() {
			if !configs.IsNamespaceSupported(nsType) {
				continue
			}
			if _, ok := state.NamespacePaths[nsType]; !ok {
				ns := configs.Namespace{Type: nsType}
				state.NamespacePaths[ns.Type] = ns.GetPath(pid)
			}
		}
	}
	return state, nil
}

// orderNamespacePaths sorts namespace paths into a list of paths that we
// can setns in order.
func (c *linuxContainer) orderNamespacePaths(namespaces map[configs.NamespaceType]string) ([]string, error) {
	paths := []string{}
	nsTypes := []configs.NamespaceType{
		configs.NEWIPC,
		configs.NEWUTS,
		configs.NEWNET,
		configs.NEWPID,
		configs.NEWNS,
	}
	// join userns if the init process explicitly requires NEWUSER
	if c.config.Namespaces.Contains(configs.NEWUSER) {
		nsTypes = append(nsTypes, configs.NEWUSER)
	}
	for _, nsType := range nsTypes {
		if p, ok := namespaces[nsType]; ok && p != "" {
			// check if the requested namespace is supported
			if !configs.IsNamespaceSupported(nsType) {
				return nil, newSystemError(fmt.Errorf("namespace %s is not supported", nsType))
			}
			// only set to join this namespace if it exists
			if _, err := os.Lstat(p); err != nil {
				return nil, newSystemError(err)
			}
			// do not allow namespace path with comma as we use it to separate
			// the namespace paths
			if strings.ContainsRune(p, ',') {
				return nil, newSystemError(fmt.Errorf("invalid path %s", p))
			}
			paths = append(paths, p)
		}
	}
	return paths, nil
}

func encodeIDMapping(idMap []configs.IDMap) ([]byte, error) {
	data := bytes.NewBuffer(nil)
	for _, im := range idMap {
		line := fmt.Sprintf("%d %d %d\n", im.ContainerID, im.HostID, im.Size)
		if _, err := data.WriteString(line); err != nil {
			return nil, err
		}
	}
	return data.Bytes(), nil
}

// bootstrapData encodes the necessary data in netlink binary format
// as a io.Reader.
// Consumer can write the data to a bootstrap program
// such as one that uses nsenter package to bootstrap the container's
// init process correctly, i.e. with correct namespaces, uid/gid
// mapping etc.
func (c *linuxContainer) bootstrapData(cloneFlags uintptr, nsMaps map[configs.NamespaceType]string, consolePath string) (io.Reader, error) {
	// create the netlink message
	r := nl.NewNetlinkRequest(int(InitMsg), 0)

	// write cloneFlags
	r.AddData(&Int32msg{
		Type:  CloneFlagsAttr,
		Value: uint32(cloneFlags),
	})

	// write console path
	if consolePath != "" {
		r.AddData(&Bytemsg{
			Type:  ConsolePathAttr,
			Value: []byte(consolePath),
		})
	}

	// write custom namespace paths
	if len(nsMaps) > 0 {
		nsPaths, err := c.orderNamespacePaths(nsMaps)
		if err != nil {
			return nil, err
		}
		r.AddData(&Bytemsg{
			Type:  NsPathsAttr,
			Value: []byte(strings.Join(nsPaths, ",")),
		})
	}

	// write namespace paths only when we are not joining an existing user ns
	_, joinExistingUser := nsMaps[configs.NEWUSER]
	if !joinExistingUser {
		// write uid mappings
		if len(c.config.UidMappings) > 0 {
			b, err := encodeIDMapping(c.config.UidMappings)
			if err != nil {
				return nil, err
			}
			r.AddData(&Bytemsg{
				Type:  UidmapAttr,
				Value: b,
			})
		}

		// write gid mappings
		if len(c.config.GidMappings) > 0 {
			b, err := encodeIDMapping(c.config.GidMappings)
			if err != nil {
				return nil, err
			}
			r.AddData(&Bytemsg{
				Type:  GidmapAttr,
				Value: b,
			})
			// check if we have CAP_SETGID to setgroup properly
			pid, err := capability.NewPid(os.Getpid())
			if err != nil {
				return nil, err
			}
			if !pid.Get(capability.EFFECTIVE, capability.CAP_SETGID) {
				r.AddData(&Boolmsg{
					Type:  SetgroupAttr,
					Value: true,
				})
			}
		}
	}

	return bytes.NewReader(r.Serialize()), nil
}
