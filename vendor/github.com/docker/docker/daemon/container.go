package daemon

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/opencontainers/runc/libcontainer/label"

	"github.com/Sirupsen/logrus"
	"github.com/docker/docker/daemon/execdriver"
	"github.com/docker/docker/daemon/logger"
	"github.com/docker/docker/daemon/logger/jsonfilelog"
	"github.com/docker/docker/daemon/network"
	"github.com/docker/docker/image"
	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/broadcastwriter"
	"github.com/docker/docker/pkg/fileutils"
	"github.com/docker/docker/pkg/ioutils"
	"github.com/docker/docker/pkg/jsonlog"
	"github.com/docker/docker/pkg/mount"
	"github.com/docker/docker/pkg/nat"
	"github.com/docker/docker/pkg/promise"
	"github.com/docker/docker/pkg/symlink"
	"github.com/docker/docker/runconfig"
	"github.com/docker/docker/volume"
)

var (
	ErrNotATTY                 = errors.New("The PTY is not a file")
	ErrNoTTY                   = errors.New("No PTY found")
	ErrContainerStart          = errors.New("The container failed to start. Unknown error")
	ErrContainerStartTimeout   = errors.New("The container failed to start due to timed out.")
	ErrContainerRootfsReadonly = errors.New("container rootfs is marked read-only")
)

type StreamConfig struct {
	stdout    *broadcastwriter.BroadcastWriter
	stderr    *broadcastwriter.BroadcastWriter
	stdin     io.ReadCloser
	stdinPipe io.WriteCloser
}

// CommonContainer holds the settings for a container which are applicable
// across all platforms supported by the daemon.
type CommonContainer struct {
	StreamConfig

	*State `json:"State"` // Needed for remote api version <= 1.11
	root   string         // Path to the "home" of the container, including metadata.
	basefs string         // Path to the graphdriver mountpoint

	ID                       string
	Created                  time.Time
	Path                     string
	Args                     []string
	Config                   *runconfig.Config
	ImageID                  string `json:"Image"`
	NetworkSettings          *network.Settings
	ResolvConfPath           string
	HostnamePath             string
	HostsPath                string
	LogPath                  string
	Name                     string
	Driver                   string
	ExecDriver               string
	MountLabel, ProcessLabel string
	RestartCount             int
	UpdateDns                bool
	HasBeenStartedBefore     bool

	MountPoints map[string]*mountPoint
	Volumes     map[string]string // Deprecated since 1.7, kept for backwards compatibility
	VolumesRW   map[string]bool   // Deprecated since 1.7, kept for backwards compatibility

	hostConfig *runconfig.HostConfig
	command    *execdriver.Command

	monitor      *containerMonitor
	execCommands *execStore
	daemon       *Daemon
	// logDriver for closing
	logDriver logger.Logger
	logCopier *logger.Copier
}

func (container *Container) FromDisk() error {
	pth, err := container.jsonPath()
	if err != nil {
		return err
	}

	jsonSource, err := os.Open(pth)
	if err != nil {
		return err
	}
	defer jsonSource.Close()

	dec := json.NewDecoder(jsonSource)

	// Load container settings
	// udp broke compat of docker.PortMapping, but it's not used when loading a container, we can skip it
	if err := dec.Decode(container); err != nil && !strings.Contains(err.Error(), "docker.PortMapping") {
		return err
	}

	if err := label.ReserveLabel(container.ProcessLabel); err != nil {
		return err
	}
	return container.readHostConfig()
}

func (container *Container) toDisk() error {
	data, err := json.Marshal(container)
	if err != nil {
		return err
	}

	pth, err := container.jsonPath()
	if err != nil {
		return err
	}

	if err := ioutil.WriteFile(pth, data, 0666); err != nil {
		return err
	}

	return container.WriteHostConfig()
}

func (container *Container) ToDisk() error {
	container.Lock()
	err := container.toDisk()
	container.Unlock()
	return err
}

func (container *Container) readHostConfig() error {
	container.hostConfig = &runconfig.HostConfig{}
	// If the hostconfig file does not exist, do not read it.
	// (We still have to initialize container.hostConfig,
	// but that's OK, since we just did that above.)
	pth, err := container.hostConfigPath()
	if err != nil {
		return err
	}

	_, err = os.Stat(pth)
	if os.IsNotExist(err) {
		return nil
	}

	f, err := os.Open(pth)
	if err != nil {
		return err
	}
	defer f.Close()

	return json.NewDecoder(f).Decode(&container.hostConfig)
}

func (container *Container) WriteHostConfig() error {
	data, err := json.Marshal(container.hostConfig)
	if err != nil {
		return err
	}

	pth, err := container.hostConfigPath()
	if err != nil {
		return err
	}

	return ioutil.WriteFile(pth, data, 0666)
}

func (container *Container) LogEvent(action string) {
	d := container.daemon
	d.EventsService.Log(
		action,
		container.ID,
		container.Config.Image,
	)
}

// Evaluates `path` in the scope of the container's basefs, with proper path
// sanitisation. Symlinks are all scoped to the basefs of the container, as
// though the container's basefs was `/`.
//
// The basefs of a container is the host-facing path which is bind-mounted as
// `/` inside the container. This method is essentially used to access a
// particular path inside the container as though you were a process in that
// container.
//
// NOTE: The returned path is *only* safely scoped inside the container's basefs
//       if no component of the returned path changes (such as a component
//       symlinking to a different path) between using this method and using the
//       path. See symlink.FollowSymlinkInScope for more details.
func (container *Container) GetResourcePath(path string) (string, error) {
	// IMPORTANT - These are paths on the OS where the daemon is running, hence
	// any filepath operations must be done in an OS agnostic way.
	cleanPath := filepath.Join(string(os.PathSeparator), path)
	r, e := symlink.FollowSymlinkInScope(filepath.Join(container.basefs, cleanPath), container.basefs)
	return r, e
}

// Evaluates `path` in the scope of the container's root, with proper path
// sanitisation. Symlinks are all scoped to the root of the container, as
// though the container's root was `/`.
//
// The root of a container is the host-facing configuration metadata directory.
// Only use this method to safely access the container's `container.json` or
// other metadata files. If in doubt, use container.GetResourcePath.
//
// NOTE: The returned path is *only* safely scoped inside the container's root
//       if no component of the returned path changes (such as a component
//       symlinking to a different path) between using this method and using the
//       path. See symlink.FollowSymlinkInScope for more details.
func (container *Container) GetRootResourcePath(path string) (string, error) {
	// IMPORTANT - These are paths on the OS where the daemon is running, hence
	// any filepath operations must be done in an OS agnostic way.
	cleanPath := filepath.Join(string(os.PathSeparator), path)
	return symlink.FollowSymlinkInScope(filepath.Join(container.root, cleanPath), container.root)
}

func (container *Container) Start() (err error) {
	container.Lock()
	defer container.Unlock()

	if container.Running {
		return nil
	}

	if container.removalInProgress || container.Dead {
		return fmt.Errorf("Container is marked for removal and cannot be started.")
	}

	// if we encounter an error during start we need to ensure that any other
	// setup has been cleaned up properly
	defer func() {
		if err != nil {
			container.setError(err)
			// if no one else has set it, make sure we don't leave it at zero
			if container.ExitCode == 0 {
				container.ExitCode = 128
			}
			container.toDisk()
			container.cleanup()
			container.LogEvent("die")
		}
	}()

	if err := container.Mount(); err != nil {
		return err
	}

	// No-op if non-Windows. Once the container filesystem is mounted,
	// prepare the layer to boot using the Windows driver.
	if err := container.PrepareStorage(); err != nil {
		return err
	}

	if err := container.initializeNetworking(); err != nil {
		return err
	}
	linkedEnv, err := container.setupLinkedContainers()
	if err != nil {
		return err
	}
	if err := container.setupWorkingDirectory(); err != nil {
		return err
	}
	env := container.createDaemonEnvironment(linkedEnv)
	if err := populateCommand(container, env); err != nil {
		return err
	}

	mounts, err := container.setupMounts()
	if err != nil {
		return err
	}

	container.command.Mounts = mounts
	return container.waitForStart()
}

func (container *Container) Run() error {
	if err := container.Start(); err != nil {
		return err
	}
	container.HasBeenStartedBefore = true
	container.WaitStop(-1 * time.Second)
	return nil
}

func (container *Container) Output() (output []byte, err error) {
	pipe := container.StdoutPipe()
	defer pipe.Close()
	if err := container.Start(); err != nil {
		return nil, err
	}
	output, err = ioutil.ReadAll(pipe)
	container.WaitStop(-1 * time.Second)
	return output, err
}

// StreamConfig.StdinPipe returns a WriteCloser which can be used to feed data
// to the standard input of the container's active process.
// Container.StdoutPipe and Container.StderrPipe each return a ReadCloser
// which can be used to retrieve the standard output (and error) generated
// by the container's active process. The output (and error) are actually
// copied and delivered to all StdoutPipe and StderrPipe consumers, using
// a kind of "broadcaster".

func (streamConfig *StreamConfig) StdinPipe() io.WriteCloser {
	return streamConfig.stdinPipe
}

func (streamConfig *StreamConfig) StdoutPipe() io.ReadCloser {
	reader, writer := io.Pipe()
	streamConfig.stdout.AddWriter(writer, "")
	return ioutils.NewBufReader(reader)
}

func (streamConfig *StreamConfig) StderrPipe() io.ReadCloser {
	reader, writer := io.Pipe()
	streamConfig.stderr.AddWriter(writer, "")
	return ioutils.NewBufReader(reader)
}

func (streamConfig *StreamConfig) StdoutLogPipe() io.ReadCloser {
	reader, writer := io.Pipe()
	streamConfig.stdout.AddWriter(writer, "stdout")
	return ioutils.NewBufReader(reader)
}

func (streamConfig *StreamConfig) StderrLogPipe() io.ReadCloser {
	reader, writer := io.Pipe()
	streamConfig.stderr.AddWriter(writer, "stderr")
	return ioutils.NewBufReader(reader)
}

func (container *Container) isNetworkAllocated() bool {
	return container.NetworkSettings.IPAddress != ""
}

// cleanup releases any network resources allocated to the container along with any rules
// around how containers are linked together.  It also unmounts the container's root filesystem.
func (container *Container) cleanup() {
	container.ReleaseNetwork()

	disableAllActiveLinks(container)

	if err := container.CleanupStorage(); err != nil {
		logrus.Errorf("%v: Failed to cleanup storage: %v", container.ID, err)
	}

	if err := container.Unmount(); err != nil {
		logrus.Errorf("%v: Failed to umount filesystem: %v", container.ID, err)
	}

	for _, eConfig := range container.execCommands.s {
		container.daemon.unregisterExecCommand(eConfig)
	}

	container.UnmountVolumes(false)
}

func (container *Container) KillSig(sig int) error {
	logrus.Debugf("Sending %d to %s", sig, container.ID)
	container.Lock()
	defer container.Unlock()

	// We could unpause the container for them rather than returning this error
	if container.Paused {
		return fmt.Errorf("Container %s is paused. Unpause the container before stopping", container.ID)
	}

	if !container.Running {
		return fmt.Errorf("Container %s is not running", container.ID)
	}

	// signal to the monitor that it should not restart the container
	// after we send the kill signal
	container.monitor.ExitOnNext()

	// if the container is currently restarting we do not need to send the signal
	// to the process.  Telling the monitor that it should exit on it's next event
	// loop is enough
	if container.Restarting {
		return nil
	}

	if err := container.daemon.Kill(container, sig); err != nil {
		return err
	}
	container.LogEvent("kill")
	return nil
}

// Wrapper aroung KillSig() suppressing "no such process" error.
func (container *Container) killPossiblyDeadProcess(sig int) error {
	err := container.KillSig(sig)
	if err == syscall.ESRCH {
		logrus.Debugf("Cannot kill process (pid=%d) with signal %d: no such process.", container.GetPid(), sig)
		return nil
	}
	return err
}

func (container *Container) Pause() error {
	container.Lock()
	defer container.Unlock()

	// We cannot Pause the container which is not running
	if !container.Running {
		return fmt.Errorf("Container %s is not running, cannot pause a non-running container", container.ID)
	}

	// We cannot Pause the container which is already paused
	if container.Paused {
		return fmt.Errorf("Container %s is already paused", container.ID)
	}

	if err := container.daemon.execDriver.Pause(container.command); err != nil {
		return err
	}
	container.Paused = true
	container.LogEvent("pause")
	return nil
}

func (container *Container) Unpause() error {
	container.Lock()
	defer container.Unlock()

	// We cannot unpause the container which is not running
	if !container.Running {
		return fmt.Errorf("Container %s is not running, cannot unpause a non-running container", container.ID)
	}

	// We cannot unpause the container which is not paused
	if !container.Paused {
		return fmt.Errorf("Container %s is not paused", container.ID)
	}

	if err := container.daemon.execDriver.Unpause(container.command); err != nil {
		return err
	}
	container.Paused = false
	container.LogEvent("unpause")
	return nil
}

func (container *Container) Kill() error {
	if !container.IsRunning() {
		return fmt.Errorf("Container %s is not running", container.ID)
	}

	// 1. Send SIGKILL
	if err := container.killPossiblyDeadProcess(9); err != nil {
		// While normally we might "return err" here we're not going to
		// because if we can't stop the container by this point then
		// its probably because its already stopped. Meaning, between
		// the time of the IsRunning() call above and now it stopped.
		// Also, since the err return will be exec driver specific we can't
		// look for any particular (common) error that would indicate
		// that the process is already dead vs something else going wrong.
		// So, instead we'll give it up to 2 more seconds to complete and if
		// by that time the container is still running, then the error
		// we got is probably valid and so we return it to the caller.

		if container.IsRunning() {
			container.WaitStop(2 * time.Second)
			if container.IsRunning() {
				return err
			}
		}
	}

	// 2. Wait for the process to die, in last resort, try to kill the process directly
	if err := killProcessDirectly(container); err != nil {
		return err
	}

	container.WaitStop(-1 * time.Second)
	return nil
}

func (container *Container) Stop(seconds int) error {
	if !container.IsRunning() {
		return nil
	}

	// 1. Send a SIGTERM
	if err := container.killPossiblyDeadProcess(15); err != nil {
		logrus.Infof("Failed to send SIGTERM to the process, force killing")
		if err := container.killPossiblyDeadProcess(9); err != nil {
			return err
		}
	}

	// 2. Wait for the process to exit on its own
	if _, err := container.WaitStop(time.Duration(seconds) * time.Second); err != nil {
		logrus.Infof("Container %v failed to exit within %d seconds of SIGTERM - using the force", container.ID, seconds)
		// 3. If it doesn't, then send SIGKILL
		if err := container.Kill(); err != nil {
			container.WaitStop(-1 * time.Second)
			return err
		}
	}

	container.LogEvent("stop")
	return nil
}

func (container *Container) Restart(seconds int) error {
	// Avoid unnecessarily unmounting and then directly mounting
	// the container when the container stops and then starts
	// again
	if err := container.Mount(); err == nil {
		defer container.Unmount()
	}

	if err := container.Stop(seconds); err != nil {
		return err
	}

	if err := container.Start(); err != nil {
		return err
	}

	container.LogEvent("restart")
	return nil
}

func (container *Container) Resize(h, w int) error {
	if !container.IsRunning() {
		return fmt.Errorf("Cannot resize container %s, container is not running", container.ID)
	}
	if err := container.command.ProcessConfig.Terminal.Resize(h, w); err != nil {
		return err
	}
	container.LogEvent("resize")
	return nil
}

func (container *Container) Export() (archive.Archive, error) {
	if err := container.Mount(); err != nil {
		return nil, err
	}

	archive, err := archive.Tar(container.basefs, archive.Uncompressed)
	if err != nil {
		container.Unmount()
		return nil, err
	}
	arch := ioutils.NewReadCloserWrapper(archive, func() error {
		err := archive.Close()
		container.Unmount()
		return err
	})
	container.LogEvent("export")
	return arch, err
}

func (container *Container) Mount() error {
	return container.daemon.Mount(container)
}

func (container *Container) changes() ([]archive.Change, error) {
	return container.daemon.Changes(container)
}

func (container *Container) Changes() ([]archive.Change, error) {
	container.Lock()
	defer container.Unlock()
	return container.changes()
}

func (container *Container) GetImage() (*image.Image, error) {
	if container.daemon == nil {
		return nil, fmt.Errorf("Can't get image of unregistered container")
	}
	return container.daemon.graph.Get(container.ImageID)
}

func (container *Container) Unmount() error {
	return container.daemon.Unmount(container)
}

func (container *Container) hostConfigPath() (string, error) {
	return container.GetRootResourcePath("hostconfig.json")
}

func (container *Container) jsonPath() (string, error) {
	return container.GetRootResourcePath("config.json")
}

// This method must be exported to be used from the lxc template
// This directory is only usable when the container is running
func (container *Container) RootfsPath() string {
	return container.basefs
}

func validateID(id string) error {
	if id == "" {
		return fmt.Errorf("Invalid empty id")
	}
	return nil
}

func (container *Container) Copy(resource string) (rc io.ReadCloser, err error) {
	container.Lock()

	defer func() {
		if err != nil {
			// Wait to unlock the container until the archive is fully read
			// (see the ReadCloseWrapper func below) or if there is an error
			// before that occurs.
			container.Unlock()
		}
	}()

	if err := container.Mount(); err != nil {
		return nil, err
	}

	defer func() {
		if err != nil {
			// unmount any volumes
			container.UnmountVolumes(true)
			// unmount the container's rootfs
			container.Unmount()
		}
	}()

	if err := container.mountVolumes(); err != nil {
		return nil, err
	}

	basePath, err := container.GetResourcePath(resource)
	if err != nil {
		return nil, err
	}
	stat, err := os.Stat(basePath)
	if err != nil {
		return nil, err
	}
	var filter []string
	if !stat.IsDir() {
		d, f := filepath.Split(basePath)
		basePath = d
		filter = []string{f}
	} else {
		filter = []string{filepath.Base(basePath)}
		basePath = filepath.Dir(basePath)
	}
	archive, err := archive.TarWithOptions(basePath, &archive.TarOptions{
		Compression:  archive.Uncompressed,
		IncludeFiles: filter,
	})
	if err != nil {
		return nil, err
	}

	if err := container.PrepareStorage(); err != nil {
		container.Unmount()
		return nil, err
	}

	reader := ioutils.NewReadCloserWrapper(archive, func() error {
		err := archive.Close()
		container.CleanupStorage()
		container.UnmountVolumes(true)
		container.Unmount()
		container.Unlock()
		return err
	})
	container.LogEvent("copy")
	return reader, nil
}

// Returns true if the container exposes a certain port
func (container *Container) Exposes(p nat.Port) bool {
	_, exists := container.Config.ExposedPorts[p]
	return exists
}

func (container *Container) HostConfig() *runconfig.HostConfig {
	return container.hostConfig
}

func (container *Container) SetHostConfig(hostConfig *runconfig.HostConfig) {
	container.hostConfig = hostConfig
}

func (container *Container) getLogConfig() runconfig.LogConfig {
	cfg := container.hostConfig.LogConfig
	if cfg.Type != "" || len(cfg.Config) > 0 { // container has log driver configured
		if cfg.Type == "" {
			cfg.Type = jsonfilelog.Name
		}
		return cfg
	}
	// Use daemon's default log config for containers
	return container.daemon.defaultLogConfig
}

func (container *Container) getLogger() (logger.Logger, error) {
	cfg := container.getLogConfig()
	if err := logger.ValidateLogOpts(cfg.Type, cfg.Config); err != nil {
		return nil, err
	}
	c, err := logger.GetLogDriver(cfg.Type)
	if err != nil {
		return nil, fmt.Errorf("Failed to get logging factory: %v", err)
	}
	ctx := logger.Context{
		Config:              cfg.Config,
		ContainerID:         container.ID,
		ContainerName:       container.Name,
		ContainerEntrypoint: container.Path,
		ContainerArgs:       container.Args,
		ContainerImageID:    container.ImageID,
		ContainerImageName:  container.Config.Image,
		ContainerCreated:    container.Created,
	}

	// Set logging file for "json-logger"
	if cfg.Type == jsonfilelog.Name {
		ctx.LogPath, err = container.GetRootResourcePath(fmt.Sprintf("%s-json.log", container.ID))
		if err != nil {
			return nil, err
		}
	}
	return c(ctx)
}

func (container *Container) startLogging() error {
	cfg := container.getLogConfig()
	if cfg.Type == "none" {
		return nil // do not start logging routines
	}

	l, err := container.getLogger()
	if err != nil {
		return fmt.Errorf("Failed to initialize logging driver: %v", err)
	}

	copier, err := logger.NewCopier(container.ID, map[string]io.Reader{"stdout": container.StdoutPipe(), "stderr": container.StderrPipe()}, l)
	if err != nil {
		return err
	}
	container.logCopier = copier
	copier.Run()
	container.logDriver = l

	// set LogPath field only for json-file logdriver
	if jl, ok := l.(*jsonfilelog.JSONFileLogger); ok {
		container.LogPath = jl.LogPath()
	}

	return nil
}

func (container *Container) waitForStart() error {
	container.monitor = newContainerMonitor(container, container.hostConfig.RestartPolicy)

	// block until we either receive an error from the initial start of the container's
	// process or until the process is running in the container
	select {
	case <-container.monitor.startSignal:
	case err := <-promise.Go(container.monitor.Start):
		return err
	}

	return nil
}

func (container *Container) GetProcessLabel() string {
	// even if we have a process label return "" if we are running
	// in privileged mode
	if container.hostConfig.Privileged {
		return ""
	}
	return container.ProcessLabel
}

func (container *Container) GetMountLabel() string {
	if container.hostConfig.Privileged {
		return ""
	}
	return container.MountLabel
}

func (container *Container) Stats() (*execdriver.ResourceStats, error) {
	return container.daemon.Stats(container)
}

func (c *Container) LogDriverType() string {
	c.Lock()
	defer c.Unlock()
	if c.hostConfig.LogConfig.Type == "" {
		return c.daemon.defaultLogConfig.Type
	}
	return c.hostConfig.LogConfig.Type
}

func (container *Container) GetExecIDs() []string {
	return container.execCommands.List()
}

func (container *Container) Exec(execConfig *execConfig) error {
	container.Lock()
	defer container.Unlock()

	waitStart := make(chan struct{})

	callback := func(processConfig *execdriver.ProcessConfig, pid int) {
		if processConfig.Tty {
			// The callback is called after the process Start()
			// so we are in the parent process. In TTY mode, stdin/out/err is the PtySlave
			// which we close here.
			if c, ok := processConfig.Stdout.(io.Closer); ok {
				c.Close()
			}
		}
		close(waitStart)
	}

	// We use a callback here instead of a goroutine and an chan for
	// syncronization purposes
	cErr := promise.Go(func() error { return container.monitorExec(execConfig, callback) })

	// Exec should not return until the process is actually running
	select {
	case <-waitStart:
	case err := <-cErr:
		return err
	}

	return nil
}

func (container *Container) monitorExec(execConfig *execConfig, callback execdriver.StartCallback) error {
	var (
		err      error
		exitCode int
	)
	pipes := execdriver.NewPipes(execConfig.StreamConfig.stdin, execConfig.StreamConfig.stdout, execConfig.StreamConfig.stderr, execConfig.OpenStdin)
	exitCode, err = container.daemon.Exec(container, execConfig, pipes, callback)
	if err != nil {
		logrus.Errorf("Error running command in existing container %s: %s", container.ID, err)
	}
	logrus.Debugf("Exec task in container %s exited with code %d", container.ID, exitCode)
	if execConfig.OpenStdin {
		if err := execConfig.StreamConfig.stdin.Close(); err != nil {
			logrus.Errorf("Error closing stdin while running in %s: %s", container.ID, err)
		}
	}
	if err := execConfig.StreamConfig.stdout.Clean(); err != nil {
		logrus.Errorf("Error closing stdout while running in %s: %s", container.ID, err)
	}
	if err := execConfig.StreamConfig.stderr.Clean(); err != nil {
		logrus.Errorf("Error closing stderr while running in %s: %s", container.ID, err)
	}
	if execConfig.ProcessConfig.Terminal != nil {
		if err := execConfig.ProcessConfig.Terminal.Close(); err != nil {
			logrus.Errorf("Error closing terminal while running in container %s: %s", container.ID, err)
		}
	}
	// remove the exec command from the container's store only and not the
	// daemon's store so that the exec command can be inspected.
	container.execCommands.Delete(execConfig.ID)
	return err
}

func (c *Container) Attach(stdin io.ReadCloser, stdout io.Writer, stderr io.Writer) chan error {
	return attach(&c.StreamConfig, c.Config.OpenStdin, c.Config.StdinOnce, c.Config.Tty, stdin, stdout, stderr)
}

func (c *Container) AttachWithLogs(stdin io.ReadCloser, stdout, stderr io.Writer, logs, stream bool) error {

	if logs {
		logDriver, err := c.getLogger()
		if err != nil {
			logrus.Errorf("Error obtaining the logger %v", err)
			return err
		}
		if _, ok := logDriver.(logger.Reader); !ok {
			logrus.Errorf("cannot read logs for [%s] driver", logDriver.Name())
		} else {
			if cLog, err := logDriver.(logger.Reader).ReadLog(); err != nil {
				logrus.Errorf("Error reading logs %v", err)
			} else {
				dec := json.NewDecoder(cLog)
				for {
					l := &jsonlog.JSONLog{}

					if err := dec.Decode(l); err == io.EOF {
						break
					} else if err != nil {
						logrus.Errorf("Error streaming logs: %s", err)
						break
					}
					if l.Stream == "stdout" && stdout != nil {
						io.WriteString(stdout, l.Log)
					}
					if l.Stream == "stderr" && stderr != nil {
						io.WriteString(stderr, l.Log)
					}
				}
			}
		}
	}

	c.LogEvent("attach")

	//stream
	if stream {
		var stdinPipe io.ReadCloser
		if stdin != nil {
			r, w := io.Pipe()
			go func() {
				defer w.Close()
				defer logrus.Debugf("Closing buffered stdin pipe")
				io.Copy(w, stdin)
			}()
			stdinPipe = r
		}
		<-c.Attach(stdinPipe, stdout, stderr)
		// If we are in stdinonce mode, wait for the process to end
		// otherwise, simply return
		if c.Config.StdinOnce && !c.Config.Tty {
			c.WaitStop(-1 * time.Second)
		}
	}
	return nil
}

func attach(streamConfig *StreamConfig, openStdin, stdinOnce, tty bool, stdin io.ReadCloser, stdout io.Writer, stderr io.Writer) chan error {
	var (
		cStdout, cStderr io.ReadCloser
		cStdin           io.WriteCloser
		wg               sync.WaitGroup
		errors           = make(chan error, 3)
	)

	if stdin != nil && openStdin {
		cStdin = streamConfig.StdinPipe()
		wg.Add(1)
	}

	if stdout != nil {
		cStdout = streamConfig.StdoutPipe()
		wg.Add(1)
	}

	if stderr != nil {
		cStderr = streamConfig.StderrPipe()
		wg.Add(1)
	}

	// Connect stdin of container to the http conn.
	go func() {
		if stdin == nil || !openStdin {
			return
		}
		logrus.Debugf("attach: stdin: begin")
		defer func() {
			if stdinOnce && !tty {
				cStdin.Close()
			} else {
				// No matter what, when stdin is closed (io.Copy unblock), close stdout and stderr
				if cStdout != nil {
					cStdout.Close()
				}
				if cStderr != nil {
					cStderr.Close()
				}
			}
			wg.Done()
			logrus.Debugf("attach: stdin: end")
		}()

		var err error
		if tty {
			_, err = copyEscapable(cStdin, stdin)
		} else {
			_, err = io.Copy(cStdin, stdin)

		}
		if err == io.ErrClosedPipe {
			err = nil
		}
		if err != nil {
			logrus.Errorf("attach: stdin: %s", err)
			errors <- err
			return
		}
	}()

	attachStream := func(name string, stream io.Writer, streamPipe io.ReadCloser) {
		if stream == nil {
			return
		}
		defer func() {
			// Make sure stdin gets closed
			if stdin != nil {
				stdin.Close()
			}
			streamPipe.Close()
			wg.Done()
			logrus.Debugf("attach: %s: end", name)
		}()

		logrus.Debugf("attach: %s: begin", name)
		_, err := io.Copy(stream, streamPipe)
		if err == io.ErrClosedPipe {
			err = nil
		}
		if err != nil {
			logrus.Errorf("attach: %s: %v", name, err)
			errors <- err
		}
	}

	go attachStream("stdout", stdout, cStdout)
	go attachStream("stderr", stderr, cStderr)

	return promise.Go(func() error {
		wg.Wait()
		close(errors)
		for err := range errors {
			if err != nil {
				return err
			}
		}
		return nil
	})
}

// Code c/c from io.Copy() modified to handle escape sequence
func copyEscapable(dst io.Writer, src io.ReadCloser) (written int64, err error) {
	buf := make([]byte, 32*1024)
	for {
		nr, er := src.Read(buf)
		if nr > 0 {
			// ---- Docker addition
			// char 16 is C-p
			if nr == 1 && buf[0] == 16 {
				nr, er = src.Read(buf)
				// char 17 is C-q
				if nr == 1 && buf[0] == 17 {
					if err := src.Close(); err != nil {
						return 0, err
					}
					return 0, nil
				}
			}
			// ---- End of docker
			nw, ew := dst.Write(buf[0:nr])
			if nw > 0 {
				written += int64(nw)
			}
			if ew != nil {
				err = ew
				break
			}
			if nr != nw {
				err = io.ErrShortWrite
				break
			}
		}
		if er == io.EOF {
			break
		}
		if er != nil {
			err = er
			break
		}
	}
	return written, err
}

func (container *Container) networkMounts() []execdriver.Mount {
	var mounts []execdriver.Mount
	if container.ResolvConfPath != "" {
		label.SetFileLabel(container.ResolvConfPath, container.MountLabel)
		mounts = append(mounts, execdriver.Mount{
			Source:      container.ResolvConfPath,
			Destination: "/etc/resolv.conf",
			Writable:    !container.hostConfig.ReadonlyRootfs,
			Private:     true,
		})
	}
	if container.HostnamePath != "" {
		label.SetFileLabel(container.HostnamePath, container.MountLabel)
		mounts = append(mounts, execdriver.Mount{
			Source:      container.HostnamePath,
			Destination: "/etc/hostname",
			Writable:    !container.hostConfig.ReadonlyRootfs,
			Private:     true,
		})
	}
	if container.HostsPath != "" {
		label.SetFileLabel(container.HostsPath, container.MountLabel)
		mounts = append(mounts, execdriver.Mount{
			Source:      container.HostsPath,
			Destination: "/etc/hosts",
			Writable:    !container.hostConfig.ReadonlyRootfs,
			Private:     true,
		})
	}
	return mounts
}

func (container *Container) addBindMountPoint(name, source, destination string, rw bool) {
	container.MountPoints[destination] = &mountPoint{
		Name:        name,
		Source:      source,
		Destination: destination,
		RW:          rw,
	}
}

func (container *Container) addLocalMountPoint(name, destination string, rw bool) {
	container.MountPoints[destination] = &mountPoint{
		Name:        name,
		Driver:      volume.DefaultDriverName,
		Destination: destination,
		RW:          rw,
	}
}

func (container *Container) addMountPointWithVolume(destination string, vol volume.Volume, rw bool) {
	container.MountPoints[destination] = &mountPoint{
		Name:        vol.Name(),
		Driver:      vol.DriverName(),
		Destination: destination,
		RW:          rw,
		Volume:      vol,
	}
}

func (container *Container) isDestinationMounted(destination string) bool {
	return container.MountPoints[destination] != nil
}

func (container *Container) prepareMountPoints() error {
	for _, config := range container.MountPoints {
		if len(config.Driver) > 0 {
			v, err := createVolume(config.Name, config.Driver)
			if err != nil {
				return err
			}
			config.Volume = v
		}
	}
	return nil
}

func (container *Container) removeMountPoints() error {
	for _, m := range container.MountPoints {
		if m.Volume != nil {
			if err := removeVolume(m.Volume); err != nil {
				return err
			}
		}
	}
	return nil
}

func (container *Container) shouldRestart() bool {
	return container.hostConfig.RestartPolicy.Name == "always" ||
		(container.hostConfig.RestartPolicy.Name == "on-failure" && container.ExitCode != 0)
}

func (container *Container) mountVolumes() error {
	mounts, err := container.setupMounts()
	if err != nil {
		return err
	}

	for _, m := range mounts {
		dest, err := container.GetResourcePath(m.Destination)
		if err != nil {
			return err
		}

		var stat os.FileInfo
		stat, err = os.Stat(m.Source)
		if err != nil {
			return err
		}
		if err = fileutils.CreateIfNotExists(dest, stat.IsDir()); err != nil {
			return err
		}

		opts := "rbind,ro"
		if m.Writable {
			opts = "rbind,rw"
		}

		if err := mount.Mount(m.Source, dest, "bind", opts); err != nil {
			return err
		}
	}

	return nil
}

func (container *Container) copyImagePathContent(v volume.Volume, destination string) error {
	rootfs, err := symlink.FollowSymlinkInScope(filepath.Join(container.basefs, destination), container.basefs)
	if err != nil {
		return err
	}

	if _, err = ioutil.ReadDir(rootfs); err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	path, err := v.Mount()
	if err != nil {
		return err
	}

	if err := copyExistingContents(rootfs, path); err != nil {
		return err
	}

	return v.Unmount()
}
