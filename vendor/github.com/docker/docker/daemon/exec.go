package daemon

import (
	"fmt"
	"io"
	"io/ioutil"
	"strings"
	"sync"
	"time"

	"github.com/Sirupsen/logrus"
	"github.com/docker/docker/daemon/execdriver"
	"github.com/docker/docker/pkg/broadcastwriter"
	"github.com/docker/docker/pkg/ioutils"
	"github.com/docker/docker/pkg/pools"
	"github.com/docker/docker/pkg/stringid"
	"github.com/docker/docker/runconfig"
)

type execConfig struct {
	sync.Mutex
	ID            string
	Running       bool
	ExitCode      int
	ProcessConfig *execdriver.ProcessConfig
	StreamConfig
	OpenStdin  bool
	OpenStderr bool
	OpenStdout bool
	Container  *Container
	canRemove  bool
}

type execStore struct {
	s map[string]*execConfig
	sync.RWMutex
}

func newExecStore() *execStore {
	return &execStore{s: make(map[string]*execConfig, 0)}
}

func (e *execStore) Add(id string, execConfig *execConfig) {
	e.Lock()
	e.s[id] = execConfig
	e.Unlock()
}

func (e *execStore) Get(id string) *execConfig {
	e.RLock()
	res := e.s[id]
	e.RUnlock()
	return res
}

func (e *execStore) Delete(id string) {
	e.Lock()
	delete(e.s, id)
	e.Unlock()
}

func (e *execStore) List() []string {
	var IDs []string
	e.RLock()
	for id := range e.s {
		IDs = append(IDs, id)
	}
	e.RUnlock()
	return IDs
}

func (execConfig *execConfig) Resize(h, w int) error {
	return execConfig.ProcessConfig.Terminal.Resize(h, w)
}

func (d *Daemon) registerExecCommand(execConfig *execConfig) {
	// Storing execs in container in order to kill them gracefully whenever the container is stopped or removed.
	execConfig.Container.execCommands.Add(execConfig.ID, execConfig)
	// Storing execs in daemon for easy access via remote API.
	d.execCommands.Add(execConfig.ID, execConfig)
}

func (d *Daemon) getExecConfig(name string) (*execConfig, error) {
	if execConfig := d.execCommands.Get(name); execConfig != nil {
		if !execConfig.Container.IsRunning() {
			return nil, fmt.Errorf("Container %s is not running", execConfig.Container.ID)
		}
		return execConfig, nil
	}

	return nil, fmt.Errorf("No such exec instance '%s' found in daemon", name)
}

func (d *Daemon) unregisterExecCommand(execConfig *execConfig) {
	execConfig.Container.execCommands.Delete(execConfig.ID)
	d.execCommands.Delete(execConfig.ID)
}

func (d *Daemon) getActiveContainer(name string) (*Container, error) {
	container, err := d.Get(name)
	if err != nil {
		return nil, err
	}

	if !container.IsRunning() {
		return nil, fmt.Errorf("Container %s is not running", name)
	}
	if container.IsPaused() {
		return nil, fmt.Errorf("Container %s is paused, unpause the container before exec", name)
	}
	return container, nil
}

func (d *Daemon) ContainerExecCreate(config *runconfig.ExecConfig) (string, error) {
	// Not all drivers support Exec (LXC for example)
	if err := checkExecSupport(d.execDriver.Name()); err != nil {
		return "", err
	}

	container, err := d.getActiveContainer(config.Container)
	if err != nil {
		return "", err
	}

	cmd := runconfig.NewCommand(config.Cmd...)
	entrypoint, args := d.getEntrypointAndArgs(runconfig.NewEntrypoint(), cmd)

	user := config.User
	if len(user) == 0 {
		user = container.Config.User
	}

	processConfig := &execdriver.ProcessConfig{
		Tty:        config.Tty,
		Entrypoint: entrypoint,
		Arguments:  args,
		User:       user,
	}

	execConfig := &execConfig{
		ID:            stringid.GenerateRandomID(),
		OpenStdin:     config.AttachStdin,
		OpenStdout:    config.AttachStdout,
		OpenStderr:    config.AttachStderr,
		StreamConfig:  StreamConfig{},
		ProcessConfig: processConfig,
		Container:     container,
		Running:       false,
	}

	d.registerExecCommand(execConfig)

	container.LogEvent("exec_create: " + execConfig.ProcessConfig.Entrypoint + " " + strings.Join(execConfig.ProcessConfig.Arguments, " "))

	return execConfig.ID, nil

}

func (d *Daemon) ContainerExecStart(execName string, stdin io.ReadCloser, stdout io.Writer, stderr io.Writer) error {

	var (
		cStdin           io.ReadCloser
		cStdout, cStderr io.Writer
	)

	execConfig, err := d.getExecConfig(execName)
	if err != nil {
		return err
	}

	func() {
		execConfig.Lock()
		defer execConfig.Unlock()
		if execConfig.Running {
			err = fmt.Errorf("Error: Exec command %s is already running", execName)
		}
		execConfig.Running = true
	}()
	if err != nil {
		return err
	}

	logrus.Debugf("starting exec command %s in container %s", execConfig.ID, execConfig.Container.ID)
	container := execConfig.Container

	container.LogEvent("exec_start: " + execConfig.ProcessConfig.Entrypoint + " " + strings.Join(execConfig.ProcessConfig.Arguments, " "))

	if execConfig.OpenStdin {
		r, w := io.Pipe()
		go func() {
			defer w.Close()
			defer logrus.Debugf("Closing buffered stdin pipe")
			pools.Copy(w, stdin)
		}()
		cStdin = r
	}
	if execConfig.OpenStdout {
		cStdout = stdout
	}
	if execConfig.OpenStderr {
		cStderr = stderr
	}

	execConfig.StreamConfig.stderr = broadcastwriter.New()
	execConfig.StreamConfig.stdout = broadcastwriter.New()
	// Attach to stdin
	if execConfig.OpenStdin {
		execConfig.StreamConfig.stdin, execConfig.StreamConfig.stdinPipe = io.Pipe()
	} else {
		execConfig.StreamConfig.stdinPipe = ioutils.NopWriteCloser(ioutil.Discard) // Silently drop stdin
	}

	attachErr := attach(&execConfig.StreamConfig, execConfig.OpenStdin, true, execConfig.ProcessConfig.Tty, cStdin, cStdout, cStderr)

	execErr := make(chan error)

	// Note, the execConfig data will be removed when the container
	// itself is deleted.  This allows us to query it (for things like
	// the exitStatus) even after the cmd is done running.

	go func() {
		if err := container.Exec(execConfig); err != nil {
			execErr <- fmt.Errorf("Cannot run exec command %s in container %s: %s", execName, container.ID, err)
		}
	}()
	select {
	case err := <-attachErr:
		if err != nil {
			return fmt.Errorf("attach failed with error: %s", err)
		}
		break
	case err := <-execErr:
		return err
	}

	return nil
}

func (d *Daemon) Exec(c *Container, execConfig *execConfig, pipes *execdriver.Pipes, startCallback execdriver.StartCallback) (int, error) {
	exitStatus, err := d.execDriver.Exec(c.command, execConfig.ProcessConfig, pipes, startCallback)

	// On err, make sure we don't leave ExitCode at zero
	if err != nil && exitStatus == 0 {
		exitStatus = 128
	}

	execConfig.ExitCode = exitStatus
	execConfig.Running = false

	return exitStatus, err
}

// execCommandGC runs a ticker to clean up the daemon references
// of exec configs that are no longer part of the container.
func (d *Daemon) execCommandGC() {
	for range time.Tick(5 * time.Minute) {
		var (
			cleaned          int
			liveExecCommands = d.containerExecIds()
		)
		for id, config := range d.execCommands.s {
			if config.canRemove {
				cleaned++
				d.execCommands.Delete(id)
			} else {
				if _, exists := liveExecCommands[id]; !exists {
					config.canRemove = true
				}
			}
		}
		if cleaned > 0 {
			logrus.Debugf("clean %d unused exec commands", cleaned)
		}
	}
}

// containerExecIds returns a list of all the current exec ids that are in use
// and running inside a container.
func (d *Daemon) containerExecIds() map[string]struct{} {
	ids := map[string]struct{}{}
	for _, c := range d.containers.List() {
		for _, id := range c.execCommands.List() {
			ids[id] = struct{}{}
		}
	}
	return ids
}
