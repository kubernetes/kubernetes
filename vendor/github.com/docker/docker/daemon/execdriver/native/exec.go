// +build linux

package native

import (
	"fmt"
	"os"
	"os/exec"
	"syscall"

	"github.com/docker/docker/daemon/execdriver"
	"github.com/opencontainers/runc/libcontainer"
	_ "github.com/opencontainers/runc/libcontainer/nsenter"
	"github.com/opencontainers/runc/libcontainer/utils"
)

// TODO(vishh): Add support for running in privileged mode.
func (d *driver) Exec(c *execdriver.Command, processConfig *execdriver.ProcessConfig, pipes *execdriver.Pipes, startCallback execdriver.StartCallback) (int, error) {
	active := d.activeContainers[c.ID]
	if active == nil {
		return -1, fmt.Errorf("No active container exists with ID %s", c.ID)
	}

	p := &libcontainer.Process{
		Args: append([]string{processConfig.Entrypoint}, processConfig.Arguments...),
		Env:  c.ProcessConfig.Env,
		Cwd:  c.WorkingDir,
		User: processConfig.User,
	}

	config := active.Config()
	if err := setupPipes(&config, processConfig, p, pipes); err != nil {
		return -1, err
	}

	if err := active.Start(p); err != nil {
		return -1, err
	}

	if startCallback != nil {
		pid, err := p.Pid()
		if err != nil {
			p.Signal(os.Kill)
			p.Wait()
			return -1, err
		}
		startCallback(&c.ProcessConfig, pid)
	}

	ps, err := p.Wait()
	if err != nil {
		exitErr, ok := err.(*exec.ExitError)
		if !ok {
			return -1, err
		}
		ps = exitErr.ProcessState
	}
	return utils.ExitStatus(ps.Sys().(syscall.WaitStatus)), nil
}
