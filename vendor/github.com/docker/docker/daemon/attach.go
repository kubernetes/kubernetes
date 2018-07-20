package daemon

import (
	"context"
	"fmt"
	"io"

	"github.com/docker/docker/api/types/backend"
	"github.com/docker/docker/container"
	"github.com/docker/docker/container/stream"
	"github.com/docker/docker/daemon/logger"
	"github.com/docker/docker/pkg/stdcopy"
	"github.com/docker/docker/pkg/term"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
)

// ContainerAttach attaches to logs according to the config passed in. See ContainerAttachConfig.
func (daemon *Daemon) ContainerAttach(prefixOrName string, c *backend.ContainerAttachConfig) error {
	keys := []byte{}
	var err error
	if c.DetachKeys != "" {
		keys, err = term.ToBytes(c.DetachKeys)
		if err != nil {
			return validationError{errors.Errorf("Invalid detach keys (%s) provided", c.DetachKeys)}
		}
	}

	container, err := daemon.GetContainer(prefixOrName)
	if err != nil {
		return err
	}
	if container.IsPaused() {
		err := fmt.Errorf("container %s is paused, unpause the container before attach", prefixOrName)
		return stateConflictError{err}
	}
	if container.IsRestarting() {
		err := fmt.Errorf("container %s is restarting, wait until the container is running", prefixOrName)
		return stateConflictError{err}
	}

	cfg := stream.AttachConfig{
		UseStdin:   c.UseStdin,
		UseStdout:  c.UseStdout,
		UseStderr:  c.UseStderr,
		TTY:        container.Config.Tty,
		CloseStdin: container.Config.StdinOnce,
		DetachKeys: keys,
	}
	container.StreamConfig.AttachStreams(&cfg)

	inStream, outStream, errStream, err := c.GetStreams()
	if err != nil {
		return err
	}
	defer inStream.Close()

	if !container.Config.Tty && c.MuxStreams {
		errStream = stdcopy.NewStdWriter(errStream, stdcopy.Stderr)
		outStream = stdcopy.NewStdWriter(outStream, stdcopy.Stdout)
	}

	if cfg.UseStdin {
		cfg.Stdin = inStream
	}
	if cfg.UseStdout {
		cfg.Stdout = outStream
	}
	if cfg.UseStderr {
		cfg.Stderr = errStream
	}

	if err := daemon.containerAttach(container, &cfg, c.Logs, c.Stream); err != nil {
		fmt.Fprintf(outStream, "Error attaching: %s\n", err)
	}
	return nil
}

// ContainerAttachRaw attaches the provided streams to the container's stdio
func (daemon *Daemon) ContainerAttachRaw(prefixOrName string, stdin io.ReadCloser, stdout, stderr io.Writer, doStream bool, attached chan struct{}) error {
	container, err := daemon.GetContainer(prefixOrName)
	if err != nil {
		return err
	}
	cfg := stream.AttachConfig{
		UseStdin:   stdin != nil,
		UseStdout:  stdout != nil,
		UseStderr:  stderr != nil,
		TTY:        container.Config.Tty,
		CloseStdin: container.Config.StdinOnce,
	}
	container.StreamConfig.AttachStreams(&cfg)
	close(attached)
	if cfg.UseStdin {
		cfg.Stdin = stdin
	}
	if cfg.UseStdout {
		cfg.Stdout = stdout
	}
	if cfg.UseStderr {
		cfg.Stderr = stderr
	}

	return daemon.containerAttach(container, &cfg, false, doStream)
}

func (daemon *Daemon) containerAttach(c *container.Container, cfg *stream.AttachConfig, logs, doStream bool) error {
	if logs {
		logDriver, logCreated, err := daemon.getLogger(c)
		if err != nil {
			return err
		}
		if logCreated {
			defer func() {
				if err = logDriver.Close(); err != nil {
					logrus.Errorf("Error closing logger: %v", err)
				}
			}()
		}
		cLog, ok := logDriver.(logger.LogReader)
		if !ok {
			return logger.ErrReadLogsNotSupported{}
		}
		logs := cLog.ReadLogs(logger.ReadConfig{Tail: -1})
		defer logs.Close()

	LogLoop:
		for {
			select {
			case msg, ok := <-logs.Msg:
				if !ok {
					break LogLoop
				}
				if msg.Source == "stdout" && cfg.Stdout != nil {
					cfg.Stdout.Write(msg.Line)
				}
				if msg.Source == "stderr" && cfg.Stderr != nil {
					cfg.Stderr.Write(msg.Line)
				}
			case err := <-logs.Err:
				logrus.Errorf("Error streaming logs: %v", err)
				break LogLoop
			}
		}
	}

	daemon.LogContainerEvent(c, "attach")

	if !doStream {
		return nil
	}

	if cfg.Stdin != nil {
		r, w := io.Pipe()
		go func(stdin io.ReadCloser) {
			defer w.Close()
			defer logrus.Debug("Closing buffered stdin pipe")
			io.Copy(w, stdin)
		}(cfg.Stdin)
		cfg.Stdin = r
	}

	if !c.Config.OpenStdin {
		cfg.Stdin = nil
	}

	if c.Config.StdinOnce && !c.Config.Tty {
		// Wait for the container to stop before returning.
		waitChan := c.Wait(context.Background(), container.WaitConditionNotRunning)
		defer func() {
			<-waitChan // Ignore returned exit code.
		}()
	}

	ctx := c.InitAttachContext()
	err := <-c.StreamConfig.CopyStreams(ctx, cfg)
	if err != nil {
		if _, ok := err.(term.EscapeError); ok {
			daemon.LogContainerEvent(c, "detach")
		} else {
			logrus.Errorf("attach failed with error: %v", err)
		}
	}

	return nil
}
