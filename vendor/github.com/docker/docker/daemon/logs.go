package daemon

import (
	"errors"
	"strconv"
	"time"

	"golang.org/x/net/context"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/backend"
	containertypes "github.com/docker/docker/api/types/container"
	timetypes "github.com/docker/docker/api/types/time"
	"github.com/docker/docker/container"
	"github.com/docker/docker/daemon/logger"
	"github.com/sirupsen/logrus"
)

// ContainerLogs copies the container's log channel to the channel provided in
// the config. If ContainerLogs returns an error, no messages have been copied.
// and the channel will be closed without data.
//
// if it returns nil, the config channel will be active and return log
// messages until it runs out or the context is canceled.
func (daemon *Daemon) ContainerLogs(ctx context.Context, containerName string, config *types.ContainerLogsOptions) (<-chan *backend.LogMessage, bool, error) {
	lg := logrus.WithFields(logrus.Fields{
		"module":    "daemon",
		"method":    "(*Daemon).ContainerLogs",
		"container": containerName,
	})

	if !(config.ShowStdout || config.ShowStderr) {
		return nil, false, validationError{errors.New("You must choose at least one stream")}
	}
	container, err := daemon.GetContainer(containerName)
	if err != nil {
		return nil, false, err
	}

	if container.RemovalInProgress || container.Dead {
		return nil, false, stateConflictError{errors.New("can not get logs from container which is dead or marked for removal")}
	}

	if container.HostConfig.LogConfig.Type == "none" {
		return nil, false, logger.ErrReadLogsNotSupported{}
	}

	cLog, cLogCreated, err := daemon.getLogger(container)
	if err != nil {
		return nil, false, err
	}
	if cLogCreated {
		defer func() {
			if err = cLog.Close(); err != nil {
				logrus.Errorf("Error closing logger: %v", err)
			}
		}()
	}

	logReader, ok := cLog.(logger.LogReader)
	if !ok {
		return nil, false, logger.ErrReadLogsNotSupported{}
	}

	follow := config.Follow && !cLogCreated
	tailLines, err := strconv.Atoi(config.Tail)
	if err != nil {
		tailLines = -1
	}

	var since time.Time
	if config.Since != "" {
		s, n, err := timetypes.ParseTimestamps(config.Since, 0)
		if err != nil {
			return nil, false, err
		}
		since = time.Unix(s, n)
	}

	var until time.Time
	if config.Until != "" && config.Until != "0" {
		s, n, err := timetypes.ParseTimestamps(config.Until, 0)
		if err != nil {
			return nil, false, err
		}
		until = time.Unix(s, n)
	}

	readConfig := logger.ReadConfig{
		Since:  since,
		Until:  until,
		Tail:   tailLines,
		Follow: follow,
	}

	logs := logReader.ReadLogs(readConfig)

	// past this point, we can't possibly return any errors, so we can just
	// start a goroutine and return to tell the caller not to expect errors
	// (if the caller wants to give up on logs, they have to cancel the context)
	// this goroutine functions as a shim between the logger and the caller.
	messageChan := make(chan *backend.LogMessage, 1)
	go func() {
		// set up some defers
		defer logs.Close()

		// close the messages channel. closing is the only way to signal above
		// that we're doing with logs (other than context cancel i guess).
		defer close(messageChan)

		lg.Debug("begin logs")
		for {
			select {
			// i do not believe as the system is currently designed any error
			// is possible, but we should be prepared to handle it anyway. if
			// we do get an error, copy only the error field to a new object so
			// we don't end up with partial data in the other fields
			case err := <-logs.Err:
				lg.Errorf("Error streaming logs: %v", err)
				select {
				case <-ctx.Done():
				case messageChan <- &backend.LogMessage{Err: err}:
				}
				return
			case <-ctx.Done():
				lg.Debugf("logs: end stream, ctx is done: %v", ctx.Err())
				return
			case msg, ok := <-logs.Msg:
				// there is some kind of pool or ring buffer in the logger that
				// produces these messages, and a possible future optimization
				// might be to use that pool and reuse message objects
				if !ok {
					lg.Debug("end logs")
					return
				}
				m := msg.AsLogMessage() // just a pointer conversion, does not copy data

				// there could be a case where the reader stops accepting
				// messages and the context is canceled. we need to check that
				// here, or otherwise we risk blocking forever on the message
				// send.
				select {
				case <-ctx.Done():
					return
				case messageChan <- m:
				}
			}
		}
	}()
	return messageChan, container.Config.Tty, nil
}

func (daemon *Daemon) getLogger(container *container.Container) (l logger.Logger, created bool, err error) {
	container.Lock()
	if container.State.Running {
		l = container.LogDriver
	}
	container.Unlock()
	if l == nil {
		created = true
		l, err = container.StartLogger()
	}
	return
}

// mergeLogConfig merges the daemon log config to the container's log config if the container's log driver is not specified.
func (daemon *Daemon) mergeAndVerifyLogConfig(cfg *containertypes.LogConfig) error {
	if cfg.Type == "" {
		cfg.Type = daemon.defaultLogConfig.Type
	}

	if cfg.Config == nil {
		cfg.Config = make(map[string]string)
	}

	if cfg.Type == daemon.defaultLogConfig.Type {
		for k, v := range daemon.defaultLogConfig.Config {
			if _, ok := cfg.Config[k]; !ok {
				cfg.Config[k] = v
			}
		}
	}

	return logger.ValidateLogOpts(cfg.Type, cfg.Config)
}
