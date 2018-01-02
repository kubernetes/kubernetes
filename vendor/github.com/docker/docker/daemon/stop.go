package daemon

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/docker/docker/api/errors"
	containerpkg "github.com/docker/docker/container"
	"github.com/sirupsen/logrus"
)

// ContainerStop looks for the given container and terminates it,
// waiting the given number of seconds before forcefully killing the
// container. If a negative number of seconds is given, ContainerStop
// will wait for a graceful termination. An error is returned if the
// container is not found, is already stopped, or if there is a
// problem stopping the container.
func (daemon *Daemon) ContainerStop(name string, seconds *int) error {
	container, err := daemon.GetContainer(name)
	if err != nil {
		return err
	}
	if !container.IsRunning() {
		err := fmt.Errorf("Container %s is already stopped", name)
		return errors.NewErrorWithStatusCode(err, http.StatusNotModified)
	}
	if seconds == nil {
		stopTimeout := container.StopTimeout()
		seconds = &stopTimeout
	}
	if err := daemon.containerStop(container, *seconds); err != nil {
		return fmt.Errorf("Cannot stop container %s: %v", name, err)
	}
	return nil
}

// containerStop halts a container by sending a stop signal, waiting for the given
// duration in seconds, and then calling SIGKILL and waiting for the
// process to exit. If a negative duration is given, Stop will wait
// for the initial signal forever. If the container is not running Stop returns
// immediately.
func (daemon *Daemon) containerStop(container *containerpkg.Container, seconds int) error {
	if !container.IsRunning() {
		return nil
	}

	daemon.stopHealthchecks(container)

	stopSignal := container.StopSignal()
	// 1. Send a stop signal
	if err := daemon.killPossiblyDeadProcess(container, stopSignal); err != nil {
		// While normally we might "return err" here we're not going to
		// because if we can't stop the container by this point then
		// it's probably because it's already stopped. Meaning, between
		// the time of the IsRunning() call above and now it stopped.
		// Also, since the err return will be environment specific we can't
		// look for any particular (common) error that would indicate
		// that the process is already dead vs something else going wrong.
		// So, instead we'll give it up to 2 more seconds to complete and if
		// by that time the container is still running, then the error
		// we got is probably valid and so we force kill it.
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()

		if status := <-container.Wait(ctx, containerpkg.WaitConditionNotRunning); status.Err() != nil {
			logrus.Infof("Container failed to stop after sending signal %d to the process, force killing", stopSignal)
			if err := daemon.killPossiblyDeadProcess(container, 9); err != nil {
				return err
			}
		}
	}

	// 2. Wait for the process to exit on its own
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(seconds)*time.Second)
	defer cancel()

	if status := <-container.Wait(ctx, containerpkg.WaitConditionNotRunning); status.Err() != nil {
		logrus.Infof("Container %v failed to exit within %d seconds of signal %d - using the force", container.ID, seconds, stopSignal)
		// 3. If it doesn't, then send SIGKILL
		if err := daemon.Kill(container); err != nil {
			// Wait without a timeout, ignore result.
			_ = <-container.Wait(context.Background(), containerpkg.WaitConditionNotRunning)
			logrus.Warn(err) // Don't return error because we only care that container is stopped, not what function stopped it
		}
	}

	daemon.LogContainerEvent(container, "stop")
	return nil
}
