package dockerfile

import (
	"fmt"
	"io"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/builder"
	containerpkg "github.com/docker/docker/container"
	"github.com/docker/docker/pkg/stringid"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context"
)

type containerManager struct {
	tmpContainers map[string]struct{}
	backend       builder.ExecBackend
}

// newContainerManager creates a new container backend
func newContainerManager(docker builder.ExecBackend) *containerManager {
	return &containerManager{
		backend:       docker,
		tmpContainers: make(map[string]struct{}),
	}
}

// Create a container
func (c *containerManager) Create(runConfig *container.Config, hostConfig *container.HostConfig, platform string) (container.ContainerCreateCreatedBody, error) {
	container, err := c.backend.ContainerCreate(types.ContainerCreateConfig{
		Config:     runConfig,
		HostConfig: hostConfig,
		Platform:   platform,
	})
	if err != nil {
		return container, err
	}
	c.tmpContainers[container.ID] = struct{}{}
	return container, nil
}

var errCancelled = errors.New("build cancelled")

// Run a container by ID
func (c *containerManager) Run(ctx context.Context, cID string, stdout, stderr io.Writer) (err error) {
	attached := make(chan struct{})
	errCh := make(chan error)
	go func() {
		errCh <- c.backend.ContainerAttachRaw(cID, nil, stdout, stderr, true, attached)
	}()
	select {
	case err := <-errCh:
		return err
	case <-attached:
	}

	finished := make(chan struct{})
	cancelErrCh := make(chan error, 1)
	go func() {
		select {
		case <-ctx.Done():
			logrus.Debugln("Build cancelled, killing and removing container:", cID)
			c.backend.ContainerKill(cID, 0)
			c.removeContainer(cID, stdout)
			cancelErrCh <- errCancelled
		case <-finished:
			cancelErrCh <- nil
		}
	}()

	if err := c.backend.ContainerStart(cID, nil, "", ""); err != nil {
		close(finished)
		logCancellationError(cancelErrCh, "error from ContainerStart: "+err.Error())
		return err
	}

	// Block on reading output from container, stop on err or chan closed
	if err := <-errCh; err != nil {
		close(finished)
		logCancellationError(cancelErrCh, "error from errCh: "+err.Error())
		return err
	}

	waitC, err := c.backend.ContainerWait(ctx, cID, containerpkg.WaitConditionNotRunning)
	if err != nil {
		close(finished)
		logCancellationError(cancelErrCh, fmt.Sprintf("unable to begin ContainerWait: %s", err))
		return err
	}

	if status := <-waitC; status.ExitCode() != 0 {
		close(finished)
		logCancellationError(cancelErrCh,
			fmt.Sprintf("a non-zero code from ContainerWait: %d", status.ExitCode()))
		return &statusCodeError{code: status.ExitCode(), err: err}
	}

	close(finished)
	return <-cancelErrCh
}

func logCancellationError(cancelErrCh chan error, msg string) {
	if cancelErr := <-cancelErrCh; cancelErr != nil {
		logrus.Debugf("Build cancelled (%v): ", cancelErr, msg)
	}
}

type statusCodeError struct {
	code int
	err  error
}

func (e *statusCodeError) Error() string {
	return e.err.Error()
}

func (e *statusCodeError) StatusCode() int {
	return e.code
}

func (c *containerManager) removeContainer(containerID string, stdout io.Writer) error {
	rmConfig := &types.ContainerRmConfig{
		ForceRemove:  true,
		RemoveVolume: true,
	}
	if err := c.backend.ContainerRm(containerID, rmConfig); err != nil {
		fmt.Fprintf(stdout, "Error removing intermediate container %s: %v\n", stringid.TruncateID(containerID), err)
		return err
	}
	return nil
}

// RemoveAll containers managed by this container manager
func (c *containerManager) RemoveAll(stdout io.Writer) {
	for containerID := range c.tmpContainers {
		if err := c.removeContainer(containerID, stdout); err != nil {
			return
		}
		delete(c.tmpContainers, containerID)
		fmt.Fprintf(stdout, "Removing intermediate container %s\n", stringid.TruncateID(containerID))
	}
}
