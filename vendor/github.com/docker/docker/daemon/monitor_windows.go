package daemon

import (
	"context"

	"github.com/docker/docker/container"
	"github.com/docker/docker/libcontainerd"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
)

// postRunProcessing starts a servicing container if required
func (daemon *Daemon) postRunProcessing(c *container.Container, ei libcontainerd.EventInfo) error {
	if ei.ExitCode == 0 && ei.UpdatePending {
		spec, err := daemon.createSpec(c)
		if err != nil {
			return err
		}
		// Turn on servicing
		spec.Windows.Servicing = true

		copts, err := daemon.getLibcontainerdCreateOptions(c)
		if err != nil {
			return err
		}

		// Create a new servicing container, which will start, complete the
		// update, and merge back the results if it succeeded, all as part of
		// the below function call.
		ctx := context.Background()
		svcID := c.ID + "_servicing"
		logger := logrus.WithField("container", svcID)
		if err := daemon.containerd.Create(ctx, svcID, spec, copts); err != nil {
			c.SetExitCode(-1)
			return errors.Wrap(err, "post-run update servicing failed")
		}
		_, err = daemon.containerd.Start(ctx, svcID, "", false, nil)
		if err != nil {
			logger.WithError(err).Warn("failed to run servicing container")
			if err := daemon.containerd.Delete(ctx, svcID); err != nil {
				logger.WithError(err).Warn("failed to delete servicing container")
			}
		} else {
			if _, _, err := daemon.containerd.DeleteTask(ctx, svcID); err != nil {
				logger.WithError(err).Warn("failed to delete servicing container task")
			}
			if err := daemon.containerd.Delete(ctx, svcID); err != nil {
				logger.WithError(err).Warn("failed to delete servicing container")
			}
		}
	}
	return nil
}
