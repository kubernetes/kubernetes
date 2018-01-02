package container

import (
	"time"

	"github.com/sirupsen/logrus"
)

const (
	loggerCloseTimeout = 10 * time.Second
)

// Reset puts a container into a state where it can be restarted again.
func (container *Container) Reset(lock bool) {
	if lock {
		container.Lock()
		defer container.Unlock()
	}

	if err := container.CloseStreams(); err != nil {
		logrus.Errorf("%s: %s", container.ID, err)
	}

	// Re-create a brand new stdin pipe once the container exited
	if container.Config.OpenStdin {
		container.StreamConfig.NewInputPipes()
	}

	if container.LogDriver != nil {
		if container.LogCopier != nil {
			exit := make(chan struct{})
			go func() {
				container.LogCopier.Wait()
				close(exit)
			}()
			select {
			case <-time.After(loggerCloseTimeout):
				logrus.Warn("Logger didn't exit in time: logs may be truncated")
			case <-exit:
			}
		}
		container.LogDriver.Close()
		container.LogCopier = nil
		container.LogDriver = nil
	}
}
