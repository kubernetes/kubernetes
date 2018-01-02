package daemon

import (
	"fmt"
	"io/ioutil"
	"os"
	"testing"

	"github.com/docker/docker/api/types"
	containertypes "github.com/docker/docker/api/types/container"
	"github.com/docker/docker/container"
	"github.com/docker/docker/pkg/testutil"
	"github.com/stretchr/testify/require"
)

func newDaemonWithTmpRoot(t *testing.T) (*Daemon, func()) {
	tmp, err := ioutil.TempDir("", "docker-daemon-unix-test-")
	require.NoError(t, err)
	d := &Daemon{
		repository: tmp,
		root:       tmp,
	}
	d.containers = container.NewMemoryStore()
	return d, func() { os.RemoveAll(tmp) }
}

func newContainerWithState(state *container.State) *container.Container {
	return &container.Container{
		ID:     "test",
		State:  state,
		Config: &containertypes.Config{},
	}

}

// TestContainerDelete tests that a useful error message and instructions is
// given when attempting to remove a container (#30842)
func TestContainerDelete(t *testing.T) {
	tt := []struct {
		errMsg        string
		fixMsg        string
		initContainer func() *container.Container
	}{
		// a paused container
		{
			errMsg: "cannot remove a paused container",
			fixMsg: "Unpause and then stop the container before attempting removal or force remove",
			initContainer: func() *container.Container {
				return newContainerWithState(&container.State{Paused: true, Running: true})
			}},
		// a restarting container
		{
			errMsg: "cannot remove a restarting container",
			fixMsg: "Stop the container before attempting removal or force remove",
			initContainer: func() *container.Container {
				c := newContainerWithState(container.NewState())
				c.SetRunning(0, true)
				c.SetRestarting(&container.ExitStatus{})
				return c
			}},
		// a running container
		{
			errMsg: "cannot remove a running container",
			fixMsg: "Stop the container before attempting removal or force remove",
			initContainer: func() *container.Container {
				return newContainerWithState(&container.State{Running: true})
			}},
	}

	for _, te := range tt {
		c := te.initContainer()
		d, cleanup := newDaemonWithTmpRoot(t)
		defer cleanup()
		d.containers.Add(c.ID, c)

		err := d.ContainerRm(c.ID, &types.ContainerRmConfig{ForceRemove: false})
		testutil.ErrorContains(t, err, te.errMsg)
		testutil.ErrorContains(t, err, te.fixMsg)
	}
}

func TestContainerDoubleDelete(t *testing.T) {
	c := newContainerWithState(container.NewState())

	// Mark the container as having a delete in progress
	c.SetRemovalInProgress()

	d, cleanup := newDaemonWithTmpRoot(t)
	defer cleanup()
	d.containers.Add(c.ID, c)

	// Try to remove the container when its state is removalInProgress.
	// It should return an error indicating it is under removal progress.
	err := d.ContainerRm(c.ID, &types.ContainerRmConfig{ForceRemove: true})
	testutil.ErrorContains(t, err, fmt.Sprintf("removal of container %s is already in progress", c.ID))
}
