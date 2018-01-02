// +build !windows

package container

import (
	"testing"
	"time"

	containertypes "github.com/docker/docker/api/types/container"
	"github.com/docker/docker/container"
	"github.com/docker/docker/daemon"
	"github.com/docker/docker/daemon/events"
	"github.com/docker/swarmkit/api"
	"golang.org/x/net/context"
)

func TestHealthStates(t *testing.T) {

	// set up environment: events, task, container ....
	e := events.New()
	_, l, _ := e.Subscribe()
	defer e.Evict(l)

	task := &api.Task{
		ID:        "id",
		ServiceID: "sid",
		Spec: api.TaskSpec{
			Runtime: &api.TaskSpec_Container{
				Container: &api.ContainerSpec{
					Image: "image_name",
					Labels: map[string]string{
						"com.docker.swarm.task.id": "id",
					},
				},
			},
		},
		Annotations: api.Annotations{Name: "name"},
	}

	c := &container.Container{
		ID:   "id",
		Name: "name",
		Config: &containertypes.Config{
			Image: "image_name",
			Labels: map[string]string{
				"com.docker.swarm.task.id": "id",
			},
		},
	}

	daemon := &daemon.Daemon{
		EventsService: e,
	}

	controller, err := newController(daemon, task, nil)
	if err != nil {
		t.Fatalf("create controller fail %v", err)
	}

	errChan := make(chan error, 1)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// fire checkHealth
	go func() {
		err := controller.checkHealth(ctx)
		select {
		case errChan <- err:
		case <-ctx.Done():
		}
	}()

	// send an event and expect to get expectedErr
	// if expectedErr is nil, shouldn't get any error
	logAndExpect := func(msg string, expectedErr error) {
		daemon.LogContainerEvent(c, msg)

		timer := time.NewTimer(1 * time.Second)
		defer timer.Stop()

		select {
		case err := <-errChan:
			if err != expectedErr {
				t.Fatalf("expect error %v, but get %v", expectedErr, err)
			}
		case <-timer.C:
			if expectedErr != nil {
				t.Fatal("time limit exceeded, didn't get expected error")
			}
		}
	}

	// events that are ignored by checkHealth
	logAndExpect("health_status: running", nil)
	logAndExpect("health_status: healthy", nil)
	logAndExpect("die", nil)

	// unhealthy event will be caught by checkHealth
	logAndExpect("health_status: unhealthy", ErrContainerUnhealthy)
}
