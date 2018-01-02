package daemon

import (
	"testing"
	"time"

	containertypes "github.com/docker/docker/api/types/container"
	eventtypes "github.com/docker/docker/api/types/events"
	"github.com/docker/docker/container"
	"github.com/docker/docker/daemon/events"
)

func TestLogContainerEventCopyLabels(t *testing.T) {
	e := events.New()
	_, l, _ := e.Subscribe()
	defer e.Evict(l)

	container := &container.Container{
		ID:   "container_id",
		Name: "container_name",
		Config: &containertypes.Config{
			Image: "image_name",
			Labels: map[string]string{
				"node": "1",
				"os":   "alpine",
			},
		},
	}
	daemon := &Daemon{
		EventsService: e,
	}
	daemon.LogContainerEvent(container, "create")

	if _, mutated := container.Config.Labels["image"]; mutated {
		t.Fatalf("Expected to not mutate the container labels, got %q", container.Config.Labels)
	}

	validateTestAttributes(t, l, map[string]string{
		"node": "1",
		"os":   "alpine",
	})
}

func TestLogContainerEventWithAttributes(t *testing.T) {
	e := events.New()
	_, l, _ := e.Subscribe()
	defer e.Evict(l)

	container := &container.Container{
		ID:   "container_id",
		Name: "container_name",
		Config: &containertypes.Config{
			Labels: map[string]string{
				"node": "1",
				"os":   "alpine",
			},
		},
	}
	daemon := &Daemon{
		EventsService: e,
	}
	attributes := map[string]string{
		"node": "2",
		"foo":  "bar",
	}
	daemon.LogContainerEventWithAttributes(container, "create", attributes)

	validateTestAttributes(t, l, map[string]string{
		"node": "1",
		"foo":  "bar",
	})
}

func validateTestAttributes(t *testing.T, l chan interface{}, expectedAttributesToTest map[string]string) {
	select {
	case ev := <-l:
		event, ok := ev.(eventtypes.Message)
		if !ok {
			t.Fatalf("Unexpected event message: %q", ev)
		}
		for key, expected := range expectedAttributesToTest {
			actual, ok := event.Actor.Attributes[key]
			if !ok || actual != expected {
				t.Fatalf("Expected value for key %s to be %s, but was %s (event:%v)", key, expected, actual, event)
			}
		}
	case <-time.After(10 * time.Second):
		t.Fatal("LogEvent test timed out")
	}
}
