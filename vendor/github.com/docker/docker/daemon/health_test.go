package daemon

import (
	"testing"
	"time"

	"github.com/docker/docker/api/types"
	containertypes "github.com/docker/docker/api/types/container"
	eventtypes "github.com/docker/docker/api/types/events"
	"github.com/docker/docker/container"
	"github.com/docker/docker/daemon/events"
)

func reset(c *container.Container) {
	c.State = &container.State{}
	c.State.Health = &container.Health{}
	c.State.Health.Status = types.Starting
}

func TestNoneHealthcheck(t *testing.T) {
	c := &container.Container{
		ID:   "container_id",
		Name: "container_name",
		Config: &containertypes.Config{
			Image: "image_name",
			Healthcheck: &containertypes.HealthConfig{
				Test: []string{"NONE"},
			},
		},
		State: &container.State{},
	}
	store, err := container.NewViewDB()
	if err != nil {
		t.Fatal(err)
	}
	daemon := &Daemon{
		containersReplica: store,
	}

	daemon.initHealthMonitor(c)
	if c.State.Health != nil {
		t.Error("Expecting Health to be nil, but was not")
	}
}

// FIXME(vdemeester) This takes around 3s… This is *way* too long
func TestHealthStates(t *testing.T) {
	e := events.New()
	_, l, _ := e.Subscribe()
	defer e.Evict(l)

	expect := func(expected string) {
		select {
		case event := <-l:
			ev := event.(eventtypes.Message)
			if ev.Status != expected {
				t.Errorf("Expecting event %#v, but got %#v\n", expected, ev.Status)
			}
		case <-time.After(1 * time.Second):
			t.Errorf("Expecting event %#v, but got nothing\n", expected)
		}
	}

	c := &container.Container{
		ID:   "container_id",
		Name: "container_name",
		Config: &containertypes.Config{
			Image: "image_name",
		},
	}

	store, err := container.NewViewDB()
	if err != nil {
		t.Fatal(err)
	}

	daemon := &Daemon{
		EventsService:     e,
		containersReplica: store,
	}

	c.Config.Healthcheck = &containertypes.HealthConfig{
		Retries: 1,
	}

	reset(c)

	handleResult := func(startTime time.Time, exitCode int) {
		handleProbeResult(daemon, c, &types.HealthcheckResult{
			Start:    startTime,
			End:      startTime,
			ExitCode: exitCode,
		}, nil)
	}

	// starting -> failed -> success -> failed

	handleResult(c.State.StartedAt.Add(1*time.Second), 1)
	expect("health_status: unhealthy")

	handleResult(c.State.StartedAt.Add(2*time.Second), 0)
	expect("health_status: healthy")

	handleResult(c.State.StartedAt.Add(3*time.Second), 1)
	expect("health_status: unhealthy")

	// Test retries

	reset(c)
	c.Config.Healthcheck.Retries = 3

	handleResult(c.State.StartedAt.Add(20*time.Second), 1)
	handleResult(c.State.StartedAt.Add(40*time.Second), 1)
	if c.State.Health.Status != types.Starting {
		t.Errorf("Expecting starting, but got %#v\n", c.State.Health.Status)
	}
	if c.State.Health.FailingStreak != 2 {
		t.Errorf("Expecting FailingStreak=2, but got %d\n", c.State.Health.FailingStreak)
	}
	handleResult(c.State.StartedAt.Add(60*time.Second), 1)
	expect("health_status: unhealthy")

	handleResult(c.State.StartedAt.Add(80*time.Second), 0)
	expect("health_status: healthy")
	if c.State.Health.FailingStreak != 0 {
		t.Errorf("Expecting FailingStreak=0, but got %d\n", c.State.Health.FailingStreak)
	}

	// Test start period

	reset(c)
	c.Config.Healthcheck.Retries = 2
	c.Config.Healthcheck.StartPeriod = 30 * time.Second

	handleResult(c.State.StartedAt.Add(20*time.Second), 1)
	if c.State.Health.Status != types.Starting {
		t.Errorf("Expecting starting, but got %#v\n", c.State.Health.Status)
	}
	if c.State.Health.FailingStreak != 0 {
		t.Errorf("Expecting FailingStreak=0, but got %d\n", c.State.Health.FailingStreak)
	}
	handleResult(c.State.StartedAt.Add(50*time.Second), 1)
	if c.State.Health.Status != types.Starting {
		t.Errorf("Expecting starting, but got %#v\n", c.State.Health.Status)
	}
	if c.State.Health.FailingStreak != 1 {
		t.Errorf("Expecting FailingStreak=1, but got %d\n", c.State.Health.FailingStreak)
	}
	handleResult(c.State.StartedAt.Add(80*time.Second), 0)
	expect("health_status: healthy")
	if c.State.Health.FailingStreak != 0 {
		t.Errorf("Expecting FailingStreak=0, but got %d\n", c.State.Health.FailingStreak)
	}
}
